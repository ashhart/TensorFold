"""Keys and Wasserman LoRA Hub: style LoRAs for MiniMax H3; no weights ship, each comes from its author's page."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from fastapi import APIRouter, HTTPException

import lora_flows
import lora_layout

# presets: lora-presets.json; user recipes: ~/TensorFoldStudio/loras/hub.json; a Mac rewrites files (convert-lora.py)
PRESETS = Path(__file__).with_name("lora-presets.json")
FIELDS = ("name", "author", "url", "page", "sha256", "trigger", "styles", "strength", "range", "base", "licence", "notes")
LIMIT = 4  # LoRAs on one clip
# the hub's sections, in the order they are shown (lora-groups.json, which the essential bundle's folders follow)
SECTIONS = json.loads(Path(__file__).with_name("lora-groups.json").read_text())
GROUPS = tuple((g["id"], g["name"]) for g in SECTIONS) + (("mine", "My LoRAs"),)
FOLDERS = {g["id"]: g["folder"] for g in SECTIONS}


# hosted builds for both engines (lora-host.json, LORA_HOST_BASE): install tries bundle, hosted build, author's file
HOSTED = json.loads(Path(__file__).with_name("lora-host.json").read_text()).get("loras", {})


def hosted(lora_id: str, platform: str) -> dict | None:
    base = os.environ.get("LORA_HOST_BASE", "").strip().rstrip("/")
    build = (HOSTED.get(lora_id) or {}).get(platform)
    return {"url": f"{base}/{build['file']}", "sha256": build["sha256"]} if base and build else None


def token() -> str:
    """The user's own Hugging Face token, for a file its author gates. Sent to huggingface.co only."""

    found = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
    saved = Path.home() / ".cache/huggingface/token"
    return found.strip() or (saved.read_text().strip() if saved.is_file() else "")


def check(path: Path, target: str = "video") -> int:
    """Adapted module count, after checking the file is a low-rank H3 LoRA (Qwen-Image for `image`)."""

    names = list(lora_layout.header(path)[0])
    if target == "image":
        pairs = [name for name in names if re.search(r"(^|\.)transformer_blocks\.\d+\..*lora_(A|down)", name)]
        if not pairs:
            raise ValueError("not a LoRA for Qwen-Image-2.1's transformer (no lora_A tensors on transformer_blocks.N)")
        return len(pairs)
    found = lora_layout.modules(names)
    if lora_layout.layout(names) == "unknown" or not any(re.search(r"(^|\.)blocks\.\d+\.(attn|mlp)\.", name) for name in found):
        raise ValueError("not a LoRA for MiniMax H3's transformer (no lora_A tensors on blocks.N.attn / blocks.N.mlp)")
    return len(found)


def rewritten(path: Path) -> bool:
    """On a DGX Spark: whether ComfyUI needs this file rewritten (a PEFT save, or a pair on an adaln projection)."""

    names = list(lora_layout.header(path)[0])
    return lora_layout.layout(names) == "peft" or any(lora_layout.ADALN.search(name) for name in lora_layout.modules(names))


def adaln_basis(folder: Path):
    """(basis, offset) carrying an adaln pair onto the pruned Ref2VA checkpoint (lora_layout.curve_basis), cached."""

    import numpy as np

    kept = folder / "ref2va-adaln-basis.npy"
    where = bundle()
    for place in (kept, Path(where).expanduser() / kept.name if where and not where.startswith(("http://", "https://")) else None):
        if place is not None and place.is_file():
            both = np.load(place)
            return both[:-1], both[-1]
    models = Path(os.environ.get("SPARK_MODELS", str(Path.home() / "spark-studio-models")))
    checkpoint = next(iter(sorted(models.glob("**/minimax_h3_ref2va_pruned_bf16.safetensors"))), None)
    table = lora_layout.curve_table(checkpoint) if checkpoint else None
    if table is None:
        return None, None  # no pruned Ref2VA here: a checkpoint with the full time embedder takes the pair as it is
    local = Path(os.environ.get("H3_MODEL_DIR", str(Path.home() / "h3-models/MiniMax-H3"))) / "Ref2VA/transformer/model-00001-of-00013.safetensors"
    if not local.is_file() and os.environ.get("LORA_OFFLINE") == "1":
        raise RuntimeError("this LoRA adapts the model's time layers, and moving it onto the DGX Spark's checkpoint takes 64 MB of "
                           "MiniMax H3 from MiniMax's page once (or ref2va-adaln-basis.npy in the LoRA bundle); LORA_OFFLINE=1 keeps this off the network")
    basis, offset, error = lora_layout.curve_basis(table, lora_layout.embedder(local if local.is_file() else None))
    if error > 1e-3:
        raise RuntimeError(f"the checkpoint's time table does not follow MiniMax H3's time embedder (fit error {error:.1e}): the LoRA is not moved onto it")
    np.save(kept, np.concatenate([basis, offset[None]], axis=0))
    return basis, offset


def bundle() -> str:
    """Where a LoRA bundle is: LORA_BUNDLE (a folder or base address), else lora-bundle in the studio's home."""

    named = os.environ.get("LORA_BUNDLE", "").strip()
    if named:
        return named.rstrip("/")
    local = Path(os.environ.get("STUDIO_HOME", str(Path.home() / "TensorFoldStudio"))) / "lora-bundle"
    return str(local) if local.is_dir() else ""


def bundled(recipe: dict) -> Path | None:
    """The recipe's file in a bundle folder on this machine, when it is there."""

    where = bundle()
    if not where or where.startswith(("http://", "https://")) or not recipe.get("url"):
        return None
    name, first = recipe["url"].rsplit("/", 1)[-1], (recipe.get("category") or ["mine"])[0]
    for found in (Path(where).expanduser() / FOLDERS.get(first, "") / recipe["id"] / name, Path(where).expanduser() / recipe["id"] / name):
        if found.is_file():
            return found
    return None


def download(recipe: dict, target: Path, state: dict) -> None:
    """The recipe's file into `target`, checked by sha256 and layout: from a LoRA bundle, else the web."""

    part = target.with_name(target.name + ".part")
    headers = {"User-Agent": "TensorFoldStudio"}
    if recipe.get("status") == "gated" and "huggingface.co/" in recipe["url"] and token():
        headers["Authorization"] = "Bearer " + token()
    try:
        digest = hashlib.sha256()
        local, where = bundled(recipe), bundle()
        try:
            if local:
                reply = open(local, "rb")
                state["of"] = local.stat().st_size
            else:
                reply = None
                if where.startswith(("http://", "https://")):
                    try:
                        mirror = f"{where}/{FOLDERS.get((recipe.get('category') or ['mine'])[0], '')}/{recipe['id']}/{recipe['url'].rsplit('/', 1)[-1]}"
                        reply = urllib.request.urlopen(urllib.request.Request(mirror, headers={"User-Agent": "TensorFoldStudio"}), timeout=60)
                    except (urllib.error.URLError, OSError):
                        reply = None  # not in that bundle: the author's page is next
                if reply is None and hosted(recipe["id"], "spark") and os.environ.get("LORA_OFFLINE") != "1":
                    try:  # the hosted copy of the author's file (the same bytes, checked below)
                        reply = urllib.request.urlopen(urllib.request.Request(hosted(recipe["id"], "spark")["url"], headers={"User-Agent": "TensorFoldStudio"}), timeout=60)
                    except (urllib.error.URLError, OSError):
                        reply = None
                if reply is None:
                    if os.environ.get("LORA_OFFLINE") == "1":
                        raise RuntimeError("not in the LoRA bundle, and LORA_OFFLINE=1 keeps this off the network")
                    reply = urllib.request.urlopen(urllib.request.Request(recipe["url"], headers=headers), timeout=60)
        except urllib.error.HTTPError as error:
            if error.code in (401, 403):
                raise RuntimeError("its author gates this file: " + (recipe.get("why") or "ask for access on the LoRA's page")) from None
            raise
        with reply, open(part, "wb") as out:
            if not local:
                state["of"] = int(reply.headers.get("Content-Length") or 0)
            while chunk := reply.read(1 << 20):
                out.write(chunk)
                digest.update(chunk)
                state["done"] += len(chunk)
        if recipe.get("sha256") and digest.hexdigest() != recipe["sha256"]:
            raise RuntimeError("the downloaded file does not match the recipe's sha256")
        check(part, recipe.get("target", "video"))
        part.replace(target)
    finally:
        part.unlink(missing_ok=True)


def convert(source: Path, ready: Path, pack: Path) -> None:
    """On a Mac: the author's file rewritten once into the layout the MLX engine merges at load."""

    model = os.environ.get("H3_MODEL_DIR", str(Path.home() / "h3-models/MiniMax-H3"))
    part = ready.with_name(ready.name.replace(".mlx.safetensors", ".part.safetensors"))  # the writer adds .safetensors to any other name
    done = subprocess.run([sys.executable, str(pack / "scripts/convert-lora.py"), str(source), str(part), model],
                          capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError((done.stderr or done.stdout).strip()[-300:])
    part.replace(ready)


def attach(home: Path, pack: Path, spark: bool):
    folder = home / "loras"
    folder.mkdir(parents=True, exist_ok=True)
    book = folder / "hub.json"
    router = APIRouter()
    busy: dict[str, dict] = {}  # id -> {"stage", "done", "of"} while a file is being fetched
    lock = threading.Lock()

    def added() -> list[dict]:
        try:
            return json.loads(book.read_text())
        except (OSError, ValueError):
            return []

    def recipes() -> list[dict]:
        built = [{"status": "ready", "category": ["mine"], **r, "builtin": True} for r in json.loads(PRESETS.read_text())]
        # a Library action's LoRA (actions.py) installs like the others; where the action cannot run, its card says why
        built = [r if r.get("status") != "action" or spark or "mac" in r.get("platforms", ()) else {**r, "status": "unsupported", "why": r.get("mac", r.get("why", ""))} for r in built]
        taken = {r["id"] for r in built}
        where = bundle()
        if where and not where.startswith(("http://", "https://")):  # a bundle newer than this pack brings its own cards
            try:
                extra = json.loads((Path(where).expanduser() / "bundle.json").read_text()).get("loras", [])
            except (OSError, ValueError, AttributeError):
                extra = []
            built += [{"status": "ready", "needs": "text", **r, "builtin": True} for r in extra if r.get("id") and r.get("url") and r["id"] not in taken]
            taken = {r["id"] for r in built}
        return built + [{"status": "ready", "category": ["mine"], "needs": "text", **r, "builtin": False} for r in added() if r.get("id") not in taken]

    def find(lora_id: str) -> dict:
        found = next((r for r in recipes() if r["id"] == lora_id), None)
        if found is None:
            raise HTTPException(404, f"no LoRA {lora_id} in the hub")
        return found

    def source(lora_id: str) -> Path:
        return folder / f"{lora_id}.safetensors"

    def images(lora_id: str) -> bool:
        """Whether this is a LoRA for the image model (Edit a picture), not for a clip."""

        return any(r["id"] == lora_id and r.get("target") == "image" for r in recipes())

    def ready(lora_id: str) -> Path:
        """The file a render reads: the author's on a Spark, the rewritten one on a Mac; an image LoRA as is."""

        if images(lora_id):
            return source(lora_id)
        if not spark:
            return folder / f"{lora_id}.mlx.safetensors"
        given, made = source(lora_id), folder / f"{lora_id}.comfy.safetensors"
        try:
            return made if made.is_file() or (given.is_file() and rewritten(given)) else given
        except (OSError, ValueError):
            return given

    def public(recipe: dict) -> dict:
        state = busy.get(recipe["id"])
        installed = ready(recipe["id"]).is_file() and not state
        # usable: its file is here and this machine has what it runs on (a clip, a Library action or Edit a picture)
        return {**recipe, "installed": installed, "installing": state, "usable": installed and recipe.get("status") != "unsupported",
                "flow": lora_flows.public(recipe["id"]) if recipe.get("flow") else None}

    def prepare(lora_id: str) -> None:
        if ready(lora_id).is_file() or images(lora_id):
            return
        if not spark:
            convert(source(lora_id), ready(lora_id), pack)
        elif source(lora_id).is_file() and rewritten(source(lora_id)):
            # ComfyUI's layout, and any adaln pair moved onto the pruned checkpoint (lora_layout.py)
            names = lora_layout.modules(list(lora_layout.header(source(lora_id))[0]))
            basis, offset = adaln_basis(folder) if any(lora_layout.ADALN.search(name) for name in names) else (None, None)
            lora_layout.to_comfy(source(lora_id), ready(lora_id), basis, offset)

    def prebuilt(recipe: dict, state: dict) -> bool:
        """On a Mac with no bundle file: the hosted build already in the MLX engine's layout, when there is one."""

        if recipe.get("target") == "image":
            return False  # read as its author wrote it: there is no build to fetch
        if not spark and not source(recipe["id"]).is_file():  # a Mac build an essential package put in the bundle
            where, name = bundle(), recipe["id"] + ".mlx.safetensors"
            local = None if not where or where.startswith(("http://", "https://")) else Path(where).expanduser() / FOLDERS.get((recipe.get("category") or ["mine"])[0], "") / recipe["id"] / name
            wanted = ((HOSTED.get(recipe["id"]) or {}).get("mac") or {}).get("sha256")
            if local and local.is_file() and wanted and hashlib.sha256(local.read_bytes()).hexdigest() == wanted:
                shutil.copyfile(local, ready(recipe["id"]))
                return True
        build = None if spark or source(recipe["id"]).is_file() or bundled(recipe) else hosted(recipe["id"], "mac")
        if not build or os.environ.get("LORA_OFFLINE") == "1":
            return False
        part = ready(recipe["id"]).with_name(recipe["id"] + ".hosted.part")
        try:
            digest = hashlib.sha256()
            with urllib.request.urlopen(urllib.request.Request(build["url"], headers={"User-Agent": "TensorFoldStudio"}), timeout=60) as reply, open(part, "wb") as out:
                state["of"] = int(reply.headers.get("Content-Length") or 0)
                while chunk := reply.read(1 << 20):
                    out.write(chunk)
                    digest.update(chunk)
                    state["done"] += len(chunk)
            if digest.hexdigest() != build["sha256"]:
                return False
            part.replace(ready(recipe["id"]))
            return True
        except (urllib.error.URLError, OSError):
            return False
        finally:
            part.unlink(missing_ok=True)

    def adopt() -> None:
        """At start: every usable preset a local bundle holds becomes installed, with no click and no network."""

        for recipe in recipes():
            lora_id = recipe["id"]
            packed = not spark and bundle() and not bundle().startswith("http") and (Path(bundle()).expanduser() / FOLDERS.get((recipe.get("category") or ["mine"])[0], "") / lora_id / (lora_id + ".mlx.safetensors")).is_file()
            if recipe.get("status") == "unsupported" or ready(lora_id).is_file() or not (bundled(recipe) or source(lora_id).is_file() or packed):
                continue
            with lock:
                if lora_id in busy:
                    continue
                busy[lora_id] = {"stage": "preparing", "done": 0, "of": 0}
            fetch(recipe)
            if busy.get(lora_id, {}).get("stage") == "failed":
                busy.pop(lora_id, None)  # a preset nobody asked for does not show an error; Install says why

    def fetch(recipe: dict) -> None:
        lora_id, state = recipe["id"], busy[recipe["id"]]
        try:
            if not prebuilt(recipe, state):
                if not source(lora_id).is_file():
                    download(recipe, source(lora_id), state)
                state["stage"] = "preparing"
                prepare(lora_id)
            busy.pop(lora_id, None)
        except Exception as error:  # noqa: BLE001 - shown on the hub's button
            busy[lora_id] = {"stage": "failed", "error": str(error)[-300:], "done": 0, "of": 0}

    @router.get("/api/loras")
    def listing() -> list[dict]:
        return [public(r) for r in recipes()]

    @router.get("/api/loras/groups")
    def groups() -> list[dict]:
        return [{"id": key, "name": name} for key, name in GROUPS]

    @router.post("/api/loras")
    def add(body: dict) -> dict:
        name = (body.get("name") or "").strip()
        url, upload = (body.get("url") or "").strip(), (body.get("file") or "").strip()
        if not name or not (url or upload):
            raise HTTPException(400, "a LoRA needs a name and either a download address or an uploaded file")
        if url and not url.startswith(("https://", "http://")):
            raise HTTPException(400, "the download address starts with https://")
        lora_id = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:48] or "lora"
        with lock:
            taken = {r["id"] for r in recipes()}
            if lora_id in taken:
                lora_id += "-" + time.strftime("%H%M%S")
            styles = body.get("styles") if isinstance(body.get("styles"), dict) else {}
            try:
                strength = float(body.get("strength") or 1.0)
            except (TypeError, ValueError):
                raise HTTPException(400, "strength is a number")
            recipe = {"id": lora_id, "name": name, "author": (body.get("author") or "").strip(), "url": url,
                      "page": (body.get("page") or "").strip(), "trigger": (body.get("trigger") or "").strip(),
                      "styles": {str(k).strip(): str(v).strip() for k, v in styles.items() if str(k).strip() and str(v).strip()},
                      "strength": strength, "base": "MiniMax H3", "licence": (body.get("licence") or "").strip(),
                      "notes": (body.get("notes") or "").strip()}
            if upload:
                given = (home / upload).resolve()
                if home.resolve() not in given.parents or not given.is_file():
                    raise HTTPException(400, "the uploaded file is not in the studio's folder")
                try:
                    check(given)
                    shutil.move(str(given), source(lora_id))
                    prepare(lora_id)
                except (ValueError, RuntimeError) as error:
                    source(lora_id).unlink(missing_ok=True)
                    raise HTTPException(400, f"that file cannot be used: {error}")
            book.write_text(json.dumps(added() + [recipe], indent=1))
        return public({**recipe, "builtin": False})

    @router.post("/api/loras/{lora_id}/install")
    def install(lora_id: str) -> dict:
        recipe = find(lora_id)
        if recipe.get("status") == "unsupported":
            raise HTTPException(400, f"{recipe['name']} is not supported in the Studio yet. {recipe.get('why', '')}".strip())
        with lock:
            state = busy.get(lora_id)
            if not ready(lora_id).is_file() and (state is None or state["stage"] == "failed"):
                if not recipe.get("url") and not source(lora_id).is_file():
                    raise HTTPException(400, "this recipe has no download address")
                busy[lora_id] = {"stage": "downloading", "done": 0, "of": 0}
                threading.Thread(target=fetch, args=(recipe,), daemon=True).start()
        return public(recipe)

    @router.delete("/api/loras/{lora_id}")
    def remove(lora_id: str) -> dict:
        """A user's recipe leaves the hub; a preset only loses its downloaded file. Files go to the trash folder."""

        recipe = find(lora_id)
        with lock:
            trash = home / "trash"
            trash.mkdir(exist_ok=True)
            for path in (source(lora_id), ready(lora_id)):
                if path.is_file():
                    shutil.move(str(path), trash / (time.strftime("%Y%m%d-%H%M%S-") + path.name))
            busy.pop(lora_id, None)
            if not recipe["builtin"]:
                book.write_text(json.dumps([r for r in added() if r.get("id") != lora_id], indent=1))
        return {"removed": lora_id, "recipe": not recipe["builtin"]}

    def resolve(picked, target: str = "video") -> list[dict]:
        """A job's `loras` as [{"path", "strength", "lead"}]; 400 if unknown, missing, or for the other `target`."""

        out = []
        for item in picked or []:
            recipe = find(str(item.get("id")))
            if recipe.get("target", "video") != target:
                raise HTTPException(400, f"{recipe['name']} is a LoRA for " + ("pictures: use it in Create, Edit a picture" if target == "video" else "clips, not for the image model"))
            if not ready(recipe["id"]).is_file():
                raise HTTPException(400, f"the LoRA {recipe['name']} is not installed: install it in the LoRA hub first")
            try:
                strength = float(item.get("strength") if item.get("strength") not in (None, "") else recipe.get("strength", 1.0))
            except (TypeError, ValueError):
                raise HTTPException(400, "a LoRA's strength is a number")
            low, high = (list(recipe.get("range") or []) + [0.1, 2])[:2] if len(recipe.get("range") or []) == 2 else (0.1, 2)
            low, high = (low if low < 0 else 0.0), max(high, 2)  # a slider LoRA takes minus values; the rest go up to 2
            if not strength or not low <= strength <= high:
                raise HTTPException(400, f"the strength of {recipe['name']} is between {low:g} and {high:g}, and not 0")
            if recipe.get("status") == "unsupported":
                raise HTTPException(400, f"{recipe['name']} is not supported in the Studio yet. {recipe.get('why', '')}".strip())
            if recipe.get("status") == "action":
                raise HTTPException(400, f"{recipe['name']} is not for a clip: {recipe.get('why', 'the Library has an action for it')}")
            phrase = (recipe.get("styles") or {}).get(str(item.get("style") or ""), "")
            lead = ", ".join(part for part in (recipe.get("trigger", ""), phrase) if part)
            out.append({"id": recipe["id"], "name": recipe["name"], "path": str(ready(recipe["id"])), "strength": strength, "trigger": recipe.get("trigger", ""), "lead": lead,
                        "comma": bool(recipe.get("comma"))})
        if len(out) > LIMIT:
            raise HTTPException(400, f"at most {LIMIT} LoRAs on one clip")
        return out

    def lead(prompt: str, picked: list[dict]) -> str:
        """The prompt with each LoRA's trigger and look phrase at the start of the shot, unless it is there already."""

        # most triggers open the shot as a sentence; one whose author writes "trigger, subject, ..." keeps its comma
        words = " ".join(p["lead"] + "," if p.get("comma") and not p["lead"].endswith(".") else p["lead"] if p["lead"].endswith(".") else p["lead"] + "."
                         for p in picked if p["lead"] and (not p["trigger"] or p["trigger"] not in prompt))
        if not words:
            return prompt
        return prompt.replace("[Shot 1] ", f"[Shot 1] {words} ", 1) if "[Shot 1] " in prompt else f"{words} {prompt}"

    threading.Thread(target=adopt, daemon=True).start()
    return router, resolve, lead


def main(argv: list[str]) -> int:
    """`loras.py install GROUP[,GROUP...]|all`: the ready presets of those groups, from their authors' pages."""

    if len(argv) < 2 or argv[0] != "install":
        print("usage: loras.py install all|" + ",".join(key for key, _ in GROUPS[:-1]), file=sys.stderr)
        return 2
    wanted = {key for key, _ in GROUPS} if argv[1] == "all" else set(argv[1].split(","))
    unknown = wanted - {key for key, _ in GROUPS}
    if unknown:
        print("no LoRA group " + ", ".join(sorted(unknown)), file=sys.stderr)
        return 2
    folder = Path(os.environ.get("STUDIO_HOME", str(Path.home() / "TensorFoldStudio"))) / "loras"
    folder.mkdir(parents=True, exist_ok=True)
    failed = 0
    for recipe in json.loads(PRESETS.read_text()):
        if not wanted & set(recipe.get("category") or []) or recipe.get("status", "ready") == "unsupported":
            continue
        target = folder / f"{recipe['id']}.safetensors"
        where = bundle()
        build = None if not where or where.startswith(("http://", "https://")) else Path(where).expanduser() / FOLDERS.get(recipe["category"][0], "") / recipe["id"] / f"{recipe['id']}.mlx.safetensors"
        made = folder / f"{recipe['id']}.mlx.safetensors"
        if sys.platform == "darwin" and not made.is_file() and build and build.is_file() \
                and hashlib.sha256(build.read_bytes()).hexdigest() == ((HOSTED.get(recipe["id"]) or {}).get("mac") or {}).get("sha256"):
            shutil.copyfile(build, made)  # an essential package's Mac build: installed as it is
            print(f"got   {recipe['name']} by {recipe['author']} ({made.stat().st_size >> 20} MB), the Mac build from the package")
            continue
        if made.is_file() and not target.is_file():
            print(f"have  {recipe['name']}")
            continue
        if not target.is_file() and not bundled(recipe) and os.environ.get("LORA_OFFLINE") == "1":
            continue  # only what the bundle holds
        if recipe.get("status") == "gated" and not target.is_file() and not bundled(recipe) and not token():
            print(f"gated {recipe['name']}: its author approves each download ({recipe['page']}); set HF_TOKEN once you have access")
            continue
        mlx = folder / f"{recipe['id']}.mlx.safetensors"
        try:
            if target.is_file():
                print(f"have  {recipe['name']}")
            else:
                came = "the bundle" if bundled(recipe) or bundle().startswith("http") else recipe["page"]
                download(recipe, target, {"done": 0, "of": 0})
                print(f"got   {recipe['name']} by {recipe['author']} ({target.stat().st_size >> 20} MB), from {came}")
            if sys.platform == "darwin" and not mlx.is_file() and recipe.get("target") != "image":
                convert(target, mlx, Path(__file__).resolve().parent.parent)
        except Exception as error:  # noqa: BLE001 - one author's page being down does not stop the rest
            failed += 1
            print(f"skip  {recipe['name']}: {error}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
