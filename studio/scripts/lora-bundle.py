"""The essential LoRA bundle: every preset of the Keys and Wasserman LoRA Hub in one folder, by the hub's groups."""
# Exit 0: every LoRA asked for is present; 1: some are unavailable; 2: usage or disk. Each file keeps its own licence.

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "app"
CARDS = ("README.md", "LICENSE", "LICENSE.md", "NOTICE", "WEIGHTS_LICENSE.md")
KEEP = ("id", "name", "author", "page", "url", "what", "needs", "trigger", "styles", "strength", "range", "base", "licence",
        "licence_tag", "notes", "status", "why", "comma", "sha256")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 22):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, target: Path, token: str, retries: int) -> None:
    """Download with resume: a .part file is continued with a Range request, and a failed try is repeated."""

    part = target.with_name(target.name + ".part")
    for attempt in range(retries + 1):
        headers = {"User-Agent": "TensorFoldStudio"}
        if token and "huggingface.co/" in url:
            headers["Authorization"] = "Bearer " + token
        have = part.stat().st_size if part.exists() else 0
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as reply:
                with open(part, "ab" if have and reply.status == 206 else "wb") as out:
                    while chunk := reply.read(1 << 20):
                        out.write(chunk)
            part.replace(target)
            return
        except urllib.error.HTTPError as error:
            if error.code == 416 and have:  # the part was already whole
                part.replace(target)
                return
            if error.code in (401, 403, 404, 410) or attempt == retries:
                raise
        except (urllib.error.URLError, OSError):
            if attempt == retries:
                raise
        time.sleep(2 * (attempt + 1))


def make_package(args, dest: Path, presets: list, hosted: dict, folder_of: dict, groups: list, say) -> int:
    """One plain tar per platform with the hosted set, a .sha256 beside each; parts of 1.9 GB above 2 GB."""

    import tarfile
    out = Path(args.make_package).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    notes = Path(args.notes_dir).expanduser() if args.notes_dir else None
    stamp, made = time.strftime("%Y-%m-%d"), 0
    for platform in ("spark", "mac"):
        members, menu = [], []
        for recipe in presets:
            if recipe["id"] not in hosted or platform not in hosted[recipe["id"]]:
                continue
            home = dest / folder_of[recipe["category"][0]] / recipe["id"]
            inside = f"{folder_of[recipe['category'][0]]}/{recipe['id']}"
            if platform == "spark" or recipe.get("target") == "image":  # a picture LoRA is read as its author wrote it, on a Mac too
                build, name = home / recipe["url"].rsplit("/", 1)[-1], recipe["url"].rsplit("/", 1)[-1]
            else:
                name = recipe["id"] + ".mlx.safetensors"
                build = next((d / name for d in ([Path(args.mlx_dir).expanduser()] if args.mlx_dir else []) + [dest.parent / "loras", home] if (d / name).is_file()), home / name)
            if not build.is_file():
                continue
            members.append((build, f"{inside}/{name}"))
            for card in CARDS:
                if (home / card).is_file():
                    members.append((home / card, f"{inside}/{card}"))
            if notes and (notes / inside / "SOURCE.md").is_file():
                members.append((notes / inside / "SOURCE.md", f"{inside}/SOURCE.md"))
            menu.append({**{k: recipe[k] for k in KEEP if k in recipe}, "category": recipe["category"], "folder": inside, "file": name,
                         "size": build.stat().st_size, "build_sha256": sha(build)})
        if not menu:
            say(f"  no {platform} builds here: no {platform} package")
            continue
        index = out / f".bundle-{platform}.json"
        index.write_text(json.dumps({"version": 1, "platform": platform, "built": stamp, "groups": groups, "loras": menu}, indent=1, ensure_ascii=False) + "\n")
        members.append((index, "bundle.json"))
        for extra in ("README.md", "LICENCES.md"):
            if notes and (notes / extra).is_file():
                members.append((notes / extra, extra))
        archive = out / f"keys-studio-lora-essential-{platform}-{args.tag or stamp}.tar"
        with tarfile.open(archive, "w") as tar:
            for path, name in members:
                tar.add(path, arcname=name)
        index.unlink()
        digest = sha(archive)
        archive.with_name(archive.name + ".sha256").write_text(f"{digest}  {archive.name}\n")
        say(f"  {archive}  {archive.stat().st_size / 1e9:.2f} GB  {len(menu)} LoRAs  sha256 {digest}")
        if archive.stat().st_size > 2_000_000_000:  # a GitHub release takes 2 GB an asset
            with open(archive, "rb") as whole:
                for n in range(99):
                    chunk = whole.read(1_900_000_000)
                    if not chunk:
                        break
                    archive.with_name(f"{archive.name}.part{n:02d}").write_bytes(chunk)
            say(f"    over 2 GB: also written as {archive.name}.partNN (cat them in order, then check the .sha256)")
        made += 1
    return 0 if made else 1


def unpack(package: str, dest: Path, presets: list, folder_of: dict, retries: int, say) -> str:
    """Fetch (resumable), check against its .sha256 and unpack an essential package; what went wrong, or ""."""

    import tarfile
    if package.startswith(("http://", "https://")):
        local = dest / package.rsplit("/", 1)[-1]
        try:
            if not local.is_file():
                fetch(package, local, "", retries)
            if not Path(str(local) + ".sha256").is_file():
                fetch(package + ".sha256", Path(str(local) + ".sha256"), "", retries)
        except (urllib.error.URLError, OSError) as error:
            return f"could not fetch {package} ({getattr(error, 'reason', error)})"
    else:
        local = Path(package).expanduser()
        if not local.is_file():
            parts = sorted(local.parent.glob(local.name + ".part*"))
            if not parts:
                return f"no file {local}"
            with open(local, "wb") as whole:
                for part in parts:
                    whole.write(part.read_bytes())
    stated = Path(str(local) + ".sha256")
    if not stated.is_file():
        return f"no {stated.name} beside the package: it is not unpacked unchecked"
    if sha(local) != stated.read_text().split()[0]:
        return "the package does not match its sha256"
    base = dest.resolve()
    with tarfile.open(local) as tar:
        for member in tar.getmembers():
            target = (dest / member.name).resolve()
            if not member.isfile() or base not in target.parents:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if member.name == "bundle.json":
                target = dest / "package.json"
            if not target.is_file() or target.stat().st_size != member.size:
                with tar.extractfile(member) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out, 1 << 22)
    say(f"  unpacked the package {local.name} into {dest}")
    return ""


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="lora-bundle.py", description="Build or check the essential LoRA bundle.")
    parser.add_argument("--dest", default=os.path.join(os.environ.get("STUDIO_HOME", str(Path.home() / "TensorFoldStudio")), "lora-bundle"))
    parser.add_argument("--categories", default="", help="group ids, comma separated (default: all)")
    parser.add_argument("--only", default="", help="LoRA ids, comma separated")
    parser.add_argument("--source", default=os.environ.get("LORA_BUNDLE", ""), help="another bundle to copy from: a folder or a base address")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--package", default=os.environ.get("LORA_PACKAGE", ""), help="an essential package (.tar file or address) to unpack into the bundle first")
    parser.add_argument("--make-package", default="", metavar="OUTDIR", help="write the package archives of the hosted set from this bundle")
    parser.add_argument("--tag", default="", help="with --make-package: the archives' name ends in this (default: today's date)")
    parser.add_argument("--host-json", default="", help="with --make-package: the hosted set to pack, when it is not app/lora-host.json yet")
    parser.add_argument("--mlx-dir", default="", help="with --make-package: where the Mac builds (ID.mlx.safetensors) are")
    parser.add_argument("--notes-dir", default="", help="with --make-package: a folder laid out like the bundle with SOURCE.md files, README.md and LICENCES.md to include")
    try:
        args = parser.parse_args(argv)
    except SystemExit:
        return 2
    groups = json.loads((APP / "lora-groups.json").read_text())
    folder_of = {g["id"]: g["folder"] for g in groups}
    presets = json.loads((APP / "lora-presets.json").read_text())
    wanted, only = set(filter(None, args.categories.split(","))), set(filter(None, args.only.split(",")))
    if wanted - set(folder_of) or only - {r["id"] for r in presets}:
        print("unknown: " + ", ".join(sorted((wanted - set(folder_of)) | (only - {r['id'] for r in presets})))
              + "\ngroups: " + ", ".join(folder_of), file=sys.stderr)
        return 2
    chosen = [r for r in presets if (not wanted or wanted & set(r["category"])) and (not only or r["id"] in only)]
    dest = Path(args.dest).expanduser()
    say = (lambda *a: None) if args.json else (lambda *a: print(*a, flush=True))
    saved = Path.home() / ".cache/huggingface/token"  # your own account's token, for a file its author gates
    token = os.environ.get("HF_TOKEN", "") or (saved.read_text().strip() if saved.is_file() else "")
    source = args.source.rstrip("/")
    # a LoRA with several adapters (variants) is one card and several files: each is fetched and checked like any other
    presets = [dict(r) for r in presets]
    extra = []
    for r in presets:
        for label, v in list((r.get("variants") or {}).items())[1:]:
            extra.append({**r, "url": v["url"], "sha256": v["sha256"], "size": v["size"], "id": r["id"], "_variant": label, "name": r["name"] + ": " + label.split(" (")[0]})
        if r.get("variants"):
            first = next(iter(r["variants"].values())); r.update(url=first["url"], sha256=first["sha256"], size=first["size"])
    chosen = [r for r in presets if (not wanted or wanted & set(r["category"])) and (not only or r["id"] in only)]
    chosen_extra = [r for r in extra if any(c["id"] == r["id"] for c in chosen)]
    if source and not source.startswith(("http://", "https://")) and Path(source).expanduser().resolve() == dest.resolve():
        source = ""

    def place(recipe: dict) -> Path:
        return dest / folder_of[recipe["category"][0]] / recipe["id"] / recipe["url"].rsplit("/", 1)[-1]

    def good(recipe: dict, path: Path) -> bool:
        return path.is_file() and (not recipe.get("sha256") or sha(path) == recipe["sha256"])

    hosted = json.loads((Path(args.host_json).expanduser() if args.host_json else APP / "lora-host.json").read_text()).get("loras", {})
    if args.make_package:
        return make_package(args, dest, presets, hosted, folder_of, groups, say)
    if args.package and not args.verify and not args.dry_run:
        dest.mkdir(parents=True, exist_ok=True)
        problem = unpack(args.package, dest, presets, folder_of, args.retries, say)
        if problem:
            say(f"  ! the package was not used: {problem}")
    if not args.verify and not args.dry_run:
        dest.mkdir(parents=True, exist_ok=True)
        for recipe in chosen:  # a bundle laid out flat by an earlier version moves into its group's folder
            flat = dest / recipe["id"]
            if flat.is_dir() and not place(recipe).parent.exists():
                place(recipe).parent.parent.mkdir(parents=True, exist_ok=True)
                flat.rename(place(recipe).parent)
    def packed(recipe: dict) -> bool:
        """A Mac build from an essential package stands in for the author's file (the hub installs it as it is)."""
        build = place(recipe).parent / (recipe["id"] + ".mlx.safetensors")
        return sys.platform == "darwin" and build.is_file() and sha(build) == ((hosted.get(recipe["id"]) or {}).get("mac") or {}).get("sha256")

    missing = [r for r in chosen if not place(r).is_file() and not packed(r)]
    need = sum(r.get("size") or 0 for r in missing)
    total = sum(r.get("size") or 0 for r in chosen)
    say(f"Essential LoRA bundle: {len(chosen)} LoRAs, {total / 1e9:.1f} GB in all; {len(missing)} to fetch ({need / 1e9:.1f} GB) into {dest}")
    if not args.verify and not args.dry_run:
        free = shutil.disk_usage(dest).free
        if need * 1.05 + (1 << 30) > free:
            print(f"not enough disk at {dest}: {need / 1e9:.1f} GB to fetch, {free / 1e9:.1f} GB free", file=sys.stderr)
            return 2
    states, started = {}, time.time()
    for recipe in chosen:
        target, name = place(recipe), recipe["url"].rsplit("/", 1)[-1]
        if target.is_file():
            states[recipe["id"]] = "already there" if good(recipe, target) else "checksum mismatch"
        elif packed(recipe):
            states[recipe["id"]] = "already there"
        elif args.verify or args.dry_run:
            states[recipe["id"]] = "to fetch" if args.dry_run else "missing"
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            tries = []
            if source.startswith(("http://", "https://")):
                tries.append((f"{source}/{folder_of[recipe['category'][0]]}/{recipe['id']}/{name}", ""))
            elif source:
                for local in (Path(source).expanduser() / folder_of[recipe["category"][0]] / recipe["id"] / name, Path(source).expanduser() / recipe["id"] / name):
                    if local.is_file():
                        shutil.copyfile(local, target)
                        for card in CARDS:
                            if (local.parent / card).is_file():
                                shutil.copyfile(local.parent / card, target.parent / card)
                        break
            if not target.is_file() and not (source and os.environ.get("LORA_BUNDLE_ONLY") == "1") and os.environ.get("LORA_OFFLINE") != "1":
                tries.append((recipe["url"], token))
            why = "unavailable at source"
            for url, auth in tries:
                if target.is_file():
                    break
                try:
                    fetch(url, target, auth, args.retries)
                except urllib.error.HTTPError as error:
                    why = "gated by its author" if error.code in (401, 403) else f"unavailable at source (HTTP {error.code})"
                except (urllib.error.URLError, OSError) as error:
                    why = f"unavailable at source ({getattr(error, 'reason', error)})"
            if not target.is_file():
                states[recipe["id"]] = why
            elif not good(recipe, target):
                target.rename(target.with_name(target.name + ".bad"))
                states[recipe["id"]] = "checksum mismatch"
            else:
                states[recipe["id"]] = "ok"
                if "huggingface.co/" in recipe["url"]:  # the author's card and licence, kept beside the file
                    base = recipe["url"].split("/resolve/")[0] + "/resolve/main/"
                    for card in CARDS:
                        if not (target.parent / card).is_file():
                            try:
                                fetch(base + card, target.parent / card, token, 0)
                            except (urllib.error.URLError, OSError):
                                pass
        if states[recipe["id"]] in ("missing", "unavailable at source") and (recipe.get("status") == "gated" or "gated" in (recipe.get("licence_tag") or "").lower()):
            states[recipe["id"]] = "gated by its author"  # not in this bundle because its author approves each download
        say(f"  {states[recipe['id']]:22} {recipe['name']} ({(recipe.get('size') or 0) / 1e6:.0f} MB)")
    for recipe in chosen_extra:
        target = place(recipe)
        if not target.is_file() and not args.verify and not args.dry_run and states.get(recipe["id"]) in ("ok", "already there"):
            try:
                fetch(recipe["url"], target, token, args.retries)
                if not good(recipe, target):
                    target.rename(target.with_name(target.name + ".bad"))
            except (urllib.error.URLError, OSError):
                pass
        say(f"  {'already there' if target.is_file() else 'missing':22} {recipe['name']} ({recipe['size'] / 1e6:.0f} MB)")
    present = lambda r: states.get(r["id"]) in ("ok", "already there")
    entries = []
    for order, recipe in enumerate(presets):
        if recipe not in chosen and not place(recipe).is_file():
            continue
        here = place(recipe).is_file() or packed(recipe)
        entries.append({**{k: recipe[k] for k in KEEP if k in recipe}, "category": recipe["category"], "order": order,
                        "folder": f"{folder_of[recipe['category'][0]]}/{recipe['id']}", "file": (recipe["url"].rsplit("/", 1)[-1] if place(recipe).is_file() else recipe["id"] + ".mlx.safetensors") if here else None,
                        "size": place(recipe).stat().st_size if place(recipe).is_file() else recipe.get("size"),
                        "fetched": time.strftime("%Y-%m-%d") if here else None,
                        "state": states.get(recipe["id"], "already there" if here else "missing")})
    if not args.verify and not args.dry_run:
        (dest / "bundle.json").write_text(json.dumps({"version": 1, "groups": groups, "loras": entries}, indent=1, ensure_ascii=False) + "\n")
    here = "mac" if sys.platform == "darwin" else "spark"

    def studio_use(r: dict) -> str:
        """How the Studio uses a preset: on a clip, in Edit a picture, or as a Library action (and on which machine)."""

        if r.get("status") == "gated":
            return "its author approves each download"
        if r.get("status") == "action":
            return "a Library action" + ("" if "mac" in r.get("platforms", ()) else " (DGX Spark only)")
        if r.get("status", "ready") != "ready":
            return "needs a Studio feature"
        return "works in Edit a picture" if r.get("target") == "image" else "works in the Studio"

    def usable(r: dict) -> bool:
        runs = r.get("status", "ready") in ("ready", "gated") or (r.get("status") == "action" and (here == "spark" or "mac" in r.get("platforms", ())))
        return runs and present(r)

    menu = [{"id": g["id"], "name": g["name"], "folder": g["folder"],
             "loras": [{"id": r["id"], "name": r["name"], "author": r["author"], "state": states[r["id"]],
                        "studio": studio_use(r)}
                       for r in chosen if g["id"] in r["category"]]} for g in groups]
    menu = [g for g in menu if g["loras"]]
    bad = [r["id"] for r in chosen if not present(r) and states[r["id"]] != "gated by its author" and states[r["id"]] != "to fetch"]
    gated = [r["id"] for r in chosen if states[r["id"]] == "gated by its author"]
    summary = {"dest": str(dest), "loras": len(chosen), "present": sum(present(r) for r in chosen), "fetched": sum(states[r["id"]] == "ok" for r in chosen),
               "bytes": sum(place(r).stat().st_size for r in chosen if place(r).is_file()), "seconds": round(time.time() - started, 1),
               "unavailable": bad, "gated": gated, "usable_here": sum(usable(r) for r in chosen), "platform": here, "states": states, "menu": menu}
    if args.json:
        print(json.dumps(summary, indent=1, ensure_ascii=False))
    else:
        say("\nThe hub's menu:")
        for g in menu:
            say(f"  {g['name']}")
            for item in g["loras"]:
                say(f"    {item['name']:26} by {item['author']:22} {item['state']}" + ("" if item["studio"] == "works in the Studio" else f" · {item['studio']}"))
        say(f"\n{summary['present']} of {summary['loras']} present, {summary['fetched']} fetched now, {summary['bytes'] / 1e9:.1f} GB, {summary['seconds']} s"
            + (f"; gated by their authors: {', '.join(gated)}" if gated else "") + (f"; UNAVAILABLE: {', '.join(bad)}" if bad else "")
            + f"; {summary['usable_here']} of {summary['loras']} usable on this {'Mac' if here == 'mac' else 'DGX Spark'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
