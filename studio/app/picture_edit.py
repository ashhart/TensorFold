"""Edit a picture: the picture model re-draws a given picture as the prompt says (scripts/edit.sh)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import edit_plan  # noqa: E402  (the sizes and the noise seed, shared with the two render scripts)

# its author's 12 directions (30 degrees apart, clockwise seen from above) and 4 heights
DIRECTIONS = [("front", "front view", 0), ("front-right", "front-right view", 30), ("front-right-quarter", "front-right quarter view", 60),
              ("right", "right side view", 90), ("back-right-quarter", "back-right quarter view", 120), ("back-right", "back-right view", 150),
              ("back", "back view", 180), ("back-left", "back-left view", 210), ("back-left-quarter", "back-left quarter view", 240),
              ("left", "left side view", 270), ("front-left-quarter", "front-left quarter view", 300), ("front-left", "front-left view", 330)]
HEIGHTS = [("eye", "eye-level shot", 0), ("elevated", "elevated shot", 30), ("high", "high-angle shot", 60), ("top", "top-down shot", 90)]
# sets a person asks for by name: (direction, height, close-up)
SETS = {
    "turn": ("Five views around", [("front", "eye", False), ("front-right-quarter", "eye", False), ("right", "eye", False), ("back", "eye", False), ("left", "eye", False)]),
    "orbit": ("Full orbit, 12 views", [(d[0], "eye", False) for d in DIRECTIONS]),
    "heights": ("Four heights", [("front-right-quarter", h[0], False) for h in HEIGHTS]),
    "cover": ("Wide and close", [("front", "eye", False), ("front", "eye", True), ("front-right-quarter", "elevated", False), ("front-right-quarter", "elevated", True)]),
}
LIMIT = 12  # results from one job
STEPS = {"qwen": "8 steps"}
PICTURES = edit_plan.PICTURES  # pictures in one edit: the edited one and nine beside it
KINDS = (".png", ".jpg", ".jpeg", ".webp")
ANGLE_LORA = "qwen-multiple-angles"


def angle_prompt(direction: str, height: str, close: bool, trigger: str = "<mva>") -> str:
    """The caption for one camera position, as the LoRA's author writes it."""

    words = dict((key, text) for key, text, _ in DIRECTIONS), dict((key, text) for key, text, _ in HEIGHTS)
    if direction not in words[0] or height not in words[1]:
        raise HTTPException(400, "unknown camera direction or height")
    return f"{trigger} {words[0][direction]}, {words[1][height]}{' close-up' if close else ''}".strip()


def label(direction: str, height: str, close: bool) -> str:
    return f"{direction}_{height}{'_close' if close else ''}"


def attach(home: Path, pack: Path, spark: bool, lora_resolve, inside, relative, models=lambda: []):
    """``models``: the installed picture models, [{"id", "label", "note"}, ...], Qwen first (picture_models.py)."""

    router = APIRouter()
    prefix = Path(os.environ.get("PREFIX", Path.home() / ".local/opt/tensorfold-studio"))

    def model_dirs() -> tuple[Path, Path]:
        turbo = Path(os.environ.get("QWEN_MODEL_DIR", Path.home() / "qwen-models/Qwen-Image-2.1-Turbo"))
        return turbo, Path(os.environ.get("QWEN_BASE_DIR", turbo.parent / "Qwen-Image-2.1"))

    def ready(model: str = "qwen") -> tuple[bool, str]:
        """Whether a picture can be edited here with ``model``, and if not, what is missing."""

        if spark:
            models = Path(os.environ.get("SPARK_MODELS", Path.home() / "spark-studio-models")) / "Qwen-Image-2.1-Comfy"
            if not (models / "diffusion_models/qwen_image_2.1_turbo_bf16.safetensors").is_file():
                return False, "Qwen-Image-2.1-Turbo is not installed here: run oneshot-setup-spark.sh"
            return True, ""
        turbo, _ = model_dirs()
        if not (turbo / "processor").is_dir() or not (turbo / "vae").is_dir():
            return False, f"editing needs the whole Qwen-Image-2.1-Turbo checkpoint (with its processor folder) at {turbo}: run oneshot-setup.sh"
        site = next((prefix / "venv/lib").glob("python3*/site-packages"), None)
        if site is None or not (site / "mflux/models/qwen21/variants/edit/qwen_image_21_edit.py").is_file():
            return False, "the installed mflux has no Qwen-Image-2.1 edit pipeline: run oneshot-setup.sh --upgrade"
        return True, ""

    def base_ready() -> bool:
        if spark:
            models = Path(os.environ.get("SPARK_MODELS", Path.home() / "spark-studio-models")) / "Qwen-Image-2.1-Comfy"
            return (models / "diffusion_models/qwen_image_2.1_bf16.safetensors").is_file()
        return (model_dirs()[1] / "transformer").is_dir() and (model_dirs()[1] / "processor").is_dir()

    def memory_gib() -> int:
        try:
            if sys.platform == "darwin":
                import subprocess
                return int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout or 0) >> 30
            return next(int(line.split()[1]) for line in open("/proc/meminfo") if line.startswith("MemTotal")) >> 20
        except (OSError, ValueError, StopIteration):
            return 0

    @router.get("/api/edit/state")
    def state() -> dict:
        qwen, why = ready()
        offered = [{"id": "qwen", "label": "Qwen 2.1 Turbo", "note": "8 steps; image LoRAs and camera angles"}] if qwen else []
        return {"ready": bool(offered), "why": "" if offered else why, "models": offered, "qwen": qwen,
                "base": qwen and base_ready(), "limit": LIMIT, "pictures": PICTURES, "angle_lora": ANGLE_LORA,
                "directions": [{"id": key, "label": text.replace(" view", ""), "degrees": degrees} for key, text, degrees in DIRECTIONS],
                "heights": [{"id": key, "label": text.replace(" shot", ""), "degrees": degrees} for key, text, degrees in HEIGHTS],
                "sets": {key: {"label": name, "angles": [{"direction": d, "height": h, "close": c} for d, h, c in angles]} for key, (name, angles) in SETS.items()},
                "note": "Qwen-Image-2.1-Turbo, 8 steps a picture" + ("; the 40-step base model is installed too" if base_ready() else "")}

    def build(params: dict, job_id: str) -> tuple[list[str], dict, list[str]]:
        """(command, environment, output paths) for one edit job."""

        model = str(params.get("image_model") or "qwen")
        if model not in STEPS:
            raise HTTPException(400, f"unknown picture model {model}: one of {', '.join(STEPS)}")
        ok, why = ready(model)
        if not ok:
            raise HTTPException(400, why)
        source = inside(str(params.get("image") or ""))
        if not source.is_file() or source.suffix.lower() not in KINDS:
            raise HTTPException(400, "pick the picture to edit: a png, jpg or webp in the studio's folder")
        # further pictures the prompt draws on: image 2, image 3, ... to the model, in this order
        beside = [inside(str(path)) for path in (params.get("images") or []) if path]
        if len(beside) > PICTURES - 1:
            raise HTTPException(400, f"at most {PICTURES} pictures in one edit: the one that is edited and {PICTURES - 1} beside it")
        for path in beside:
            if not path.is_file() or path.suffix.lower() not in KINDS:
                raise HTTPException(400, f"no picture at {relative(path)}: a further picture is a png, jpg or webp in the studio's folder")
        picked = lora_resolve(params.get("loras"), "image")
        angles = params.get("angles") or []
        prompt = (params.get("prompt") or "").strip()
        folder = home / "scouts" / job_id
        if angles:
            if not any(p["id"] == ANGLE_LORA for p in picked):
                raise HTTPException(400, "a set of camera angles needs the Multiple Angles LoRA: install it in the LoRA hub (Fun) and tick it here")
            trigger = next(p["trigger"] for p in picked if p["id"] == ANGLE_LORA) or "<mva>"
            edits = []
            for angle in angles:
                direction, height, close = str(angle.get("direction")), str(angle.get("height") or "eye"), bool(angle.get("close"))
                edits.append((angle_prompt(direction, height, close, trigger), f"edit_{label(direction, height, close)}.png"))
        else:
            if not prompt:
                raise HTTPException(400, "say what changes in the picture, or pick camera angles")
            for p in picked:  # a LoRA's trigger opens the prompt unless it is there already
                if p["trigger"] and p["trigger"] not in prompt:
                    prompt = f"{p['trigger']} {prompt}"
            edits = [(prompt, "edit.png")]
        if len(edits) > LIMIT or len({name for _, name in edits}) != len(edits):
            raise HTTPException(400, f"at most {LIMIT} different results from one job")
        try:
            resolution = int(params.get("resolution") or 1024)
            seed = int(params.get("seed") or 0)
        except (TypeError, ValueError):
            raise HTTPException(400, "seed and size are numbers")
        if resolution % 32 or not 512 <= resolution <= 1536:
            raise HTTPException(400, "the size is a multiple of 32 between 512 and 1536")
        # the result's size: the picture's own unless the job says otherwise
        size = str(params.get("size") or "source")
        if size not in ("source", "model"):
            raise HTTPException(400, 'size is "source" (the picture\'s own size) or "model" (about the given resolution squared)')
        asked = {}
        if params.get("width") or params.get("height"):
            try:
                asked = {"WIDTH": int(params["width"]), "HEIGHT": int(params["height"])}
            except (KeyError, TypeError, ValueError):
                raise HTTPException(400, "a size of your own needs a width and a height")
            if edit_plan.refused(asked["WIDTH"], asked["HEIGHT"]):
                raise HTTPException(400, edit_plan.refused(asked["WIDTH"], asked["HEIGHT"]))
        base = bool(params.get("base"))
        if base and not base_ready():
            raise HTTPException(400, "the 40-step base model is not installed here: oneshot-setup" + ("-spark" if spark else "") + ".sh --image-edit fetches it")
        folder.mkdir(parents=True, exist_ok=True)
        plan = folder / "edits.json"
        plan.write_text(json.dumps([[text, str(folder / name)] for text, name in edits]))
        env = {"SET_FILE": str(plan), "IMAGE_MODEL": model, "SEED": seed, "RESOLUTION": resolution, "SIZE": size, "BASE": "1" if base else "",
               "LORA": "\n".join(f"{p['path']}:{p['strength']:g}" for p in picked),
               "IMAGES": "\n".join(str(path) for path in beside), **asked}
        if not spark and model == "qwen" and 0 < memory_gib() < 96:
            env["QUANTIZE"] = "8"  # bfloat16 peaks at 43 GiB with the text encoder and the picture's rows
            # memory grows with the pixels, so a large picture is edited at 1280x1280 at most there (not measured)
            env["MOST_PIXELS"] = 1280 * 1280
        return ["bash", str(pack / "scripts/edit.sh"), str(source)], env, [str(folder / name) for _, name in edits]

    return router, build
