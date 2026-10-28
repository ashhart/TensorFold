"""Library actions on hub LoRAs (turnaround sheet, upscale, find/remove a person) as jobs; a Mac: sheet only."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from fastapi import APIRouter, HTTPException

KINDS = ("turnaround", "upscale", "mask", "remove")
PRESETS = Path(__file__).with_name("lora-presets.json")
# the LoRA each action is built on (its id in the hub), and what the action is called on the page
LORA = {"turnaround": "turnaround", "upscale": "video-upscaler", "remove": "person-remover", "mask": None}
TITLES = {"turnaround": "Turnaround sheet", "upscale": "Upscale", "remove": "Remove a person", "mask": "Find a person"}
SIZES = (512, 1024, 2048)                 # a turnaround view's side: 512 is what the LoRA was trained at
MEGAPIXELS = (1.0, 2.0, 3.0, 4.0)         # an upscale's area: 2 is about 1920x1080
WINDOWS = (22, 39, 56)                    # the remover's window: 22 is the author's proven default
MAC = {
    "upscale": "DGX Spark only. The upscaler is a ComfyUI node pack in PyTorch (a latent upscaler network, then a "
               "re-draw in tiles guided through the LoRA's own layout); the Mac build has no ComfyUI and its MLX "
               "engine has neither. On a Mac the 2K and QHD sizes double a clip with the 2x decoder.",
    "remove": "DGX Spark only. The removal needs SAM 3.1 to track the person, a source clip fed in as a frame-aligned "
              "guide and windows that carry 18 frames of history; the Mac's MLX engine has no segmentation model, no "
              "aligned guide and no history relay.",
}
MAC["mask"] = MAC["remove"]


def attach(home: Path, pack: Path, spark: bool, inside, relative):
    router = APIRouter()
    prefix = Path(os.environ.get("PREFIX", Path.home() / ".local/opt/tensorfold-studio"))
    models = Path(os.environ.get("SPARK_MODELS", Path.home() / "spark-studio-models"))

    def recipe(lora_id: str) -> dict:
        return next((r for r in json.loads(PRESETS.read_text()) if r["id"] == lora_id), {})

    def lora_file(lora_id: str) -> Path | None:
        """The LoRA's own file: in the hub (the MLX rewrite on a Mac), or on a Spark still in a local bundle."""

        hub = home / "loras" / (f"{lora_id}.safetensors" if spark else f"{lora_id}.mlx.safetensors")
        if not spark:
            return hub if hub.is_file() else None
        if hub.is_file():
            return hub
        name = recipe(lora_id).get("url", "").rsplit("/", 1)[-1]
        where = Path(os.environ.get("LORA_BUNDLE") or home / "lora-bundle").expanduser()
        if name and where.is_dir():
            return next(iter(where.glob(f"*/{lora_id}/{name}")), None)
        return None

    def missing(kind: str) -> str:
        """Why an action cannot run here, or '' when it can."""

        if not spark:
            if kind != "turnaround":
                return MAC[kind]
            model = Path(os.environ.get("H3_MODEL_DIR", Path.home() / "h3-models/MiniMax-H3"))
            if not (model / "Ref2VA/transformer").is_dir():
                return "MiniMax H3 Ref2VA is not installed: run `bash oneshot-setup.sh --ref2va`."
            return "" if lora_file("turnaround") else "Install Turnaround in the LoRA hub first."
        nodes = prefix / "ComfyUI" / "custom_nodes"
        need = {"turnaround": ["ComfyUI-H3-ContactSheet"], "upscale": ["ComfyUI-H3-Video-Upsampler"],
                "remove": ["h3-relay"], "mask": ["h3-relay"]}[kind]
        files = {"upscale": ["latent_upscale_models/minimax_h3_latent_upscaler_3d_bf16.safetensors"],
                 "remove": ["checkpoints/sam3.1_multiplex_fp16.safetensors"],
                 "mask": ["checkpoints/sam3.1_multiplex_fp16.safetensors"]}.get(kind, [])
        if any(not (nodes / n).is_dir() for n in need) or any(not (models / "Workflows-Comfy" / f).is_file() for f in files):
            return "Not installed yet: run `bash spark/setup-workflows.sh` in the pack's folder (three node packs and 2.5 GB of models), then restart ComfyUI."
        if not (models / "MiniMax-H3-Comfy/diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors").is_file():
            return "MiniMax H3 Ref2VA is not installed: run `bash oneshot-setup-spark.sh --ref2va`."
        if LORA[kind] and lora_file(LORA[kind]) is None:
            return f"Install {recipe(LORA[kind]).get('name', LORA[kind])} in the LoRA hub first."
        return ""

    @router.get("/api/actions")
    def state() -> dict:
        out = {}
        for kind in KINDS:
            why, card = missing(kind), recipe(LORA[kind] or "person-remover")
            out[kind] = {"title": TITLES[kind], "ready": not why, "why": why, "lora": LORA[kind],
                         "lora_name": card.get("name"), "author": card.get("author")}
        out["turnaround"].update(sizes=SIZES, size=1024)
        out["upscale"].update(megapixels=MEGAPIXELS, megapixel=2.0)
        out["remove"].update(windows=WINDOWS, window=22, steps=12)
        return out

    def media(params: dict, key: str, suffixes: tuple) -> Path:
        given = params.get(key)
        if not given:
            raise HTTPException(400, f"choose a {'clip' if key == 'video' else 'picture'} from the Library first")
        path = inside(given)
        if not path.is_file() or path.suffix.lower() not in suffixes:
            raise HTTPException(400, f"{given} is not a {'clip' if key == 'video' else 'picture'} in the studio folder")
        return path

    def number(params: dict, key: str, default, kind=float):
        try:
            return kind(params.get(key) if params.get(key) not in (None, "") else default)
        except (TypeError, ValueError):
            raise HTTPException(400, f"{key} is a number")

    def build(kind: str, params: dict, job_id: str) -> tuple[list[str], dict, list[str]]:
        """(command, environment, output paths) for one action's job; 400 with the reason when it cannot run."""

        why = missing(kind)
        if why:
            raise HTTPException(400, why)
        python = str(prefix / "venv/bin/python")
        script = [python, str(pack / "spark/spark_workflows.py"), kind]
        env = {"PREFIX": str(prefix)} if spark else {}
        lora = lora_file(LORA[kind]) if LORA[kind] else None
        if kind == "turnaround":
            picture = media(params, "image", (".png", ".jpg", ".jpeg", ".webp"))
            size = number(params, "size", 1024, int)
            if size not in SIZES:
                raise HTTPException(400, "a view is 512, 1024 or 2048 on a side")
            strength = number(params, "strength", 1.0)
            if not 0.3 <= strength <= 1.5:
                raise HTTPException(400, "the strength is between 0.3 and 1.5 (1.0 turns furthest, about 0.7 keeps the look closest)")
            folder = home / "scouts" / job_id
            # the same script on both machines: the MLX engine on a Mac, the author's ComfyUI nodes on a Spark
            env.update(SIZE=size, STRENGTH=f"{strength:g}", SEED=number(params, "seed", 0, int), TURNAROUND_LORA=str(lora))
            if (params.get("prompt") or "").strip():
                text = params["prompt"].strip()
                if "<Picture 1>" not in text:
                    raise HTTPException(400, "the instruction names the picture as <Picture 1>")
                env["PROMPT"] = text
            return (["bash", str(pack / "scripts/turnaround.sh"), str(picture), str(folder / "turn")], env,
                    [str(folder / "turn_sheet.png")] + [str(folder / f"turn_view{i}.png") for i in range(1, 6)])
        clip = media(params, "video", (".mp4", ".mov"))
        if kind == "upscale":
            area = number(params, "megapixels", 2.0)
            if area not in MEGAPIXELS:
                raise HTTPException(400, "the size is 1, 2, 3 or 4 megapixels")
            redraw = number(params, "denoise", 0.1)
            if not 0 <= redraw <= 0.4:
                raise HTTPException(400, "the re-draw is between 0 (only enlarge) and 0.4; 0.1 is the tested setting")
            target = home / "renders" / f"{job_id}.mp4"
            return script + ["--video", str(clip), "--lora", str(lora), "--megapixels", f"{area:g}", "--denoise", f"{redraw:g}",
                             "-o", str(target)], env, [str(target)]
        person = " ".join((params.get("person") or "").split())
        if not person:
            raise HTTPException(400, "say who to remove in a few words: man in a gray shirt")
        if len(person) > 200:
            raise HTTPException(400, "describe the person in a few words")
        folder = home / "masks" / job_id
        if kind == "mask":
            return script + ["--video", str(clip), "--person", person, "-o", str(folder / "person")], env, \
                [str(folder / "person_mask.mp4"), str(folder / "person_matte.mp4"), str(folder / "person_plate.png")]
        window, steps = number(params, "window", 22, int), number(params, "steps", 12, int)
        if window not in WINDOWS or not 4 <= steps <= 30:
            raise HTTPException(400, "the window is 22, 39 or 56 frames and the passes are between 4 and 30")
        target = home / "renders" / f"{job_id}.mp4"
        command = script + ["--video", str(clip), "--person", person, "--lora", str(lora), "--window", str(window),
                            "--steps", str(steps), "--seed", str(number(params, "seed", 904234, int)),
                            "--keep", str(folder / "person"), "-o", str(target)]
        if params.get("clean_frame"):
            command += ["--clean-frame", str(media(params, "clean_frame", (".png", ".jpg", ".jpeg", ".webp")))]
        if params.get("silent"):
            command.append("--silent")
        return command, env, [str(target), str(folder / "person_mask.mp4"), str(folder / "person_clean.png")]

    def title(kind: str, params: dict) -> str:
        name = Path(str(params.get("image") or params.get("video") or "")).name
        return f"{TITLES[kind]}: {name}" if name else TITLES[kind]

    def source_of(kind: str, params: dict) -> str | None:
        """The Library file an action's result came from, as the page shows it beside the result."""

        path = params.get("image") if kind == "turnaround" else params.get("video")
        return relative(inside(path)) if path else None

    return router, build, title, source_of
