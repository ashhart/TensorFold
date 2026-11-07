"""The picture models a person picks from: ``image_model`` on a job, IMAGE_MODEL to scripts/image.sh and edit.sh."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import HTTPException

from settings import SPARK, ZIG_ENGINE


# Qwen 2.1 Turbo is every install's; FLUX3-K adds FLUX 3 Image Klein's distilled (16 steps) and base (50) checkpoints.
IMAGE_MODELS = {"qwen": ("Qwen 2.1 Turbo", "8 steps, the quickest draft; Qwen Research License (non-commercial)"),
                "flux3k-fast": ("FLUX3-K Fast", "FLUX 3 Image Klein, 16 steps: sharper text, whole figures, better prompt following"),
                "flux3k-full": ("FLUX3-K Full", "FLUX 3 Image Klein base, 50 steps: the most detail, about 5x slower than Fast")}
FLUX3_FILES = {"flux3k-fast": "flux3_image_klein_sd_bf16.safetensors", "flux3k-full": "flux3_image_klein_base_bf16.safetensors"}


def image_models() -> list[str]:
    """The picture models installed here, Qwen first: FLUX3-K's ComfyUI files on a Spark, flux3-mlx's on a Mac."""

    found = ["qwen"]
    for name, unet in FLUX3_FILES.items():
        if SPARK:
            folder = Path(os.environ.get("SPARK_MODELS", Path.home() / "spark-studio-models")) / "FLUX3-K-Comfy"
            ready = (folder / "diffusion_models" / unet).is_file() and (folder / "system_prompts.json").is_file()
        else:
            folder = Path(os.environ.get("FLUX3_MODEL_DIR", Path.home() / "flux3-models"))
            ready = ((ZIG_ENGINE.parent / "flux3-venv" / "bin" / "python").exists()
                     and (folder / ("FLUX3-K-Fast" if name == "flux3k-fast" else "FLUX3-K-Full") / "model.safetensors").is_file())
        if ready:
            found.append(name)
    return found


def image_model_env(params: dict) -> dict:
    """IMAGE_MODEL for a job's picture, refused when that model is not installed here."""

    name = params.get("image_model") or "qwen"
    if name not in IMAGE_MODELS:
        raise HTTPException(400, f"unknown picture model {name}: one of {', '.join(IMAGE_MODELS)}")
    if name not in image_models():
        raise HTTPException(400, f"{IMAGE_MODELS[name][0]} is not installed here: oneshot-setup.sh --flux3 adds it")
    return {"IMAGE_MODEL": name}


def offered() -> list[dict]:
    """The installed picture models as the page lists them: [{"id", "label", "note"}, ...], Qwen first."""

    return [{"id": k, "label": IMAGE_MODELS[k][0], "note": IMAGE_MODELS[k][1]} for k in image_models()]
