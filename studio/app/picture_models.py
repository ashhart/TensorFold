"""The picture models a person picks from: ``image_model`` on a job, IMAGE_MODEL to scripts/image.sh and edit.sh."""

from __future__ import annotations

from fastapi import HTTPException

# Qwen 2.1 Turbo is every install's picture model.
IMAGE_MODELS = {"qwen": ("Qwen 2.1 Turbo", "8 steps, the quickest draft; Qwen Research License (non-commercial)")}


def image_models() -> list[str]:
    """The picture models installed here, Qwen first."""

    return ["qwen"]


def image_model_env(params: dict) -> dict:
    """IMAGE_MODEL for a job's picture, refused when that model is not installed here."""

    name = params.get("image_model") or "qwen"
    if name not in IMAGE_MODELS:
        raise HTTPException(400, f"unknown picture model {name}: one of {', '.join(IMAGE_MODELS)}")
    if name not in image_models():
        raise HTTPException(400, f"{IMAGE_MODELS[name][0]} is not installed here")
    return {"IMAGE_MODEL": name}


def offered() -> list[dict]:
    """The installed picture models as the page lists them: [{"id", "label", "note"}, ...], Qwen first."""

    return [{"id": k, "label": IMAGE_MODELS[k][0], "note": IMAGE_MODELS[k][1]} for k in image_models()]
