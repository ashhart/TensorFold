"""Qwen-Image-2.1 text-to-image generation (MLX); a development path, no engine entry point yet."""

from __future__ import annotations

# Diffusers layout: `model_index.json` at the root and one `config.json` per component, so there is no
# `config.json["model_type"]`. Not a lane family and no `load`: the model denoises image latents.
MODEL_TYPES = ("qwen_image_21",)
TITLE = "Qwen-Image-2.1"
LANES = False
MODELS = ()
PIPELINE_CLASS = "QwenImage21Pipeline"
REQUIRED_COMPONENTS = ("transformer", "text_encoder", "processor", "vae")


def is_checkpoint(model_dir) -> bool:
    """Whether ``model_dir`` is a Qwen-Image-2.1 pipeline folder."""

    from .config import pipeline_root

    return pipeline_root(model_dir) is not None
