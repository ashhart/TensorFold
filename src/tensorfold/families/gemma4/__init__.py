"""Gemma 4 (model_type ``gemma4``, text-only ``gemma4_text``), e.g. Gemma 4 26B-A4B.

``model``: mlx_lm's Gemma 4 blocks with the backbone and the tied vocabulary head apart, decoded by the serial
engine one token a round (no drafts). Prompts run mlx_lm's forward; one-token steps of the MoE models (26B-A4B)
run the fused kernels in ``kernels/gemma/v1`` (``TF_GEMMA4_FUSED=0``: mlx_lm's forward throughout). The vision
and audio towers are dropped by mlx_lm's sanitize (text only).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_TYPES = ("gemma4", "gemma4_text")
TITLE = "Gemma 4"
LANES = False
MODELS = ("mlx-community/gemma-4-26b-a4b-it-4bit",)
KERNEL_PACKAGE = "tensorfold.kernels.gemma.v1"
KERNEL_VERSION = "v1"


def load(model_dir: Path, **_: Any) -> tuple[Any, Any]:
    from tensorfold.families.gemma4.model import load as load_model

    return load_model(Path(model_dir))
