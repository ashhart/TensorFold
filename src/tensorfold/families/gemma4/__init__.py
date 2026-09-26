"""Gemma 4 (model_type ``gemma4``, text-only ``gemma4_text``), e.g. Gemma 4 26B-A4B.

``model``: mlx_lm's Gemma 4 blocks with the backbone and the tied vocabulary head apart, decoded by the serial
engine. No TensorFold kernels and no drafts: the output is mlx_lm's forward, one token a round. The vision and
audio towers are dropped by mlx_lm's sanitize (text only).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_TYPES = ("gemma4", "gemma4_text")
TITLE = "Gemma 4"
LANES = False
MODELS = ("mlx-community/gemma-4-26b-a4b-it-4bit",)


def load(model_dir: Path, **_: Any) -> tuple[Any, Any]:
    from tensorfold.families.gemma4.model import load as load_model

    return load_model(Path(model_dir))
