"""Qwen-Image-2.1 checkpoint layout and transformer configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

LATENT_STRIDE = 16  # pixels per latent token along each side
PIXEL_STEP = 16     # width and height must be multiples of this


def pipeline_root(model_dir) -> Path | None:
    """The folder holding `model_index.json` for a Qwen-Image-2.1 pipeline, or None."""

    root = Path(model_dir)
    index = root / "model_index.json"
    if not index.is_file():
        return None
    with open(index) as handle:
        name = json.load(handle).get("_class_name")
    return root if name == "QwenImage21Pipeline" else None


@dataclass(frozen=True)
class DiTConfig:
    in_channels: int = 64
    out_channels: int = 64
    num_layers: int = 32
    attention_head_dim: int = 128
    num_attention_heads: int = 32
    context_in_dim: int = 4096
    mlp_ratio: int = 3
    axes_dims_rope: tuple[int, int, int] = (16, 56, 56)
    eps: float = 1e-6
    rope_theta: float = 10000.0
    timestep_dim: int = 256

    @property
    def hidden_size(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    @classmethod
    def from_checkpoint(cls, model_dir) -> "DiTConfig":
        root = pipeline_root(model_dir)
        if root is None:
            raise FileNotFoundError(f"{model_dir} is not a Qwen-Image-2.1 pipeline folder")
        with open(root / "transformer" / "config.json") as handle:
            raw = json.load(handle)
        if not raw.get("causal_condition", False) or raw.get("patch_size", 1) != 1:
            raise ValueError("only the causal-condition, unpatched Qwen-Image-2.1 transformer is supported")
        fields = {name: raw[name] for name in ("in_channels", "out_channels", "num_layers", "attention_head_dim",
                                               "num_attention_heads", "context_in_dim", "mlp_ratio", "eps")
                  if name in raw}
        if "axes_dims_rope" in raw:
            fields["axes_dims_rope"] = tuple(raw["axes_dims_rope"])
        config = cls(**fields)
        if sum(config.axes_dims_rope) != config.attention_head_dim:
            raise ValueError("the rotary axes must fill the attention head")
        return config


def latent_grid(width: int, height: int) -> tuple[int, int]:
    """(rows, columns) of latent tokens for an image of ``width`` x ``height`` pixels."""

    if width % PIXEL_STEP or height % PIXEL_STEP or width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive multiples of {PIXEL_STEP}, got {width}x{height}")
    return height // LATENT_STRIDE, width // LATENT_STRIDE
