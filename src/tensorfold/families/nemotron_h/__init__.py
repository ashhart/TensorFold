"""Nemotron-H (model_type ``nemotron_h``), e.g. Nemotron 3.5 Lightning 30B-A3B.

``model``: mlx_lm's blocks with the backbone and head apart, decoded by the lane engine's family rounds
(``engine.lane_family``) through ``kernels`` (TensorFold's fused decode); ``mtp``: the MTP head (converted from
the BF16 release with ``mtp.convert``; the tested checkpoint ships it as ``mtp-4bit.safetensors``), whose chained
drafts every round verifies, as many as pay at the measured window costs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_TYPES = ("nemotron_h",)
TITLE = "Nemotron 3.5 Lightning"
LANES = True
MODELS = ("Vontra/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit",)
REQUIRED_FILES = {MODELS[0]: ("mtp-4bit.safetensors",)}
KERNEL_PACKAGE = "tensorfold.kernels.nemotron.lightning.v1"
KERNEL_VERSION = "v1"
# The long-context path also calls Qwen dense's lane attention kernel.
KERNEL_DEPENDENCIES = ("tensorfold.kernels.qwen.dense.v1.lane_attention",)
# MLX command buffers: up to 200 ops a buffer (MLX's default commits more often on this model's many small
# kernels); set by the CLI before MLX starts, unless the environment already sets them
MLX_ENV = {"MLX_MAX_OPS_PER_BUFFER": "200", "MLX_MAX_MB_PER_BUFFER": "100000"}


def load(model_dir: Path, *, mtp_head: str = "", mtp_drafts: int | None = None, **_: Any) -> tuple[Any, Any]:
    from tensorfold.families.nemotron_h.model import load as load_model

    return load_model(Path(model_dir), mtp_head=mtp_head, mtp_drafts=mtp_drafts)


def engine_settings(model: Any) -> dict[str, Any]:
    """Rows a round verifies at most: the widest window checked exact at load."""

    width = int(getattr(model, "exact_width", 1) or 1)
    return {"max_rows": width, "max_draft": max(0, width - 1)}
