"""Kolibri 1 (model_type ``kolibri1``): Aleph Alpha's 78B-A3.5B MoE, MLX affine checkpoints, on the lane engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_TYPES = ("kolibri1",)
TITLE = "Kolibri 1"
LANES = True
MODELS = ("velaia/Kolibri-1-MLX-4bit",)
KERNEL_PACKAGE = "tensorfold.kernels.kolibri.v1"
KERNEL_VERSION = "v1"
# the decode runs Gemma 4's attention, glue and expert kernels and their matmuls
KERNEL_DEPENDENCIES = ("tensorfold.kernels.gemma.v1", "tensorfold.kernels.qwen.dense.v1.lane_qmm",
                       "tensorfold.kernels.nemotron.lightning.v1.rows")


def check(model_dir: str | Path) -> None:
    """Refuse, from config.json alone, a checkpoint the vendored forward does not read (MLX affine, 4 or 8 bits)."""

    from tensorfold.families import OWN_MODEL_HELP, describe_quantization, quantization, read_config

    config = read_config(model_dir)
    bits, group = quantization(config)
    if bits not in (4, 8) or group not in (32, 64):
        raise ValueError(f"Kolibri 1 reads MLX affine 4-bit or 8-bit weights in groups of 32 or 64 ({MODELS[0]}); "
                         f"this checkpoint has {describe_quantization(config)}. {OWN_MODEL_HELP}")
    kinds = set(config.get("layer_types") or ())
    if not kinds <= {"sliding_attention", "full_attention"}:
        raise ValueError(f"Kolibri 1's layers are sliding-window or full attention; this checkpoint has "
                         f"{', '.join(sorted(kinds))}. {OWN_MODEL_HELP}")


def load(model_dir: Path, *, lane_kernels: str = "auto", **_: Any) -> tuple[Any, Any]:
    """Decode rows through the row kernels: MLX's qmv loop a row by default; ``lane_kernels`` "on": the lane matmul.
    8-bit checkpoints decode through mlx_lm's forward, one row a step (the row kernels read 4-bit weights)."""

    from tensorfold.families import quantization, read_config
    from tensorfold.families.kolibri1.model import load as load_model

    bits, _group = quantization(read_config(model_dir))
    backend = None if bits != 4 else ("lane" if str(lane_kernels) == "on" else "rows")
    return load_model(Path(model_dir), backend=backend)


def engine_settings(model: Any) -> dict[str, Any]:
    """Rows a round verifies at most: the widest window checked exact at load."""

    width = int(getattr(model, "exact_width", 1) or 1)
    return {"max_rows": width, "max_draft": max(0, width - 1)}
