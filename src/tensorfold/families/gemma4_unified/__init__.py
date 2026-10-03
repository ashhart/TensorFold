"""Gemma 4 unified (model_type ``gemma4_unified``, the 12B): its dense text decoder on the lane engine; text only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_TYPES = ("gemma4_unified", "gemma4_unified_text")
TITLE = "Gemma 4 unified (12B)"
LANES = True
MODELS = ("unigilby/gemma-4-12B-it-oQ8e",)   # an oQe in groups of 64 from google/gemma-4-12B-it
DRAFTER = "mlx-community/gemma-4-12B-it-qat-assistant-4bit"
KERNEL_PACKAGE = "tensorfold.kernels.gemma.dense.v1"
KERNEL_VERSION = "v1"
# the model, caches, attention and RoPE are Gemma 4's; the projections run the Qwen dense lane and row kernels
KERNEL_DEPENDENCIES = ("tensorfold.families.gemma4.model", "tensorfold.families.gemma4.cache",
                       "tensorfold.kernels.gemma.v1.attention", "tensorfold.kernels.gemma.v1.glue",
                       "tensorfold.kernels.gemma.v1.decode", "tensorfold.kernels.qwen.dense.v1.lane_qmm",
                       "tensorfold.kernels.qwen.dense.v1.row_matmul", "tensorfold.kernels.qwen.dense.v1.simd_qmm",
                       "tensorfold.kernels.qwen.dense.v1.simd_qmm_bits", "tensorfold.kernels.qwen.dense.v1.affine_rows")

# widths the decode projections read (MLX affine): every oQ width; groups of 64 (4-bit also 32)
_READS = {2: (64,), 3: (64,), 4: (32, 64), 5: (64,), 6: (64,), 8: (64,)}


def _text(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("text_config") or config


def check(model_dir: str | Path) -> None:
    """Refuse, from config.json alone, a checkpoint the dense kernels do not read (MoE, per-layer inputs, a width)."""

    from tensorfold.families import OWN_MODEL_HELP, describe_quantization, layer_quantization, quantization, read_config

    config = read_config(model_dir)
    text = _text(config)
    missing = [what for what, ok in (
        ("a dense MLP in every layer (no MoE block)", not text.get("enable_moe_block")),
        ("no per-layer inputs", not int(text.get("hidden_size_per_layer_input") or 0)),
        ("no shared-KV layers", not int(text.get("num_kv_shared_layers") or 0)),
        ("head dims a multiple of 64", all(int(text.get(k) or 64) % 64 == 0 for k in ("head_dim", "global_head_dim"))),
        ("a hidden size a multiple of 256", int(text.get("hidden_size") or 0) % 256 == 0),
    ) if not ok]
    if missing:
        raise ValueError(f"TensorFold's Gemma 4 unified kernels cover the dense 12B ({MODELS[0]}); this checkpoint "
                         "lacks " + ", ".join(missing) + f". {OWN_MODEL_HELP}")
    bits, group = quantization(config)
    if bits is None or group not in _READS.get(bits, ()):
        raise ValueError(f"Gemma 4 unified's kernels read MLX affine weights of 2-8 bits in groups of 64 (4-bit: 32 "
                         f"or 64); this checkpoint has {describe_quantization(config)}. {OWN_MODEL_HELP}")
    for path, (b, g, mode) in layer_quantization(config).items():
        if mode != "affine" or g not in _READS.get(b, ()):
            raise ValueError(f"Gemma 4 unified's kernels read MLX affine 2-8-bit weights; {path} has {b}-bit "
                             f"{mode} weights in groups of {g}. {OWN_MODEL_HELP}")


def load(model_dir: Path, *, lane_kernels: str = "auto", drafter: str = "", mtp_drafts: int = 0,
         **_: Any) -> tuple[Any, Any]:
    """The lane matmul with tensor units (``lane_kernels``), else simd rows; ``drafter``: an MTP assistant."""

    from tensorfold.families.gemma4_unified.model import load as load_model
    from tensorfold.kernels.gemma.dense.v1.matmul import tensor_units

    choice = str(lane_kernels)
    if choice == "on" and not tensor_units():
        raise ValueError("--lane-kernels on needs a GPU with tensor units (M5 generation)")
    backend = "lane" if choice == "on" or (choice == "auto" and tensor_units()) else "rows"
    return load_model(Path(model_dir), backend=backend, drafter=drafter, drafts=int(mtp_drafts or 0))


def engine_settings(model: Any) -> dict[str, Any]:
    """Rows a round verifies at most: the widest window checked exact at load."""

    width = int(getattr(model, "exact_width", 1) or 1)
    return {"max_rows": width, "max_draft": max(0, width - 1)}
