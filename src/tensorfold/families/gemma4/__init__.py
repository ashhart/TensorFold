"""Gemma 4 (``gemma4`` or text-only ``gemma4_text``): MoE and dense checkpoints, uniform or oQ widths, on lanes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_TYPES = ("gemma4", "gemma4_text")
TITLE = "Gemma 4"
LANES = True
MODELS = ("mlx-community/gemma-4-26b-a4b-it-4bit", "unigilby/gemma-4-26B-A4B-it-qat-oQ4e",
          "unigilby/gemma-4-31B-it-qat-oQ8e")
KERNEL_PACKAGE = "tensorfold.kernels.gemma.v1"
KERNEL_VERSION = "v1"
# the decode projections run another family's matmul kernels
KERNEL_DEPENDENCIES = ("tensorfold.kernels.qwen.dense.v1.lane_qmm", "tensorfold.kernels.qwen.dense.v1.affine_rows",
                       "tensorfold.kernels.nemotron.lightning.v1.rows")


# the widths and groups the decode projections read (kernels.gemma.v1.matmul; listed here so a check needs no MLX)
PROJECTION_BITS = (2, 3, 4, 5, 6, 8)
PROJECTION_GROUPS = (32, 64, 128)
EXPERT_GROUPS = (32, 64, 128)            # the expert kernels read 4-bit weights only


def check(model_dir: str | Path) -> None:
    """Refuse, from config.json alone, a checkpoint the kernels do not read (layout, MLX affine widths and groups)."""

    from tensorfold.families import OWN_MODEL_HELP, read_config

    config = read_config(model_dir)
    text = config.get("text_config") or config
    missing = [what for what, ok in (
        ("no per-layer inputs", not int(text.get("hidden_size_per_layer_input") or 0)),
        ("no shared-KV layers", not int(text.get("num_kv_shared_layers") or 0)),
        ("head dims a multiple of 64", all(int(text.get(k) or 64) % 64 == 0 for k in ("head_dim", "global_head_dim"))),
        ("no double-wide MLP", not text.get("use_double_wide_mlp")),
    ) if not ok]
    if missing:
        raise ValueError(f"TensorFold's Gemma 4 kernels cover the 26B-A4B MoE and 31B dense layouts ({MODELS[0]}); "
                         "this one lacks " + ", ".join(missing) + f". {OWN_MODEL_HELP}")
    why = quantization_refusal(config)
    if why:
        raise ValueError(f"Gemma 4's kernels cannot read this checkpoint's weights: {why}. {OWN_MODEL_HELP}")


def quantization_refusal(config: dict[str, Any]) -> str:
    """Why the decode kernels cannot read this config's MLX quantization (oQ's per-layer overrides included), or ''."""

    from tensorfold.families import describe_quantization, layer_quantization, quant_method, quantization

    if quant_method(config) != "mlx":
        return f"they read MLX affine-quantized weights, this checkpoint has {describe_quantization(config)}"
    bits, group = quantization(config)
    block = config.get("quantization") or config.get("quantization_config") or {}
    default_mode = str(block.get("mode") or "affine").lower()
    text = config.get("text_config") or config
    moe = bool(text.get("enable_moe_block"))
    overrides = layer_quantization(config)

    def readable(b: int, g: int, mode: str) -> bool:
        return mode == "affine" and b in PROJECTION_BITS and g in PROJECTION_GROUPS

    if not readable(int(bits), int(group), default_mode):
        return (f"the default is {bits}-bit {default_mode} in groups of {group}; the projections read affine "
                f"{'/'.join(map(str, PROJECTION_BITS))}-bit weights in groups of "
                f"{'/'.join(map(str, PROJECTION_GROUPS))}")
    for path, (b, g, mode) in sorted(overrides.items()):
        if "vision" in path or "audio" in path:               # towers the text decoder never reads
            continue
        if not readable(b, g, mode):
            return f"{path} is {b}-bit {mode} in groups of {g}"
    if moe:
        expert = [(p, s) for p, s in overrides.items() if ".experts." in p or "switch_glu" in p or "switch_mlp" in p]
        if (int(bits) != 4 or int(group) not in EXPERT_GROUPS) and not expert:
            return f"the experts are {bits}-bit in groups of {group}; the expert kernels read 4-bit weights"
        for path, (b, g, _) in expert:
            if b != 4 or g not in EXPERT_GROUPS:
                return f"{path} is {b}-bit in groups of {g}; the expert kernels read 4-bit weights"
    return ""


def load(model_dir: Path, *, lane_kernels: str = "auto", drafter: str = "", drafter_bits: int = 8,
         **_: Any) -> tuple[Any, Any]:
    """Before M5 each linear's faster of rows and matrix by shape; ``lane_kernels`` "on": lane; ``drafter``: DFlash."""

    import os

    from tensorfold.families.gemma4.model import load as load_model
    from tensorfold.kernels.gemma.v1.matmul import tensor_units

    backend = "lane" if str(lane_kernels) == "on" else ("rows" if tensor_units() else "auto")
    backend = os.environ.get("TF_GEMMA_DENSE") or backend      # rows, matrix or auto: for measurements
    return load_model(Path(model_dir), backend=backend, drafter=drafter, drafter_bits=drafter_bits)


def engine_settings(model: Any) -> dict[str, Any]:
    """Rows a round verifies at most: the widest window checked exact at load."""

    width = int(getattr(model, "exact_width", 1) or 1)
    return {"max_rows": width, "max_draft": max(0, width - 1)}
