"""Laguna (poolside, model_type ``laguna``): Laguna-S-2.1's gated-attention sigmoid MoE, on the lane engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_TYPES = ("laguna",)
TITLE = "Laguna"
LANES = True
MODELS = ("mlx-community/Laguna-S-2.1-oQ4e",)
DRAFTER = "poolside/Laguna-S-2.1-DFlash"      # DFlash (v1): chains of each position's own argmax
KERNEL_PACKAGE = "tensorfold.kernels.laguna.v1"
KERNEL_VERSION = "v1"
# the engine interface and caches are Gemma 4's; the decode runs Gemma's attention and expert kernels and Qwen's matmuls
KERNEL_DEPENDENCIES = ("tensorfold.families.gemma4", "tensorfold.kernels.gemma.v1", "tensorfold.drafters",
                       "tensorfold.kernels.qwen.dense.v1.lane_qmm", "tensorfold.kernels.qwen.dense.v1.affine_rows")
# the lane kernels read every affine width; prompts run mlx-lm's own (any width)
QUANT_METHODS = {"mlx": ("mlx",)}

_EXPERTS = ("switch_mlp.gate_proj", "switch_mlp.up_proj", "switch_mlp.down_proj")


def _text(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("text_config") or config


def check(model_dir: str | Path) -> None:
    """Refuse, from config.json alone, a checkpoint the kernels do not read."""

    from tensorfold.families import OWN_MODEL_HELP, read_config

    config = read_config(model_dir)
    text = _text(config)
    heads = text.get("num_attention_heads_per_layer") or [text.get("num_attention_heads")] * int(
        text.get("num_hidden_layers") or 0)
    kv = int(text.get("num_key_value_heads") or 0)
    layer_types = set(text.get("layer_types") or ["full_attention"])
    rope = text.get("rope_parameters") or {}
    missing = [what for what, ok in (
        ("the per-head attention gate", str(text.get("gating", True)).replace("_", "-") in ("per-head", "True")),
        ("sigmoid routing", str(text.get("moe_router_score_func") or "sigmoid") == "sigmoid"),
        ("normalised top-k weights", bool(text.get("norm_topk_prob", True))),
        ("router weights applied after the experts", not text.get("moe_apply_router_weight_on_input")),
        ("full and sliding attention layers only", layer_types <= {"full_attention", "sliding_attention"}),
        ("a head dim a multiple of 64", int(text.get("head_dim") or 0) % 64 == 0),
        ("query heads a multiple of the key heads", kv > 0 and all(int(h) % kv == 0 for h in heads)),
        ("RoPE over a multiple of 64 dims", all(
            int(int(text.get("head_dim") or 0) * float((rope.get(t) or {}).get("partial_rotary_factor", 1.0))) % 64
            == 0 for t in layer_types)),
        ("an expert width of 512 to 1,024", 512 <= int(text.get("moe_intermediate_size") or 0) <= 1024),
        ("a hidden size a multiple of 256", int(text.get("hidden_size") or 0) % 256 == 0),
    ) if not ok]
    if missing:
        raise ValueError(f"TensorFold's Laguna kernels cover Laguna-S-2.1 ({MODELS[0]}); this checkpoint lacks "
                         + ", ".join(missing) + f". {OWN_MODEL_HELP}")
    check_quantization(config, "mlx")


def check_quantization(config: dict[str, Any], backend: str = "mlx") -> None:
    """MLX affine weights of any width the row kernels read; the routed experts at 4 bits (Gemma's expert kernels)."""

    from tensorfold.families import (OWN_MODEL_HELP, describe_quantization, layer_quantization, quant_method,
                                     quantization)

    if backend != "mlx":
        raise ValueError(f"Laguna runs on Apple Silicon (MLX) only. {OWN_MODEL_HELP}")
    if quant_method(config) is None:
        raise ValueError("Laguna's decode kernels read MLX-quantized weights; this checkpoint is unquantized. "
                         f"{OWN_MODEL_HELP}")
    bits, group = quantization(config)
    widths = {path: spec for path, spec in layer_quantization(config).items()}
    base = (int(bits or 0), int(group or 0), "affine")
    bad = [f"{path} ({b}-bit {mode}, groups of {g})" for path, (b, g, mode) in [("default", base), *widths.items()]
           if mode != "affine" or b not in (2, 3, 4, 5, 6, 8) or g not in (32, 64, 128)]
    experts = [path for path, (b, _, _) in widths.items() if path.endswith(_EXPERTS) and b != 4]
    if base[0] != 4 and not any(path.endswith(_EXPERTS) for path in widths):
        experts.append(f"the routed experts ({describe_quantization(config)})")
    if bad or experts:
        raise ValueError("Laguna's decode kernels read MLX affine 2- to 8-bit weights in groups of 32, 64 or 128, "
                         "routed experts at 4 bits; this checkpoint has "
                         + ", ".join(bad + [f"{e} not 4-bit" for e in experts]) + f". {OWN_MODEL_HELP}")


def load(model_dir: Path, *, lane_kernels: str = "auto", drafter: str = "", drafter_bits: int = 8,
         **_: Any) -> tuple[Any, Any]:
    """The lane matmul with tensor units (``lane_kernels``), else the affine row kernel; ``drafter``: Laguna DFlash."""

    from tensorfold.families.laguna.model import load as load_model
    from tensorfold.kernels.laguna.v1.matmul import tensor_units

    mode = str(lane_kernels)
    backend = "lane" if mode == "on" or (mode == "auto" and tensor_units()) else "rows"
    return load_model(Path(model_dir), backend=backend, drafter=drafter, drafter_bits=drafter_bits)


def engine_settings(model: Any) -> dict[str, Any]:
    """Rows a round verifies at most: the widest window checked exact at load."""

    width = int(getattr(model, "exact_width", 1) or 1)
    return {"max_rows": width, "max_draft": max(0, width - 1)}
