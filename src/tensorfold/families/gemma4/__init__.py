"""Gemma 4 (model_type ``gemma4`` or text-only ``gemma4_text``): the MoE checkpoints, on the lane engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_TYPES = ("gemma4", "gemma4_text")
TITLE = "Gemma 4"
LANES = True
MODELS = ("mlx-community/gemma-4-26b-a4b-it-4bit",)
KERNEL_PACKAGE = "tensorfold.kernels.gemma.v1"
KERNEL_VERSION = "v1"
# the decode projections run another family's matmul kernels
KERNEL_DEPENDENCIES = ("tensorfold.kernels.qwen.dense.v1.lane_qmm", "tensorfold.kernels.nemotron.lightning.v1.rows")


def check(model_dir: str | Path) -> None:
    """Refuse, from config.json alone, a checkpoint the kernels do not read (MoE layout, 4-bit, 8-bit router)."""

    from tensorfold.families import OWN_MODEL_HELP, describe_quantization, quantization, read_config

    config = read_config(model_dir)
    text = config.get("text_config") or config
    missing = [what for what, ok in (
        ("a MoE block in every layer", bool(text.get("enable_moe_block"))),
        ("no per-layer inputs", not int(text.get("hidden_size_per_layer_input") or 0)),
        ("no shared-KV layers", not int(text.get("num_kv_shared_layers") or 0)),
        ("head dims a multiple of 64", all(int(text.get(k) or 64) % 64 == 0 for k in ("head_dim", "global_head_dim"))),
    ) if not ok]
    if missing:
        raise ValueError(f"TensorFold's Gemma 4 kernels cover the MoE checkpoints ({MODELS[0]}); this one lacks "
                         + ", ".join(missing) + f". {OWN_MODEL_HELP}")
    bits, group = quantization(config)
    if bits != 4 or group not in (32, 64):
        raise ValueError(f"Gemma 4's kernels read MLX 4-bit weights in groups of 32 or 64 ({MODELS[0]}); this "
                         f"checkpoint has {describe_quantization(config)}. {OWN_MODEL_HELP}")
    found = config.get("quantization") or config.get("quantization_config") or {}
    for name, spec in found.items():
        if isinstance(spec, dict) and not (name.endswith("router.proj") and int(spec.get("bits", 0)) == 8):
            raise ValueError(f"Gemma 4's kernels read 4-bit projections with an 8-bit router; {name} has "
                             f"{spec.get('bits')}-bit weights. {OWN_MODEL_HELP}")


def load(model_dir: Path, *, lane_kernels: str = "auto", drafter: str = "", drafter_bits: int = 8,
         **_: Any) -> tuple[Any, Any]:
    """MLX's qmv loop a row by default; ``lane_kernels`` "on": the lane matmul; ``drafter``: a DFlash model's chains."""

    from tensorfold.families.gemma4.model import load as load_model

    backend = "lane" if str(lane_kernels) == "on" else "rows"
    return load_model(Path(model_dir), backend=backend, drafter=drafter, drafter_bits=drafter_bits)


def engine_settings(model: Any) -> dict[str, Any]:
    """Rows a round verifies at most: the widest window checked exact at load."""

    width = int(getattr(model, "exact_width", 1) or 1)
    return {"max_rows": width, "max_draft": max(0, width - 1)}
