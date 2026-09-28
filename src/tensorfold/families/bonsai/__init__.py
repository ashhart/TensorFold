"""Ternary Bonsai 2: Qwen3.8 dense stored as Prism's rotated 2-bit pack, on the Qwen3.8 lane engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tensorfold.families.bonsai.pack import MODEL_TYPE, contract

MODEL_TYPES = (MODEL_TYPE,)
TITLE = "Ternary Bonsai 2"
LANES = True
MODELS = ("prism-ml/Ternary-Bonsai-2-27B-mlx-2bit",)
DRAFTER = "z-lab/Qwen3.8-27B-DFlash2"
REQUIRED_FILES = {MODELS[0]: ("hadamard.json",)}
KERNEL_PACKAGE = "tensorfold.kernels.qwen.prism.v1"
KERNEL_VERSION = "v1"
KERNEL_DEPENDENCIES = ("tensorfold.families.qwen3_5", "tensorfold.kernels.qwen.dense.v1")


def check(model_dir: str | Path) -> None:
    """Refuse, from config.json and hadamard.json alone, a pack outside the Prism contract this family reads."""

    contract(model_dir)


def load(model_dir: Path, *, lane_kernels: str = "auto", drafter: str = "", drafter_bits: int = 4,
         **_: Any) -> tuple[Any, Any]:
    """2-bit lanes with tensor units; before M5 codes widened to 4 bits, or the pack's 2-bit rows where those don't fit."""

    import mlx.core as mx
    from mlx_lm.utils import load_tokenizer

    from tensorfold.families.bonsai import pack
    from tensorfold.families.qwen3_5 import lane_family, tensor_units
    from tensorfold.server.memory_budget import memory_limit_bytes

    if lane_kernels == "on" and not tensor_units():
        raise SystemExit(f"[tensorfold] {TITLE}: --lane-kernels on needs Metal 4 tensor units (an M5-generation GPU)")
    lanes = lane_kernels == "on" or (lane_kernels == "auto" and tensor_units())
    form = "lanes" if lanes else pack.pre_m5_form(model_dir, memory_limit_bytes(mx))
    if form == "packed":
        print(f"[tensorfold] {TITLE}: the codes widened to 4 bits would leave too little of this Mac's memory "
              "budget, so the row decoder reads the pack's 2-bit weights as stored (slower rows)", flush=True)
    model = pack.build(model_dir, form=form)
    eos = json.loads((Path(model_dir) / "generation_config.json").read_text()).get("eos_token_id")
    tokenizer = load_tokenizer(Path(model_dir), eos_token_ids=eos if isinstance(eos, list) else None)
    family = lane_family(model, lanes=lanes, drafter=drafter, drafter_bits=drafter_bits, title=TITLE, use=MODELS[0])
    family.pack_fingerprint = f"{pack.fingerprint(model_dir)}-{form}"
    return family, tokenizer


def engine_settings(model: Any) -> dict[str, Any]:
    from tensorfold.families.qwen3_5 import engine_settings as dense

    return dense(model)


def kernel_version(model: Any) -> str:
    """The decoder's kernels, the rotation kernels and this family's code, plus the pack's transform identity."""

    from tensorfold.families import families, kernel_source_version
    from tensorfold.families.qwen3_5 import kernel_version as dense

    decoder = dense(model) if model is not None else "none"      # no model: this family's own sources alone
    return f"{kernel_source_version(families()[MODEL_TYPE])}-{getattr(model, 'pack_fingerprint', 'none')}-{decoder}"


__all__ = ["DRAFTER", "MODELS", "MODEL_TYPES", "TITLE", "check", "engine_settings", "kernel_version", "load"]
