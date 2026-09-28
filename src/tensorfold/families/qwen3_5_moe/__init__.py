"""Qwen3.6 MoE (qwen3_5_moe) on CUDA: the 27B's DeltaNet and attention with routed experts, MTP drafts on the lanes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_TYPES = ("qwen3_5_moe",)
TITLE = "Qwen3.6 MoE"
LANES = True
# MLX 4-bit, groups of 64, routers 8-bit, MTP layer in mtp-4bit.safetensors (mlx-community's files take it too)
MODELS = ("Vontra/Qwen3.6-35B-A3B-MLX-4bit-MTP",)
REQUIRED_FILES = {MODELS[0]: ("mtp-4bit.safetensors",)}
# the CUDA engine's kernels read MLX affine weights of this (bits, group size)
CUDA_QUANTIZATION = (4, 64)


def check(model_dir: str | Path) -> None:
    """One GPU, MLX 4-bit weights in groups of 64."""

    from tensorfold.families import OWN_MODEL_HELP, describe_quantization, quantization, read_config

    if quantization(read_config(model_dir)) != CUDA_QUANTIZATION:
        raise ValueError(f"{TITLE}'s CUDA engine reads MLX 4-bit weights in groups of 64 ({MODELS[0]}); this "
                         f"checkpoint has {describe_quantization(read_config(model_dir))}. {OWN_MODEL_HELP}")


def cuda_engine(model_dir: str | Path, *, drafter: str = "", tp: int = 1, rank: int = 0, master: str = "",
                master_port: int = 29551, no_drafts: bool = False, mtp_drafts: int | None = None,
                context: int | None = None, **options: Any):
    """The one-GPU engine: MTP chains verified exactly, or the serial reference with ``no_drafts``."""

    if drafter:
        raise ValueError(f"{TITLE} drafts with its own MTP layer on CUDA: a separate draft model does not apply")
    if int(tp) != 1:
        raise ValueError(f"{TITLE} runs on one GPU: drop --tp")
    if int(options.get("parallel") or 1) > 1:
        raise ValueError(f"{TITLE} serves one request at a time for now: drop --parallel")
    from .cuda import DEPTH
    from .cuda.engine import Qwen36Engine

    depth = 0 if no_drafts else DEPTH if mtp_drafts is None else int(mtp_drafts)
    return Qwen36Engine(Path(model_dir), depth=depth, context=context, context_explicit=options.get("context_explicit"))
