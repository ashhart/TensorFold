"""Qwen3.6 MoE (qwen3_5_moe): the 27B's DeltaNet and attention with routed experts. On CUDA, MTP drafts on the lanes;
on Macs, the Qwen3.8 dense row decoder (``row_forward.moe`` for the experts) with DFlash (v1) drafts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_TYPES = ("qwen3_5_moe",)
TITLE = "Qwen3.6 MoE"
LANES = True
# MLX 4-bit, groups of 64, routers 8-bit, MTP layer in mtp-4bit.safetensors (mlx-community's files take it too)
MODELS = ("Vontra/Qwen3.6-35B-A3B-MLX-4bit-MTP", "mlx-community/Qwen3.6-35B-A3B-4bit")
REQUIRED_FILES = {MODELS[0]: ("mtp-4bit.safetensors",)}
DRAFTER = "z-lab/Qwen3.6-35B-A3B-DFlash"      # Macs: DFlash (v1), chains of each position's own argmax
CUDA_DRAFTER = ""                             # CUDA: the checkpoint's own MTP layer
KERNEL_PACKAGE = "tensorfold.kernels.qwen.dense.v1"
KERNEL_VERSION = "v1"
# the CUDA engine's kernels read MLX affine weights of this (bits, group size)
CUDA_QUANTIZATION = (4, 64)


def check(model_dir: str | Path) -> None:
    """One GPU, MLX 4-bit weights in groups of 64."""

    from tensorfold.families import OWN_MODEL_HELP, describe_quantization, quantization, read_config

    if quantization(read_config(model_dir)) != CUDA_QUANTIZATION:
        raise ValueError(f"{TITLE}'s CUDA engine reads MLX 4-bit weights in groups of 64 ({MODELS[0]}); this "
                         f"checkpoint has {describe_quantization(read_config(model_dir))}. {OWN_MODEL_HELP}")


def load(model_dir: Path, *, drafter: str = "", drafter_bits: int = 4, **_: Any) -> tuple[Any, Any]:
    """The row decoder on every Mac: the lane kernels (M5) take no routed experts, whose rows MLX's gather_qmm
    computes one row's bits at any width (``row_forward.moe``). Text only: mlx_lm drops the vision weights."""

    from tensorfold.families.qwen3_5 import lane_family, load_lane_model

    model, tokenizer = load_lane_model(Path(model_dir))
    return lane_family(model, lanes=False, drafter=drafter, drafter_bits=drafter_bits, title=TITLE,
                       use=MODELS[1]), tokenizer


def engine_settings(model: Any) -> dict[str, Any]:
    from tensorfold.families.qwen3_5 import engine_settings as dense

    return dense(model)


def kernel_version(model: Any) -> str:
    """Names the row decoder that computed a prefix snapshot; TF_MOE_ROWS's two paths give other bits."""

    import hashlib

    from tensorfold.kernels.qwen.dense.v1 import row_forward, row_matmul

    parts = [getattr(row_matmul.BACKEND, "name", "simd_qmm"), f"row_attention={row_forward.ROW_ATTENTION}",
             f"moe_rows={row_forward.MOE_ROWS}",
             *(path.read_text() for path in sorted(Path(row_forward.__file__).parent.glob("*.py")))]
    return f"qwen3_5_moe-{KERNEL_VERSION}-" + hashlib.sha256("\n".join(parts).encode()).hexdigest()[:12]


def cuda_engine(model_dir: str | Path, *, drafter: str = "", tp: int = 1, rank: int = 0, master: str = "",
                master_port: int = 29551, no_drafts: bool = False, mtp_drafts: int | None = None,
                context: int | None = None, **options: Any):
    """The one-GPU engine: MTP chains verified exactly, or the serial reference with ``no_drafts``; ``parallel`` > 1 decodes that many requests together."""

    if drafter:
        raise ValueError(f"{TITLE} drafts with its own MTP layer on CUDA: a separate draft model does not apply")
    if int(tp) != 1:
        raise ValueError(f"{TITLE} runs on one GPU: drop --tp")
    from .cuda import DEPTH
    from .cuda.engine import Qwen36Engine

    depth = 0 if no_drafts else DEPTH if mtp_drafts is None else int(mtp_drafts)
    streams = max(1, int(options.get("parallel") or 1))
    if streams > 1 and not 0 <= depth <= 15:
        raise ValueError(f"--parallel verifies up to 16 rows a stream: --mtp-drafts 0 to 15, not {depth}")
    return Qwen36Engine(Path(model_dir), depth=depth, context=context, context_explicit=options.get("context_explicit"),
                        streams=streams)
