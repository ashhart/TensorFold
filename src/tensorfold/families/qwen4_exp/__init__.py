"""Qwen3.8 Flash Next forward pass and fused decode with exact MTP drafting on Metal or one or two CUDA GPUs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_TYPES = ("qwen4_exp",)
TITLE = "Qwen3.8 Flash Next"
LANES = True
# 4-bit weights in groups of 32 (what the fused kernels read), with the checkpoint's MTP head kept
MODELS = ("Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP",)
KERNEL_PACKAGE = "tensorfold.kernels.qwen.flash_next.v1"
KERNEL_VERSION = "v1"
# The CLI sets these before MLX starts, respecting environment overrides, to keep expert bindings from ending each command buffer.
MLX_ENV = {"MLX_MAX_OPS_PER_BUFFER": "200", "MLX_MAX_MB_PER_BUFFER": "100000"}


def has_mtp(model_dir: Path) -> bool:
    """Whether the checkpoint kept the MTP head's weights (``mtp.*``)."""

    import json

    index = Path(model_dir) / "model.safetensors.index.json"
    if not index.is_file():
        return False
    return any(".mtp." in name or name.startswith("mtp.") for name in json.loads(index.read_text())["weight_map"])


def check(model_dir: Path) -> None:
    from tensorfold.families import quantization, read_config

    bits, group = quantization(read_config(model_dir))
    if (bits, group) != (4, 32):
        raise ValueError(f"TensorFold's Flash Next kernels read 4-bit weights in groups of 32; this checkpoint has "
                         f"{bits}-bit weights in groups of {group}. Use {MODELS[0]}.")
    # Config-only preflight cannot establish whether the MTP head is missing; wait for weights or their index.
    if ((Path(model_dir) / "model.safetensors.index.json").is_file()
            or any(Path(model_dir).glob("model*.safetensors"))) and not has_mtp(model_dir):
        print(f"[tensorfold] this checkpoint has no MTP head: decoding without MTP drafts ({MODELS[0]} has one)",
              flush=True)


def load(model_dir: Path, *, mtp_drafts: int | None = None, **_: Any) -> tuple[Any, Any]:
    from tensorfold.families.qwen4_exp.runtime import load as load_runtime

    return load_runtime(Path(model_dir), drafts=mtp_drafts if has_mtp(Path(model_dir)) else 0)


def engine_settings(model: Any) -> dict[str, Any]:
    """Rows a round verifies at most: the widest window checked exact at load."""

    width = int(getattr(model, "exact_width", 1) or 1)
    return {"max_rows": width, "max_draft": max(0, width - 1)}


def kernel_version(model: Any) -> str:
    """Include the loaded model's prompt-attention selection modes in its kernel fingerprint."""

    from tensorfold.families import families, kernel_source_version

    modes = [str(int(bool(getattr(layer.self_attn, "kernel_select", False))))
             for layer in getattr(model, "layers", ()) if hasattr(layer, "self_attn")]
    source = kernel_source_version(families()["qwen4_exp"])
    prefill = getattr(model, "prefill_key", None)                     # the prefill path, matmul route and GPU
    return f"{source}|prompt_attention={','.join(modes)}" + (f"|{prefill}" if prefill else "")


# the CUDA engine's kernels read MLX affine weights of this (bits, group size)
CUDA_QUANTIZATION = (4, 32)

def cuda_engine(model_dir: str | Path, *, drafter: str = "", tp: int = 1, rank: int = 0, master: str = "",
                master_port: int = 29551, no_drafts: bool = False, mtp_drafts: int | None = None,
                context: int | None = None, **options: Any):
    """The CUDA engine: MTP chains verified exactly on one GPU or two (``tp=2``; start rank 1 first)."""

    if drafter:
        raise ValueError(f"{TITLE} drafts with its own MTP head on CUDA: a separate draft model does not apply")
    from .cuda import DEPTH
    from .cuda.engine import FlashNextEngine

    depth = 0 if no_drafts else DEPTH if mtp_drafts is None else int(mtp_drafts)
    if depth and not has_mtp(Path(model_dir)):
        raise ValueError(f"this checkpoint has no MTP head, which {TITLE}'s CUDA engine drafts with ({MODELS[0]} "
                         "has one): without it every round would decode one token. Serve a checkpoint with the "
                         "head, or pass --no-drafts for the serial reference")
    return FlashNextEngine(Path(model_dir), depth=depth, max_len=context,
                           context_explicit=options.get("context_explicit"), tp=int(tp), rank=int(rank),
                           master=master, port=int(master_port), streams=max(1, int(options.get("parallel") or 1)))
