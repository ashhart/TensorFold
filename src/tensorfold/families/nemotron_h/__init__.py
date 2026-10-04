"""Nemotron-H family support with fused decode kernels and verified MTP draft chains."""

from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_TYPES = ("nemotron_h",)
TITLE = "Nemotron 3.5 Lightning"
LANES = True
MODELS = ("TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit",)
REQUIRED_FILES = {MODELS[0]: ("mtp-4bit.safetensors",)}
KERNEL_PACKAGE = "tensorfold.kernels.nemotron.lightning.v1"
KERNEL_VERSION = "v1"
# The long-context path also calls Qwen dense's lane attention kernel.
KERNEL_DEPENDENCIES = ("tensorfold.kernels.qwen.dense.v1.lane_attention",)
# Set command-buffer limits before MLX starts, preserving explicit environment settings.
MLX_ENV = {"MLX_MAX_OPS_PER_BUFFER": "200", "MLX_MAX_MB_PER_BUFFER": "100000"}


GROUPS = (32, 64)          # 4-bit groups every chip's kernels read: lane_qmm on M5, rows' matvecs and experts


def refusal(config: dict[str, Any]) -> str | None:
    """Why the kernels cannot read a checkpoint, from config.json: they read MLX 4-bit weights in GROUPS only."""

    from tensorfold.families import describe_quantization, layer_quantization, quantization

    bits, group = quantization(config)
    odd = sorted({f"{b}-bit g{g}" + ("" if m == "affine" else f" {m}") for path, (b, g, m)    # embeddings: a lookup
                  in layer_quantization(config).items()
                  if not (b == 4 and g in GROUPS and m == "affine") and not path.endswith("embeddings")})
    if bits == 4 and group in GROUPS and not odd:
        return None
    found = describe_quantization(config) + (f", with layers at {', '.join(odd)}" if odd else "")
    return (f"its kernels read MLX 4-bit weights in groups of 32 or 64 in every projection and expert (the expert "
            f"kernel reads any other width as 4-bit); this checkpoint has {found}")


def check(model_dir: str | Path) -> None:
    """Refuse from config.json alone, before any weight downloads, a checkpoint the kernels cannot read."""

    from tensorfold.families import read_config

    why = refusal(read_config(model_dir))
    if why:
        raise ValueError(f"{TITLE} cannot run this checkpoint: {why}. Use {MODELS[0]}")


def unreadable(model: Any) -> dict[str, int]:
    """An mlx_lm model's quantized linears and expert tables, by kind, other than 4-bit affine in GROUPS (bf16)."""

    import mlx.core as mx
    import mlx.nn as nn

    tables = [fc for layer in model.layers if getattr(layer, "block_type", "") == "E"
              for fc in (layer.mixer.switch_mlp.fc1, layer.mixer.switch_mlp.fc2)]
    linears = [m for _, m in model.named_modules() if isinstance(m, nn.QuantizedLinear)]
    counts: dict[str, int] = {}
    for part, found in (("expert tables", tables), ("linears", linears)):
        for m in found:
            mode, scales = getattr(m, "mode", "affine"), m["scales"].dtype
            if m.bits == 4 and m.group_size in GROUPS and mode == "affine" and scales == mx.bfloat16:
                continue
            kind = (f"{m.bits}-bit g{m.group_size}" + ("" if mode == "affine" else f" {mode}")
                    + ("" if scales == mx.bfloat16 else f" {scales} scales") + f" {part}")
            counts[kind] = counts.get(kind, 0) + 1
    return counts


def load(model_dir: Path, *, mtp_head: str = "", mtp_drafts: int | None = None, **_: Any) -> tuple[Any, Any]:
    from tensorfold.families import read_config
    from tensorfold.families.nemotron_h.model import load as load_model

    why = refusal(read_config(model_dir))
    if why:
        raise SystemExit(f"[tensorfold] {TITLE} cannot run this checkpoint: {why}. Use {MODELS[0]}")
    model, tokenizer = load_model(Path(model_dir), mtp_head=mtp_head, mtp_drafts=mtp_drafts)
    missed = unreadable(model.model)          # the loaded modules, in case config.json did not name a width
    if missed:
        kinds = ", ".join(f"{n} {kind}" for kind, n in sorted(missed.items()))
        raise SystemExit(f"[tensorfold] {TITLE}: the kernels do not read this checkpoint's {kinds}. Use {MODELS[0]}")
    return model, tokenizer


def engine_settings(model: Any) -> dict[str, Any]:
    """Rows a round verifies at most and, with tensor units, prompt chunks of up to 8,192 tokens where memory allows."""

    from tensorfold.families.qwen3_5 import tensor_units

    width = int(getattr(model, "exact_width", 1) or 1)
    settings: dict[str, Any] = {"max_rows": width, "max_draft": max(0, width - 1)}
    if tensor_units():
        settings["prefill_steps"] = (8192, 4096, 2048)
    return settings


# the CUDA engine's kernels read MLX affine weights of this (bits, group size)
CUDA_QUANTIZATION = (4, 64)


def cuda_engine(model_dir: str | Path, *, drafter: str = "", tp: int = 1, rank: int = 0, master: str = "",
                master_port: int = 29571, no_drafts: bool = False, mtp_drafts: int | None = None,
                mtp_confidence: float | None = None, context: int | None = None, **options: Any):
    """The CUDA engine: MTP chains verified exactly on one GPU or two (``tp=2``; start rank 1 first)."""

    if drafter:
        raise ValueError(f"{TITLE} drafts with its own MTP head on CUDA: a separate draft model does not apply")
    from .cuda import CONFIDENCE, CONTEXT, DRAFTS
    from .cuda.app import NemotronEngine

    drafts = 0 if no_drafts else DRAFTS if mtp_drafts is None else int(mtp_drafts)
    confidence = CONFIDENCE if mtp_confidence is None else float(mtp_confidence)
    explicit = bool(options.get("context_explicit"))
    return NemotronEngine(Path(model_dir), drafts=drafts, confidence=confidence,
                          context=context if explicit else CONTEXT, context_explicit=explicit, tp=int(tp),
                          rank=int(rank), master=master, port=int(master_port))
