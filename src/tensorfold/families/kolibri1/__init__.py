"""Kolibri 1 (kolibri1): Aleph Alpha's MoE with sliding-window and position-free full attention, on CUDA."""
# Every MLP routes 6 of 384 experts by sigmoid weight beside an ungated shared expert; FP8 is read as it ships.

from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_TYPES = ("kolibri1",)
TITLE = "Kolibri 1"
MODELS = ("Aleph-Alpha/Kolibri-1",)
QUANT_METHODS = {"cuda": ("fp8",)}


def check(model_dir: str | Path) -> None:
    """The FP8 checkpoint: 128 x 128 blocks, fp32 scales."""

    from .cuda.weights import Config

    Config.read(model_dir)


def cuda_engine(model_dir: str | Path, *, drafter: str = "", tp: int = 1, rank: int = 0, master: str = "",
                master_port: int = 29551, no_drafts: bool = False, mtp_drafts: int | None = None,
                context: int | None = None, **options: Any):
    """One GPU, ``--parallel`` requests decoded together (Kolibri 1 ships no draft head); the context fits memory."""

    if drafter:
        raise ValueError(f"{TITLE} has no draft model on CUDA yet: drop --drafter")
    if int(tp) != 1:
        raise ValueError(f"{TITLE} runs on one GPU: drop --tp")
    from .cuda.engine import Kolibri1Engine

    return Kolibri1Engine(Path(model_dir), context=int(context) if context else None,
                          explicit=bool(options.get("context_explicit")),
                          streams=max(1, int(options.get("parallel") or 1)))
