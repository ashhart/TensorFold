"""Qwen3.8 Flash Next (model_type ``qwen4_exp``).

``model``: TensorFold's forward pass for the checkpoint (hyper-connections, Gated DeltaNet, sparse attention,
512-expert MoE, hashed n-gram embedding); ``kernels`` and ``decode``: the fused decode step; ``mtp``: the
checkpoint's MTP head; ``runtime``: what the serial engine serves (fused decode + exact MTP drafting).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_TYPES = ("qwen4_exp",)
TITLE = "Qwen3.8 Flash Next"
LANES = False


def load(model_dir: Path, *, mtp_drafts: int | None = None, **_: Any) -> tuple[Any, Any]:
    from tensorfold.families.qwen4_exp.runtime import load as load_runtime

    return load_runtime(Path(model_dir), drafts=mtp_drafts)
