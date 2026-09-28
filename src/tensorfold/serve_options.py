"""Serve options a backend or family has no path for, refused before any weight is downloaded."""

from __future__ import annotations

import argparse
import inspect
from typing import Any


def check(args: argparse.Namespace, family: Any, backend: str) -> None:
    """Refuse a KV cache or draft rule the backend or family has no path for, before any weight is downloaded."""

    kv = getattr(args, "kv_dtype", "bf16")
    if kv != "bf16" and backend != "cuda":
        raise ValueError(f"--kv-dtype {kv} is a CUDA engine option: the MLX path caches keys and values as bf16")
    supported = getattr(family.package, "CUDA_KV_DTYPES", ("bf16",))
    if kv not in supported:
        raise ValueError(f"{family.title} on CUDA serves a {' or '.join(supported)} KV cache, not --kv-dtype {kv}")
    confidence = getattr(args, "mtp_confidence", None)
    if confidence is None:
        return
    engine = getattr(family.package, "cuda_engine", None) if backend == "cuda" else None
    if engine is None or "mtp_confidence" not in inspect.signature(engine).parameters:
        raise ValueError(f"--mtp-confidence sets where a CUDA engine's MTP chains stop; {family.title} on "
                         f"{'CUDA' if backend == 'cuda' else 'MLX'} has no such rule")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"--mtp-confidence is a probability from 0 to 1, not {confidence}")


__all__ = ["check"]
