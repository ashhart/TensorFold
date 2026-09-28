"""Serve options a backend or family has no path for, refused before any weight is downloaded."""

from __future__ import annotations

import argparse
import inspect
from typing import Any


def check(args: argparse.Namespace, family: Any, backend: str, config_dir: Any = None) -> None:
    """Refuse a KV cache, draft rule or image option the backend or family has no path for, before any download."""

    if getattr(args, "vision_urls", False) and not getattr(args, "vision", False):
        raise ValueError("--vision-urls needs --vision")
    if getattr(args, "vision", False):             # only --vision reads the config here
        from tensorfold.families import read_config
        from tensorfold.vision.config import validate_vision_config

        validate_vision_config(read_config(config_dir) if config_dir else {}, family.model_type)
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


def vision_options(args: argparse.Namespace) -> dict[str, Any]:
    """``--vision`` and ``--vision-urls`` as a family's load options."""

    if not getattr(args, "vision", False):
        return {}
    return {"vision": True, "vision_urls": bool(getattr(args, "vision_urls", False))}


__all__ = ["check", "vision_options"]
