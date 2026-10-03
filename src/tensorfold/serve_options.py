"""Serve options a backend or family has no path for, refused before any weight is downloaded."""

from __future__ import annotations

import argparse
import inspect
from typing import Any


def check(args: argparse.Namespace, family: Any, backend: str, config_dir: Any = None) -> None:
    """Refuse KV cache, draft rule, image, share, slot and precision options the backend or family can't serve."""

    if getattr(args, "vision_urls", False) and not getattr(args, "vision", False):
        raise ValueError("--vision-urls needs --vision")
    images = getattr(args, "vision_max_images", None)
    if images is not None:
        if not isinstance(images, int) or isinstance(images, bool) or images < 1:
            raise ValueError("--vision-max-images must be a positive integer")
        if not getattr(args, "vision", False):
            raise ValueError("--vision-max-images needs --vision")
    tokens = getattr(args, "vision_image_tokens", None)
    if tokens is not None:
        if not isinstance(tokens, int) or isinstance(tokens, bool) or not 1 <= tokens <= 65536:
            raise ValueError("--vision-image-tokens is a number of tokens from 1 to 65,536")
        if not getattr(args, "vision", False):
            raise ValueError("--vision-image-tokens needs --vision")
        if backend != "cuda":
            raise ValueError("--vision-image-tokens sets the CUDA Qwen image budget; the MLX towers size their "
                             "workspace for 4,096 visual tokens")
    if getattr(args, "vision_offload", False):
        if not getattr(args, "vision", False):
            raise ValueError("--vision-offload needs --vision")
        if backend != "cuda":
            raise ValueError("--vision-offload is for the CUDA backend; the Mac's image tower already shares host memory")
    if getattr(args, "vision", False):             # only --vision reads the config here
        if family.model_type == "glm5_next" and backend != "mlx":
            raise ValueError("GLM-5.3-Flash image input is currently MLX-only")
        from tensorfold.families import read_config
        from tensorfold.vision.config import validate_vision_config

        validate_vision_config(read_config(config_dir) if config_dir else {}, family.model_type)
        if family.model_type == "qwen4_exp" and backend != "cuda":
            raise ValueError("--vision for Flash Next runs on the CUDA engine; the MLX path has no image tower yet")
    share = getattr(args, "decode_share", None)
    if share is not None and backend == "cuda" and not getattr(family.package, "CUDA_DECODE_SHARE", False):
        raise ValueError("--decode-share sets the Mac server's share, and Flash Next's on CUDA; this CUDA engine runs "
                         "a round after each 1,024 prompt rows")
    if share is not None and share < 0:
        raise ValueError(f"--decode-share is 0 (whole prompts first) or more, not {share}")
    kv = getattr(args, "kv_dtype", "bf16")
    if kv != "bf16" and backend != "cuda":
        raise ValueError(f"--kv-dtype {kv} is a CUDA engine option: the MLX path caches keys and values as bf16")
    supported = getattr(family.package, "CUDA_KV_DTYPES", ("bf16",))
    if kv not in supported:
        raise ValueError(f"{family.title} on CUDA serves a {' or '.join(supported)} KV cache, not --kv-dtype {kv}")
    slots = getattr(args, "checkpoint_slots", None)
    if slots is not None and backend == "cuda" and getattr(family.package, "CUDA_CHECKPOINT_SLOTS", False):
        if slots < 1:
            raise ValueError(f"--checkpoint-slots is 1 or more, not {slots}")
        if _cuda_streams(getattr(args, "parallel", "auto")) < 2:
            raise ValueError(f"--checkpoint-slots sets the prompt states {family.title}'s concurrent decoder keeps on "
                             "CUDA (--parallel 2 or more); one stream keeps 4, which share its attention buffer")
    fp8 = getattr(family.package, "CUDA_PREFILL_FP8", False) and backend == "cuda"
    if getattr(args, "prefill_fp8", None) and not fp8:              # asked for by name, not a default
        raise ValueError(f"--prefill-fp8 picks FP8 prompt kernels on CUDA; {family.title} on "
                         f"{'CUDA' if backend == 'cuda' else 'MLX'} has none (its prompts run bf16 activations)")
    cost = getattr(args, "mtp_cost", None)
    if cost is not None:
        engine = getattr(family.package, "cuda_engine", None) if backend == "cuda" else None
        if engine is None or "mtp_cost" not in inspect.signature(engine).parameters:
            raise ValueError(f"--mtp-cost prices a CUDA engine's MTP drafts; {family.title} on "
                             f"{'CUDA' if backend == 'cuda' else 'MLX'} has no such rule")
        if cost < 0:
            raise ValueError(f"--mtp-cost is tokens per ms, 0 (off) or more, not {cost}")
        if cost > 0 and _cuda_streams(getattr(args, "parallel", "auto")) > 1:
            raise ValueError("--mtp-cost prices one stream's rounds; drop it or --parallel")
    confidence = getattr(args, "mtp_confidence", None)
    if confidence is None:
        return
    engine = getattr(family.package, "cuda_engine", None) if backend == "cuda" else None
    if engine is None or "mtp_confidence" not in inspect.signature(engine).parameters:
        raise ValueError(f"--mtp-confidence sets where a CUDA engine's MTP chains stop; {family.title} on "
                         f"{'CUDA' if backend == 'cuda' else 'MLX'} has no such rule")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"--mtp-confidence is a probability from 0 to 1, not {confidence}")


def _cuda_streams(value: Any) -> int:
    """The streams a CUDA engine serves for ``--parallel`` (auto: one), or 2 for a value the serve command refuses itself."""

    text = str(value).strip().lower()
    if text == "auto":
        return 1
    try:
        return max(1, int(text))
    except ValueError:
        return 2


def vision_options(args: argparse.Namespace) -> dict[str, Any]:
    """``--vision``, ``--vision-urls`` and ``--vision-offload`` as a family's load options."""

    if not getattr(args, "vision", False):
        return {}
    return {"vision": True, "vision_urls": bool(getattr(args, "vision_urls", False)),
            "vision_offload": bool(getattr(args, "vision_offload", False))}


__all__ = ["check", "vision_options"]
