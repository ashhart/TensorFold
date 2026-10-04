"""Count retained packed words at their actual precision before loading the CUDA family."""

from __future__ import annotations

from functools import lru_cache
import json
import math
from pathlib import Path


def weight_transform(model_dir, *, one_gpu: bool = False):
    """Tensor bytes; the head counts twice (the drafter's copy), once on ``one_gpu`` where the drafter views it."""

    from tensorfold.cuda.geometry import linear_weights, size
    from tensorfold.cuda.capacity import headers
    from tensorfold.quantization import resolve_affine

    @lru_cache(maxsize=1)
    def metadata():
        path = Path(model_dir)
        return json.loads((path / "config.json").read_text()), headers(path)

    def tiled(path: str) -> bool:
        """4-, 5-, 6- or 8-bit words in groups of 64 with bf16 scales and biases: what loading tiles."""

        config, tensors = metadata()
        spec = resolve_affine(config, path)
        scales, biases = (tensors.get(path + suffix) for suffix in (".scales", ".biases"))
        return (spec is not None and scales is not None and biases is not None and
                spec.bits in (4, 5, 6, 8) and
                (spec.group_size, scales["dtype"], biases["dtype"]) == (64, "BF16", "BF16"))

    def stored(name, info):
        if info["dtype"] not in ("U32", "I32") or not name.endswith(".weight"):
            return linear_weights(name, info)
        config, tensors = metadata()
        path = name.removesuffix(".weight")
        spec = resolve_affine(config, path)
        if spec is None:
            raise ValueError(f"packed weight has disabled affine metadata: {path}")
        scales, biases = (tensors.get(path + suffix) for suffix in (".scales", ".biases"))
        if scales is None or biases is None:
            raise ValueError(f"packed weight is missing affine scales or biases: {path}")
        from tensorfold.quantization import validate_shapes

        validate_shapes(info["shape"], scales["shape"], biases["shape"], spec)
        if tiled(path):
            return linear_weights(name, info)
        return size(info) * (2 if "lm_head." in name else 1), 0

    def transform(name, info):
        if name.startswith("vision_tower") or ".mtp." in name or name.startswith("mtp."):
            return 0, 0
        amount, host = stored(name, info)
        if one_gpu and "lm_head." in name and tiled(name.rsplit(".", 1)[0]):
            amount //= 2                               # the drafter's rows are views of the target's head
        return amount, host
    return transform


def packed_draft(name: str, shape) -> bool:
    """Whether ``DFlash2`` packs this checkpoint tensor to 4 bits (``q4``) instead of keeping it as stored."""

    return (len(shape) == 2 and name.endswith(".weight") and shape[0] % 64 == 0 and shape[1] % 64 == 0
            and shape[0] * shape[1] >= 1 << 20)


def draft_weights(draft_dir):
    """The drafter's bytes and load peak: quantize4 holds 14 bytes an element of its largest tensor while packing."""

    from tensorfold.cuda.capacity import Weights, estimate_weights, headers

    held = estimate_weights(draft_dir, draft_bytes)
    largest = max((math.prod(info["shape"]) for name, info in headers(draft_dir).items()
                   if packed_draft(name, info["shape"])), default=0)
    return Weights(held.resident, max(held.staging, 14 * largest), held.mapped)


def draft_bytes(name: str, info: dict) -> tuple[int, int]:
    """GPU bytes of a draft tensor as the 4-bit drafter holds it (q4 words and scales; k and v twice)."""

    from tensorfold.cuda.capacity import itemsize

    shape = [int(n) for n in info["shape"]]
    count = math.prod(shape)
    if name.startswith("candidate_selector.") and name.endswith("_codebook"):
        return 0, 0
    if not packed_draft(name, shape):
        return count * itemsize(info, name), 0
    packed = count // 2 + count // 64 * 4
    return packed * (2 if name.endswith(("k_proj.weight", "v_proj.weight")) else 1), 0
