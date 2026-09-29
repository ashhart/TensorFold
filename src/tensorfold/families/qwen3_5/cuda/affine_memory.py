"""Count retained packed words at their actual precision before loading the CUDA family."""

from __future__ import annotations

from functools import lru_cache
import json
import math
from pathlib import Path


def weight_transform(model_dir):
    from tensorfold.cuda.geometry import linear_weights, size
    from tensorfold.cuda.capacity import headers
    from tensorfold.quantization import resolve_affine

    @lru_cache(maxsize=1)
    def metadata():
        path = Path(model_dir)
        return json.loads((path / "config.json").read_text()), headers(path)

    def transform(name, info):
        if name.startswith("vision_tower") or ".mtp." in name or name.startswith("mtp."):
            return 0, 0
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
        if (spec.bits, spec.group_size, scales["dtype"], biases["dtype"]) == (4, 64, "BF16", "BF16"):
            return linear_weights(name, info)
        return size(info) * (2 if "lm_head." in name else 1), 0
    return transform


def packed_draft(name: str, shape) -> bool:
    """Whether ``DFlash2`` packs this checkpoint tensor to 4 bits (``q4``) instead of keeping it as stored."""

    return (len(shape) == 2 and name.endswith(".weight") and shape[0] % 64 == 0 and shape[1] % 64 == 0
            and shape[0] * shape[1] >= 1 << 20)


def draft_bytes(name: str, info: dict) -> tuple[int, int]:
    """GPU bytes of one checkpoint tensor once the 4-bit drafter holds it (for ``capacity.admit``): packed words plus
    bf16 scales and biases, the fused path's second [k | v] copy, other tensors as stored, codebooks on the host."""

    from tensorfold.cuda.capacity import itemsize

    shape = [int(n) for n in info["shape"]]
    count = math.prod(shape)
    if name.startswith("candidate_selector.") and name.endswith("_codebook"):
        return 0, 0
    if not packed_draft(name, shape):
        return count * itemsize(info, name), 0
    packed = count // 2 + count // 64 * 4
    return packed * (2 if name.endswith(("k_proj.weight", "v_proj.weight")) else 1), 0
