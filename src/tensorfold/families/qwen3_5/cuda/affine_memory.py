"""Count retained packed words at their actual precision before loading the CUDA family."""

from __future__ import annotations

from functools import lru_cache
import json
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
