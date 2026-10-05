"""The cache-value roundings of DeepSeek-V4.1: deterministic elementwise FP8/FP4 quantization, dequantized in place.

Exactly the reference port's ``quantize_cache`` (which the official kernels' arithmetic pins): the
window ring's keys are FP8 e8m0 in groups of 32, pool rows FP4 e4m3-scaled in groups of 16, index
keys and index queries FP4 e8m0 in groups of 32. Values are rounded and dequantized back, so the
cache stores bf16 arrays whose values carry the quantization roundings.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

# FP4's level set with ties-to-even: the official kernel's e2m1 codes reordered even codes first
_LEVELS = mx.array([0, 1, 2, 4, 0.5, 1.5, 3, 6], dtype=mx.float32)
_MADE: dict[tuple, mx.array] = {}


def _levels() -> mx.array:
    return _LEVELS


def round_fp8(x: mx.array) -> mx.array:
    """FP8 e4m3 values without a scale: step 2^(floor(log2 a) - 3), capped at +-448 (power-of-two grid)."""
    a = mx.abs(x)
    step = 2.0 ** (mx.maximum(mx.floor(mx.log2(mx.maximum(a, 2.0 ** -20))), -6.0) - 3.0)
    return (mx.sign(x) * mx.minimum(mx.round(a / step) * step, 448.0)).astype(mx.float32)


def round_fp4(x: mx.array) -> mx.array:
    """FP4 e2m1 levels {0, .5, 1, 1.5, 2, 3, 4, 6} with ties-to-even, sign kept."""
    a = mx.abs(x)[..., None]
    chosen = mx.argmin(mx.abs(a - _levels()), axis=-1)
    return mx.sign(x) * mx.take(_levels(), chosen.reshape(x.shape))


def _maybe(shape: tuple[int, ...], dtype: Any, fill: float) -> mx.array:
    key = (shape, str(dtype), fill)
    found = _MADE.get(key)
    if found is None:
        if len(_MADE) > 64:
            _MADE.clear()
        found = _MADE[key] = mx.full(shape, fill, dtype=dtype)
    return found


def quantize_cache(x: mx.array, bits: int, group: int, scale_format: str = "e8m0") -> mx.array:
    """The reference's cache rounding: values quantized per ``group`` and dequantized back in place."""
    shape = x.shape
    v = x.astype(mx.float32).reshape(*shape[:-1], -1, group)
    maximum = mx.max(mx.abs(v), axis=-1, keepdims=True)
    if bits == 8:
        scale = 2.0 ** mx.ceil(mx.log2(mx.maximum(maximum, 1e-4) / 448.0))
        out = round_fp8(v / scale) * scale
    else:
        if scale_format == "e4m3":
            scale = round_fp8(mx.maximum(maximum, 6.0 * 2.0 ** -9) / 6.0)
        else:
            scale = 2.0 ** mx.ceil(mx.log2(mx.maximum(maximum, 6.0 * 2.0 ** -126) / 6.0))
        out = round_fp4(v / scale) * scale
    return out.reshape(shape).astype(x.dtype)
