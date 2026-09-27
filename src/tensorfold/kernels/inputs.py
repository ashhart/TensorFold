"""Pad per-call Metal inputs to keep address spaces stable and avoid recompilation freeing an in-flight pipeline."""

from __future__ import annotations

from typing import Sequence

import mlx.core as mx

MIN_ELEMENTS = 8


def ints(values: Sequence[int], dtype: mx.Dtype = mx.int32) -> mx.array:
    """``values`` as a kernel input, zero-padded to MIN_ELEMENTS (a kernel reads only its own entries)."""

    out = [int(v) for v in values]
    out += [0] * max(0, MIN_ELEMENTS - len(out))
    return mx.array(out, dtype=dtype)


def floats(values: Sequence[float]) -> mx.array:
    """``values`` as a float32 kernel input, zero-padded to MIN_ELEMENTS (no pad op on the GPU)."""

    out = [float(v) for v in values]
    out += [0.0] * max(0, MIN_ELEMENTS - len(out))
    return mx.array(out, dtype=mx.float32)


def padded(a: mx.array) -> mx.array:
    """``a`` when it has MIN_ELEMENTS or more, else flattened and zero-padded to MIN_ELEMENTS."""

    if a.size >= MIN_ELEMENTS:
        return a
    return mx.concatenate([a.reshape(-1), mx.zeros((MIN_ELEMENTS - a.size,), dtype=a.dtype)])
