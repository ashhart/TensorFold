"""Kernel inputs whose length follows the window, kept to one Metal signature.

``mx.fast.metal_kernel`` declares an input of fewer than MIN_DEVICE_ELEMENTS elements in the ``constant`` address
space and a larger one as ``device``, under one kernel name. Before MLX 0.32 a call with the other signature
recompiled the kernel and evicted the pipeline an uncommitted command buffer may still use (ml-explore/mlx#3662):
that dispatch was lost, its outputs never written (NaN logits from the lane decoder's rounds, 2026-09-26). Inputs
whose length follows the window (tree parents, attention depths and paths, accepted rows, conv windows) therefore
keep at least this many elements, padded at the end, so each kernel binds them one way; the kernels read only a
window's own entries.
"""

from __future__ import annotations

from typing import Sequence

import mlx.core as mx

MIN_DEVICE_ELEMENTS = 8


def device_ints(values: Sequence[int]) -> mx.array:
    """``values`` as int32, zero-padded to MIN_DEVICE_ELEMENTS (kernels read only the first ``len(values)``)."""

    vals = [int(v) for v in values]
    return mx.array(vals + [0] * (MIN_DEVICE_ELEMENTS - len(vals)), dtype=mx.int32)


__all__ = ["MIN_DEVICE_ELEMENTS", "device_ints"]
