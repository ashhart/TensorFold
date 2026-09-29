"""Bonsai's layers: projections that rotate their rows first, the rotated embedding and the unrotated fp32 gates."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from tensorfold.kernels.qwen.prism.v1 import rotate

# signs id -> (the last input, its rotation): projections that share an input rotate it once
_last: dict[int, tuple[mx.array, mx.array]] = {}


def rotated(x: mx.array, signs: mx.array) -> mx.array:
    """x in the rotated basis of its width's signs, reusing the previous result for the same input array."""

    hit = _last.get(id(signs))
    if hit is not None and hit[0] is x:
        return hit[1]
    y = rotate.rotate_rows(x, signs)
    _last[id(signs)] = (x, y)
    return y


class RotatedLinear(nn.Module):
    """A projection stored in the rotated basis: rows are rotated, then ``inner`` (the lane or row matmul) runs."""

    def __init__(self, inner: nn.QuantizedLinear, signs: mx.array) -> None:
        super().__init__()
        self.inner = inner
        self.signs = signs

    def rotate(self, x: mx.array) -> mx.array:
        return rotated(x, self.signs)

    def __call__(self, x: mx.array) -> mx.array:
        # MLX promotes bf16 rows against the pack's fp16 scales to fp32; caches and row kernels keep bf16
        return self.inner(rotated(x, self.signs)).astype(x.dtype)

    def project_rows(self, x: mx.array) -> mx.array:
        """The row decoder's projection (``row_matmul.project``) of the rotated rows."""

        from tensorfold.kernels.qwen.dense.v1 import row_matmul

        return row_matmul.project(self.inner, rotated(x, self.signs))


class RotatedEmbedding(nn.Module):
    """A 2-bit embedding stored in the rotated basis: a lookup dequantizes and rotates back, row by row."""

    def __init__(self, weight: mx.array, scales: mx.array, biases: mx.array, signs: mx.array, group: int) -> None:
        super().__init__()
        self.weight, self.scales, self.biases, self.signs = weight, scales, biases, signs
        self.group = int(group)

    def __call__(self, ids: mx.array) -> mx.array:
        return rotate.embed_rows(ids, self.weight, self.scales, self.biases, self.signs, self.group)


# the widest call any decode window makes (the lane kernels' 128 rows); prompt chunks past it take MLX's matmul
ROW_EXACT_ROWS = 128


class RowDense(nn.Module):
    """An unquantized fp32 projection (the recurrent layers' a and b): decode windows' rows independent of the count."""

    def __init__(self, weight: mx.array) -> None:
        super().__init__()
        self.weight = weight

    def __call__(self, x: mx.array) -> mx.array:
        if x.size // int(x.shape[-1]) > ROW_EXACT_ROWS:
            return (x.astype(mx.float32) @ self.weight.T).astype(mx.bfloat16)
        return rotate.dense_rows(x, self.weight)

    def project_rows(self, x: mx.array) -> mx.array:
        return rotate.dense_rows(x, self.weight)


def inner_of(module: Any) -> Any:
    """The matmul module under a rotated projection, else the module itself."""

    return module.inner if isinstance(module, RotatedLinear) else module


__all__ = ["RotatedEmbedding", "RotatedLinear", "RowDense", "inner_of", "rotated"]
