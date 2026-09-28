"""Quantized linears in the checkpoint's MLX affine layout, applied with one-row bits on the decode path."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from tensorfold.kernels.glm.flash.v1 import kernels as K


class Q:
    """A quantized linear's weights (MLX affine layout: [out, in * bits / 32] uint32, scales/biases [out, groups])."""

    def __init__(self, weight: mx.array, scales: mx.array, biases: mx.array) -> None:
        self.weight, self.scales, self.biases = weight, scales, biases
        groups = int(scales.shape[-1])
        packed = int(weight.shape[-1])
        # in_dims = groups * group; bits = packed * 32 / in_dims, with group in {32, 64, 128}
        for group in (64, 32, 128):
            ins = groups * group
            if (packed * 32) % ins == 0 and (packed * 32) // ins in (2, 3, 4, 5, 6, 8):
                self.group, self.bits, self.ins = group, (packed * 32) // ins, ins
                break
        else:
            raise ValueError(f"cannot infer the quantization of a {weight.shape} weight with {scales.shape} scales")

    @property
    def outs(self) -> int:
        return int(self.weight.shape[-2])

    def arrays(self) -> list[mx.array]:
        return [self.weight, self.scales, self.biases]

    def __call__(self, x: mx.array) -> mx.array:
        return mx.quantized_matmul(x, self.weight, self.scales, self.biases, transpose=True, group_size=self.group,
                                   bits=self.bits)

    @classmethod
    def stack(cls, parts: list["Q"]) -> "Q":
        """Projections that read the same input as one matrix (rows concatenated)."""

        return cls(mx.concatenate([p.weight for p in parts]), mx.concatenate([p.scales for p in parts]),
                   mx.concatenate([p.biases for p in parts]))


def _rows(q: Q, lo: int, hi: int) -> Q:
    """Output rows lo .. hi - 1 of a quantized linear (views of its arrays)."""

    return Q(q.weight[lo:hi], q.scales[lo:hi], q.biases[lo:hi])


def project(x: mx.array, q: Q, *, rows_exact: bool) -> mx.array:
    """x [R, K] through a quantized linear. One row: MLX's quantized matmul. Several rows on the decode path: the
    rows share the weight reads through ``kernels.qmv_rows`` (MLX's one-row bits) where the weights fit it, else
    one MLX call per row."""

    rows = int(x.shape[0])
    if rows == 1 or not rows_exact:
        return q(x)
    if K.metal() and K.qmv_rows_fits(q, rows):
        return K.qmv_rows(x, q)
    return mx.concatenate([q(x[r:r + 1]) for r in range(rows)])


def per_row(fn: Any, x: mx.array, rows_exact: bool) -> mx.array:
    rows = int(x.shape[0])
    if rows == 1 or not rows_exact:
        return fn(x)
    return mx.concatenate([fn(x[r:r + 1]) for r in range(rows)])


def silu(x: mx.array) -> mx.array:
    return nn.silu(x)

