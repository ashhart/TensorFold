"""Quantized linears in the checkpoint's MLX affine layout, applied with one-row bits on the decode path."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from tensorfold.families.glm5_next.config import BITS, GROUPS
from tensorfold.kernels.glm.flash.v1 import kernels as K


class Q:
    """A quantized linear in MLX's affine layout at its stated format (shapes can't tell 4-bit g32 from 2-bit g64)."""

    def __init__(self, weight: mx.array, scales: mx.array, biases: mx.array, *, bits: int, group: int) -> None:
        packed, groups = int(weight.shape[-1]), int(scales.shape[-1])
        if (bits not in BITS or group not in GROUPS or packed * 32 != groups * group * bits
                or tuple(scales.shape) != tuple(biases.shape)):
            raise ValueError(f"{bits}-bit weights in groups of {group} do not fit a {tuple(weight.shape)} matrix with "
                             f"{tuple(scales.shape)} scales")
        self.weight, self.scales, self.biases = weight, scales, biases
        self.bits, self.group, self.ins = int(bits), int(group), groups * group

    @property
    def outs(self) -> int:
        return int(self.weight.shape[-2])

    def arrays(self) -> list[mx.array]:
        return [self.weight, self.scales, self.biases]

    def __call__(self, x: mx.array) -> mx.array:
        return mx.quantized_matmul(x, self.weight, self.scales, self.biases, transpose=True, group_size=self.group,
                                   bits=self.bits)

    @classmethod
    def stack(cls, parts: list["Q"]) -> "Q | QSplit":
        """Projections of one input as one matrix (rows concatenated); parts of different formats stay separate."""

        if len({(p.bits, p.group) for p in parts}) > 1:
            return QSplit(parts)
        bits, group = one_format(parts)
        return cls(mx.concatenate([p.weight for p in parts]), mx.concatenate([p.scales for p in parts]),
                   mx.concatenate([p.biases for p in parts]), bits=bits, group=group)


class QSplit:
    """Projections of one input at different formats: each part its own call, outputs concatenated in order."""

    bits = None
    group = None

    def __init__(self, parts: list[Q]) -> None:
        self.parts = list(parts)
        self.cuts: list[int] = []
        at = 0
        for p in self.parts:
            at += p.outs
            self.cuts.append(at)

    @property
    def outs(self) -> int:
        return self.cuts[-1]

    def arrays(self) -> list[mx.array]:
        return [a for p in self.parts for a in p.arrays()]

    def __call__(self, x: mx.array) -> mx.array:
        return mx.concatenate([p(x) for p in self.parts], axis=-1)

    def part(self, lo: int, hi: int) -> Q:
        """The part whose output rows are exactly lo .. hi - 1."""

        start = 0
        for p, end in zip(self.parts, self.cuts):
            if (start, end) == (lo, hi):
                return p
            start = end
        raise ValueError(f"QSplit: rows {lo}:{hi} are not one part (cuts {self.cuts})")


class Dense:
    """An unquantized bf16 linear [out, in] behind ``Q``'s interface; the decode path runs it a row at a time."""

    bits = None
    group = None

    def __init__(self, weight: mx.array) -> None:
        self.weight = weight

    @property
    def outs(self) -> int:
        return int(self.weight.shape[0])

    @property
    def ins(self) -> int:
        return int(self.weight.shape[1])

    def arrays(self) -> list[mx.array]:
        return [self.weight]

    def __call__(self, x: mx.array) -> mx.array:
        return mx.matmul(x, self.weight.T)


def one_format(parts: list[Q]) -> tuple[int, int]:
    """The (bits, group) of linears stacked into one matrix, which must share it."""

    formats = sorted({(p.bits, p.group) for p in parts})
    if len(formats) != 1:
        raise ValueError(f"projections that read the same input are stacked into one matrix, so they need one format; "
                         f"these are stored as {', '.join(f'{b}-bit in groups of {g}' for b, g in formats)}")
    return formats[0]


def _rows(q: Q | QSplit, lo: int, hi: int) -> Q:
    """Output rows lo .. hi - 1 of a quantized linear (views of its arrays)."""

    if isinstance(q, QSplit):
        return q.part(lo, hi)
    return Q(q.weight[lo:hi], q.scales[lo:hi], q.biases[lo:hi], bits=q.bits, group=q.group)


def project(x: mx.array, q: Any, *, rows_exact: bool) -> mx.array:
    """x [R, K] through a linear: one row by MLX's call, a decode window by ``qmv_rows`` where it fits, else by row."""

    rows = int(x.shape[0])
    if rows == 1 or not rows_exact:
        return q(x)
    if isinstance(q, QSplit):
        return mx.concatenate([project(x, p, rows_exact=True) for p in q.parts], axis=-1)
    if isinstance(q, Q) and x.dtype == mx.bfloat16 and K.metal() and K.qmv_rows_fits(q, rows):
        return K.qmv_rows(x, q)
    return mx.concatenate([q(x[r:r + 1]) for r in range(rows)])


def bf16_if_exact(a: mx.array) -> mx.array:
    """An fp32 tensor as bf16 when no value changes (mlx-lm stores bf16 mixes and routers as fp32), else as is."""

    if a.dtype != mx.float32:
        return a
    b = a.astype(mx.bfloat16)
    if bool(mx.array_equal(b.astype(mx.float32), a).item()):
        return b
    return a


def per_row(fn: Any, x: mx.array, rows_exact: bool) -> mx.array:
    rows = int(x.shape[0])
    if rows == 1 or not rows_exact:
        return fn(x)
    return mx.concatenate([fn(x[r:r + 1]) for r in range(rows)])


def silu(x: mx.array) -> mx.array:
    return nn.silu(x)

