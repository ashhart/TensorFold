"""Quantized linears in the checkpoint's MLX affine layout, applied with one-row bits on the decode path."""

from __future__ import annotations

import os
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from tensorfold.families.glm5_next.config import BITS, GROUPS
from tensorfold.kernels.glm.flash.v1 import kernels as K
from tensorfold.kernels.qwen.flash_next.v1 import prefill_mm as PM


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
        if x.ndim == 2:                                     # prompt rows take MLX's qmm with wider tiles: its bits
            return PM.matmul(x, self.weight, self.scales, self.biases, group=self.group, bits=self.bits).astype(x.dtype)
        return mx.quantized_matmul(x, self.weight, self.scales, self.biases, transpose=True, group_size=self.group,
                                   bits=self.bits).astype(x.dtype)

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


def kernel_q(*qs: Any) -> bool:
    """Weights the prompt kernels were proven on: ``Q``s of 4 or 8 bits with bf16 scales and biases (not Dense)."""

    return all(isinstance(q, Q) and q.bits in (4, 8) and q.scales.dtype == q.biases.dtype == mx.bfloat16 for q in qs)


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


# "rows" (default): MLX's one-row arithmetic (a window by qmv_rows).  "matrix": the decode path's dense projections
# on the matrix units at every row count, one row included -- simd_qmm / simd_qmm_bits / affine_rows before M5
# (the Qwen dense backend, each shape checked once), lane_qmm on M5 -- so a row's bits do not depend on the window.
DENSE = os.environ.get("TF_GLM_DENSE", "rows")
_MATRIX: dict[int, tuple] = {}            # id(Q) -> (weight, prepared arrays)
_BACKEND: list = []


def _tensor_units() -> bool:
    from tensorfold.families.qwen3_5 import tensor_units

    return bool(tensor_units())


def matrix(x: mx.array, q: "Q") -> mx.array:
    """x [R, K] bf16 through a 2-D affine linear on the matrix units (see ``DENSE``); groups of 128 read as 2 x 64."""

    if not _BACKEND:
        if _tensor_units():
            _BACKEND.append("lane")
        else:
            from tensorfold.kernels.qwen.dense.v1 import row_matmul

            _BACKEND.append(row_matmul.simd_qmm_backend())
    backend = _BACKEND[0]
    hit = _MATRIX.get(id(q))
    if hit is None or hit[0] is not q.weight:
        scales, biases, group = q.scales, q.biases, q.group
        if group == 128:
            scales, biases, group = mx.repeat(scales, 2, axis=1), mx.repeat(biases, 2, axis=1), 64
        if backend == "lane":
            from tensorfold.kernels.qwen.dense.v1 import lane_qmm

            n = q.outs
            nt = 64 if (q.bits == 4 and n % 64 == 0) else 32 if n % 32 == 0 else 0
            tiled = lane_qmm.tile_weight(q.weight, nt, group, bits=q.bits) if nt else q.weight
            sbt = lane_qmm.pack_scales(scales, biases)
            mx.eval(tiled, sbt)
            hit = (q.weight, tiled, sbt, nt, group)
        else:
            mx.eval(scales, biases)
            backend.prepare([(q.weight, scales, biases, group, q.bits)])
            hit = (q.weight, scales, biases, group)
        _MATRIX[id(q)] = hit
    dtype = x.dtype
    xb = x if dtype == mx.bfloat16 else x.astype(mx.bfloat16)
    if backend == "lane":
        from tensorfold.kernels.qwen.dense.v1 import lane_qmm

        _, tiled, sbt, nt, group = hit
        kwargs = {"tiled": bool(nt), "nt": nt or lane_qmm.NT, "group": group}
        rows, most = int(xb.shape[0]), lane_qmm.MAX_ROWS
        y = (lane_qmm.lane_matmul(xb, tiled, sbt, **kwargs) if rows <= most else
             mx.concatenate([lane_qmm.lane_matmul(xb[i:i + most], tiled, sbt, **kwargs) for i in range(0, rows, most)]))
    else:
        _, scales, biases, group = hit
        y = backend(xb, q.weight, scales, biases, group, q.bits)
    return y if dtype == mx.bfloat16 else y.astype(dtype)


def project(x: mx.array, q: Any, *, rows_exact: bool) -> mx.array:
    """x [R, K] through a linear: one row by MLX's call, a decode window by ``qmv_rows`` where it fits, else by row."""

    rows = int(x.shape[0])
    if DENSE == "matrix" and rows_exact and K.metal():                 # Metal kernels: the CPU keeps MLX's
        if isinstance(q, QSplit):
            return mx.concatenate([project(x, p, rows_exact=True) for p in q.parts], axis=-1)
        if isinstance(q, Q) and q.weight.ndim == 2:
            return matrix(x, q)
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


class ChunkQueue:
    """A prompt pass's chunks two at a time: queue one, wait for the one before, so its buffers serve the next."""

    def __init__(self) -> None:
        self.last: tuple[mx.array, ...] | None = None

    def push(self, *arrays: mx.array) -> None:
        mx.async_eval(*arrays)
        if self.last is not None:
            mx.eval(*self.last)
        self.last = arrays

