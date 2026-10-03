"""Row-exact decode projections of any MLX affine width: the lane matmul, or the packed affine row kernel."""

from __future__ import annotations

from typing import Any, Sequence

import mlx.core as mx

from tensorfold.kernels import device
from tensorfold.kernels.qwen.dense.v1 import affine_rows, lane_qmm

BACKENDS = ("lane", "rows")


def tensor_units() -> bool:
    """Whether this GPU has the M5 generation's tensor units (applegpu_g17 and later)."""

    return device.tensor_units()


def fmt(linear: Any) -> tuple[int, int, str]:
    """A quantized linear's (bits, group size, mode)."""

    return int(getattr(linear, "bits", 0) or 0), int(getattr(linear, "group_size", 0) or 0), \
        str(getattr(linear, "mode", "affine"))


def regroup(scales: mx.array, biases: mx.array, group: int, to: int) -> tuple[mx.array, mx.array]:
    """Scales and biases of groups of ``group`` as groups of ``to`` (``group`` a multiple): each repeated in place."""

    if group == to:
        return scales, biases
    times = group // to
    return mx.repeat(scales, times, axis=-1), mx.repeat(biases, times, axis=-1)


class Projection:
    """One affine linear, or several of one format that read the same input stacked along the output, for decode."""

    def __init__(self, linears: Sequence[Any], backend: str) -> None:
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}")
        formats = {fmt(l) for l in linears}
        if len(formats) != 1:
            raise ValueError(f"stacked projections must share one format, got {sorted(formats)}")
        self.bits, self.group, mode = formats.pop()
        if mode != "affine" or not affine_rows.readable(self.bits, self.group):
            raise ValueError(f"Laguna's decode projections read MLX affine 2- to 8-bit weights in groups of 32, 64 "
                             f"or 128, got {self.bits}-bit {mode} in groups of {self.group}")
        parts = [(l["weight"], l["scales"], l["biases"]) for l in linears]
        weight, scales, biases = (mx.concatenate([p[i] for p in parts]) if len(parts) > 1 else parts[0][i]
                                  for i in range(3))
        self.n = int(weight.shape[0])
        self.k = int(weight.shape[1]) * 32 // self.bits
        self.cuts = [int(sum(int(p[0].shape[0]) for p in parts[:i + 1])) for i in range(len(parts) - 1)]
        self.backend = backend if backend == "rows" or self._lane_fits() else "rows"
        if self.backend == "lane":
            group = 64 if self.group == 128 else self.group
            scales, biases = regroup(scales, biases, self.group, group)
            self.lane_group = group
            self.tiled = self.n % lane_qmm.NT == 0
            self.nt = 64 if self.bits == 4 and self.n % 64 == 0 else lane_qmm.NT
            self.weight = (lane_qmm.tile_weight(weight, self.nt, group, bits=self.bits) if self.tiled
                           else mx.contiguous(weight))
            self.sbt = lane_qmm.pack_scales(scales, biases)
            mx.eval(self.weight, self.sbt)
        else:
            self.weight, self.scales, self.biases = weight, scales, biases
            if len(parts) > 1:
                mx.eval(self.weight, self.scales, self.biases)

    def _lane_fits(self) -> bool:
        group = 64 if self.group == 128 else self.group
        if not lane_qmm.reads(self.bits, group) or self.k % 64:
            return False
        if self.bits == 4:
            return self.n % lane_qmm.NT == 0
        return self.n % 4 == 0

    def __call__(self, x: mx.array) -> mx.array:
        if self.backend == "lane":
            rows = 1
            for d in x.shape[:-1]:
                rows *= int(d)
            if rows <= lane_qmm.MAX_ROWS:
                return lane_qmm.lane_matmul(x, self.weight, self.sbt, tiled=self.tiled, nt=self.nt,
                                            group=self.lane_group)
            x2 = x.reshape(-1, self.k)        # rows are independent: longer inputs in calls of MAX_ROWS rows
            out = mx.concatenate([self(x2[i:i + lane_qmm.MAX_ROWS]) for i in range(0, rows, lane_qmm.MAX_ROWS)])
            return out.reshape(*x.shape[:-1], self.n)
        return affine_rows.qmm(x, self.weight, self.scales, self.biases, self.group, self.bits)

    def split(self, y: mx.array) -> list[mx.array]:
        return mx.split(y, self.cuts, axis=-1) if self.cuts else [y]


def stacks(linears: Sequence[Any], backend: str) -> list[tuple[Projection, list[int]]]:
    """Linears that read one input as projections, one per format (in first-seen order), each with its members."""

    groups: dict[tuple[int, int, str], list[int]] = {}
    for i, linear in enumerate(linears):
        groups.setdefault(fmt(linear), []).append(i)
    return [(Projection([linears[i] for i in members], backend), members) for members in groups.values()]


class Projections:
    """Several linears of one input in as few format-uniform matmuls as their widths allow, outputs in order."""

    def __init__(self, linears: Sequence[Any], backend: str) -> None:
        self.parts = stacks(linears, backend)
        self.count = len(linears)

    def __call__(self, x: mx.array) -> list[mx.array]:
        out: list[Any] = [None] * self.count
        for projection, members in self.parts:
            for i, y in zip(members, projection.split(projection(x))):
                out[i] = y
        return out

    def joined(self, x: mx.array) -> mx.array:
        """The outputs side by side [..., sum N] (one matmul's output as it is when one format covers them all)."""

        if len(self.parts) == 1:
            return self.parts[0][0](x)
        return mx.concatenate(self(x), axis=-1)

    @property
    def backends(self) -> list[str]:
        return [p.backend for p, _ in self.parts]


__all__ = ["BACKENDS", "Projection", "Projections", "fmt", "regroup", "stacks", "tensor_units"]
