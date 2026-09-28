"""Row-exact 4-bit decode projections: the tensor-unit lane matmul (M5 on) or MLX's qmv loop a row (any Mac)."""

from __future__ import annotations

from typing import Any, Sequence

import mlx.core as mx

from tensorfold.kernels.nemotron.lightning.v1 import rows as row_kernels
from tensorfold.kernels.qwen.dense.v1 import lane_qmm

BACKENDS = ("lane", "rows")


def tensor_units() -> bool:
    """Whether this GPU has the M5 generation's tensor units (applegpu_g17 and later)."""

    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    digits = "".join(ch for ch in str(info.get("architecture", "")).removeprefix("applegpu_g") if ch.isdigit())
    return bool(digits) and int(digits) >= 17


def _parts(linear: Any) -> tuple[mx.array, mx.array, mx.array]:
    return linear["weight"], linear["scales"], linear["biases"]


class Projection:
    """One 4-bit linear, or several that read the same input stacked along the output, for decode rows."""

    def __init__(self, linears: Sequence[Any], backend: str) -> None:
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}")
        first = linears[0]
        self.group = int(first.group_size)
        if any(int(l.bits) != 4 or int(l.group_size) != self.group or getattr(l, "mode", "affine") != "affine"
               for l in linears):
            raise ValueError("Gemma's decode projections read MLX affine 4-bit weights of one group size")
        parts = [_parts(l) for l in linears]
        weight, scales, biases = (mx.concatenate([p[i] for p in parts]) if len(parts) > 1 else parts[0][i]
                                  for i in range(3))
        self.n, self.k = int(weight.shape[0]), int(weight.shape[1]) * 8
        self.cuts = [int(sum(int(p[0].shape[0]) for p in parts[:i + 1])) for i in range(len(parts) - 1)]
        self.backend = backend
        if backend == "lane":
            if self.group not in (32, 64) or self.k % 64 or self.n % 32:
                raise ValueError("the lane matmul takes groups of 32 or 64, K a multiple of 64, N a multiple of 32")
            self.nt = 64 if self.n % 64 == 0 else 32
            self.weight = lane_qmm.tile_weight(weight, self.nt, self.group, bits=4)
            self.sbt = lane_qmm.pack_scales(scales, biases)
            mx.eval(self.weight, self.sbt)
        else:
            if not row_kernels.fits(weight, scales, self.group, 4):
                raise ValueError("the row kernels take groups of 32, 64 or 128, K a multiple of 64, N a multiple of 8")
            self.weight, self.scales, self.biases = weight, scales, biases
            if len(parts) > 1:
                mx.eval(self.weight, self.scales, self.biases)

    def __call__(self, x: mx.array) -> mx.array:
        if self.backend == "lane":
            return lane_qmm.lane_matmul(x, self.weight, self.sbt, tiled=True, nt=self.nt, group=self.group)
        return row_kernels.qmv(x, self.weight, self.scales, self.biases, self.group)

    def split(self, y: mx.array) -> list[mx.array]:
        return mx.split(y, self.cuts, axis=-1) if self.cuts else [y]


__all__ = ["BACKENDS", "Projection", "tensor_units"]
