"""Row-exact affine decode projections of any width oQ writes: the M5 lane matmul, or the simd row kernels (any Mac)."""

from __future__ import annotations

from typing import Any, Sequence

import mlx.core as mx

from tensorfold.kernels import device
from tensorfold.kernels.qwen.dense.v1 import lane_qmm

BACKENDS = ("lane", "rows")

_rows_backend: Any = None


def tensor_units() -> bool:
    """Whether this GPU has the M5 generation's tensor units (the lane matmul needs them)."""

    return device.tensor_units()


def rows_backend() -> Any:
    """``row_matmul``'s simd backend: 4-bit simd_qmm, 5/6/8-bit simd_qmm_bits, else affine_rows (checked per shape)."""

    global _rows_backend
    if _rows_backend is None:
        from tensorfold.kernels.qwen.dense.v1.row_matmul import simd_qmm_backend

        _rows_backend = simd_qmm_backend()
    return _rows_backend


def spec(linear: Any) -> tuple[int, int, str]:
    """(bits, group size, mode) of an MLX quantized linear or embedding."""

    return int(linear.bits), int(linear.group_size), str(getattr(linear, "mode", "affine"))


class _Stack:
    """Linears of one width and group read as one weight (rows of the outputs concatenated)."""

    def __init__(self, linears: Sequence[Any], backend: str) -> None:
        bits, group, mode = spec(linears[0])
        if mode != "affine":
            raise ValueError(f"Gemma's dense decode reads MLX affine weights, not {mode}")
        parts = [(l["weight"], l["scales"], l["biases"]) for l in linears]
        weight, scales, biases = (mx.concatenate([p[i] for p in parts]) if len(parts) > 1 else parts[0][i]
                                  for i in range(3))
        self.bits, self.group, self.backend = bits, group, backend
        self.n, self.k = int(weight.shape[0]), int(weight.shape[1]) * 32 // bits
        if backend == "lane":
            if not lane_qmm.readable(bits, group) or self.k % 64 or self.n % 32:
                raise ValueError(f"the lane matmul does not read {bits}-bit weights in groups of {group} "
                                 f"(N {self.n}, K {self.k})")
            self.nt = 64 if bits == 4 and self.n % 64 == 0 else 32
            self.weight = lane_qmm.tile_weight(weight, self.nt, group, bits=bits)
            self.sbt = lane_qmm.pack_scales(scales, biases)
            mx.eval(self.weight, self.sbt)
        else:
            self.weight, self.scales, self.biases = weight, scales, biases
            if len(parts) > 1:
                mx.eval(self.weight, self.scales, self.biases)
            rows_backend().prepare([(self.weight, self.scales, self.biases, group, bits)])

    def __call__(self, x: mx.array) -> mx.array:
        if self.backend == "lane":
            return lane_qmm.lane_matmul(x, self.weight, self.sbt, tiled=True, nt=self.nt, group=self.group)
        return rows_backend()(x, self.weight, self.scales, self.biases, self.group, self.bits)


class Projection:
    """One linear, or several that read the same input stacked along the output; widths may differ between them."""

    def __init__(self, linears: Sequence[Any], backend: str) -> None:
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}")
        self.backend = backend
        runs: list[list[Any]] = []
        for linear in linears:                       # consecutive linears of one width share a weight
            if runs and spec(runs[-1][0]) == spec(linear):
                runs[-1].append(linear)
            else:
                runs.append([linear])
        self.stacks = [_Stack(run, backend) for run in runs]
        sizes = [int(l["weight"].shape[0]) for l in linears]
        self.n = sum(sizes)
        self.cuts = [sum(sizes[:i + 1]) for i in range(len(sizes) - 1)]

    def __call__(self, x: mx.array) -> mx.array:
        outs = [stack(x) for stack in self.stacks]
        return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=-1)

    def split(self, y: mx.array) -> list[mx.array]:
        return mx.split(y, self.cuts, axis=-1) if self.cuts else [y]


__all__ = ["BACKENDS", "Projection", "rows_backend", "spec", "tensor_units"]
