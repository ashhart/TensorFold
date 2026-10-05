"""Row-exact affine decode projections at any MLX width: the tensor-unit lane matmul (M5 on) or a row matvec."""

from __future__ import annotations

from typing import Any, Sequence

import mlx.core as mx

from tensorfold.kernels import device
from tensorfold.kernels.nemotron.lightning.v1 import rows as row_kernels
from tensorfold.kernels.qwen.dense.v1 import affine_rows, lane_qmm, simd_qmm, simd_qmm_bits

BACKENDS = ("lane", "rows", "matrix", "auto")
# widths and groups some decode kernel here reads (MLX affine)
BITS = affine_rows.BITS
GROUP_SIZES = affine_rows.GROUP_SIZES


def tensor_units() -> bool:
    """Whether this GPU has the M5 generation's tensor units (applegpu_g17 and later)."""

    return device.tensor_units()


def readable(bits: int, group_size: int, mode: str = "affine") -> bool:
    """Whether a decode projection here reads MLX weights of this width, group and mode (on some backend)."""

    return affine_rows.readable(int(bits), int(group_size), str(mode or "affine"))


def _parts(linear: Any) -> tuple[mx.array, mx.array, mx.array]:
    return linear["weight"], linear["scales"], linear["biases"]


def _spec(linear: Any) -> tuple[int, int]:
    mode = getattr(linear, "mode", "affine")
    bits, group = int(getattr(linear, "bits", 0) or 0), int(getattr(linear, "group_size", 0) or 0)
    if not readable(bits, group, mode):
        raise ValueError(f"Gemma's decode projections read MLX affine {'/'.join(map(str, BITS))}-bit weights in groups "
                         f"of {'/'.join(map(str, GROUP_SIZES))}; got {bits}-bit {mode} in groups of {group}")
    return bits, group


def share(linears: Sequence[Any], stacked: tuple[mx.array, mx.array, mx.array]) -> None:
    """Point each linear at its rows of the stacked arrays, so mlx_lm's prompt forward reads the stack's bytes."""

    start = 0
    for linear in linears:
        rows = int(linear["weight"].shape[0])
        for name, array in zip(("weight", "scales", "biases"), stacked):
            setattr(linear, name, array[start:start + rows])
        start += rows
    mx.eval([linear[name] for linear in linears for name in ("weight", "scales", "biases")])


def matrix_kind(weight: mx.array, scales: mx.array, biases: mx.array, bits: int, group: int) -> str:
    """The simdgroup matrix kernel that reads this weight ("simd" or "simd_bits"), or "" when neither does."""

    n, k = int(weight.shape[0]), int(weight.shape[1]) * 32 // bits
    if (bits == 4 and group in (32, 64) and scales.dtype == mx.bfloat16 and biases.dtype == mx.bfloat16
            and n % 8 == 0 and k % 64 == 0):
        return "simd"
    return "simd_bits" if simd_qmm_bits.reads(weight, scales, biases, group, bits) and k % 64 == 0 else ""


def prefers_matrix(bits: int, group: int, n: int, k: int) -> bool:
    """Whether this format and shape is faster on the matrix kernels over 2 to 16 rows and level at one."""

    if bits == 4 and group in (32, 64):
        return n * k >= 1 << 22         # a 128-wide router stays on the row matvec
    return True


class _Run:
    """Linears of one width and group stacked along the output: one matmul."""

    def __init__(self, linears: Sequence[Any], bits: int, group: int, backend: str) -> None:
        parts = [_parts(l) for l in linears]
        weight, scales, biases = (mx.concatenate([p[i] for p in parts]) if len(parts) > 1 else parts[0][i]
                                  for i in range(3))
        self.bits, self.group = bits, group
        self.n, self.k = int(weight.shape[0]), int(weight.shape[1]) * 32 // bits
        if backend == "lane" and not (lane_qmm.reads(bits, group) and self.k % 64 == 0 and self.n % 32 == 0
                                      and scales.dtype == mx.bfloat16):
            backend = "rows"                    # a width or shape the lane matmul does not take: the row matvec
        self.one = ""
        if backend in ("matrix", "auto"):
            matrix = matrix_kind(weight, scales, biases, bits, group)
            if matrix and (backend == "matrix" or prefers_matrix(bits, group, self.n, self.k)):
                self.kind, self.weight, self.scales, self.biases = matrix, weight, scales, biases
                self._check()
                if len(parts) > 1:
                    mx.eval(self.weight, self.scales, self.biases)
                    share(linears, (weight, scales, biases))
                return
            backend = "rows"
        if backend == "lane":
            self.kind = "lane"
            self.nt = 64 if (bits == 4 and self.n % 64 == 0) else 32
            self.weight = lane_qmm.tile_weight(weight, self.nt, group, bits=bits)
            self.sbt = lane_qmm.pack_scales(scales, biases)
            mx.eval(self.weight, self.sbt)
            return
        self.kind = "q4" if bits == 4 and row_kernels.fits(weight, scales, group, 4) else "affine"
        if self.kind == "affine":
            affine_rows.shape(weight, scales, biases, group, bits)     # refuses a shape it cannot address
        self.weight, self.scales, self.biases = weight, scales, biases
        if len(parts) > 1:
            mx.eval(self.weight, self.scales, self.biases)
            share(linears, (weight, scales, biases))

    def _check(self) -> None:
        """A one-row call takes the scalar twin only where it gives the matrix kernel's bits for this weight."""

        w, s, b = self.weight, self.scales, self.biases
        if self.kind == "simd":
            if not simd_qmm.check(w, s, b, group_size=self.group):
                simd_qmm.mma_one_row.add((self.n, self.k, self.group))
        elif not simd_qmm_bits.check(w, s, b, self.bits, self.group):
            self.one = "mma"

    def __call__(self, x: mx.array) -> mx.array:
        if self.kind == "simd":
            return simd_qmm.qmm(x, self.weight, self.scales, self.biases, self.group)
        if self.kind == "simd_bits":
            return simd_qmm_bits.qmm(x, self.weight, self.scales, self.biases, self.bits, self.group,
                                     kind=self.one or None)
        if self.kind == "lane":
            return lane_qmm.lane_matmul(x, self.weight, self.sbt, tiled=True, nt=self.nt, group=self.group)
        if self.kind == "q4":
            return row_kernels.qmv(x, self.weight, self.scales, self.biases, self.group)
        return affine_rows.qmm(x, self.weight, self.scales, self.biases, self.group, self.bits)


class Projection:
    """One affine linear, or several that read the same input stacked along the output, for decode rows."""

    def __init__(self, linears: Sequence[Any], backend: str) -> None:
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}")
        specs = [_spec(l) for l in linears]
        runs: list[tuple[tuple[int, int], list[Any]]] = []
        for linear, spec in zip(linears, specs):
            if runs and runs[-1][0] == spec:
                runs[-1][1].append(linear)
            else:
                runs.append((spec, [linear]))
        self.runs = [_Run(members, bits, group, backend) for (bits, group), members in runs]
        self.n, self.k = sum(r.n for r in self.runs), self.runs[0].k
        widths = [int(l["weight"].shape[0]) for l in linears]
        self.cuts = [sum(widths[:i + 1]) for i in range(len(widths) - 1)]
        self.backend = backend
        # one run of 4-bit row matvec: the fused q|k|v kernel reads its weights directly
        only = self.runs[0] if len(self.runs) == 1 else None
        self.q4_rows = only is not None and only.kind == "q4"
        if only is not None and only.kind != "lane":
            self.weight, self.scales, self.biases, self.group = only.weight, only.scales, only.biases, only.group
        else:
            self.group = only.group if only is not None else 0
        self.bits = tuple(r.bits for r in self.runs)
        self.kinds = tuple(r.kind for r in self.runs)

    def __call__(self, x: mx.array) -> mx.array:
        if len(self.runs) == 1:
            return self.runs[0](x)
        return mx.concatenate([run(x) for run in self.runs], axis=-1)

    def split(self, y: mx.array) -> list[mx.array]:
        return mx.split(y, self.cuts, axis=-1) if self.cuts else [y]


__all__ = ["BACKENDS", "BITS", "GROUP_SIZES", "Projection", "matrix_kind", "prefers_matrix", "readable",
           "tensor_units"]
