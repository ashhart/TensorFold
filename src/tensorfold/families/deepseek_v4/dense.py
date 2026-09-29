"""Decode rows' dense 4-bit projections through simd_qmm on Metal: every row its one-row bits, MMA from 3 rows."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from tensorfold.families.glm5_next.linear import Q, project
from tensorfold.kernels.glm.flash.v1 import kernels as GK
from tensorfold.kernels.qwen.dense.v1 import simd_qmm as SQ


def fits(q: Any) -> bool:
    return (GK.metal() and isinstance(q, Q) and q.bits == 4 and q.group in (32, 64)
            and q.scales.dtype == mx.bfloat16 and q.outs % 8 == 0 and q.ins % q.group == 0)


def prepare(qs: list[Any]) -> int:
    """Check each shape's scalar and MMA kernels agree (else one row takes the MMA kernel too); shapes checked."""

    seen: set[tuple[int, int, int]] = set()
    for q in qs:
        if not fits(q):
            continue
        shape = (q.outs, q.ins, q.group)
        if shape in seen:
            continue
        seen.add(shape)
        if not SQ.check(q.weight, q.scales, q.biases, group_size=q.group):
            SQ.mma_one_row.add(shape)
    return len(seen)


def dense(x: mx.array, q: Any, rows_exact: bool) -> mx.array:
    """x [R, K] through a dense projection: simd_qmm for decode rows on Metal, else GLM's ``project``."""

    if rows_exact and fits(q):
        return SQ.qmm(x, q.weight, q.scales, q.biases, q.group)
    return project(x, q, rows_exact=rows_exact)
