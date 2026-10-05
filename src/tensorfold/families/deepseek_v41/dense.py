"""Decode rows' affine multi-bit projections: per-row quantized matmul (exact; speed is a later milestone)."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from tensorfold.families.glm5_next.linear import Q, project


def fits(q: Any) -> bool:
    return isinstance(q, Q)


def prepare(qs: list[Any]) -> int:
    """Nothing to build for M1: every projection runs through MLX's per-row quantized matmul."""
    return 0


def dense(x: mx.array, q: Any, rows_exact: bool) -> mx.array:
    """x [R, K] through a dense projection: each decode row its one-row quantized_matmul call."""
    return project(x, q, rows_exact=rows_exact)
