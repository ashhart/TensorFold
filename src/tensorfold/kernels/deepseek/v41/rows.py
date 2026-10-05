"""Affine row kernels for DeepSeek-V4.1: the 3-bit g64 row path (M1: the scalar per-row route, exact).

The fused MMA-from-rows lanes (shared with the Qwen multi-bit infrastructure) are a later milestone;
until then every decode row runs its one-row ``mx.quantized_matmul`` call, which is bit-exact with the
serial path by construction.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from tensorfold.families.glm5_next.linear import Q, project


def fits(q: Any) -> bool:
    return isinstance(q, Q)


def prepare(qs: list[Any]) -> int:
    """Nothing to build for M1: every projection runs through MLX's per-row quantized matmul."""
    return 0


def rows(x: mx.array, q: Any, rows_exact: bool) -> mx.array:
    """x [R, K] through an affine projection: each decode row its one-row call (exact), prompts one call."""
    return project(x, q, rows_exact=rows_exact)
