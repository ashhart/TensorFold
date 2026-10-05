"""Row-exact RMSNorm for the V4.1 family: the reference port's manual formula, one row a
call when exact rows are asked.

Two reasons it exists:
- ``mx.fast.rms_norm`` rounds bf16 inputs differently than the reference's explicit ops
  (f * rsqrt(mean(f*f) + eps) * weight, one bf16 rounding at the end); the family must
  produce the port's bits.
- batched CPU reductions round differently than a single row's; decode windows keep their
  one-row bits by taking the boundary one row a call (Metal's ops are per-row already).

Used at every decode-path norm site so a window's rows keep their one-row bits.
"""
from __future__ import annotations

import mlx.core as mx


def _rms_one(x: mx.array, weight: mx.array | None, eps: float) -> mx.array:
    f = x.astype(mx.float32)
    y = f * mx.rsqrt(mx.mean(f * f, axis=-1, keepdims=True) + eps)
    if weight is not None:
        y = y * weight
    return y.astype(x.dtype)


def rms_rows(x: mx.array, weight: mx.array | None, eps: float, rows_exact: bool = False) -> mx.array:
    """x [R, D] (any trailing shape flattens into the row) RMS-normed; row-exact when asked."""
    rows = int(x.shape[0])
    if rows == 1 or not rows_exact:
        return _rms_one(x, weight, eps)
    return mx.concatenate([_rms_one(x[r:r + 1], weight, eps) for r in range(rows)])
