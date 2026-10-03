"""Decode rows' unquantized matmuls in MLX's one-row gemv order, the rows sharing the weight reads."""

from __future__ import annotations

from functools import cache
from typing import Any

import mlx.core as mx

from tensorfold.kernels.deepseek.v41.rows import metal

# -- decode rows' unquantized matmuls: MLX's gemv per-row order, (TN, BN) matched on first use (``gemv_config``)
_GEMV_ROWS = r"""
  const uint lane = thread_index_in_simdgroup;
  const int s = int(simdgroup_index_in_threadgroup);
  constexpr int OUTS = THREADS / 32 / BN;
  const int n = int(threadgroup_position_in_grid.x) * OUTS + s / BN;
  const int sgN = s % BN;
  constexpr int blockN = 32 * BN * TN;
  threadgroup float part[THREADS / 32][R];
  float result[R];
  for (int r = 0; r < R; ++r) result[r] = 0.0f;
  if (n < N) {
    const device W_T* mat = W + size_t(n) * K;
    const device X_T* v = X + (n / NPG) * K;
    int bn = (sgN * 32 + int(lane)) * TN;
    for (int i = 0; i < K / blockN; ++i) {
      float m[TN];
      for (int t = 0; t < TN; ++t) m[t] = float(mat[bn + t]);
      for (int r = 0; r < R; ++r) {
        float acc = result[r];
        for (int t = 0; t < TN; ++t) acc = fma(m[t], float(v[size_t(r) * XW + bn + t]), acc);
        result[r] = acc;
      }
      bn += blockN;
    }
    if (K % blockN) {
      float m[TN];
      for (int t = 0; t < TN; ++t) m[t] = bn + t < K ? float(mat[bn + t]) : 0.0f;
      for (int r = 0; r < R; ++r) {
        float acc = result[r];
        for (int t = 0; t < TN; ++t) acc = fma(m[t], bn + t < K ? float(v[size_t(r) * XW + bn + t]) : 0.0f, acc);
        result[r] = acc;
      }
    }
  }
  for (int r = 0; r < R; ++r)
    for (ushort sn = 16; sn >= 1; sn >>= 1) result[r] += simd_shuffle_down(result[r], sn);
  if (BN > 1) {
    if (lane == 0) for (int r = 0; r < R; ++r) part[s][r] = result[r];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0 && sgN == 0)
      for (int b = 1; b < BN; ++b) for (int r = 0; r < R; ++r) result[r] += part[s + b][r];
  }
  if (lane == 0 && sgN == 0 && n < N)
    for (int r = 0; r < R; ++r) OUT[size_t(r) * N + n] = OUT_T(result[r]);
"""


@cache
def _gemv_kernel() -> Any:
    return mx.fast.metal_kernel(name="tf_dsv41_gemv_rows", input_names=["X", "W"], output_names=["OUT"],
                                source=_GEMV_ROWS)


def gemv_rows(x: mx.array, w: mx.array, out_dtype: Any, tn: int, bn: int, groups: int = 1) -> mx.array:
    rows, n, k = int(x.shape[0]), int(w.shape[0]), int(w.shape[1])
    threads = 256 if bn <= 8 else 32 * bn
    outs = threads // 32 // bn
    return _gemv_kernel()(inputs=[mx.contiguous(x), w],
                          template=[("R", rows), ("K", k), ("N", n), ("NPG", n // groups), ("XW", int(x.shape[1])),
                                    ("TN", tn), ("BN", bn), ("THREADS", threads), ("W_T", w.dtype), ("X_T", x.dtype),
                                    ("OUT_T", out_dtype)],
                          grid=(-(-n // outs) * threads, 1, 1), threadgroup=(threads, 1, 1), output_shapes=[(rows, n)],
                          output_dtypes=[out_dtype])[0]


_GEMV_CONFIGS: dict[tuple, tuple[int, int] | None] = {}
# shapes whose MLX one-row order a strong random test pins down uniquely (bf16 outputs hide small differences)
GEMV_CHECKED = {mx.float32}


def gemv_config(w: mx.array, x_dtype: Any, out_dtype: Any, groups: int) -> tuple[int, int] | None:
    """The (TN, BN) whose rows equal MLX's one-row matmul for this weight's shape, checked once (None: none does)."""

    key = (tuple(w.shape), w.dtype, x_dtype, out_dtype, groups)
    if key in _GEMV_CONFIGS:
        return _GEMV_CONFIGS[key]
    n, k = int(w.shape[0]), int(w.shape[1])
    found = None
    if metal() and (w.dtype in GEMV_CHECKED or groups > 1) and n % groups == 0:
        g = (mx.random.normal((n, k), key=mx.random.key(7)) * 0.05).astype(w.dtype)
        x = (mx.random.normal((32, groups * k), key=mx.random.key(8)) *
             mx.random.uniform(0.1, 10, (32, 1), key=mx.random.key(9))).astype(x_dtype)

        def one(a: mx.array) -> mx.array:
            if groups == 1:
                return mx.matmul(a, g.T).astype(out_dtype)
            return mx.einsum("lgd,grd->lgr", a.reshape(1, groups, -1), g.reshape(groups, n // groups, -1)) \
                .reshape(1, -1).astype(out_dtype)

        ref = mx.concatenate([one(x[r:r + 1]) for r in range(32)])
        for tn, bn in ((4, 1), (4, 8), (4, 4), (4, 2), (4, 16), (1, 1), (8, 1)):
            if all(bool(mx.array_equal(gemv_rows(x[i:i + 16], g, out_dtype, tn, bn, groups), ref[i:i + 16]).item())
                   for i in (0, 16)):
                found = (tn, bn)
                break
    _GEMV_CONFIGS[key] = found
    return found
