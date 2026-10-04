"""H3 fused QKV: int8 projection with the q/k RMSNorm and rotary applied in the kernel, written head-major."""

# Folding the q/k normalisation and RoPE into the int8 QKV projection follows antirez's h3.c (MIT,
# https://github.com/antirez/h3.c, revision 8974cc0) and Sol-H3's "QKNorm + partial RoPE" fusion. The kernel is
# written for `mx.fast.metal_kernel`; no source is copied.

from __future__ import annotations

import hashlib
from typing import Any

import mlx.core as mx

from .mlp_int8 import _HEADER, _PRELUDE, TILE, _check, _mdims, quantize_rows, quantize_weight

# The checkpoint's QKV rows are interleaved per head, [h0: q, k, v][h1: q, k, v]..., so one 128-column tile is
# exactly q, k or v of one head for 128 rows. The projection lands in its final head-major place first; the q
# and k tiles are then normalised and rotated in place, two threads per row. Both read the whole row before
# either writes, and each writes only the channel pairs it owns.
# Y: (3, HEADS, M, 128) bfloat.  NW: (2, 128) q and k norm weights.  CS, SN: (M, ROT) rotary tables.
_QKV = _PRELUDE + r"""
  constexpr int HEADS = N / (3 * T);
  constexpr int HALF = ROT / 2;
  const int tile = threadgroup_position_in_grid.x;
  const int head = tile / 3, kind = tile % 3;
  const int64_t base = (int64_t)(kind * HEADS + head) * M * T;
  auto acc = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), int32_t>();
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < CAP; i++) acc[i] = 0;
  for (int t = 0; t < K / T; t++) {
    auto a = x.slice<T, T>(t * T, r0);
    auto b = w.slice<T, T>(t * T, n0);
    op.run(a, b, acc);
  }
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < CAP; i++) {
    auto ids = acc.get_multidimensional_index(i);
    const int row = r0 + ids[1];
    if (row < M) Y[base + (int64_t)row * T + ids[0]] = static_cast<bfloat>(float(acc[i]) * sc[ids[1]] * WS[n0 + ids[0]]);
  }
  threadgroup_barrier(mem_flags::mem_device);
  const int row = r0 + (thread_position_in_threadgroup.x >> 1);
  const int part = thread_position_in_threadgroup.x & 1;
  const bool live = kind < 2 && row < M;
  float v[T];
  if (live) {
    float squares = 0.0f;
    for (int c = 0; c < T; c++) { v[c] = float(Y[base + (int64_t)row * T + c]); squares = fma(v[c], v[c], squares); }
    const float inverse = rsqrt(squares / float(T) + EPS);
    for (int c = 0; c < T; c++) v[c] *= inverse * NW[kind * T + c];
  }
  threadgroup_barrier(mem_flags::mem_device);
  if (live) {
    const int64_t at = base + (int64_t)row * T;
    const int64_t rot = (int64_t)row * ROT;
    for (int c = part * (HALF / 2); c < (part + 1) * (HALF / 2); c++) {
      const float lo = v[c], hi = v[c + HALF];
      Y[at + c] = static_cast<bfloat>(lo * CS[rot + c] - hi * SN[rot + c]);
      Y[at + c + HALF] = static_cast<bfloat>(hi * CS[rot + c + HALF] + lo * SN[rot + c + HALF]);
    }
    const int rest = (T - ROT) / 2;
    for (int c = ROT + part * rest; c < ROT + (part + 1) * rest; c++) Y[at + c] = static_cast<bfloat>(v[c]);
  }
"""

_compiled: dict[tuple, Any] = {}


def _kernel(n: int, k: int, rotary: int, eps: float) -> Any:
    key = (n, k, rotary, eps)
    run = _compiled.get(key)
    if run is None:
        source = (f"  constexpr int N = {n};\n  constexpr int K = {k};\n  constexpr int GROUP = {k};\n"
                  f"  constexpr int ROT = {rotary};\n  constexpr float EPS = {eps!r}f;\n") + _QKV
        name = "h3_int8_qkv_" + hashlib.sha256((_HEADER + source).encode()).hexdigest()[:16]
        run = _compiled[key] = mx.fast.metal_kernel(
            name=name, input_names=["X", "XS", "W", "WS", "NW", "CS", "SN", "mdims"], output_names=["Y"],
            source=source, header=_HEADER)
    return run


class Int8QKV:
    """``Attention.qkv`` in one kernel: int8 projection, q/k RMSNorm and rotary, head-major outputs."""

    def __init__(self, qkv_weight: mx.array, q_norm_weight: mx.array, k_norm_weight: mx.array, heads: int,
                 head_dim: int, eps: float = 1e-5) -> None:
        self.heads, self.head_dim, self.eps = heads, head_dim, float(eps)
        self.hidden = qkv_weight.shape[1]
        if head_dim != TILE or qkv_weight.shape[0] != 3 * heads * head_dim or self.hidden % 256:
            raise ValueError(f"the fused QKV needs {TILE}-channel heads, a (3 x heads x {TILE}, K) weight and K in "
                             f"multiples of 256, got {tuple(qkv_weight.shape)} for {heads} heads of {head_dim}")
        self.w, self.s = quantize_weight(qkv_weight)
        self.norms = mx.stack([q_norm_weight, k_norm_weight]).astype(mx.float32)
        mx.eval(self.w, self.s, self.norms)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """q, k, v as (1, heads, rows, head_dim) from x (1, rows, hidden) and rotary tables (rows, rotary)."""

        if x.ndim != 3 or x.shape[0] != 1:
            raise ValueError(f"the fused QKV takes one sequence (1, rows, {self.hidden}), got {tuple(x.shape)}")
        rotary = cos.shape[-1]
        if rotary % 4 or rotary > TILE or (TILE - rotary) % 2 or cos.shape != sin.shape:
            raise ValueError(f"rotary tables must be (rows, R) with R a multiple of 4 up to {TILE}, got {cos.shape}")
        xq, xs, rows = quantize_rows(x[0], self.hidden)
        if cos.shape[0] != rows:
            raise ValueError(f"rotary tables cover {cos.shape[0]} rows, the sequence has {rows}")
        padded, n = xq.shape[0], self.w.shape[0]
        _check(padded, self.hidden, n, self.hidden)
        out = _kernel(n, self.hidden, rotary, self.eps)(
            inputs=[xq, xs, self.w, self.s, self.norms, cos.astype(mx.float32), sin.astype(mx.float32),
                    _mdims(rows, padded)],
            grid=(n // TILE * 256, padded // TILE, 1), threadgroup=(256, 1, 1),
            output_shapes=[(3, self.heads, rows, TILE)], output_dtypes=[mx.bfloat16])[0].astype(x.dtype)
        return out[0][None], out[1][None], out[2][None]
