"""The dense Gemma 4 layer's norms and residuals around its matmuls: one threadgroup a row, Gemma v1's rounding."""

from __future__ import annotations

import mlx.core as mx

from tensorfold.kernels.gemma.v1.base import Kernel
from tensorfold.kernels.gemma.v1.glue import _reduce, threads

# hn = h + RMSNorm(o) * wa; n1 = RMSNorm(hn) * w1 (the MLP's input): every load before the first reduction
_ATTN_TAIL = r"""
  const uint t = thread_position_in_threadgroup.x;
  const uint r = threadgroup_position_in_grid.x;
  constexpr int PER = D / T;
  threadgroup float p1[T / 32], p2[T / 32];
  float ov[PER], hin[PER], wa[PER], w1[PER];
  for (int i = 0; i < PER; i++) {
    const int c = int(t) + i * T;
    ov[i] = float(O[int(r) * D + c]); hin[i] = float(H[int(r) * D + c]);
    wa[i] = float(WA[c]); w1[i] = float(W1[c]);
  }
  float ss = 0.0f;
  for (int i = 0; i < PER; i++) ss = fma(ov[i], ov[i], ss);
  float total1;
  REDUCE1
  const float inv1 = metal::precise::rsqrt(total1 / float(D) + eps[0]);
  float hv[PER];
  float ss2 = 0.0f;
  for (int i = 0; i < PER; i++) {
    const int c = int(t) + i * T;
    const float a = float(bfloat(wa[i] * float(bfloat(ov[i] * inv1))));
    const bfloat hn = bfloat(hin[i] + a);
    HN[int(r) * D + c] = hn;
    hv[i] = float(hn);
    ss2 = fma(hv[i], hv[i], ss2);
  }
  float total2;
  REDUCE2
  const float inv2 = metal::precise::rsqrt(total2 / float(D) + eps[0]);
  for (int i = 0; i < PER; i++) {
    const int c = int(t) + i * T;
    N1[int(r) * D + c] = bfloat(w1[i] * float(bfloat(hv[i] * inv2)));
  }
""".replace("REDUCE1", _reduce("ss", "p1", "total1")).replace("REDUCE2", _reduce("ss2", "p2", "total2"))

# hn = (h + RMSNorm(y) * wp) * scalar; next = RMSNorm(hn) * wn (the next layer's input, or the final norm)
_MLP_TAIL = r"""
  const uint t = thread_position_in_threadgroup.x;
  const uint r = threadgroup_position_in_grid.x;
  constexpr int PER = D / T;
  threadgroup float p1[T / 32], p2[T / 32];
  float yv[PER], hin[PER], wp[PER], wn[PER];
  for (int i = 0; i < PER; i++) {
    const int c = int(t) + i * T;
    yv[i] = float(Y[int(r) * D + c]); hin[i] = float(H[int(r) * D + c]);
    wp[i] = float(WP[c]); wn[i] = float(WN[c]);
  }
  const float sc = float(SC[0]);
  float ss = 0.0f;
  for (int i = 0; i < PER; i++) ss = fma(yv[i], yv[i], ss);
  float total1;
  REDUCE1
  const float inv1 = metal::precise::rsqrt(total1 / float(D) + eps[0]);
  float hv[PER];
  float ss2 = 0.0f;
  for (int i = 0; i < PER; i++) {
    const int c = int(t) + i * T;
    const float b = float(bfloat(wp[i] * float(bfloat(yv[i] * inv1))));
    const bfloat hs = bfloat(hin[i] + b);
    const bfloat hn = bfloat(float(hs) * sc);
    HN[int(r) * D + c] = hn;
    hv[i] = float(hn);
    ss2 = fma(hv[i], hv[i], ss2);
  }
  float total2;
  REDUCE2
  const float inv2 = metal::precise::rsqrt(total2 / float(D) + eps[0]);
  for (int i = 0; i < PER; i++) {
    const int c = int(t) + i * T;
    NEXT[int(r) * D + c] = bfloat(wn[i] * float(bfloat(hv[i] * inv2)));
  }
""".replace("REDUCE1", _reduce("ss", "p1", "total1")).replace("REDUCE2", _reduce("ss2", "p2", "total2"))

_attn_tail = Kernel("gemma_dense_attn_tail", _ATTN_TAIL, ["H", "O", "WA", "W1", "eps"], ["HN", "N1"])
_mlp_tail = Kernel("gemma_dense_mlp_tail", _MLP_TAIL, ["H", "Y", "WP", "SC", "WN", "eps"], ["HN", "NEXT"])


def attn_tail(h: mx.array, o: mx.array, w_attn: mx.array, w_mlp: mx.array, eps: mx.array) -> tuple[mx.array, mx.array]:
    """hn = h + RMSNorm(o) * w_attn and RMSNorm(hn) * w_mlp: (hn, the MLP's input), [R, D] bf16."""

    rows, dims = h.shape
    count = threads(dims)
    return _attn_tail((("D", dims), ("T", count)), inputs=[h, o, w_attn, w_mlp, eps],
                      grid=(count * rows, 1, 1), threadgroup=(count, 1, 1),
                      output_shapes=[(rows, dims)] * 2, output_dtypes=[mx.bfloat16] * 2)


def mlp_tail(h: mx.array, y: mx.array, w_post: mx.array, scalar: mx.array, w_next: mx.array,
             eps: mx.array) -> tuple[mx.array, mx.array]:
    """hn = (h + RMSNorm(y) * w_post) * scalar and RMSNorm(hn) * w_next: [R, D] bf16."""

    rows, dims = h.shape
    count = threads(dims)
    return _mlp_tail((("D", dims), ("T", count)), inputs=[h, y, w_post, scalar, w_next, eps],
                     grid=(count * rows, 1, 1), threadgroup=(count, 1, 1),
                     output_shapes=[(rows, dims)] * 2, output_dtypes=[mx.bfloat16] * 2)


__all__ = ["attn_tail", "mlp_tail"]
