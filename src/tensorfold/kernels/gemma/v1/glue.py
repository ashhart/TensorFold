"""Gemma 4's norms, residuals and RoPE around its matmuls: each row its own simdgroup or threadgroup."""

from __future__ import annotations

import mlx.core as mx

from tensorfold.kernels.gemma.v1.base import Kernel
from tensorfold.kernels.nemotron.lightning.v1 import rows as row_kernels

# a head's raw values xv[PER] (lane l: elements l, l + 32, ...) out: q (KIND 0), k (1) normed and roped, v (2) normed
_EMIT = r"""
  {
    float ss = 0.0f;
    for (int i = 0; i < PER; i++) ss = fma(xv[i], xv[i], ss);
    ss = simd_sum(ss);
    const float inv = metal::precise::rsqrt(ss / float(DH) + eps[0]);
    if (KIND == 2) {
      for (int i = 0; i < PER; i++) V[(HEAD * R + r) * DH + int(lane) + 32 * i] = bfloat(xv[i] * inv);
    } else {
      const device bfloat* w = KIND == 0 ? QW : KW;
      float y[PER];
      for (int i = 0; i < PER; i++) y[i] = float(bfloat(float(w[int(lane) + 32 * i]) * float(bfloat(xv[i] * inv))));
      const float pos = float(POS[r]);
      for (int i = 0; i < PER / 2; i++) {
        const float theta = pos * INVF[int(lane) + 32 * i];
        const float c = metal::fast::cos(theta), s = metal::fast::sin(theta);
        const float x1 = y[i], x2 = y[i + PER / 2];
        y[i] = x1 * c - x2 * s;
        y[i + PER / 2] = x1 * s + x2 * c;
      }
      if (KIND == 0)
        for (int i = 0; i < PER; i++) Q[(r * NQ + HEAD) * DH + int(lane) + 32 * i] = bfloat(y[i]);
      else
        for (int i = 0; i < PER; i++) K[(HEAD * R + r) * DH + int(lane) + 32 * i] = bfloat(y[i]);
    }
  }
"""


def _emit(kind: str, head: str) -> str:
    return _EMIT.replace("KIND", kind).replace("HEAD", head)


# one simdgroup per (row, head slot) of the stacked output: q heads, k heads, then v heads (VK: the raw keys again)
_QKV_PREP = r"""
  const uint lane = thread_index_in_simdgroup;
  const int slot = int(threadgroup_position_in_grid.x);
  const int r = int(threadgroup_position_in_grid.y);
  const int R = int(threadgroups_per_grid.y);
  constexpr int PER = DH / 32;
  const int kind = slot < NQ ? 0 : (slot < NQ + NK ? 1 : 2);
  const int head = kind == 0 ? slot : (kind == 1 ? slot - NQ : slot - NQ - NK);
  const int src = kind == 2 ? (VK ? NQ * DH : (NQ + NK) * DH) + head * DH : slot * DH;
  float xv[PER];
  for (int i = 0; i < PER; i++) xv[i] = float(QKV[r * W + src + int(lane) + 32 * i]);
""" + _emit("kind", "head")

# threadgroup (head slot, row): the head's DH rows by rows.qmv's loop (its bits), then qkv_prep's emit (VK: k emits v)
_QKV_ROWS = r"""
  const uint lane = thread_index_in_simdgroup;
  const int sg = int(simdgroup_index_in_threadgroup);
  const int slot = int(threadgroup_position_in_grid.x);
  const int r = int(threadgroup_position_in_grid.y);
  const int R = int(threadgroups_per_grid.y);
  constexpr int PER = DH / 32;
  threadgroup float raw[DH];
  for (int p = sg; p < DH / RPS; p += SG) {
    const int row0 = slot * DH + p * RPS;
    float acc[RPS];
    tf_rowdot<KD, GS, RPS>((const device uint8_t*)W + size_t(row0) * (KD / 2), S + size_t(row0) * (KD / GS),
                           B + size_t(row0) * (KD / GS), X + size_t(r) * KD, lane, acc);
    if (lane == 0)
      for (int j = 0; j < RPS; j++) raw[p * RPS + j] = float(bfloat(acc[j]));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;
  const int kind = slot < NQ ? 0 : (slot < NQ + NK ? 1 : 2);
  const int head = kind == 0 ? slot : (kind == 1 ? slot - NQ : slot - NQ - NK);
  float xv[PER];
  for (int i = 0; i < PER; i++) xv[i] = raw[int(lane) + 32 * i];
""" + _emit("kind", "head") + r"""
  if (VK && kind == 1) {
""" + _emit("2", "head") + r"""
  }
"""

# sum of squares over the threadgroup's T threads into `total` (P: a threadgroup array of T / 32 floats)
_REDUCE = r"""
  {
    float s = simd_sum(ACC);
    if (thread_index_in_simdgroup == 0) P[simdgroup_index_in_threadgroup] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    OUTV = 0.0f;
    for (int q = 0; q < T / 32; q++) OUTV += P[q];
  }
"""


def _reduce(acc: str, partial: str, out: str) -> str:
    return _REDUCE.replace("ACC", acc).replace("P[", f"{partial}[").replace("OUTV", out)


# one threadgroup of T threads per row, thread t owning elements t, t + T, ...; every load before the first reduction
_ATTN_TAIL = r"""
  const uint t = thread_position_in_threadgroup.x;
  const uint r = threadgroup_position_in_grid.x;
  constexpr int PER = D / T;
  threadgroup float p1[T / 32], p2[T / 32];
  float ov[PER], hin[PER], wa[PER], w1[PER], w2[PER], w3[PER];
  for (int i = 0; i < PER; i++) {
    const int c = int(t) + i * T;
    ov[i] = float(O[int(r) * D + c]); hin[i] = float(H[int(r) * D + c]);
    wa[i] = float(WA[c]); w1[i] = float(W1[c]); w2[i] = float(W2[c]); w3[i] = float(W3[c]);
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
    const float n = float(bfloat(hv[i] * inv2));
    N1[int(r) * D + c] = bfloat(w1[i] * n);
    N2[int(r) * D + c] = bfloat(w2[i] * n);
    N3[int(r) * D + c] = bfloat(w3[i] * n);
  }
""".replace("REDUCE1", _reduce("ss", "p1", "total1")).replace("REDUCE2", _reduce("ss2", "p2", "total2"))

_MOE_TAIL = r"""
  const uint t = thread_position_in_threadgroup.x;
  const uint r = threadgroup_position_in_grid.x;
  constexpr int PER = D / T;
  threadgroup float p1[T / 32], p2[T / 32], p3[T / 32], p4[T / 32];
  // every load before the first reduction, as in attn_tail
  float y1v[PER], h2v[PER], hin[PER], w1[PER], w2[PER], wp[PER], wn[PER];
  for (int i = 0; i < PER; i++) {
    const int c = int(t) + i * T;
    y1v[i] = float(Y1[int(r) * D + c]); h2v[i] = float(Y2[int(r) * D + c]); hin[i] = float(H[int(r) * D + c]);
    w1[i] = float(W1[c]); w2[i] = float(W2[c]); wp[i] = float(WP[c]); wn[i] = float(WN[c]);
  }
  const float sc = float(SC[0]);
  float ss1 = 0.0f, ss2 = 0.0f;
  for (int i = 0; i < PER; i++) {
    ss1 = fma(y1v[i], y1v[i], ss1);
    ss2 = fma(h2v[i], h2v[i], ss2);
  }
  float total1, total2;
  REDUCE1
  REDUCE2
  const float inv1 = metal::precise::rsqrt(total1 / float(D) + eps[0]);
  const float inv2 = metal::precise::rsqrt(total2 / float(D) + eps[0]);
  float sv[PER];
  float ss3 = 0.0f;
  for (int i = 0; i < PER; i++) {
    const float a1 = float(bfloat(w1[i] * float(bfloat(y1v[i] * inv1))));
    const float a2 = float(bfloat(w2[i] * float(bfloat(h2v[i] * inv2))));
    sv[i] = float(bfloat(a1 + a2));
    ss3 = fma(sv[i], sv[i], ss3);
  }
  float total3;
  REDUCE3
  const float inv3 = metal::precise::rsqrt(total3 / float(D) + eps[0]);
  float hv[PER];
  float ss4 = 0.0f;
  for (int i = 0; i < PER; i++) {
    const int c = int(t) + i * T;
    const float b = float(bfloat(wp[i] * float(bfloat(sv[i] * inv3))));
    const bfloat hs = bfloat(hin[i] + b);
    const bfloat hn = bfloat(float(hs) * sc);
    HN[int(r) * D + c] = hn;
    hv[i] = float(hn);
    ss4 = fma(hv[i], hv[i], ss4);
  }
  float total4;
  REDUCE4
  const float inv4 = metal::precise::rsqrt(total4 / float(D) + eps[0]);
  for (int i = 0; i < PER; i++) {
    const int c = int(t) + i * T;
    NEXT[int(r) * D + c] = bfloat(wn[i] * float(bfloat(hv[i] * inv4)));
  }
""".replace("REDUCE1", _reduce("ss1", "p1", "total1")).replace("REDUCE2", _reduce("ss2", "p2", "total2")) \
   .replace("REDUCE3", _reduce("ss3", "p3", "total3")).replace("REDUCE4", _reduce("ss4", "p4", "total4"))

_prep = Kernel("gemma_qkv_prep", _QKV_PREP, ["QKV", "QW", "KW", "INVF", "POS", "eps"], ["Q", "K", "V"])
_rows = Kernel("gemma_qkv_rows", _QKV_ROWS, ["X", "W", "S", "B", "QW", "KW", "INVF", "POS", "eps"], ["Q", "K", "V"],
               header=row_kernels.HEADER)
_attn_tail = Kernel("gemma_attn_tail", _ATTN_TAIL, ["H", "O", "WA", "W1", "W2", "W3", "eps"], ["HN", "N1", "N2", "N3"])
_moe_tail = Kernel("gemma_moe_tail", _MOE_TAIL, ["H", "Y1", "Y2", "W1", "W2", "WP", "SC", "WN", "eps"],
                   ["HN", "NEXT"])


def threads(dims: int) -> int:
    for count in (256, 128, 64, 32):
        if dims % count == 0:
            return count
    raise ValueError(f"hidden size {dims} does not split into threadgroups of 32")


def qkv_prep(qkv: mx.array, q_w: mx.array, k_w: mx.array, inv_freq: mx.array, positions: mx.array, eps: mx.array, *,
             heads: int, kv_heads: int, head_dim: int, values_are_keys: bool) -> tuple[mx.array, mx.array, mx.array]:
    """Stacked q|k|v [R, W] -> q [R, H, Dh], k and v [Hk, R, Dh] (bf16): head norms, RoPE at each row's position."""

    rows, width = qkv.shape
    if head_dim % 64:
        raise ValueError("qkv_prep: head_dim must be a multiple of 64 (a RoPE pair in one lane)")
    consts = (("DH", head_dim), ("NQ", heads), ("NK", kv_heads), ("W", width), ("VK", int(values_are_keys)))
    return _prep(consts, inputs=[qkv, q_w, k_w, inv_freq, positions, eps],
                 grid=(32 * (heads + 2 * kv_heads), rows, 1), threadgroup=(32, 1, 1),
                 output_shapes=[(rows, heads, head_dim), (kv_heads, rows, head_dim), (kv_heads, rows, head_dim)],
                 output_dtypes=[mx.bfloat16, mx.bfloat16, mx.bfloat16])


def qkv_rows(x: mx.array, weight: mx.array, scales: mx.array, biases: mx.array, group: int, q_w: mx.array,
             k_w: mx.array, inv_freq: mx.array, positions: mx.array, eps: mx.array, *, heads: int, kv_heads: int,
             head_dim: int, values_are_keys: bool, simdgroups: int = 32) -> tuple[mx.array, mx.array, mx.array]:
    """``qkv_prep`` of rows.qmv's stacked projection of x [R, K] in one kernel, bit for bit."""

    rows, dims = x.shape
    slots = heads + kv_heads * (1 if values_are_keys else 2)
    if head_dim % 64 or dims % 64:
        raise ValueError("qkv_rows: head_dim and K multiples of 64")
    consts = (("DH", head_dim), ("NQ", heads), ("NK", kv_heads), ("VK", int(values_are_keys)), ("KD", dims),
              ("GS", int(group)), ("RPS", row_kernels.RPS), ("SG", simdgroups))
    return _rows(consts, inputs=[x, weight, scales, biases, q_w, k_w, inv_freq, positions, eps],
                 grid=(32 * simdgroups * slots, rows, 1), threadgroup=(32 * simdgroups, 1, 1),
                 output_shapes=[(rows, heads, head_dim), (kv_heads, rows, head_dim), (kv_heads, rows, head_dim)],
                 output_dtypes=[mx.bfloat16, mx.bfloat16, mx.bfloat16])


def attn_tail(h: mx.array, o: mx.array, w_attn: mx.array, w1: mx.array, w2: mx.array, w3: mx.array,
              eps: mx.array) -> tuple[mx.array, ...]:
    """hn = h + RMSNorm(o) * w_attn and RMSNorm(hn) times w1, w2, w3: (hn, n1, n2, n3), [R, D] bf16."""

    rows, dims = h.shape
    count = threads(dims)
    return _attn_tail((("D", dims), ("T", count)), inputs=[h, o, w_attn, w1, w2, w3, eps],
                      grid=(count * rows, 1, 1), threadgroup=(count, 1, 1),
                      output_shapes=[(rows, dims)] * 4, output_dtypes=[mx.bfloat16] * 4)


def moe_tail(h: mx.array, y1: mx.array, y2: mx.array, w1: mx.array, w2: mx.array, w_post: mx.array,
             scalar: mx.array, w_next: mx.array, eps: mx.array) -> tuple[mx.array, mx.array]:
    """hn = (h + RMSNorm(RMSNorm(y1) w1 + RMSNorm(y2) w2) w_post) * scalar and RMSNorm(hn) * w_next: [R, D] bf16."""

    rows, dims = h.shape
    count = threads(dims)
    return _moe_tail((("D", dims), ("T", count)), inputs=[h, y1, y2, w1, w2, w_post, scalar, w_next, eps],
                     grid=(count * rows, 1, 1), threadgroup=(count, 1, 1),
                     output_shapes=[(rows, dims), (rows, dims)], output_dtypes=[mx.bfloat16, mx.bfloat16])


__all__ = ["attn_tail", "moe_tail", "qkv_prep", "qkv_rows", "threads"]
