"""Decode attention over each row's own keys, sums fixed by absolute key position: a row alone gets the same bits."""

from __future__ import annotations

from typing import Sequence

import mlx.core as mx

from tensorfold.kernels.gemma.v1.base import Kernel
from tensorfold.kernels.inputs import MIN_ELEMENTS, ints

_PARTIAL = r"""
  // threadgroup (chunk C0 + x, key head h, row r); simdgroup (g, s): query head h G + g, keys k0 + s, k0 + s + S, ...
  const uint lane = thread_index_in_simdgroup;
  const int sgi = int(simdgroup_index_in_threadgroup);
  const int g = sgi / S, s = sgi % S;
  const int C0 = META[0], NCH = META[1], RING = META[2], R = META[3];
  const int c = C0 + int(threadgroup_position_in_grid.x);
  const int h = int(threadgroup_position_in_grid.y);
  const int r = int(threadgroup_position_in_grid.z);
  const int qh = h * G + g;
  const int CAP = K_shape[2];
  const int P0 = POS[0];                                    // the call's first row: keys from P0 on are its new rows
  constexpr int DPL = D / 32;
  const int k0 = metal::max(c * CK, LO[r]), k1 = metal::min((c + 1) * CK, POS[r] + 1);
  float q[DPL], o[DPL];
  const device bfloat* qp = Q + (size_t(r) * (HK * G) + qh) * D + lane * DPL;
  for (int i = 0; i < DPL; i++) { q[i] = float(qp[i]) * SCALE; o[i] = 0.0f; }
  float m = -INFINITY, l = 0.0f;
  const device bfloat* kb = K + size_t(h) * CAP * D + lane * DPL;
  const device bfloat* vb = V + size_t(h) * CAP * D + lane * DPL;
  const device bfloat* kn = KN + size_t(h) * R * D + lane * DPL;
  const device bfloat* vn = VN + size_t(h) * R * D + lane * DPL;
  for (int base = k0 + s; base < k1; base += S * BLK) {
    float sc[BLK];
    size_t at[BLK];
    bool fresh[BLK], live[BLK];
    float bm = -INFINITY;
    for (int j = 0; j < BLK; j++) {
      const int p = base + j * S;
      live[j] = p < k1;
      fresh[j] = p >= P0;
      at[j] = size_t(fresh[j] ? p - P0 : (RING ? p % RING : p)) * D;
      float d = 0.0f;
      if (live[j]) {
        const device bfloat* kr = (fresh[j] ? kn : kb) + at[j];
        for (int i = 0; i < DPL; i++) d = fma(q[i], float(kr[i]), d);
      }
      sc[j] = simd_sum(d);
      if (live[j]) bm = metal::max(bm, sc[j]);
    }
    const float mn = metal::max(m, bm);
    const float a = metal::exp(m - mn);
    l *= a;
    for (int i = 0; i < DPL; i++) o[i] *= a;
    for (int j = 0; j < BLK; j++) {
      if (!live[j]) continue;
      const float b = metal::exp(sc[j] - mn);
      l += b;
      const device bfloat* vr = (fresh[j] ? vn : vb) + at[j];
      for (int i = 0; i < DPL; i++) o[i] = fma(b, float(vr[i]), o[i]);
    }
    m = mn;
  }
  const size_t slot = (size_t(qh) * R + r) * NCH + (c - C0);
  if (S == 1) {
    if (lane == 0) { PM[slot] = m; PL[slot] = l; }
    for (int i = 0; i < DPL; i++) PO[slot * D + lane * DPL + i] = o[i];
    return;
  }
  threadgroup float sm[G * S], sl[G * S];
  threadgroup float so[S > 1 ? G * S : 1][S > 1 ? D : 1];
  if (lane == 0) { sm[sgi] = m; sl[sgi] = l; }
  for (int i = 0; i < DPL; i++) so[sgi][lane * DPL + i] = o[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (s != 0) return;
  float top = -INFINITY;
  for (int t = 0; t < S; t++) top = metal::max(top, sm[g * S + t]);
  float lsum = 0.0f, acc[DPL];
  for (int i = 0; i < DPL; i++) acc[i] = 0.0f;
  for (int t = 0; t < S; t++) {
    const float e = sl[g * S + t] > 0.0f ? metal::exp(sm[g * S + t] - top) : 0.0f;
    lsum = fma(sl[g * S + t], e, lsum);
    for (int i = 0; i < DPL; i++) acc[i] = fma(so[g * S + t][lane * DPL + i], e, acc[i]);
  }
  if (lane == 0) { PM[slot] = top; PL[slot] = lsum; }
  for (int i = 0; i < DPL; i++) PO[slot * D + lane * DPL + i] = acc[i];
"""

_MERGE = r"""
  // one simdgroup per (query head, row): the chunks' partials in chunk order
  const uint lane = thread_index_in_simdgroup;
  const int qh = int(threadgroup_position_in_grid.y);
  const int r = int(threadgroup_position_in_grid.z);
  const int NCH = META[1], R = META[3];
  constexpr int DPL = D / 32;
  const size_t base = (size_t(qh) * R + r) * NCH;
  float top = -INFINITY;
  for (int c = 0; c < NCH; c++) top = metal::max(top, PM[base + c]);
  float lsum = 0.0f, acc[DPL];
  for (int i = 0; i < DPL; i++) acc[i] = 0.0f;
  for (int c = 0; c < NCH; c++) {
    const float e = PL[base + c] > 0.0f ? metal::exp(PM[base + c] - top) : 0.0f;
    lsum = fma(PL[base + c], e, lsum);
    for (int i = 0; i < DPL; i++) acc[i] = fma(PO[(base + c) * D + lane * DPL + i], e, acc[i]);
  }
  for (int i = 0; i < DPL; i++) OUT[(size_t(r) * H + qh) * D + lane * DPL + i] = bfloat(acc[i] / lsum);
"""

_partial = Kernel("gemma_attention_partial", _PARTIAL, ["Q", "K", "V", "KN", "VN", "POS", "LO", "META"],
                  ["PM", "PL", "PO"])
_merge = Kernel("gemma_attention_merge", _MERGE, ["PM", "PL", "PO", "META"], ["OUT"])

# (keys a chunk, simdgroups a query head, keys scored before an update) by head dim: part of the arithmetic
SHAPES = {256: (128, 4, 4), 512: (64, 1, 4)}
OTHER = (64, 4, 4)


def ranges(positions: Sequence[int], window: int) -> list[int]:
    """Each row's first key: ``window`` keys up to and including its own position (0: every key)."""

    return [max(0, int(p) - window + 1) if window else 0 for p in positions]


class Rows:
    """One stream's rows for one kind of layer: positions, first keys and chunks, kernel inputs built once a forward."""

    def __init__(self, positions: Sequence[int], window: int, ring: int, dims: int) -> None:
        chunk = SHAPES.get(dims, OTHER)[0]
        lows = ranges(positions, window)
        self.count = len(positions)
        self.first = min(lows) // chunk
        self.chunks = max(int(p) for p in positions) // chunk - self.first + 1
        self.positions, self.lows = ints(positions), ints(lows)
        self.meta = ints((self.first, self.chunks, int(ring), self.count))


def attend(q: mx.array, keys: mx.array, values: mx.array, rows: Rows, new_keys: mx.array, new_values: mx.array,
           scale: float = 1.0) -> mx.array:
    """q [R, H, D] over its rows' new keys [Hk, R, D] and the earlier ones in [1, Hk, CAP, D] buffers: [R, H, D]."""

    count, heads, dims = (int(s) for s in q.shape)
    kv_heads = int(keys.shape[1])
    if dims % 32 or heads % kv_heads or rows.count != count:
        raise ValueError(f"attend: head dim a multiple of 32, heads a multiple of key heads, {count} rows")
    chunk, split, block = SHAPES.get(dims, OTHER)
    group = heads // kv_heads
    if 32 * group * split > 1024:
        raise ValueError(f"attend: {group} query heads a key head need {32 * group * split} threads a threadgroup")
    consts = (("D", dims), ("G", group), ("HK", kv_heads), ("CK", chunk), ("S", split), ("BLK", block),
              ("SCALE", float(scale)))
    slots = heads * count * rows.chunks
    # the merge reads the partials as they are: MIN_ELEMENTS or more keeps one Metal signature at every size
    pm, pl, po = _partial(consts, inputs=[q, keys, values, new_keys, new_values, rows.positions, rows.lows, rows.meta],
                          grid=(32 * group * split * rows.chunks, kv_heads, count),
                          threadgroup=(32 * group * split, 1, 1),
                          output_shapes=[(max(slots, MIN_ELEMENTS),), (max(slots, MIN_ELEMENTS),), (slots, dims)],
                          output_dtypes=[mx.float32, mx.float32, mx.float32])
    return _merge((("D", dims), ("H", heads)), inputs=[pm, pl, po, rows.meta],
                  grid=(32, heads, count), threadgroup=(32, 1, 1),
                  output_shapes=[(count, heads, dims)], output_dtypes=[mx.bfloat16])[0]


__all__ = ["Rows", "SHAPES", "attend", "ranges"]
