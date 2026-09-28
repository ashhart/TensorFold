"""Decode attention of DeepSeek-V4-Flash: each row over its own pool rows and window, with sinks."""

from __future__ import annotations

import mlx.core as mx

from tensorfold.kernels import inputs
from tensorfold.kernels.glm.flash.v1 import kernels as GK

BK = 16                 # keys a block, staged in threadgroup memory
HEADS_TG = 8            # heads a threadgroup (4 simdgroups x 2): 8 threadgroups a row keep more cores busy
SPLITS = 4              # the split kernel: simdgroups sharing a (row, head)'s keys, 0 for the staged kernel
SPLIT_ROWS = 16         # rows up to which the split kernel runs; prompt chunks share staged keys across 8 heads

# Row r over its pool rows then its window, 16 keys a block, bf16 p in P.V; ROT: output unroped as norm_rope does
_SOURCE = r"""
  const int r = int(threadgroup_position_in_grid.y) / (64 / HTG);
  const int hg = int(threadgroup_position_in_grid.y) % (64 / HTG);
  const uint t = thread_position_in_threadgroup.x;
  constexpr int TPG = HTG * 16;
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  constexpr int D = 512;
  constexpr int H = 64;
  constexpr float NEG = -INFINITY;
  threadgroup bfloat Ks[BK * D];
  const int prefix = META[0], pm = META[1], rs = META[2];
  const float scale = SCALE[0];
  const int pc = PCOUNT[r];
  const int wlo = WPOS[2 * r], whi = WPOS[2 * r + 1];
  const int n = pc + (whi - wlo + 1);
  const int h0 = hg * HTG + int(sg) * 2;
  float q0[16], q1[16], o0[16], o1[16];
  const device bfloat* qp = Q + (size_t(r) * H + h0) * D + lane * 16;
  for (int d = 0; d < 16; d++) { q0[d] = float(qp[d]); q1[d] = float(qp[D + d]); o0[d] = 0.0f; o1[d] = 0.0f; }
  float m0 = NEG, m1 = NEG, l0 = 0.0f, l1 = 0.0f;
  for (int b = 0; b < n; b += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = int(t); e < BK * D / 8; e += TPG) {
      const int i = e / (D / 8), c = e - i * (D / 8), g = b + i;
      uint4 v = uint4(0);
      if (g < pc) {
        const int row = prefix ? g : PIDX[size_t(r) * pm + g];
        v = ((const device uint4*)(POOL + size_t(row) * D))[c];
      } else if (g < n) {
        v = ((const device uint4*)(RING + size_t((wlo + g - pc) % rs) * D))[c];
      }
      ((threadgroup uint4*)Ks)[e] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s0[BK], s1[BK];
    float mb0 = NEG, mb1 = NEG;
    for (int i = 0; i < BK; i++) {
      const threadgroup bfloat* kp = Ks + i * D + lane * 16;
      float a0 = 0.0f, a1 = 0.0f;
      for (int d = 0; d < 16; d++) { const float k = float(kp[d]); a0 = fma(q0[d], k, a0); a1 = fma(q1[d], k, a1); }
      a0 = simd_sum(a0) * scale;
      a1 = simd_sum(a1) * scale;
      s0[i] = (b + i < n) ? a0 : NEG;
      s1[i] = (b + i < n) ? a1 : NEG;
      mb0 = max(mb0, s0[i]);
      mb1 = max(mb1, s1[i]);
    }
    const float n0 = max(m0, mb0), n1 = max(m1, mb1);
    const float c0 = m0 == NEG ? 0.0f : metal::exp(m0 - n0), c1 = m1 == NEG ? 0.0f : metal::exp(m1 - n1);
    l0 *= c0; l1 *= c1;
    for (int d = 0; d < 16; d++) { o0[d] *= c0; o1[d] *= c1; }
    for (int i = 0; i < BK; i++) {
      const float p0 = s0[i] == NEG ? 0.0f : metal::exp(s0[i] - n0);
      const float p1 = s1[i] == NEG ? 0.0f : metal::exp(s1[i] - n1);
      l0 += p0; l1 += p1;
      const float b0 = float(bfloat(p0)), b1 = float(bfloat(p1));
      const threadgroup bfloat* kp = Ks + i * D + lane * 16;
      for (int d = 0; d < 16; d++) {
        const float k = float(kp[d]);
        o0[d] = fma(b0, k, o0[d]);
        o1[d] = fma(b1, k, o1[d]);
      }
    }
    m0 = n0; m1 = n1;
  }
  l0 += metal::exp(SINK[h0] - m0);
  l1 += metal::exp(SINK[h0 + 1] - m1);
  device bfloat* op = OUT + (size_t(r) * H + h0) * D + lane * 16;
  float v0[16], v1[16];
  for (int d = 0; d < 16; d++) { v0[d] = float(bfloat(o0[d] / l0)); v1[d] = float(bfloat(o1[d] / l1)); }
  if (ROT && int(lane) * 16 + 16 > D - PE) {
    const float pos = float(whi + META[3]);
    for (int d = 0; d < 16; d += 2) {
      const int dd = int(lane) * 16 + d;
      if (dd >= D - PE) {
        const float th = pos * INV[(dd - (D - PE)) / 2];
        const float c = metal::precise::cos(th);
        const float s = -metal::precise::sin(th);
        const float a0 = v0[d], b0 = v0[d + 1], a1 = v1[d], b1 = v1[d + 1];
        v0[d] = a0 * c - b0 * s;
        v0[d + 1] = a0 * s + b0 * c;
        v1[d] = a1 * c - b1 * s;
        v1[d + 1] = a1 * s + b1 * c;
      }
    }
  }
  for (int d = 0; d < 16; d++) { op[d] = bfloat(v0[d]); op[D + d] = bfloat(v1[d]); }
"""


_HEADER = r"""
inline void load16(const device bfloat* p, thread float* v) {
  const device uint4* u = (const device uint4*)p;
  const uint4 a = u[0], b = u[1];
  const uint w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
  for (int i = 0; i < 8; i++) { v[2 * i] = as_type<float>(w[i] << 16); v[2 * i + 1] = as_type<float>(w[i] & 0xffff0000u); }
}
"""

# Simdgroup g of (row r, head h) runs keys [g * per, ..) of the row's pool rows then window, 8 a block (fp32 scores,
# bf16 p in P.V); simdgroup 0 merges the S runs in order with the sink; ROT unropes the bf16 output as norm_rope does
_SPLIT = r"""
  const int r = int(threadgroup_position_in_grid.y) / 64;
  const int h = int(threadgroup_position_in_grid.y) % 64;
  const uint lane = thread_index_in_simdgroup;
  const int g = int(simdgroup_index_in_threadgroup);
  constexpr int D = 512;
  constexpr int KB = 8;
  constexpr float NEG = -INFINITY;
  const int prefix = META[0], pm = META[1], rs = META[2];
  const float scale = SCALE[0];
  const int pc = PCOUNT[r];
  const int wlo = WPOS[2 * r], whi = WPOS[2 * r + 1];
  const int n = pc + (whi - wlo + 1);
  const int per = (n + S - 1) / S;
  const int j0 = g * per, j1 = min(n, j0 + per);
  auto key = [&](int j) -> const device bfloat* {
    if (j < pc) return POOL + size_t(prefix ? j : PIDX[size_t(r) * pm + j]) * D + lane * 16;
    return RING + size_t((wlo + j - pc) % rs) * D + lane * 16;
  };
  float q[16], o[16], k[16];
  load16(Q + (size_t(r) * 64 + h) * D + lane * 16, q);
  for (int d = 0; d < 16; d++) o[d] = 0.0f;
  float m = NEG, l = 0.0f;
  for (int b = j0; b < j1; b += KB) {
    float sc[KB];
    float mb = NEG;
    for (int i = 0; i < KB; i++) {
      float a = 0.0f;
      if (b + i < j1) {
        load16(key(b + i), k);
        for (int d = 0; d < 16; d++) a = fma(q[d], k[d], a);
      }
      a = simd_sum(a) * scale;
      sc[i] = b + i < j1 ? a : NEG;
      mb = max(mb, sc[i]);
    }
    const float mn = max(m, mb);
    const float c = m == NEG ? 0.0f : metal::exp(m - mn);
    l *= c;
    for (int d = 0; d < 16; d++) o[d] *= c;
    for (int i = 0; i < KB; i++) {
      if (b + i < j1) {
        const float p = metal::exp(sc[i] - mn);
        l += p;
        const float pb = float(bfloat(p));
        load16(key(b + i), k);
        for (int d = 0; d < 16; d++) o[d] = fma(pb, k[d], o[d]);
      }
    }
    m = mn;
  }
  threadgroup float tm[S], tl[S], to[S * D];
  if (lane == 0) { tm[g] = m; tl[g] = l; }
  for (int d = 0; d < 16; d++) to[g * D + lane * 16 + d] = o[d];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (g != 0) return;
  float top = NEG;
  for (int i = 0; i < S; i++) top = max(top, tm[i]);
  float lt = 0.0f, v[16];
  for (int d = 0; d < 16; d++) v[d] = 0.0f;
  for (int i = 0; i < S; i++) {
    const float w = tm[i] == NEG ? 0.0f : metal::exp(tm[i] - top);
    lt += tl[i] * w;
    for (int d = 0; d < 16; d++) v[d] = fma(to[i * D + lane * 16 + d], w, v[d]);
  }
  lt += metal::exp(SINK[h] - top);
  for (int d = 0; d < 16; d++) v[d] = float(bfloat(v[d] / lt));
  if (ROT && int(lane) * 16 + 16 > D - PE) {
    const float pos = float(whi + META[3]);
    for (int d = 0; d < 16; d += 2) {
      const int dd = int(lane) * 16 + d;
      if (dd >= D - PE) {
        const float th = pos * INV[(dd - (D - PE)) / 2];
        const float c = metal::precise::cos(th);
        const float s = -metal::precise::sin(th);
        const float a = v[d], bb = v[d + 1];
        v[d] = a * c - bb * s;
        v[d + 1] = a * s + bb * c;
      }
    }
  }
  device bfloat* op = OUT + (size_t(r) * 64 + h) * D + lane * 16;
  for (int d = 0; d < 16; d++) op[d] = bfloat(v[d]);
"""


def fits(heads: int, dims: int) -> bool:
    return GK.metal() and (heads, dims) == (64, 512)


_MADE: dict[tuple, mx.array] = {}


def _ints(values: list[int]) -> mx.array:
    """A small int32 input, made once while the same values recur (every layer of a round passes them)."""

    key = tuple(values)
    found = _MADE.get(key)
    if found is None:
        if len(_MADE) > 1024:
            _MADE.clear()
        found = _MADE[key] = mx.array(list(key) + [0] * max(0, inputs.MIN_ELEMENTS - len(key)), dtype=mx.int32)
    return found


def _scale(value: float) -> mx.array:
    key = ("scale", float(value))
    found = _MADE.get(key)
    if found is None:
        found = _MADE[key] = mx.array([float(value)], dtype=mx.float32)
    return found


def attend_rows(q: mx.array, pool: mx.array | None, pidx: mx.array | None, counts: list[int], ring: mx.array,
                windows: list[tuple[int, int]], sink: mx.array, scale: float, inv_freq: mx.array | None = None,
                base: int = 0) -> mx.array:
    """Rows q [R, 64, 512] over pool rows (``pidx`` or the first counts) and windows; ``inv_freq``: rotated back."""

    rows = int(q.shape[0])
    prefix = pidx is None
    pm = 1 if prefix else int(pidx.shape[1])
    names = ["Q", "POOL", "PIDX", "PCOUNT", "RING", "WPOS", "SINK", "SCALE", "META", "INV"]
    pool = ring[:1] if pool is None else pool
    pidx = _ints([0]) if prefix else mx.contiguous(pidx.astype(mx.int32)).reshape(-1)
    ins = [mx.contiguous(q), pool, pidx, _ints(list(counts)), ring, _ints([v for w in windows for v in w]), sink,
           _scale(scale), _ints([int(prefix), pm, int(ring.shape[0]), base]), sink if inv_freq is None else inv_freq]
    rot = [("ROT", int(inv_freq is not None)), ("PE", 0 if inv_freq is None else 2 * int(inv_freq.shape[0]))]
    if SPLITS and rows <= SPLIT_ROWS:
        kernel = GK._kernel("ds4_attn_split", _SPLIT, names, ["OUT"], _HEADER)
        return kernel(inputs=ins, template=[("S", SPLITS), *rot], grid=(32 * SPLITS, rows * 64, 1),
                      threadgroup=(32 * SPLITS, 1, 1), output_shapes=[(rows, 64, 512)],
                      output_dtypes=[mx.bfloat16])[0]
    kernel = GK._kernel("ds4_attn_rows", _SOURCE, names, ["OUT"])
    return kernel(inputs=ins, template=[("BK", BK), ("HTG", HEADS_TG), *rot],
                  grid=(16 * HEADS_TG, rows * (64 // HEADS_TG), 1), threadgroup=(16 * HEADS_TG, 1, 1),
                  output_shapes=[(rows, 64, 512)], output_dtypes=[mx.bfloat16])[0]
