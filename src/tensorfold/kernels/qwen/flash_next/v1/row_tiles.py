"""Pre-M5 hyper-connection projections of 8+ rows on the matrix units with the per-row kernels' bits."""

from __future__ import annotations

import mlx.core as mx

from tensorfold.kernels.qwen.flash_next.v1.base import MMA_HEADER, QDOT_HEADER, QWeights, count, kernel
from tensorfold.kernels.qwen.flash_next.v1.hc import RINV

_HC_DOWN_MMA = r"""
  // _HC_DOWN_SPLIT for 8 rows a tile on the matrix units, the same bits: a group's dot is one FMA chain over its 32
  // inputs in order (4 chained 8-step MMAs), then fma(scale, dot, bias * sum), and a split's 32 groups meet in one
  // simd_sum with lane = group. Threadgroup (i, k, t): outputs 8 i .., split k, rows 8 t ..; simdgroup c: groups 4 c ..
  const uint lane = thread_index_in_simdgroup;
  const int c = int(simdgroup_index_in_threadgroup);
  const uint t = thread_position_in_threadgroup.x;
  const int fm = tile_fm(int(lane)), fn = tile_fn(int(lane));
  const int R = rows[0];
  constexpr int W = S * D, GROUPS = W / 32;
  const int nb = int(threadgroup_position_in_grid.x) * 8;
  const int k = int(threadgroup_position_in_grid.y);
  const int rb = int(threadgroup_position_in_grid.z) * 8;
  threadgroup float rinv[8 * S];
  threadgroup float sums[8][32];
  threadgroup float red[64][33];
  if (t < 8 * S) rinv[t] = stream_rinv(SSP, min(rb + int(t) / S, R - 1), int(t) % S, D / 256, S, D, eps[0]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  {
    const int r = int(lane) % 8, j = 4 * c + int(lane) / 8;
    const int row = min(rb + r, R - 1);
    const int e0 = 32 * (32 * k + j);
    float dx = 0.0f;
    for (int v = 0; v < 32; v++) {
      const int e = e0 + v;
      dx += float(bfloat((float(HN[size_t(row) * W + e]) * rinv[r * S + e / D]) * NW[e]));
    }
    sums[r][j] = dx;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int o = min(nb + fm, ND - 1);
  const int ra = min(rb + fn, R - 1), rc = min(rb + fn + 1, R - 1);
  for (int jj = 0; jj < 4; jj++) {
    const int j = 4 * c + jj, g = 32 * k + j;
    const device uint* wq = QW + (size_t(o) * GROUPS + g) * 4;
    simdgroup_matrix<float, 8, 8> P = simdgroup_matrix<float, 8, 8>(0.0f);
    for (int st = 0; st < 4; st++) {
      const uint word = wq[st];
      const int e = 32 * g + 8 * st + fm;
      simdgroup_matrix<float, 8, 8> am, bm;
      am.thread_elements()[0] = nib((word >> (4 * fn)) & 0xFu);
      am.thread_elements()[1] = nib((word >> (4 * fn + 4)) & 0xFu);
      bm.thread_elements()[0] = float(bfloat((float(HN[size_t(ra) * W + e]) * rinv[(ra - rb) * S + e / D]) * NW[e]));
      bm.thread_elements()[1] = float(bfloat((float(HN[size_t(rc) * W + e]) * rinv[(rc - rb) * S + e / D]) * NW[e]));
      simdgroup_multiply_accumulate(P, am, bm, P);
    }
    const float sc = float(QS[o * GROUPS + g]), bi = float(QB[o * GROUPS + g]);
    red[fm * 8 + fn][j] = fma(sc, P.thread_elements()[0], bi * sums[ra - rb][j]);
    red[fm * 8 + fn + 1][j] = fma(sc, P.thread_elements()[1], bi * sums[rc - rb][j]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int p = 8 * c; p < 8 * c + 8; p++) {
    const float v = simd_sum(red[p][lane]);
    const int out = nb + p / 8, row = rb + p % 8;
    if (lane == 0 && out < ND && row < R) PART[(size_t(k) * R + row) * ND + out] = v;
  }
"""

_HC_UP_MMA = r"""
  // _HC_UP2 for 8 rows a tile on the matrix units, the same bits: the prologue sums the split partials in order into
  // the SiLU inputs; an up row's dot adds its groups' qgroup_dot values (one FMA chain over 32 inputs in order, then
  // fma(scale, dot, bias * sum)) in group order. Threadgroup (i, t): dims DT i .. of every stream, rows 8 t ..;
  // simdgroup s: stream s.
  const uint lane = thread_index_in_simdgroup;
  const int s = int(simdgroup_index_in_threadgroup);
  const uint t = thread_position_in_threadgroup.x;
  const int fm = tile_fm(int(lane)), fn = tile_fn(int(lane));
  const int R = rows[0];
  constexpr int W = S * D, GPR = LOW / 32, NT = 32 * S, LP = LOW + 4;
  const int d0 = int(threadgroup_position_in_grid.x) * DT;
  const int rb = int(threadgroup_position_in_grid.y) * 8;
  const int nr = min(8, R - rb);
  threadgroup float act[8 * LP];
  threadgroup float sums[8][GPR];
  threadgroup float rinv[8 * S];
  threadgroup float prod[S][DT][8];
  if (t < 8 * S) rinv[t] = stream_rinv(SSP, min(rb + int(t) / S, R - 1), int(t) % S, D / 256, S, D, eps[0]);
  for (int i = int(t); i < 8 * ND; i += NT) {
    const int r = i / ND, cc = i % ND;
    const int row = min(rb + r, R - 1);
    float v = 0.0f;
    for (int kk = 0; kk < KS; kk++) v += PART[(size_t(kk) * R + row) * ND + cc];
    const float v4 = float(bfloat(float(bfloat(v)) / float(S)));
    if (cc < LOW) act[r * LP + cc] = bsilu(v4);
    else if (threadgroup_position_in_grid.x == 0 && r < nr) INJOUT[(rb + r) * S + (cc - LOW)] = bfloat(2.0f * bsig(v4));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = int(t); i < 8 * GPR; i += NT) {
    const int r = i / GPR, q = i % GPR;
    float dx = 0.0f;
    for (int v = 0; v < 32; v++) dx += act[r * LP + 32 * q + v];
    sums[r][q] = dx;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int tt = 0; tt < DT / 8; tt++) {
    const int orow = s * D + d0 + 8 * tt + fm;
    const device uint* wq = QW + size_t(orow) * GPR * 4;
    float u0 = 0.0f, u1 = 0.0f;
    for (int q = 0; q < GPR; q++) {
      simdgroup_matrix<float, 8, 8> P = simdgroup_matrix<float, 8, 8>(0.0f);
      for (int st = 0; st < 4; st++) {
        const uint word = wq[4 * q + st];
        const int e = 32 * q + 8 * st + fm;
        simdgroup_matrix<float, 8, 8> am, bm;
        am.thread_elements()[0] = nib((word >> (4 * fn)) & 0xFu);
        am.thread_elements()[1] = nib((word >> (4 * fn + 4)) & 0xFu);
        bm.thread_elements()[0] = act[fn * LP + e];
        bm.thread_elements()[1] = act[(fn + 1) * LP + e];
        simdgroup_multiply_accumulate(P, am, bm, P);
      }
      const float sc = float(QS[orow * GPR + q]), bi = float(QB[orow * GPR + q]);
      u0 += fma(sc, P.thread_elements()[0], bi * sums[fn][q]);
      u1 += fma(sc, P.thread_elements()[1], bi * sums[fn + 1][q]);
    }
    for (int e = 0; e < 2; e++) {
      const int r = fn + e, row = min(rb + r, R - 1);
      const float normed = float(bfloat((float(HN[size_t(row) * W + orow]) * rinv[r * S + s]) * NW[orow]));
      prod[s][8 * tt + fm][r] = float(bfloat(bsig(float(bfloat(e ? u1 : u0))) * normed));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = int(t); i < DT * 8; i += NT) {
    const int d = i / 8, r = i % 8;
    float total = 0.0f;
    for (int ss = 0; ss < S; ss++) total += prod[ss][d][r];
    if (r < nr) MIXED[size_t(rb + r) * D + d0 + d] = bfloat(total / float(S));
  }
"""

def hc_tiles(h_new: mx.array, ssp: mx.array, down: QWeights, up: QWeights, norm_scale: mx.array, *,
              eps: mx.array, streams: int, low: int, dims_a_group: int = 32) -> tuple[mx.array, mx.array]:
    """``_hc_rows`` through 8-row tiles on the matrix units (4-bit group 32): every row's bits unchanged."""

    rows, wide = h_new.shape
    dims = wide // streams
    splits = wide // 32 // 32
    tiles = -(-rows // 8)
    nd = down.rows
    header = QDOT_HEADER + RINV + MMA_HEADER
    run = kernel("q4_hc_down_tiles", _HC_DOWN_MMA, ["HN", "SSP", "NW", "QW", "QS", "QB", "eps", "rows"], ["PART"],
                 header=header)
    part = run(inputs=[h_new, ssp, norm_scale, down.weight, down.scales, down.biases, eps, count(rows)],
               template=[("S", streams), ("D", dims), ("ND", nd)],
               grid=(-(-nd // 8) * 256, splits, tiles), threadgroup=(256, 1, 1),
               output_shapes=[(splits, rows, nd)], output_dtypes=[mx.float32])[0]
    run = kernel("q4_hc_up_tiles", _HC_UP_MMA, ["HN", "SSP", "NW", "PART", "QW", "QS", "QB", "eps", "rows"],
                 ["MIXED", "INJOUT"], header=header)
    return tuple(run(inputs=[h_new, ssp, norm_scale, part, up.weight, up.scales, up.biases, eps, count(rows)],
                     template=[("S", streams), ("D", dims), ("LOW", low), ("ND", nd), ("KS", splits),
                               ("DT", dims_a_group)],
                     grid=(dims // dims_a_group * 32 * streams, tiles, 1), threadgroup=(32 * streams, 1, 1),
                     output_shapes=[(rows, dims), (max(rows, 2), streams)],
                     output_dtypes=[mx.bfloat16, mx.bfloat16]))


__all__ = ["hc_tiles"]
