"""Hyper-connections: write a block back into the residual streams, then mix them into the next block's input."""

from __future__ import annotations

import mlx.core as mx

from tensorfold.kernels.qwen.flash_next.v1.base import MMA_HEADER, QDOT_HEADER, QWeights, count, kernel

_WRITEBACK = "hv = float(bfloat(hv + float(bfloat(branch * float(INJ[r * S + s])))));"

_BRANCH_PLAIN = "const float branch = float(BR[r * D + d]);"

_BRANCH_GROUPED = r"""float routed = 0.0f;
      for (int k = 0; k < TOPK; k++) routed = fma(float(Y[(r * (TOPK + 1) + k) * D + d]), WTS[r * TOPK + k], routed);
      const float shared = float(bfloat(float(Y[(r * (TOPK + 1) + TOPK) * D + d]) * bsig(float(bfloat(LG[r * NL + NL - 1])))));
      const float branch = float(bfloat(float(bfloat(routed)) + shared));"""

_HC_NORM = r"""
  // Threadgroup (j, r): dims 256 j .. 256 j + 255 of row r in every stream: write the block's branch back into the
  // S streams (bf16 ops) and each stream's partial sum of squares over these dims (fp32, simdgroups in order).
  // Consumers take a stream's inverse RMS from its NT partials, added in j order.
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup;
  const uint g = simdgroup_index_in_threadgroup;
  const int j = int(threadgroup_position_in_grid.x);
  const int r = int(threadgroup_position_in_grid.y);
  constexpr int W = S * D;
  constexpr int NT = D / 256;
  threadgroup float part[8][S];
  const int d = j * 256 + int(t);
  float ss[S];
  BRANCH
  for (int s = 0; s < S; s++) {
    const int e = s * D + d;
    float hv = float(H[r * W + e]);
    WRITEBACK
    HN[r * W + e] = bfloat(hv);
    ss[s] = simd_sum(hv * hv);
  }
  if (lane == 0) for (int s = 0; s < S; s++) part[g][s] = ss[s];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < S) {
    float total = 0.0f;
    for (int k = 0; k < 8; k++) total += part[k][t];
    SSP[(r * NT + j) * S + t] = total;
  }
"""

RINV = r"""
inline float stream_rinv(const device float* ssp, int r, int s, int nt, int streams, int dims, float eps) {
  float total = 0.0f;
  for (int j = 0; j < nt; j++) total += ssp[(r * nt + j) * streams + s];
  return metal::rsqrt(total / float(dims) + eps);
}
"""

def hc_norm(h: mx.array, *, streams: int, write_back: str = "none", branch: tuple[mx.array, ...] = (),
            inject: mx.array | None = None) -> tuple[mx.array, mx.array]:
    """Write the previous block into residual streams and return h_new with fp32 partial sums of squares for hc_project."""

    rows, wide = h.shape
    dims = wide // streams
    if dims % 256:
        raise ValueError("hc_norm: D must be a multiple of 256")
    names, inputs = ["H"], [h]
    if write_back == "none":
        make = lambda: _HC_NORM.replace("BRANCH", "").replace("WRITEBACK", "")          # noqa: E731
    elif write_back == "plain":
        if inject is None:
            raise ValueError("hc_norm: write-back needs the previous inject gates")
        make = lambda: _HC_NORM.replace("WRITEBACK", _WRITEBACK).replace("BRANCH", _BRANCH_PLAIN)  # noqa: E731
        names += ["INJ", "BR"]
        inputs += [inject, branch[0]]
    elif write_back == "grouped":
        if inject is None:
            raise ValueError("hc_norm: write-back needs the previous inject gates")
        y, weights, logits = branch
        make = lambda: _HC_NORM.replace("WRITEBACK", _WRITEBACK).replace("BRANCH", _BRANCH_GROUPED)  # noqa: E731
        names += ["INJ", "Y", "WTS", "LG"]
        inputs += [inject, y, weights, logits]
        extra = [("TOPK", int(weights.shape[-1])), ("NL", int(logits.shape[-1]))]
    else:
        raise ValueError(f"hc_norm: unknown write-back {write_back!r}")
    run = kernel(f"q4_hc_norm_{write_back}", make, names, ["HN", "SSP"])
    template = [("S", streams), ("D", dims)] + (extra if write_back == "grouped" else [])
    return tuple(run(inputs=inputs, template=template, grid=(dims, rows, 1),
                        threadgroup=(256, 1, 1), output_shapes=[(rows, wide), (rows, dims // 256, streams)],
                        output_dtypes=[mx.bfloat16, mx.float32]))

_DOWN_MMA = r"""
  // Split-K down projection of the normed streams on the matrix units, every row reading the weights once:
  // threadgroup (i, k, t) takes outputs 8 i .. 8 i + 7 over input groups 32 k .. 32 k + 31 for rows 8 t .. 8 t + 7
  // (rows past R read row R - 1, dropped); simdgroup c of 8 takes 4 of the groups, and the 8 add in order into
  // PART[k][r][o], which the up projection sums in k order.
  const uint lane = thread_index_in_simdgroup;
  const int c = int(simdgroup_index_in_threadgroup);
  const uint t = thread_position_in_threadgroup.x;
  const int qid = int(lane) / 4;
  const int fm = (qid & 4) + ((int(lane) / 2) % 4);
  const int fn = (qid & 2) * 2 + (int(lane) % 2) * 2;
  const int R = rows[0];
  constexpr int W = S * D, G = W / 32;
  const int nb = int(threadgroup_position_in_grid.x) * 8;
  const int k = int(threadgroup_position_in_grid.y);
  const int rb = int(threadgroup_position_in_grid.z) * 8;
  threadgroup float rinv[8 * S];
  threadgroup float red[8][64];
  if (t < 8 * S) rinv[t] = stream_rinv(SSP, min(rb + int(t) / S, R - 1), int(t) % S, D / 256, S, D, eps[0]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int o = min(nb + fm, ND - 1);
  const int ra = min(rb + fn, R - 1), rc = min(rb + fn + 1, R - 1);
  const device uint* wq = QW + size_t(o) * (W / 8) + fn / 2;
  float acc0 = 0.0f, acc1 = 0.0f;
  for (int j = 0; j < 4; j++) {
    const int g = 32 * k + 4 * c + j;
    const int e0 = 32 * g + 8 * (fm / 2);            // the lane's 8 inputs (one stream: D % 8 == 0)
    const float ia = rinv[(ra - rb) * S + e0 / D], ic = rinv[(rc - rb) * S + e0 / D];
    const uint4 ha = ((const device uint4*)HN)[(size_t(ra) * W + e0) / 8];
    const uint4 hc = ((const device uint4*)HN)[(size_t(rc) * W + e0) / 8];
    float xa[8], xc[8];
    for (int i = 0; i < 8; i++) {
      const float w = NW[e0 + i];
      xa[i] = float(bfloat((bfv(ha, i) * ia) * w));
      xc[i] = float(bfloat((bfv(hc, i) * ic) * w));
    }
    mma_group(wq[4 * g], xa, xc, fm, float(QS[size_t(o) * G + g]), float(QB[size_t(o) * G + g]), acc0, acc1);
  }
  red[c][2 * lane] = acc0;
  red[c][2 * lane + 1] = acc1;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < 64) {
    float v = 0.0f;
    for (int cc = 0; cc < 8; cc++) v += red[cc][t];
    const int l = int(t) / 2, lq = l / 4;
    const int out = nb + (lq & 4) + ((l / 2) % 4), row = rb + (lq & 2) * 2 + (l % 2) * 2 + int(t) % 2;
    if (out < ND && row < R) PART[(size_t(k) * R + row) * ND + out] = v;
  }
"""

_UP_MMA = r"""
  // The up projection for dims d0 .. d0 + DT - 1 of every stream x 8 rows, every row reading the weights once.
  // Prologue: the down projection's outputs from the split-K partials (summed in chunk order) -> bf16 -> / S ->
  // bf16 -> SiLU (bf16) for the rows here; threadgroup 0 of a row tile also writes their inject gates 2 sigmoid.
  // Simdgroup s takes up rows s D + d0 .. (DT / 8 tiles of 8); then sigmoid of each (bf16) times the normed stream
  // (bf16), summed over the streams in order, / S, into MIXED.
  const uint lane = thread_index_in_simdgroup;
  const int s = int(simdgroup_index_in_threadgroup);
  const uint t = thread_position_in_threadgroup.x;
  const int qid = int(lane) / 4;
  const int fm = (qid & 4) + ((int(lane) / 2) % 4);
  const int fn = (qid & 2) * 2 + (int(lane) % 2) * 2;
  const int R = rows[0];
  constexpr int W = S * D, GL = LOW / 32, TT = DT / 8;
  const int d0 = int(threadgroup_position_in_grid.x) * DT;
  const int rb = int(threadgroup_position_in_grid.y) * 8;
  const int nr = min(8, R - rb);                       // rows of this tile
  threadgroup float rinv[8 * S];
  threadgroup float act[8][LOW];
  threadgroup float prod[S][DT][8];
  if (t < 8 * S) rinv[t] = stream_rinv(SSP, min(rb + int(t) / S, R - 1), int(t) % S, D / 256, S, D, eps[0]);
  for (int i = int(t); i < nr * ND; i += 32 * S) {
    const int r = i / ND, cc = i % ND;
    float v = 0.0f;
    for (int k = 0; k < KS; k++) v += PART[(size_t(k) * R + rb + r) * ND + cc];
    const float v4 = float(bfloat(float(bfloat(v)) / float(S)));
    if (cc < LOW) act[r][cc] = bsilu(v4);
    else if (threadgroup_position_in_grid.x == 0) INJOUT[(rb + r) * S + (cc - LOW)] = bfloat(2.0f * bsig(v4));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int ra = min(fn, nr - 1), rc = min(fn + 1, nr - 1);
  for (int tt = 0; tt < TT; tt++) {
    const int o = s * D + d0 + 8 * tt + fm;
    const device uint* wq = QW + size_t(o) * (LOW / 8) + fn / 2;
    float acc0 = 0.0f, acc1 = 0.0f;
    for (int g = 0; g < GL; g++) {
      const int e0 = 32 * g + 8 * (fm / 2);
      float xa[8], xc[8];
      for (int i = 0; i < 8; i++) { xa[i] = act[ra][e0 + i]; xc[i] = act[rc][e0 + i]; }
      mma_group(wq[4 * g], xa, xc, fm, float(QS[size_t(o) * GL + g]), float(QB[size_t(o) * GL + g]), acc0, acc1);
    }
    for (int e = 0; e < 2; e++) {
      const int row = rb + min(fn + e, nr - 1);
      const float normed = float(bfloat((float(HN[size_t(row) * W + o]) * rinv[(row - rb) * S + s]) * NW[o]));
      prod[s][8 * tt + fm][fn + e] = float(bfloat(bsig(float(bfloat(e ? acc1 : acc0))) * normed));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = int(t); i < DT * 8; i += 32 * S) {
    const int d = i / 8, r = i % 8;
    float total = 0.0f;
    for (int k = 0; k < S; k++) total += prod[k][d][r];
    if (r < nr) MIXED[size_t(rb + r) * D + d0 + d] = bfloat(total / float(S));
  }
"""


_SCALAR_HEADER = r"""
// one group's product sum on the MMA path, scalar and in its order: steps st 0..3, each an fp32 FMA chain over
// k 0..7 (input 8 (k / 2) + 2 st + k % 2, nibble left in place, input scaled by 2^-4e)
inline float scalar_group(const device uint* w, const threadgroup float* x) {
  const uint4 q = *((const device uint4*)w);
  float p = 0.0f;
  for (int st = 0; st < 4; st++) {
    for (int k = 0; k < 8; k++) {
      const int e = 2 * st + k % 2;
      p = fma(float(q[k / 2] & (0xFu << (4 * e))), x[8 * (k / 2) + e] * pre4(e), p);
    }
  }
  return p;
}
// a group's input sum as the MMA path's group_sums takes it: each run of 8 left to right, then pairs of runs
inline float scalar_sum(const threadgroup float* x) {
  float s[4];
  for (int m = 0; m < 4; m++) {
    s[m] = x[8 * m];
    for (int i = 1; i < 8; i++) s[m] += x[8 * m + i];
  }
  return (s[0] + s[1]) + (s[2] + s[3]);
}
"""

_DOWN_ROW = r"""
  // The down projection of one row, bit for bit the MMA path's: threadgroup (i, k, r) takes outputs 32 i ..
  // 32 i + 31 over input groups 32 k .. 32 k + 31 of row r. Thread (g, o) takes one group's product sum; then,
  // as the MMA path, 8 chains of 4 groups each (scale and bias folded in order) add in order into PART[k][r][o].
  const uint t = thread_position_in_threadgroup.x;
  const int R = rows[0];
  constexpr int W = S * D, G = W / 32;
  const int ob = int(threadgroup_position_in_grid.x) * 32;
  const int k = int(threadgroup_position_in_grid.y);
  const int r = int(threadgroup_position_in_grid.z);
  threadgroup float xs[1024];
  threadgroup float vs[32];
  threadgroup float ps[32][32];
  threadgroup float red[8][32];
  {
    const int e = 1024 * k + int(t);
    const float ri = stream_rinv(SSP, r, e / D, D / 256, S, D, eps[0]);
    xs[t] = float(bfloat((float(HN[size_t(r) * W + e]) * ri) * NW[e]));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < 32) vs[t] = scalar_sum(xs + 32 * t);
  const int gl = int(t) / 32, ol = int(t) % 32;
  const int o = min(ob + ol, ND - 1);
  ps[gl][ol] = scalar_group(QW + size_t(o) * (W / 8) + 4 * (32 * k + gl), xs + 32 * gl);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < 256) {
    const int c = int(t) / 32;
    float acc = 0.0f;
    for (int j = 0; j < 4; j++) {
      const int g = 32 * k + 4 * c + j;
      acc = fma(float(QB[size_t(o) * G + g]), vs[4 * c + j],
                fma(float(QS[size_t(o) * G + g]), ps[4 * c + j][ol], acc));
    }
    red[c][ol] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < 32 && ob + ol < ND) {
    float v = 0.0f;
    for (int c = 0; c < 8; c++) v += red[c][ol];
    PART[(size_t(k) * R + r) * ND + ob + ol] = v;
  }
"""

_UP_ROW = r"""
  // The up projection of one row, bit for bit the MMA path's: threadgroup (i, r) takes dims 8 i .. 8 i + 7 of the
  // S streams. The prologue is the MMA path's; thread (g, s, d) takes one group's product sum, then each output's
  // chain over the groups in order, the sigmoid times the normed stream, summed over the streams in order.
  const uint t = thread_position_in_threadgroup.x;
  const int R = rows[0];
  constexpr int W = S * D, GL = LOW / 32, NO = 8 * S;
  const int d0 = int(threadgroup_position_in_grid.x) * 8;
  const int r = int(threadgroup_position_in_grid.z);
  threadgroup float act[LOW];
  threadgroup float vs[GL];
  threadgroup float ps[GL][NO];
  threadgroup float prod[S][8];
  for (int cc = int(t); cc < ND; cc += GL * NO) {
    float v = 0.0f;
    for (int k = 0; k < KS; k++) v += PART[(size_t(k) * R + r) * ND + cc];
    const float v4 = float(bfloat(float(bfloat(v)) / float(S)));
    if (cc < LOW) act[cc] = bsilu(v4);
    else if (threadgroup_position_in_grid.x == 0) INJOUT[r * S + (cc - LOW)] = bfloat(2.0f * bsig(v4));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (int(t) < GL) vs[t] = scalar_sum(act + 32 * t);
  const int g = int(t) / NO, n = int(t) % NO;
  const int s = n / 8, o = s * D + d0 + n % 8;
  ps[g][n] = scalar_group(QW + size_t(o) * (LOW / 8) + 4 * g, act + 32 * g);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (int(t) < NO) {
    float acc = 0.0f;
    for (int gg = 0; gg < GL; gg++)
      acc = fma(float(QB[size_t(o) * GL + gg]), vs[gg], fma(float(QS[size_t(o) * GL + gg]), ps[gg][n], acc));
    const float normed = float(bfloat((float(HN[size_t(r) * W + o]) * stream_rinv(SSP, r, s, D / 256, S, D, eps[0]))
                                      * NW[o]));
    prod[s][n % 8] = float(bfloat(bsig(float(bfloat(acc))) * normed));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (int(t) < 8) {
    float total = 0.0f;
    for (int k = 0; k < S; k++) total += prod[k][t];
    MIXED[size_t(r) * D + d0 + int(t)] = bfloat(total / float(S));
  }
"""

def hc_project(h_new: mx.array, ssp: mx.array, down: QWeights, up: QWeights, norm_scale: mx.array, *,
               eps: mx.array, streams: int, low: int, dims_a_group: int = 0) -> tuple[mx.array, mx.array]:
    """Return mixed rows and inject gates with padding unset; dims_a_group changes performance without changing results."""

    rows, wide = h_new.shape
    dims = wide // streams
    groups = wide // 32
    splits = groups // 32
    dt = dims_a_group or (8 if rows <= 4 else (16 if rows <= 16 else 32))
    if groups % 32 or low % 32 or dims % dt or dt % 8:
        raise ValueError("hc_project: S * D a multiple of 1024, LOW of 32, D of the dims a threadgroup")
    nd = down.rows
    if rows <= SCALAR_ROWS:
        return _project_rows(h_new, ssp, down, up, norm_scale, eps=eps, streams=streams, low=low)
    tiles = -(-rows // 8)
    run = kernel("q4_hc_down_mma", _DOWN_MMA, ["HN", "SSP", "NW", "QW", "QS", "QB", "eps", "rows"], ["PART"],
                 header=QDOT_HEADER + RINV + MMA_HEADER)
    part = run(inputs=[h_new, ssp, norm_scale, down.weight, down.scales, down.biases, eps, count(rows)],
               template=[("S", streams), ("D", dims), ("ND", nd)],
               grid=(-(-nd // 8) * 256, splits, tiles), threadgroup=(256, 1, 1),
               output_shapes=[(splits, rows, nd)], output_dtypes=[mx.float32])[0]
    run = kernel("q4_hc_up_mma", _UP_MMA, ["HN", "SSP", "NW", "PART", "QW", "QS", "QB", "eps", "rows"],
                 ["MIXED", "INJOUT"], header=QDOT_HEADER + RINV + MMA_HEADER)
    mixed, inject = run(inputs=[h_new, ssp, norm_scale, part, up.weight, up.scales, up.biases, eps, count(rows)],
                        template=[("S", streams), ("D", dims), ("LOW", low), ("ND", nd), ("KS", splits), ("DT", dt)],
                        grid=(dims // dt * 32 * streams, tiles, 1), threadgroup=(32 * streams, 1, 1),
                        output_shapes=[(rows, dims), (max(rows, 2), streams)],
                        output_dtypes=[mx.bfloat16, mx.bfloat16])
    return mixed, inject


# rows a call takes on the scalar path, which gives each row the MMA path's bits without its 8-row tiles
SCALAR_ROWS = 2


def _project_rows(h_new: mx.array, ssp: mx.array, down: QWeights, up: QWeights, norm_scale: mx.array, *,
                  eps: mx.array, streams: int, low: int) -> tuple[mx.array, mx.array]:
    """``hc_project`` for a few rows through the scalar kernels (same outputs, bit for bit)."""

    rows, wide = h_new.shape
    dims = wide // streams
    splits = wide // 32 // 32
    nd = down.rows
    if dims % 8 or wide % 1024 or 8 * streams * (low // 32) > 1024:
        raise ValueError("hc_project: the scalar path takes D a multiple of 8, S * D of 1024, 8 S LOW / 32 <= 1024")
    header = QDOT_HEADER + RINV + MMA_HEADER + _SCALAR_HEADER
    run = kernel("q4_hc_down_row", _DOWN_ROW, ["HN", "SSP", "NW", "QW", "QS", "QB", "eps", "rows"], ["PART"],
                 header=header)
    part = run(inputs=[h_new, ssp, norm_scale, down.weight, down.scales, down.biases, eps, count(rows)],
               template=[("S", streams), ("D", dims), ("ND", nd)],
               grid=(-(-nd // 32) * 1024, splits, rows), threadgroup=(1024, 1, 1),
               output_shapes=[(splits, rows, nd)], output_dtypes=[mx.float32])[0]
    run = kernel("q4_hc_up_row", _UP_ROW, ["HN", "SSP", "NW", "PART", "QW", "QS", "QB", "eps", "rows"],
                 ["MIXED", "INJOUT"], header=header)
    return tuple(run(inputs=[h_new, ssp, norm_scale, part, up.weight, up.scales, up.biases, eps, count(rows)],
                     template=[("S", streams), ("D", dims), ("LOW", low), ("ND", nd), ("KS", splits)],
                     grid=(dims // 8 * 8 * streams * (low // 32), 1, rows),
                     threadgroup=(8 * streams * (low // 32), 1, 1),
                     output_shapes=[(rows, dims), (max(rows, 2), streams)],
                     output_dtypes=[mx.bfloat16, mx.bfloat16]))
