"""0.3.4.1's per-row projections for pre-M5 GPUs: each row runs alone in its own simdgroup or threadgroups, so its bits never depend on the row count."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from tensorfold.kernels.qwen.flash_next.v1 import row_tiles
from tensorfold.kernels.qwen.flash_next.v1.base import (AFFINE_HEADER, LANE_CODES, QDOT_HEADER, QWeights, by_rows,
                                                        count, edited, kernel)
from tensorfold.kernels.qwen.flash_next.v1.hc import RINV

_QMV_ROWS = r"""
  // simdgroup r runs MLX's one-row qmv_fast loop for input row r over outputs RPS b ..; the rows share the weight reads
  const uint lane = thread_index_in_simdgroup;
  const int r = int(simdgroup_index_in_threadgroup);
  const int row0 = int(threadgroup_position_in_grid.y) * RPS;
  constexpr int KB = K / 2;
  constexpr int KG = K / 32;
  const device uint8_t* w = (const device uint8_t*)W + size_t(row0) * KB + lane * 8;
  const device bfloat* sc = S + size_t(row0) * KG + lane / 2;
  const device bfloat* bi = B + size_t(row0) * KG + lane / 2;
  const device bfloat* x = X + r * K + lane * 16;
  float acc[RPS];
  for (int j = 0; j < RPS; j++) acc[j] = 0.0f;
  for (int k0 = 0; k0 < K; k0 += 512) {
    float xt[16];
    const float sum = load16(x, xt);
    for (int j = 0; j < RPS; j++)
      acc[j] += qdot16(w + j * KB, xt, float(sc[j * KG]), float(bi[j * KG]), sum);
    w += 256; sc += 16; bi += 16; x += 512;
  }
  for (int j = 0; j < RPS; j++) {
    const float v = simd_sum(acc[j]);
    if (lane == 0) OUT[r * N + row0 + j] = bfloat(v);
  }
"""

_HC_DOWN_SPLIT = r"""
  // split-K matvec of one row's normed streams: simdgroup o of threadgroup (i, k, r) over input groups 32 k .. 32 k + 31
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup;
  const uint g = simdgroup_index_in_threadgroup;
  const int R = rows[0];
  constexpr int W = S * D;
  constexpr int GROUPS = W / 32;
  const int o = int(threadgroup_position_in_grid.x) * 8 + int(g);
  const int k = int(threadgroup_position_in_grid.y);
  const int r = int(threadgroup_position_in_grid.z);
  const int c0 = k * 32 * 32;
  threadgroup float xs[32 * 33];                      // group j at 33 j: a lane's reads hit distinct banks
  threadgroup float rinv[S];
  if (t < S) rinv[t] = stream_rinv(SSP, r, int(t), D / 256, S, D, eps[0]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = int(t); i < 32 * 32; i += 256) {
    const int e = c0 + i;
    xs[(i / 32) * 33 + i % 32] = float(bfloat((float(HN[r * W + e]) * rinv[e / D]) * NW[e]));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (o < ND) {
    const int grp = k * 32 + int(lane);
    float x[32];
    for (int n = 0; n < 32; n++) x[n] = xs[lane * 33 + n];
    float acc = qgroup_dot(QW + (size_t(o) * GROUPS + grp) * 4, float(QS[o * GROUPS + grp]), float(QB[o * GROUPS + grp]), x);
    acc = simd_sum(acc);
    if (lane == 0) PART[(k * R + r) * ND + o] = acc;
  }
"""

_HC_UP2 = r"""
  // one row's up projection for 8 dims of every stream: thread 10 i + q takes group q of up row i
  const uint t = thread_position_in_threadgroup.x;
  const int R = rows[0];
  const int r = int(threadgroup_position_in_grid.y);
  constexpr int W = S * D;
  constexpr int GPR = LOW / 32;
  constexpr int ROWS = S * 8;
  const int d0 = int(threadgroup_position_in_grid.x) * 8;
  threadgroup float act[GPR * 33];
  threadgroup float part[ROWS][GPR];
  threadgroup float prod[ROWS];
  const int i = int(t) / GPR, q = int(t) % GPR;
  const int s = i / 8, d = d0 + i % 8;
  const int row = s * D + d;
  for (int c = int(t); c < ND; c += ROWS * GPR) {
    float v = 0.0f;
    for (int k = 0; k < KS; k++) v += PART[(k * R + r) * ND + c];
    const float v4 = float(bfloat(float(bfloat(v)) / float(S)));
    if (c < LOW) act[(c / 32) * 33 + c % 32] = bsilu(v4);
    else if (threadgroup_position_in_grid.x == 0) INJOUT[r * S + (c - LOW)] = bfloat(2.0f * bsig(v4));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  {
    float x[32];
    for (int n = 0; n < 32; n++) x[n] = act[q * 33 + n];
    part[i][q] = qgroup_dot(QW + (size_t(row) * GPR + q) * 4, float(QS[row * GPR + q]), float(QB[row * GPR + q]), x);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (q == 0) {
    float u = 0.0f;
    for (int qq = 0; qq < GPR; qq++) u += part[i][qq];
    const float rv = stream_rinv(SSP, r, s, D / 256, S, D, eps[0]);
    const float normed = float(bfloat((float(HN[r * W + row]) * rv) * NW[row]));
    prod[i] = float(bfloat(bsig(float(bfloat(u))) * normed));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < 8) {
    float total = 0.0f;
    for (int ss = 0; ss < S; ss++) total += prod[ss * 8 + int(t)];
    MIXED[r * D + d0 + int(t)] = bfloat(total / float(S));
  }
"""

_HC_DOWN_SPLIT_Q = edited(_HC_DOWN_SPLIT, [(
    "float acc = qgroup_dot(QW + (size_t(o) * GROUPS + grp) * 4, float(QS[o * GROUPS + grp]), float(QB[o * GROUPS + grp]), x);",
    "float acc = qchunk_dot<BITS>(QW + size_t(o) * (W * BITS / 32) + grp * BITS, float(QS[o * (W / GS) + grp * 32 / GS]),\n"
    "                                float(QB[o * (W / GS) + grp * 32 / GS]), x);")])
_HC_UP2_Q = edited(_HC_UP2, [(
    "part[i][q] = qgroup_dot(QW + (size_t(row) * GPR + q) * 4, float(QS[row * GPR + q]), float(QB[row * GPR + q]), x);",
    "part[i][q] = qchunk_dot<BITS>(QW + size_t(row) * (LOW * BITS / 32) + q * BITS, float(QS[row * (LOW / GS) + q * 32 / GS]),\n"
    "                                  float(QB[row * (LOW / GS) + q * 32 / GS]), x);")])

_QMV_ROWS_Q = r"""
  // qmv_rows for any width: simdgroup r runs the qmv_fast loop (VPT codes a lane a step) for input row r
  const uint lane = thread_index_in_simdgroup;
  const int r = int(simdgroup_index_in_threadgroup);
  const int row0 = int(threadgroup_position_in_grid.y) * RPS;
  constexpr int VPT = lane_values(BITS), RB = K * BITS / 8, KG = K / GS;
  const device uint8_t* w = (const device uint8_t*)W + size_t(row0) * RB;
  const device bfloat* x = X + r * K;
  float acc[RPS];
  for (int j = 0; j < RPS; j++) acc[j] = 0.0f;
  for (int v0 = int(lane) * VPT; v0 < K; v0 += 32 * VPT) {
    float xv[VPT], sum = 0.0f;
    for (int i = 0; i < VPT; i++) { xv[i] = float(x[v0 + i]); sum += xv[i]; }
    for (int j = 0; j < RPS; j++) {
      float q[VPT];
      lane_codes<BITS, VPT>(w + j * RB, v0, q);
      float d = 0.0f;
      for (int i = 0; i < VPT; i++) d = fma(q[i], xv[i], d);
      const size_t at = size_t(row0 + j) * KG + v0 / GS;
      acc[j] += fma(float(S[at]), d, float(B[at]) * sum);
    }
  }
  for (int j = 0; j < RPS; j++) {
    const float v = simd_sum(acc[j]);
    if (lane == 0) OUT[r * N + row0 + j] = bfloat(v);
  }
"""

_ROW_BLOCK_SUMS = r"""
  // qmv_rows' sum of each (row, block): block b's VPT inputs added in order from zero
  const int b = int(thread_position_in_grid.x), r = int(thread_position_in_grid.y);
  if (b >= K / VPT) return;
  const device bfloat* xp = X + size_t(r) * K + b * VPT;
  float sum = 0.0f;
  for (int i = 0; i < VPT; i++) sum += float(xp[i]);
  SUMS[size_t(r) * (K / VPT) + b] = sum;
"""

_QMV_ROWS_MMA = r"""
  // qmv_rows' per-row arithmetic for several rows on the matrix units. Threadgroup (i, t): outputs 8 i .. 8 i + 7, rows
  // 8 t .. 8 t + 7 (rows past R read row R - 1, dropped). qmv_rows' lane l at step s reads block b = 32 s + l
  // (values VPT b ..); simdgroup j of SG keeps lanes l = j (32 / SG) .. : per block an MMA from zero gives each
  // (output, row) that lane's dot, fma(scale, dot, bias * sum) joins the lane's partial in step order, and the 32
  // partials of each (output, row) meet in one simd_sum, as in qmv_rows.
  const uint lane = thread_index_in_simdgroup;
  const int sg = int(simdgroup_index_in_threadgroup);
  const int qid = int(lane) / 4;
  const int fm = (qid & 4) + ((int(lane) / 2) % 4);
  const int fn = (qid & 2) * 2 + (int(lane) % 2) * 2;
  const int R = X_shape[0];
  constexpr int VPT = lane_values(BITS), NB = K / VPT, STEPS = NB / 32, KG = K / GS, WPR = K * BITS / 32;
  constexpr int L = 32 / SG;
  constexpr bool ONE_GROUP = L * VPT <= GS;             // a simdgroup's blocks of a step share one scale and bias
  static_assert(L == 4, "the sums load a step's 4 blocks as one float4");
  threadgroup float red[64 * 33];
  const int nb = int(threadgroup_position_in_grid.x) * 8;
  const int rb = int(threadgroup_position_in_grid.y) * 8;
  const int o = min(nb + fm, N - 1);
  const device uint* wrow = W + size_t(o) * WPR;
  const int ra = min(rb + fn, R - 1), rc = min(rb + fn + 1, R - 1);
  const device bfloat* xa = X + size_t(ra) * K;
  const device bfloat* xc = X + size_t(rc) * K;
  const device float* sa = SUMS + size_t(ra) * NB;
  const device float* scs = SUMS + size_t(rc) * NB;
  float acc0[L], acc1[L];
  for (int j = 0; j < L; j++) { acc0[j] = 0.0f; acc1[j] = 0.0f; }
  constexpr uint MASK = (1u << BITS) - 1u;
  for (int t = 0; t < STEPS; t++) {
    const int b0 = 32 * t + sg * L;
    const float4 sua = *(const device float4*)(sa + b0), suc = *(const device float4*)(scs + b0);
    const float sums_a[4] = {sua.x, sua.y, sua.z, sua.w}, sums_c[4] = {suc.x, suc.y, suc.z, suc.w};
    float sc = 0.0f, bi = 0.0f;
    if (ONE_GROUP) { const size_t at = size_t(o) * KG + b0 * VPT / GS; sc = float(S[at]); bi = float(B[at]); }
    PRAGMA_UNROLL
    for (int j = 0; j < L; j++) {
      const int b = b0 + j;
      const int v0 = b * VPT;
      simdgroup_matrix<float, 8, 8> P = simdgroup_matrix<float, 8, 8>(0.0f);
      PRAGMA_UNROLL
      for (int h = 0; h < VPT / 8; h++) {
        simdgroup_matrix<float, 8, 8> am, bm;
        const int bit = (v0 + 8 * h + fn) * BITS, word = bit >> 5, shift = bit & 31;
        const uint hi = shift + 2 * BITS > 32 ? wrow[word + 1] : 0u;
        const ulong pair = ((ulong(hi) << 32) | ulong(wrow[word])) >> shift;
        am.thread_elements()[0] = float(uint(pair) & MASK);
        am.thread_elements()[1] = float(uint(pair >> BITS) & MASK);
        bm.thread_elements()[0] = float(xa[v0 + 8 * h + fm]);
        bm.thread_elements()[1] = float(xc[v0 + 8 * h + fm]);
        simdgroup_multiply_accumulate(P, am, bm, P);
      }
      if (!ONE_GROUP) { const size_t at = size_t(o) * KG + v0 / GS; sc = float(S[at]); bi = float(B[at]); }
      acc0[j] += fma(sc, P.thread_elements()[0], bi * sums_a[j]);
      acc1[j] += fma(sc, P.thread_elements()[1], bi * sums_c[j]);
    }
  }
  for (int j = 0; j < L; j++) {
    red[(fm * 8 + fn) * 33 + sg * L + j] = acc0[j];
    red[(fm * 8 + fn + 1) * 33 + sg * L + j] = acc1[j];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int e = sg; e < 64; e += SG) {
    const float v = simd_sum(red[e * 33 + int(lane)]);
    const int n = nb + e / 8, row = rb + e % 8;
    if (lane == 0 && n < N && row < R) OUT[size_t(row) * N + n] = bfloat(v);
  }
"""

ROWS_A_CALL = 32     # simdgroups a qmv_rows threadgroup: one an input row


def qmv_rows(x: mx.array, weights: Any, *, rows_per_simdgroup: int = 4) -> mx.array:
    """x [..., K] @ W.T, each row alone in a simdgroup: 4-bit group 32 with MLX's one-row bits, other widths alike."""

    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    rows, dims = int(x2.shape[0]), int(x2.shape[1])
    n = int(weights.weight.shape[0])
    bits, group = int(getattr(weights, "bits", 4)), int(getattr(weights, "group_size", 32))
    if (bits, group) == (4, 32):
        if dims % 512 or n % rows_per_simdgroup:
            raise ValueError(f"qmv_rows: needs K % 512 == 0 and N % {rows_per_simdgroup} == 0")
        run = kernel(*by_rows("q4_qmv_rows", _QMV_ROWS, rows), ["X", "W", "S", "B"], ["OUT"],
                     reserve=32 * ROWS_A_CALL)
        fmt = []
    else:
        if dims % 16:
            raise ValueError("qmv_rows: needs K % 16 == 0")
        if rows >= MMA_FROM and dims % (32 * (8 if bits in (6, 8) else 16)) == 0 and _mma_exact(weights, n, dims):
            return qmv_rows_mma(x, weights)
        rows_per_simdgroup = next(c for c in (rows_per_simdgroup, 2, 1) if n % c == 0)   # a row's bits never depend on it
        run = kernel("qa_qmv_rows", _QMV_ROWS_Q, ["X", "W", "S", "B"], ["OUT"], header=QDOT_HEADER + LANE_CODES,
                     reserve=32 * ROWS_A_CALL)
        fmt = [("BITS", bits), ("GS", group)]
    parts = []
    for lo in range(0, rows, ROWS_A_CALL):
        part = x2[lo:lo + ROWS_A_CALL]
        m = int(part.shape[0])
        parts.append(run(inputs=[part, weights.weight, weights.scales, weights.biases],
                         template=[("K", dims), ("N", n), ("RPS", rows_per_simdgroup), *fmt],
                         grid=(32 * m, n // rows_per_simdgroup, 1), threadgroup=(32 * m, 1, 1),
                         output_shapes=[(m, n)], output_dtypes=[mx.bfloat16])[0])
    out = parts[0] if len(parts) == 1 else mx.concatenate(parts)
    return out.reshape(*shape[:-1], n)


MMA_FROM = 4   # rows from which qmv_rows_mma beats the per-row loop on the M3 (level at 3, behind at 2)
_mma_ok: dict[tuple[int, int, int, int], bool] = {}   # (N, K, bits, group): its rows equal the loop's on this GPU


def _mma_exact(weights: Any, n: int, dims: int) -> bool:
    """Whether qmv_rows_mma gives this shape's rows the per-row loop's bits here (checked once, on 8 random rows)."""

    key = (n, dims, int(weights.bits), int(weights.group_size))
    if key not in _mma_ok:
        x = (mx.random.normal((8, dims), key=mx.random.key(0)) * 0.5).astype(mx.bfloat16)
        rows_alone = mx.concatenate([qmv_rows(x[r:r + 1], weights) for r in range(8)])
        _mma_ok[key] = bool(mx.array_equal(qmv_rows_mma(x, weights), rows_alone).item())
    return _mma_ok[key]


def qmv_rows_mma(x: mx.array, weights: Any, *, simdgroups: int = 8) -> mx.array:
    """qmv_rows for any width but 4-bit g32, several rows at once on the matrix units: every row qmv_rows' own bits."""

    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    rows, dims = int(x2.shape[0]), int(x2.shape[1])
    n = int(weights.weight.shape[0])
    bits, group = int(weights.bits), int(weights.group_size)
    vpt = 8 if bits in (6, 8) else 16
    blocks = dims // vpt
    sums_run = kernel("qa_row_block_sums", _ROW_BLOCK_SUMS, ["X"], ["SUMS"])
    sums = sums_run(inputs=[x2], template=[("K", dims), ("VPT", vpt)], grid=(-(-blocks // 32) * 32, rows, 1),
                    threadgroup=(32, 1, 1), output_shapes=[(rows, blocks)], output_dtypes=[mx.float32])[0]
    run = kernel("qa_qmv_rows_mma", _QMV_ROWS_MMA, ["X", "SUMS", "W", "S", "B"], ["OUT"],
                 header=QDOT_HEADER + LANE_CODES + AFFINE_HEADER + '#define PRAGMA_UNROLL _Pragma("clang loop unroll(full)")\n')
    out = run(inputs=[x2, sums, weights.weight, weights.scales, weights.biases],
              template=[("K", dims), ("N", n), ("BITS", bits), ("GS", group), ("SG", simdgroups)],
              grid=(-(-n // 8) * 32 * simdgroups, -(-rows // 8), 1), threadgroup=(32 * simdgroups, 1, 1),
              output_shapes=[(rows, n)], output_dtypes=[mx.bfloat16])[0]
    return out.reshape(*shape[:-1], n)


HC_MMA_FROM = 8     # rows from which the tiles beat the per-row kernels inside a forward on the M3 (same bits)
hc_tiles_on = True  # off while the runtime times its windows at load
_hc_mma_ok: dict[tuple[int, ...], bool] = {}   # shape: the tiles give the per-row kernels' bits on this GPU


def hc_project(h_new: mx.array, ssp: mx.array, down: QWeights, up: QWeights, norm_scale: mx.array, *,
               eps: mx.array, streams: int, low: int) -> tuple[mx.array, mx.array]:
    """hc.hc_project with every row's bits its own one-row call's: (mixed [R, D], inject gates [max(R, 2), S])."""

    if hc_tiles_on and int(h_new.shape[0]) >= HC_MMA_FROM and _hc_mma_exact(down, up, norm_scale, eps, streams, low):
        return row_tiles.hc_tiles(h_new, ssp, down, up, norm_scale, eps=eps, streams=streams, low=low)
    return _hc_rows(h_new, ssp, down, up, norm_scale, eps=eps, streams=streams, low=low)


def _hc_mma_exact(down: QWeights, up: QWeights, norm_scale: mx.array, eps: mx.array, streams: int, low: int) -> bool:
    """Whether the tiles give this shape's rows the per-row kernels' bits here (checked once, on 12 random rows)."""

    key = (down.rows, down.cols, down.bits, down.group, up.rows, up.cols, up.bits, up.group, streams, low)
    if key not in _hc_mma_ok:
        _hc_mma_ok[key] = False
        if down.q4 and up.q4 and (down.cols // streams) % 32 == 0:
            from tensorfold.kernels.qwen.flash_next.v1.hc import hc_norm

            h = (mx.random.normal((12, down.cols), key=mx.random.key(3)) * 0.3).astype(mx.bfloat16)
            hn, ssp = hc_norm(h, streams=streams)
            want = _hc_rows(hn, ssp, down, up, norm_scale, eps=eps, streams=streams, low=low)
            try:
                got = row_tiles.hc_tiles(hn, ssp, down, up, norm_scale, eps=eps, streams=streams, low=low)
                same = bool(mx.array_equal(want[0], got[0]).item())
                if down.rows > low:                   # the gates exist only with inject rows
                    same = same and bool(mx.array_equal(want[1][:12], got[1][:12]).item())
            except Exception as exc:  # noqa: BLE001 - a GPU the tiles don't build on keeps the per-row kernels
                print(f"[flash-next] hyper-connection tiles unavailable here ({type(exc).__name__}): per-row kernels",
                      flush=True)
                same = False
            _hc_mma_ok[key] = same
    return _hc_mma_ok[key]


def _hc_rows(h_new: mx.array, ssp: mx.array, down: QWeights, up: QWeights, norm_scale: mx.array, *,
             eps: mx.array, streams: int, low: int) -> tuple[mx.array, mx.array]:
    """Every row in its own threadgroups: 0.3.4.1's hyper-connection kernels."""

    rows, wide = h_new.shape
    dims = wide // streams
    groups = wide // 32
    splits = groups // 32
    if groups % 32:
        raise ValueError("hc_project: S * D must be a multiple of 1024")
    generic = not (down.q4 and up.q4)
    dq = [("BITS", down.bits), ("GS", down.group)] if generic else []
    uq = [("BITS", up.bits), ("GS", up.group)] if generic else []
    if generic:
        down_run = kernel("qa_hc_down_split", _HC_DOWN_SPLIT_Q, ["HN", "SSP", "NW", "QW", "QS", "QB", "eps", "rows"],
                          ["PART"], header=QDOT_HEADER + RINV + AFFINE_HEADER)
    else:
        down_run = kernel(*by_rows("q4_hc_down_split", _HC_DOWN_SPLIT, rows),
                          ["HN", "SSP", "NW", "QW", "QS", "QB", "eps", "rows"], ["PART"], header=QDOT_HEADER + RINV)
    part = down_run(inputs=[h_new, ssp, norm_scale, down.weight, down.scales, down.biases, eps, count(rows)],
                    template=[("S", streams), ("D", dims), ("ND", down.rows), *dq],
                    grid=(-(-down.rows // 8) * 256, splits, rows), threadgroup=(256, 1, 1),
                    output_shapes=[(splits, rows, down.rows)], output_dtypes=[mx.float32])[0]
    threads = streams * 8 * (low // 32)
    names = ["HN", "SSP", "PART", "QW", "QS", "QB", "NW", "eps", "rows"]
    if generic:
        up_run = kernel("qa_hc_up2", _HC_UP2_Q, names, ["MIXED", "INJOUT"], header=QDOT_HEADER + RINV + AFFINE_HEADER)
    else:
        up_run = kernel(*by_rows("q4_hc_up2", _HC_UP2, rows), names, ["MIXED", "INJOUT"], header=QDOT_HEADER + RINV)
    mixed, inject = up_run(inputs=[h_new, ssp, part, up.weight, up.scales, up.biases, norm_scale, eps, count(rows)],
                           template=[("S", streams), ("D", dims), ("LOW", low), ("ND", down.rows), ("KS", splits),
                                     *uq],
                           grid=(dims // 8 * threads, rows, 1), threadgroup=(threads, 1, 1),
                           output_shapes=[(rows, dims), (max(rows, 2), streams)],
                           output_dtypes=[mx.bfloat16, mx.bfloat16])
    return mixed, inject


__all__ = ["hc_project", "qmv_rows", "qmv_rows_mma"]
