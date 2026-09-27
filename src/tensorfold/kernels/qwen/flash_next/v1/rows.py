"""0.3.4.1's per-row projections for pre-M5 GPUs: each row runs alone in its own simdgroup or threadgroups, so its bits never depend on the row count."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from tensorfold.kernels.qwen.flash_next.v1.base import QDOT_HEADER, QWeights, count, kernel
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

ROWS_A_CALL = 32     # simdgroups a qmv_rows threadgroup: one an input row


def qmv_rows(x: mx.array, weights: Any, *, rows_per_simdgroup: int = 4) -> mx.array:
    """x [..., K] @ W.T for 4-bit groups of 32: every row MLX's one-row bits, whatever the row count."""

    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    rows, dims = int(x2.shape[0]), int(x2.shape[1])
    n = int(weights.weight.shape[0])
    if dims % 512 or n % rows_per_simdgroup:
        raise ValueError(f"qmv_rows: needs K % 512 == 0 and N % {rows_per_simdgroup} == 0")
    run = kernel("q4_qmv_rows", _QMV_ROWS, ["X", "W", "S", "B"], ["OUT"])
    parts = []
    for lo in range(0, rows, ROWS_A_CALL):
        part = x2[lo:lo + ROWS_A_CALL]
        m = int(part.shape[0])
        parts.append(run(inputs=[part, weights.weight, weights.scales, weights.biases],
                         template=[("K", dims), ("N", n), ("RPS", rows_per_simdgroup)],
                         grid=(32 * m, n // rows_per_simdgroup, 1), threadgroup=(32 * m, 1, 1),
                         output_shapes=[(m, n)], output_dtypes=[mx.bfloat16])[0])
    out = parts[0] if len(parts) == 1 else mx.concatenate(parts)
    return out.reshape(*shape[:-1], n)


def hc_project(h_new: mx.array, ssp: mx.array, down: QWeights, up: QWeights, norm_scale: mx.array, *,
               eps: mx.array, streams: int, low: int) -> tuple[mx.array, mx.array]:
    """hc.hc_project with every row in its own threadgroups: (mixed [R, D], inject gates [max(R, 2), S])."""

    rows, wide = h_new.shape
    dims = wide // streams
    groups = wide // 32
    splits = groups // 32
    if groups % 32:
        raise ValueError("hc_project: S * D must be a multiple of 1024")
    down_run = kernel("q4_hc_down_split", _HC_DOWN_SPLIT, ["HN", "SSP", "NW", "QW", "QS", "QB", "eps", "rows"],
                      ["PART"], header=QDOT_HEADER + RINV)
    part = down_run(inputs=[h_new, ssp, norm_scale, down.weight, down.scales, down.biases, eps, count(rows)],
                    template=[("S", streams), ("D", dims), ("ND", down.rows)],
                    grid=(-(-down.rows // 8) * 256, splits, rows), threadgroup=(256, 1, 1),
                    output_shapes=[(splits, rows, down.rows)], output_dtypes=[mx.float32])[0]
    threads = streams * 8 * (low // 32)
    up_run = kernel("q4_hc_up2", _HC_UP2, ["HN", "SSP", "PART", "QW", "QS", "QB", "NW", "eps", "rows"],
                    ["MIXED", "INJOUT"], header=QDOT_HEADER + RINV)
    mixed, inject = up_run(inputs=[h_new, ssp, part, up.weight, up.scales, up.biases, norm_scale, eps, count(rows)],
                           template=[("S", streams), ("D", dims), ("LOW", low), ("ND", down.rows), ("KS", splits)],
                           grid=(dims // 8 * threads, rows, 1), threadgroup=(threads, 1, 1),
                           output_shapes=[(rows, dims), (max(rows, 2), streams)],
                           output_dtypes=[mx.bfloat16, mx.bfloat16])
    return mixed, inject


__all__ = ["hc_project", "qmv_rows"]
