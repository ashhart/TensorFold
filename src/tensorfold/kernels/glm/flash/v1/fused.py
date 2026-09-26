"""GLM-5.3-Flash fused decode kernels: fewer, larger kernels a layer on the decode path (1-16 rows).

Every kernel reproduces the row-by-row decode path's bits (strategy B of the recipe book: MLX's own
one-row partitions and summation orders, repeated per row), so serial output is unchanged and a window's rows get
the bits one-row steps give them:

    moe_rows    the MoE block in five kernels: the router's fp32 matmul reading the stored bf16 weights once
                (MLX's gemv_t tiling for 288 experts), route + group (bias-corrected top-k in
                ``mx.argpartition``'s order, ties by id), gate/up + SwiGLU (the shared expert as one more slot),
                down, and the weighted combine in the reference's order;
    hc_step     a hyper-connection boundary in three kernels: the previous block's write-back with the stream RMS,
                the 24-way mix (from the repacked bf16 matrix), then sinkhorn + collapse + the block's RMSNorm.

The whole KDA step is ``kda.py`` (ported from mlx-vlm PR #2105).

Precision rules (from mlx-vlm PR #2105, avlp12, MIT): exp is spelled metal::precise::exp where MLX's prebuilt
kernels use the precise one, and sums of squares are kept from contracting into fma.
"""

from __future__ import annotations

import hashlib
from typing import Any

import mlx.core as mx

from tensorfold.kernels.glm.flash.v1 import kernels as K

MAX_ROWS = 16
SQ_FMA = 0
ROUTER_TG = 1024
HC_MIX_U = 8             # iterations of loads the hyper-connection mix issues ahead
SPLIT_SHARED = 1         # the shared expert in kernels of its own, beside the router (moe_rows)

_HEADER = K._HEADER + r"""
template <typename U>
inline U sigmoid_precise(U x) {
  U e = static_cast<U>(metal::precise::exp(metal::abs(x)));
  U y = static_cast<U>(1) / (static_cast<U>(1) + e);
  return (x < 0) ? y : (static_cast<U>(1) - y);
}
// nn.silu is an mx.compile'd x * sigmoid(x) on bf16: MLX's Sigmoid in bfloat arithmetic with the JIT's (fast) exp
template <typename U>
inline U sigmoid_fast(U x) {
  U e = static_cast<U>(metal::exp(metal::abs(x)));
  U y = static_cast<U>(1) / (static_cast<U>(1) + e);
  return (x < 0) ? y : (static_cast<U>(1) - y);
}
#pragma clang fp contract(off)
// MLX's rms_norm accumulates acc += x * x in its prebuilt library: FMA 1 if that contracts to an fma there
template <int FMA>
inline float sq_acc(float acc, float v) { return FMA ? fma(v, v, acc) : v * v + acc; }
inline float mul_add(float acc, float a, float b) { return a * b + acc; }
inline float add_nc(float a, float b) { return a + b; }
#pragma clang fp contract(on)
"""

_MOE_ROUTE = r"""
  // One threadgroup (MAXR simdgroups). Simdgroup r: row r's top TOPK experts by sigmoid(logit) + bias, largest
  // first, lowest id among ties (mx.argpartition's order here, checked), and their weights: scores / their sum
  // (in pick order) x SCALE. Then thread e lists the picks (r TOPK + k) of expert e, and the distinct experts get
  // places in increasing id order (expert_group's layout).
  const uint lane = thread_index_in_simdgroup;
  const uint g = simdgroup_index_in_threadgroup;
  const int e = int(thread_position_in_threadgroup.x);
  const int R = int(LOGITS_shape[0]);
  constexpr int PER = (NE + 31) / 32;
  threadgroup int picks[MAXR * TOPK];
  threadgroup int offs[32];
  if (int(g) < R) {
    const int r = int(g);
    float c[PER], sc[PER];
    for (int j = 0; j < PER; j++) {
      const int id = j * 32 + int(lane);
      if (id < NE) {
        sc[j] = sigmoid_precise(LOGITS[r * NE + id]);
        c[j] = sc[j] + BIAS[id];
      } else {
        sc[j] = 0.0f; c[j] = -INFINITY;
      }
    }
    float w[TOPK];
    for (int k = 0; k < TOPK; k++) {
      float best = -INFINITY, bsc = 0.0f;
      int bid = NE;
      for (int j = 0; j < PER; j++) {
        const int id = j * 32 + int(lane);
        if (id < NE && (c[j] > best || (c[j] == best && id < bid))) { best = c[j]; bid = id; bsc = sc[j]; }
      }
      for (int off = 16; off > 0; off /= 2) {
        const float ob = simd_shuffle_xor(best, off);
        const int oi = simd_shuffle_xor(bid, off);
        const float os = simd_shuffle_xor(bsc, off);
        if (ob > best || (ob == best && oi < bid)) { best = ob; bid = oi; bsc = os; }
      }
      w[k] = bsc;
      if (int(lane) == bid % 32) c[bid / 32] = -INFINITY;
      if (lane == 0) { picks[r * TOPK + k] = bid; PICK[r * TOPK + k] = bid; }
    }
    if (lane == 0) {
      float total = w[0];
      for (int k = 1; k < TOPK; k++) total = total + w[k];
      for (int k = 0; k < TOPK; k++) WTS[r * TOPK + k] = (w[k] / total) * SCALE[0];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int members[MAXR];
  int count = 0;
  if (e < NE)
    for (int p = 0; p < R * TOPK; p++)
      if (picks[p] == e) members[count++] = p;
  const int used = count > 0 ? 1 : 0;
  const int before = simd_prefix_exclusive_sum(used);
  if (lane == 31) offs[g] = before + used;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int base = 0;
  for (int q = 0; q < int(g); q++) base += offs[q];
  if (used) {
    const int u = base + before;
    UIDS[u] = e;
    for (int j = 0; j < MAXR; j++) UMEM[u * MAXR + j] = j < count ? members[j] : -1;
  }
  if (e == int(NT) - 1) UCOUNT[0] = base + before + used;
"""

_ROUTER = r"""
  // MLX's one-row gemv_t for x [RR, K] fp32 @ M [K, NE] (tiling BM 1, BN 2, SM 8, SN 4, TM 4, TN 4): thread (column
  // quad q, thrM m) sums rows 4 m + 32 i + tm for i in order, then the shuffle-down reduction over m. The matrix
  // comes repacked per thread (RP[q][m][i][tm][tn], its stored bf16 values: exact in fp32), so a thread streams its
  // own values, U iterations' loads issued before their use; the window's rows share the reads (each row's sums in
  // the same order).
  const uint lane = thread_index_in_simdgroup;
  const int thrM = int(lane) / 4, thrN = int(lane) % 4;
  const int q = int(threadgroup_position_in_grid.x) * 4 + thrN;       // column quad: columns 4 q .. 4 q + 3
  constexpr int ITERS = K / 32;
  float acc[RR][4];
  for (int r = 0; r < RR; r++) for (int tn = 0; tn < 4; tn++) acc[r][tn] = 0.0f;
  const device uint4* w = (const device uint4*)(RP + (size_t(q) * 8 + thrM) * ITERS * 16);
  for (int i0 = 0; i0 < ITERS; i0 += U) {
    uint4 raw[U][2];                                                   // U iterations x 16 bf16
    for (int u = 0; u < U; u++) { raw[u][0] = w[(i0 + u) * 2]; raw[u][1] = w[(i0 + u) * 2 + 1]; }
    for (int u = 0; u < U; u++) {
      float inter[4][4];
      for (int h = 0; h < 2; h++) {
        const uint4 v = raw[u][h];
        const uint words[4] = {v.x, v.y, v.z, v.w};
        for (int j = 0; j < 4; j++) {
          const int e = h * 8 + j * 2;                                 // bf16 pairs: low half first
          inter[e / 4][e % 4] = as_type<float>(words[j] << 16);
          inter[(e + 1) / 4][(e + 1) % 4] = as_type<float>(words[j] & 0xffff0000u);
        }
      }
      const int bm = 4 * thrM + 32 * (i0 + u);
      for (int r = 0; r < RR; r++) {
        float vc[4];
        for (int tm = 0; tm < 4; tm++) vc[tm] = X[size_t(r) * K + bm + tm];
        for (int tm = 0; tm < 4; tm++)
          for (int tn = 0; tn < 4; tn++) acc[r][tn] += vc[tm] * inter[tm][tn];
      }
    }
  }
  for (int r = 0; r < RR; r++)
    for (int tn = 0; tn < 4; tn++) {
      float v = acc[r][tn];
      for (ushort sm = 4; sm >= 1; sm >>= 1) v += simd_shuffle_down(v, 4 * sm);
      if (thrM == 0) OUT[size_t(r) * NE + 4 * q + tn] = v;
    }
"""

_ROUTER_TG = r"""
  // _ROUTER's arithmetic (simdgroup 0 computes, exactly as there) with the other simdgroups fetching its weights
  // into threadgroup memory, C iterations a chunk, double-buffered.
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
  const int thrM = int(lane) / 4, thrN = int(lane) % 4;
  const int g = int(threadgroup_position_in_grid.x);
  constexpr int ITERS = K / 32;
  constexpr int NCH = ITERS / C;
  constexpr int UNITS = 32 * C * 2;                                    // uint4 units a chunk
  constexpr int XS = 32 * C;                                           // x values a row a chunk
  threadgroup uint4 buf[2][UNITS];
  threadgroup float xb[2][RR][XS];
  const device uint4* rp = (const device uint4*)RP;
  auto fetch = [&](int ch, int b) {
    for (int u = int(t) - 32; u < UNITS + RR * XS; u += int(NT) - 32) {
      if (u < 0) continue;
      if (u < UNITS) {
        const int l = u / (C * 2), it = (u % (C * 2)) / 2, hh = u % 2;
        const int q = g * 4 + (l % 4), m = l / 4;
        buf[b][u] = rp[((size_t(q) * 8 + m) * ITERS + ch * C + it) * 2 + hh];
      } else {
        const int v = u - UNITS, r = v / XS, k = v % XS;
        xb[b][r][k] = X[size_t(r) * K + ch * XS + k];
      }
    }
  };
  float acc[RR][4];
  for (int r = 0; r < RR; r++) for (int tn = 0; tn < 4; tn++) acc[r][tn] = 0.0f;
  fetch(0, 0);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int ch = 0; ch < NCH; ch++) {
    const int b = ch & 1;
    if (sg != 0) {
      if (ch + 1 < NCH) fetch(ch + 1, b ^ 1);
    } else {
      for (int it = 0; it < C; it++) {
        float inter[4][4];
        for (int h = 0; h < 2; h++) {
          const uint4 v = buf[b][(int(lane) * C + it) * 2 + h];
          const uint words[4] = {v.x, v.y, v.z, v.w};
          for (int j = 0; j < 4; j++) {
            const int e = h * 8 + j * 2;
            inter[e / 4][e % 4] = as_type<float>(words[j] << 16);
            inter[(e + 1) / 4][(e + 1) % 4] = as_type<float>(words[j] & 0xffff0000u);
          }
        }
        const int bm = 4 * thrM + 32 * it;                             // within the chunk
        for (int r = 0; r < RR; r++) {
          float vc[4];
          for (int tm = 0; tm < 4; tm++) vc[tm] = xb[b][r][bm + tm];
          for (int tm = 0; tm < 4; tm++)
            for (int tn = 0; tn < 4; tn++) acc[r][tn] += vc[tm] * inter[tm][tn];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (sg != 0) return;
  const int q = g * 4 + thrN;
  for (int r = 0; r < RR; r++)
    for (int tn = 0; tn < 4; tn++) {
      float v = acc[r][tn];
      for (ushort sm = 4; sm >= 1; sm >>= 1) v += simd_shuffle_down(v, 4 * sm);
      if (thrM == 0) OUT[size_t(r) * NE + 4 * q + tn] = v;
    }
"""

_MOE_GATEUP = r"""
  // Threadgroup (b, u): simdgroup m takes the m-th pick of distinct expert u (u == MAXU: the shared expert, every
  // row, slot TOPK) and computes gate and up rows RPS b .. RPS b + RPS - 1 with the one-row qmv_fast loop (the bits
  // mx.gather_qmm and the shared expert's projection give them), then SwiGLU clamped at LIMIT as the row-by-row
  // path's bf16 ops do it. ACT[row][slot][n].
  const uint lane = thread_index_in_simdgroup;
  const int m = int(simdgroup_index_in_threadgroup);
  // PART 0: routed and shared (ACT [R][TOPK + 1][N], the shared expert in slot TOPK); PART 1: the shared expert
  // alone (ACT [R][1][N]); PART 2: the routed experts alone (ACT [R][TOPK][N]). The same arithmetic in each.
  const int u = PART == 1 ? MAXU : int(threadgroup_position_in_grid.z);
  const int R = int(X_shape[0]);
  constexpr int SLOTS = PART == 0 ? TOPK + 1 : (PART == 1 ? 1 : TOPK);
  const bool shared = u == MAXU;
  if (!shared && u >= UCOUNT[0]) return;
  const int pick = shared ? (m < R ? m * TOPK : -1) : UMEM[u * MAXR + m];
  if (pick < 0) return;
  const int r = pick / TOPK, slot = shared ? (PART == 1 ? 0 : TOPK) : pick % TOPK;
  const int row0 = int(threadgroup_position_in_grid.y) * RPS;
  constexpr int KB = K / 2;
  constexpr int KG = K / 64;
  const size_t e = shared ? 0 : size_t(UIDS[u]);
  // shared: one stacked matrix [gate (N) ; up (N)] rows
  const size_t grow = shared ? size_t(row0) : e * N + row0;
  const size_t urow = shared ? size_t(N + row0) : e * N + row0;
  const device uint8_t* gw = (const device uint8_t*)(shared ? SGU : GW) + grow * KB + lane * 8;
  const device uint8_t* uw = (const device uint8_t*)(shared ? SGU : UW) + urow * KB + lane * 8;
  const device bfloat* gs = (shared ? SGUS : GS) + grow * KG + lane / 4;
  const device bfloat* gb = (shared ? SGUB : GB) + grow * KG + lane / 4;
  const device bfloat* us = (shared ? SGUS : US) + urow * KG + lane / 4;
  const device bfloat* ub = (shared ? SGUB : UB) + urow * KG + lane / 4;
  const device bfloat* x = X + size_t(r) * K + lane * 16;
  float ag[RPS], au[RPS];
  for (int j = 0; j < RPS; j++) { ag[j] = 0.0f; au[j] = 0.0f; }
  for (int k0 = 0; k0 < K; k0 += 512) {
    float xt[16];
    const float sum = load16(x, xt);
    for (int j = 0; j < RPS; j++) {
      ag[j] += qdot16(gw + j * KB, xt, float(gs[j * KG]), float(gb[j * KG]), sum);
      au[j] += qdot16(uw + j * KB, xt, float(us[j * KG]), float(ub[j * KG]), sum);
    }
    gw += 256; uw += 256; gs += 8; gb += 8; us += 8; ub += 8; x += 512;
  }
  for (int j = 0; j < RPS; j++) {
    const float gv = simd_sum(ag[j]), uv = simd_sum(au[j]);
    if (lane == 0) {
      // minimum / clip on bf16 return one of their inputs: exact in float
      const float lim = float(bfloat(LIM[0]));
      const bfloat gt = bfloat(metal::min(float(bfloat(gv)), lim));
      const bfloat up = bfloat(metal::min(metal::max(float(bfloat(uv)), -lim), lim));
      const bfloat sl = gt * sigmoid_fast(gt);
      ACT[(size_t(r) * SLOTS + slot) * N + row0 + j] = sl * up;
    }
  }
"""

_MOE_DOWN = r"""
  // Threadgroup (b, u): simdgroup m takes the m-th pick of distinct expert u (MAXU: the shared expert) and computes
  // down rows RPS b .. RPS b + RPS - 1 of its activation with the one-row qmv_fast loop. Y[row][slot][d].
  const uint lane = thread_index_in_simdgroup;
  const int m = int(simdgroup_index_in_threadgroup);
  const int u = PART == 1 ? MAXU : int(threadgroup_position_in_grid.z);
  const int R = int(ACT_shape[0]);
  constexpr int SLOTS = PART == 0 ? TOPK + 1 : (PART == 1 ? 1 : TOPK);
  const bool shared = u == MAXU;
  if (!shared && u >= UCOUNT[0]) return;
  const int pick = shared ? (m < R ? m * TOPK : -1) : UMEM[u * MAXR + m];
  if (pick < 0) return;
  const int r = pick / TOPK, slot = shared ? (PART == 1 ? 0 : TOPK) : pick % TOPK;
  const int row0 = int(threadgroup_position_in_grid.y) * RPS;
  constexpr int KB = K / 2;
  constexpr int KG = K / 64;
  const size_t at = shared ? size_t(row0) : size_t(UIDS[u]) * N + row0;
  const device uint8_t* w = (const device uint8_t*)(shared ? SDW : DW) + at * KB + lane * 8;
  const device bfloat* sc = (shared ? SDS : DS) + at * KG + lane / 4;
  const device bfloat* bi = (shared ? SDB : DB) + at * KG + lane / 4;
  const device bfloat* x = ACT + (size_t(r) * SLOTS + slot) * K + lane * 16;
  float acc[RPS];
  for (int j = 0; j < RPS; j++) acc[j] = 0.0f;
  for (int k0 = 0; k0 < K; k0 += 512) {
    float xt[16];
    const float sum = load16(x, xt);
    for (int j = 0; j < RPS; j++) acc[j] += qdot16(w + j * KB, xt, float(sc[j * KG]), float(bi[j * KG]), sum);
    w += 256; sc += 8; bi += 8; x += 512;
  }
  for (int j = 0; j < RPS; j++) {
    const float v = simd_sum(acc[j]);
    if (lane == 0) Y[(size_t(r) * SLOTS + slot) * N + row0 + j] = bfloat(v);
  }
"""

_MOE_COMBINE_SPLIT = r"""
  // _MOE_COMBINE with the routed experts' outputs Y [R][TOPK][D] and the shared expert's YS [R][D] apart
  const uint gid = thread_position_in_grid.x;
  const int r = int(gid / uint(D)), d = int(gid % uint(D));
  if (r >= int(WTS_shape[0])) return;
  const device bfloat* y = Y + size_t(r) * TOPK * D + d;
  float acc = WTS[r * TOPK] * float(y[0]);
  for (int k = 1; k < TOPK; k++) acc = mul_add(acc, WTS[r * TOPK + k], float(y[size_t(k) * D]));
  OUT[size_t(r) * D + d] = bfloat(acc) + YS[size_t(r) * D + d];
"""

_MOE_COMBINE = r"""
  // out[r][d] = bf16(bf16(sum_k w_k y_k, fp32 in slot order, each product rounded before its add) + shared)
  const uint gid = thread_position_in_grid.x;
  const int r = int(gid / uint(D)), d = int(gid % uint(D));
  if (r >= int(WTS_shape[0])) return;
  constexpr int SLOTS = TOPK + 1;
  const device bfloat* y = Y + size_t(r) * SLOTS * D + d;
  float acc = WTS[r * TOPK] * float(y[0]);
  for (int k = 1; k < TOPK; k++) acc = mul_add(acc, WTS[r * TOPK + k], float(y[size_t(k) * D]));
  OUT[size_t(r) * D + d] = bfloat(acc) + y[size_t(TOPK) * D];
"""

# -- hyper-connections -----------------------------------------------------------------------------------------------
_HC_COMMON = r"""
  constexpr int S = 4;
  constexpr int F = S * D;                          // flattened streams
  constexpr int MIX = (2 + S) * S;                  // 24 mixes
"""

# A block boundary is three kernels. Each piece repeats the row-by-row path's MLX kernels in the same partition
# and order, so the bits are the row-by-row path's:
#   write-back   post_s b + sum_j comb[j][s] x_j: an fp32 product, the matmul as an fma chain from comb[0] x_0
#   stream RMS   rms_looped: thread t sums x^2 over its 4-value runs t 4 + 4096 k, then two simd_sum levels
#   mix          gemv (BM 1, BN 8, SN 32, TM 4, TN 4): 6 threadgroups of 8 simdgroups a row, as MLX launches it
#   split        mlx-vlm's hc_sinkhorn_collapse kernel (Apple, MIT): sigmoid pre/post, sinkhorn, collapse
#   RMSNorm      rms_single_row over D: thread t its 4 contiguous values
_HC_EXPAND = _HC_COMMON + r"""
  // Threadgroup r (1024 threads): the pending write-back (EXPAND) and the streams' RMS scale (SPLIT).
  const int r = int(threadgroup_position_in_grid.x);
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
  threadgroup float red[32];
  device const bfloat* xo = XOLD + size_t(r) * F;
  device bfloat* xn = XNEW + size_t(r) * F;
  float ss = 0.0f;
  for (int k = 0; k < F / 4096; ++k) {
    for (int i = 0; i < 4; ++i) {
      const int f = int(t) * 4 + 4096 * k + i;
      float v;
      if (EXPAND) {
        const int s = f / D, d = f - s * D;
        const float y = POST[r * S + s] * float(BRANCH[size_t(r) * D + d]);
        const device float* c = COMB + r * S * S;
        float mm = c[0 * S + s] * float(xo[0 * D + d]);
        mm = fma(c[1 * S + s], float(xo[1 * D + d]), mm);
        mm = fma(c[2 * S + s], float(xo[2 * D + d]), mm);
        mm = fma(c[3 * S + s], float(xo[3 * D + d]), mm);
        const bfloat nb = bfloat(add_nc(y, mm));
        xn[f] = nb;
        v = float(nb);
      } else {
        v = float(xo[f]);
      }
      ss = sq_acc<SQ_FMA>(ss, v);
    }
  }
  if (!SPLIT) return;
  ss = simd_sum(ss);
  if (sg == 0) red[lane] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) red[sg] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const float a = simd_sum(red[lane]);
    if (lane == 0) INV[r] = metal::precise::rsqrt(a / float(F) + EPS[0]);
  }
"""

_HC_MIX = _HC_COMMON + r"""
  // Threadgroup (og, r): MLX's gemv for mixes og 4 .. og 4 + 3 of row r on z = x inv (the rms_norm output).
  const int og = int(threadgroup_position_in_grid.x);
  const int r = int(threadgroup_position_in_grid.y);
  const uint lane = thread_index_in_simdgroup, sgn = simdgroup_index_in_threadgroup;
  threadgroup float part[8][4];
  const float inv = INV[r];
  device const bfloat* xs = X + size_t(r) * F;
  float res[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int bn = (32 * int(sgn) + int(lane)) * 4; bn < F; bn += 1024) {
    float vc[4];
    for (int tn = 0; tn < 4; tn++) vc[tn] = float(xs[bn + tn]) * inv;
    for (int tm = 0; tm < 4; tm++) {
      const device float* mrow = FN + size_t(og * 4 + tm) * F;
      float inter[4];
      for (int tn = 0; tn < 4; tn++) inter[tn] = mrow[bn + tn];
      for (int tn = 0; tn < 4; tn++) res[tm] += inter[tn] * vc[tn];
    }
  }
  for (int tm = 0; tm < 4; tm++)
    for (ushort sn = 16; sn >= 1; sn >>= 1) res[tm] += simd_shuffle_down(res[tm], sn);
  if (lane == 0) for (int tm = 0; tm < 4; tm++) part[sgn][tm] = res[tm];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgn == 0 && lane < 4) {
    float a = part[0][lane];
    for (int k = 1; k < 8; k++) a += part[k][lane];
    MIXES[r * MIX + og * 4 + lane] = a;
  }
"""

_HC_MIX_PACKED = _HC_COMMON + r"""
  // _HC_MIX's arithmetic with the stored bf16 matrix repacked per thread (FNP[og][sgn][lane][i][tm][tn]), its
  // U iterations' loads issued before their use.
  const int og = int(threadgroup_position_in_grid.x);
  const int r = int(threadgroup_position_in_grid.y);
  const uint lane = thread_index_in_simdgroup, sgn = simdgroup_index_in_threadgroup;
  constexpr int ITERS = F / 1024;
  threadgroup float part[8][4];
  const float inv = INV[r];
  device const bfloat* xs = X + size_t(r) * F;
  const device uint4* w = (const device uint4*)(FNP + ((size_t(og) * 8 + sgn) * 32 + lane) * ITERS * 16);
  const int bn0 = (32 * int(sgn) + int(lane)) * 4;
  float res[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int i0 = 0; i0 < ITERS; i0 += U) {
    uint4 raw[U][2];
    float xv[U][4];
    for (int u = 0; u < U; u++) {
      raw[u][0] = w[(i0 + u) * 2]; raw[u][1] = w[(i0 + u) * 2 + 1];
      for (int tn = 0; tn < 4; tn++) xv[u][tn] = float(xs[bn0 + 1024 * (i0 + u) + tn]);
    }
    for (int u = 0; u < U; u++) {
      float vc[4];
      for (int tn = 0; tn < 4; tn++) vc[tn] = xv[u][tn] * inv;
      float inter[4][4];
      for (int h = 0; h < 2; h++) {
        const uint4 v = raw[u][h];
        const uint words[4] = {v.x, v.y, v.z, v.w};
        for (int j = 0; j < 4; j++) {
          const int e = h * 8 + j * 2;
          inter[e / 4][e % 4] = as_type<float>(words[j] << 16);
          inter[(e + 1) / 4][(e + 1) % 4] = as_type<float>(words[j] & 0xffff0000u);
        }
      }
      for (int tm = 0; tm < 4; tm++)
        for (int tn = 0; tn < 4; tn++) res[tm] += inter[tm][tn] * vc[tn];
    }
  }
  for (int tm = 0; tm < 4; tm++)
    for (ushort sn = 16; sn >= 1; sn >>= 1) res[tm] += simd_shuffle_down(res[tm], sn);
  if (lane == 0) for (int tm = 0; tm < 4; tm++) part[sgn][tm] = res[tm];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgn == 0 && lane < 4) {
    float a = part[0][lane];
    for (int k = 1; k < 8; k++) a += part[k][lane];
    MIXES[r * MIX + og * 4 + lane] = a;
  }
"""

_HC_SPLIT_NORM = _HC_COMMON + r"""
  // Threadgroup r (1024 threads): sinkhorn and pre / post / comb (hc_sinkhorn_collapse's arithmetic), the collapse
  // and the next block's RMSNorm.
  constexpr float HC_EPS = HC_EPS_INT * 1e-9;       // as the hc_split kernel spells its eps
  const int r = int(threadgroup_position_in_grid.x);
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
  threadgroup float red[32];
  threadgroup float pre_s[S];
  threadgroup float inv_s[1];
  device const float* mixes = MIXES + r * MIX;
  device const bfloat* xs = X + size_t(r) * F;
  if (sg == 0) {
    constexpr int BASE_OFF = 2 * S;
    const float pre_scale = SCALE[0], post_scale = SCALE[1], comb_scale = SCALE[2];
    const float active = (lane < (uint)S) ? 1.0f : 0.0f;
    const uint llane = metal::min(lane, (uint)(S - 1));
    const float pre_z = mixes[llane] * pre_scale + BASEV[llane];
    const float post_z = mixes[S + llane] * post_scale + BASEV[S + llane];
    const float pre_v = 1.0f / (1.0f + metal::fast::exp(-pre_z)) + HC_EPS;
    const float post_v = 2.0f / (1.0f + metal::fast::exp(-post_z));
    if (lane < (uint)S) { pre_s[lane] = pre_v; POST_OUT[r * S + lane] = post_v; }
    float4 v = (*(const device float4*)(mixes + BASE_OFF + llane * S) * comb_scale
                + *(const device float4*)(BASEV + BASE_OFF + llane * S)) * active;
    const float row_max = metal::max(metal::max(v.x, v.y), metal::max(v.z, v.w));
    const float4 e = metal::fast::exp(v - row_max) * active;
    float4 rr = e * (1.0f / (e.x + e.y + e.z + e.w + HC_EPS)) + HC_EPS * active;
    float4 col_inv = 1.0f / (float4(simd_sum(rr.x), simd_sum(rr.y), simd_sum(rr.z), simd_sum(rr.w)) + HC_EPS);
    rr *= col_inv;
    for (int iter = 1; iter < ITERS; ++iter) {
      rr *= (1.0f / (rr.x + rr.y + rr.z + rr.w + HC_EPS)) * active;
      col_inv = 1.0f / (float4(simd_sum(rr.x), simd_sum(rr.y), simd_sum(rr.z), simd_sum(rr.w)) + HC_EPS);
      rr *= col_inv;
    }
    if (lane < (uint)S) *(device float4*)(COMB_OUT + r * S * S + lane * S) = rr;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float p0 = pre_s[0], p1 = pre_s[1], p2 = pre_s[2], p3 = pre_s[3];
  float xc[4];
  float acc = 0.0f;
  for (int i = 0; i < 4; ++i) {
    const int d = int(t) * 4 + i;
    const float res = fma(p0, float(xs[d]), fma(p1, float(xs[D + d]),
                          fma(p2, float(xs[2 * D + d]), p3 * float(xs[3 * D + d]))));
    xc[i] = float(bfloat(res));
    acc = sq_acc<SQ_FMA>(acc, xc[i]);
  }
  acc = simd_sum(acc);
  if (sg == 0) red[lane] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) red[sg] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const float a = simd_sum(red[lane]);
    if (lane == 0) inv_s[0] = metal::precise::rsqrt(a / float(D) + EPS[0]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = 0; i < 4; ++i) {
    const int d = int(t) * 4 + i;
    NORMED[size_t(r) * D + d] = NORMW[d] * bfloat(xc[i] * inv_s[0]);
  }
"""

_kernels: dict[str, Any] = {}


def metal() -> bool:
    return mx.default_device() == mx.gpu and mx.metal.is_available()


def _kernel(name: str, source: str, inputs: list[str], outputs: list[str]) -> Any:
    kernel = _kernels.get(name)
    if kernel is None:
        digest = hashlib.sha256((_HEADER + source).encode()).hexdigest()[:10]
        kernel = mx.fast.metal_kernel(name=f"tf_glm5_fused_{name}_{digest}", input_names=inputs,
                                      output_names=outputs, source=source, header=_HEADER)
        _kernels[name] = kernel
    return kernel


def sources() -> dict[str, str]:
    return {"fused_header": _HEADER, "moe_route": _MOE_ROUTE, "moe_gateup": _MOE_GATEUP,
            "moe_down": _MOE_DOWN, "moe_combine": _MOE_COMBINE,
            "moe_combine_split": _MOE_COMBINE_SPLIT, "router": _ROUTER, "router_tg": _ROUTER_TG,
            "hc_expand": _HC_EXPAND, "hc_mix": _HC_MIX, "hc_mix_packed": _HC_MIX_PACKED,
            "hc_split_norm": _HC_SPLIT_NORM}


def hc_fits(hc: Any, dims: int) -> bool:
    cfg = hc.cfg
    return (cfg.hc_mult == 4 and dims == 4096 and int(hc.fn.shape[0]) == 24 and int(hc.fn.shape[1]) == 4 * dims)


def hc_step(x: mx.array, pending: tuple[mx.array, mx.array, mx.array] | None, hc: Any | None, norm_w: mx.array | None,
            eps: float) -> tuple[mx.array, mx.array | None, mx.array | None, mx.array | None]:
    """A block boundary on streams x [R, 4, D]: the pending write-back (branch [R, D], post [R, 4], comb [R, 4, 4];
    None at the first block), then (hc given) the next block's split and its input RMSNorm with norm_w. Returns
    (streams, normed input [R, D] or None, post, comb) with the row-by-row path's bits."""

    rows, streams, dims = x.shape
    expand, split = pending is not None, hc is not None
    if expand or split:
        branch, post, comb = pending if expand else (x[:, 0], mx.zeros((rows, 4), mx.float32),
                                                       mx.zeros((rows, 4, 4), mx.float32))
        eps_arr = mx.array([hc.cfg.rms_norm_eps if split else eps], dtype=mx.float32)
        k1 = _kernel("hc_expand", _HC_EXPAND, ["XOLD", "BRANCH", "POST", "COMB", "EPS"], ["XNEW", "INV"])
        xn, inv = k1(inputs=[x, branch, post, comb, eps_arr],
                     template=[("D", dims), ("EXPAND", int(expand)), ("SPLIT", int(split)), ("SQ_FMA", SQ_FMA)],
                     grid=(1024 * rows, 1, 1), threadgroup=(1024, 1, 1),
                     output_shapes=[x.shape if expand else (1,), (rows,)], output_dtypes=[mx.bfloat16, mx.float32])
        if expand:
            x = xn
    if not split:
        return x, None, None, None
    if hc.fn_packed is not None:
        k2 = _kernel("hc_mix_packed", _HC_MIX_PACKED, ["X", "INV", "FNP"], ["MIXES"])
        mixes = k2(inputs=[x, inv, hc.fn_packed], template=[("D", dims), ("U", HC_MIX_U)], grid=(6 * 256, rows, 1),
                   threadgroup=(256, 1, 1), output_shapes=[(rows, 24)], output_dtypes=[mx.float32])[0]
    else:
        k2 = _kernel("hc_mix", _HC_MIX, ["X", "INV", "FN"], ["MIXES"])
        mixes = k2(inputs=[x, inv, hc.fn], template=[("D", dims)], grid=(6 * 256, rows, 1), threadgroup=(256, 1, 1),
                   output_shapes=[(rows, 24)], output_dtypes=[mx.float32])[0]
    k3 = _kernel("hc_split_norm", _HC_SPLIT_NORM, ["X", "MIXES", "SCALE", "BASEV", "NORMW", "EPS"],
                 ["NORMED", "POST_OUT", "COMB_OUT"])
    normed, post_o, comb_o = k3(
        inputs=[x, mixes, hc.scale, hc.base, norm_w, eps_arr],
        template=[("D", dims), ("SQ_FMA", SQ_FMA), ("ITERS", hc.cfg.hc_sinkhorn_iters),
                  ("HC_EPS_INT", round(hc.cfg.hc_eps / 1e-9))],
        grid=(1024 * rows, 1, 1), threadgroup=(1024, 1, 1),
        output_shapes=[(rows, dims), (rows, 4), (rows, 4, 4)], output_dtypes=[mx.bfloat16, mx.float32, mx.float32])
    return x, normed, post_o, comb_o


def moe_fits(moe: Any) -> bool:
    """4-bit group-64 experts whose input dims are whole 512-value blocks, a shared expert, top-k <= 32."""

    qs = [moe.gate, moe.up, moe.down] + ([moe.shared.gate_up, moe.shared.down] if moe.shared is not None else [])
    return (moe.shared is not None and moe.shared.width == moe.gate.outs and moe.cfg.norm_topk_prob
            and moe.cfg.num_experts_per_tok <= 32 and moe.cfg.n_routed_experts <= 1024
            and all(q.bits == 4 and q.group == 64 and q.ins % 512 == 0 and q.outs % 4 == 0 for q in qs))


def moe_rows(moe: Any, x: mx.array, *, rps: int = 4) -> mx.array:
    """The MoE block (routed experts + shared expert) on a window's rows x [R, D]: the router's one-row matmul,
    route + group, gate/up + SwiGLU, down, combine. Every row gets the bits the row-by-row block gives it.

    SPLIT_SHARED: the shared expert's gate/up and down run as kernels of their own that read only x, so the GPU
    runs them beside the router and the route kernel (both latency-bound) instead of after them. Same arithmetic,
    same bits."""

    rows, dims = x.shape
    cfg = moe.cfg
    top, experts = cfg.num_experts_per_tok, cfg.n_routed_experts
    inter = moe.gate.outs
    sh = moe.shared
    maxu = rows * top
    gateup = _kernel("moe_gateup", _MOE_GATEUP,
                     ["X", "GW", "GS", "GB", "UW", "US", "UB", "SGU", "SGUS", "SGUB", "UIDS", "UMEM", "UCOUNT", "LIM"],
                     ["ACT"])
    down = _kernel("moe_down", _MOE_DOWN, ["ACT", "DW", "DS", "DB", "SDW", "SDS", "SDB", "UIDS", "UMEM", "UCOUNT"],
                   ["Y"])

    def gu(part: int, slots: int, zs: int, uids: mx.array, umem: mx.array, ucount: mx.array) -> mx.array:
        return gateup(inputs=[x, moe.gate.weight, moe.gate.scales, moe.gate.biases, moe.up.weight, moe.up.scales,
                              moe.up.biases, sh.gate_up.weight, sh.gate_up.scales, sh.gate_up.biases, uids, umem,
                              ucount, moe.limit_arr],
                      template=[("K", dims), ("N", inter), ("RPS", rps), ("TOPK", top), ("MAXR", MAX_ROWS),
                                ("MAXU", maxu), ("PART", part)],
                      grid=(32 * rows, inter // rps, zs), threadgroup=(32 * rows, 1, 1),
                      output_shapes=[(rows, slots, inter)], output_dtypes=[mx.bfloat16])[0]

    def dn(act: mx.array, part: int, slots: int, zs: int, uids: mx.array, umem: mx.array,
           ucount: mx.array) -> mx.array:
        return down(inputs=[act, moe.down.weight, moe.down.scales, moe.down.biases, sh.down.weight, sh.down.scales,
                            sh.down.biases, uids, umem, ucount],
                    template=[("K", inter), ("N", dims), ("RPS", rps), ("TOPK", top), ("MAXR", MAX_ROWS),
                              ("MAXU", maxu), ("PART", part)],
                    grid=(32 * rows, dims // rps, zs), threadgroup=(32 * rows, 1, 1),
                    output_shapes=[(rows, slots, dims)], output_dtypes=[mx.bfloat16])[0]

    split = SPLIT_SHARED
    if split:
        # the shared expert first, from x alone (its group inputs are placeholders it never reads)
        none = moe.__dict__.get("_no_group")
        if none is None:
            none = moe._no_group = mx.zeros((1,), dtype=mx.int32)
            mx.eval(none)
        ys = dn(gu(1, 1, 1, none, none, none), 1, 1, 1, none, none, none)
    logits = router_rows(x.astype(mx.float32), moe)                                   # [R, E] fp32
    threads = max(32 * MAX_ROWS, -(-experts // 32) * 32)
    route = _kernel("moe_route", _MOE_ROUTE, ["LOGITS", "BIAS", "SCALE"], ["PICK", "WTS", "UIDS", "UMEM", "UCOUNT"])
    pick, wts, uids, umem, ucount = route(
        inputs=[logits, moe.bias, moe.scale_arr],
        template=[("NE", experts), ("TOPK", top), ("MAXR", MAX_ROWS), ("NT", threads)],
        grid=(threads, 1, 1), threadgroup=(threads, 1, 1),
        output_shapes=[(rows, top), (rows, top), (rows * top,), (rows * top, MAX_ROWS), (1,)],
        output_dtypes=[mx.int32, mx.float32, mx.int32, mx.int32, mx.int32])
    if split:
        y = dn(gu(2, top, maxu, uids, umem, ucount), 2, top, maxu, uids, umem, ucount)
        combine = _kernel("moe_combine_split", _MOE_COMBINE_SPLIT, ["YS", "Y", "WTS"], ["OUT"])
        return combine(inputs=[ys.reshape(rows, dims), y, wts], template=[("D", dims), ("TOPK", top)],
                       grid=(rows * dims, 1, 1), threadgroup=(256, 1, 1),
                       output_shapes=[(rows, dims)], output_dtypes=[mx.bfloat16])[0]
    slots = top + 1
    y = dn(gu(0, slots, maxu + 1, uids, umem, ucount), 0, slots, maxu + 1, uids, umem, ucount)
    combine = _kernel("moe_combine", _MOE_COMBINE, ["Y", "WTS"], ["OUT"])
    return combine(inputs=[y, wts], template=[("D", dims), ("TOPK", top)],
                   grid=(rows * dims, 1, 1), threadgroup=(256, 1, 1),
                   output_shapes=[(rows, dims)], output_dtypes=[mx.bfloat16])[0]


def pack_router(router_bf16: mx.array) -> mx.array:
    """The router [E, K] (bf16 as stored) repacked for _ROUTER: [E / 4, 8, K / 32, 4 (tm), 4 (tn)]."""

    e, k = router_bf16.shape
    m = router_bf16.T.reshape(k // 32, 8, 4, e // 4, 4)            # [i, thrM, tm, q, tn]  (row = 32 i + 4 thrM + tm)
    return mx.contiguous(m.transpose(3, 1, 0, 2, 4))              # [q, thrM, i, tm, tn]


def router_fits(moe: Any) -> bool:
    e, k = moe.router.shape[1], moe.router.shape[0]
    return (moe.router_packed is not None and e % 16 == 0 and k % 32 == 0 and K.gemv_params(True, k, e) ==
            (1, 2, 8, 4, 4, 4))


def router_rows(x: mx.array, moe: Any) -> mx.array:
    """Router logits x [R, D] fp32 @ W^T with MLX's one-row matmul bits (its gemv_t tiling for 288 experts),
    reading the stored bf16 weights once for the window."""

    if not router_fits(moe):
        return K.matmul_rows(x, moe.router, transposed=True)
    rows, dims = x.shape
    experts = int(moe.router.shape[1])
    if ROUTER_TG:
        kernel = _kernel("router_tg", _ROUTER_TG, ["X", "RP"], ["OUT"])
        return kernel(inputs=[x, moe.router_packed],
                      template=[("K", dims), ("NE", experts), ("RR", rows), ("C", 8 if rows <= 4 else 4),
                                ("NT", ROUTER_TG)],
                      grid=(ROUTER_TG * experts // 16, 1, 1), threadgroup=(ROUTER_TG, 1, 1),
                      output_shapes=[(rows, experts)], output_dtypes=[mx.float32])[0]
    kernel = _kernel("router", _ROUTER, ["X", "RP"], ["OUT"])
    return kernel(inputs=[x, moe.router_packed], template=[("K", dims), ("NE", experts), ("RR", rows), ("U", 8)],
                  grid=(32 * experts // 16, 1, 1), threadgroup=(32, 1, 1),
                  output_shapes=[(rows, experts)], output_dtypes=[mx.float32])[0]


def pack_hc_fn(fn_bf16: mx.array) -> mx.array:
    """The mix matrix [24, 16384] (bf16 as stored) repacked for _HC_MIX_PACKED: [og 6][sgn 8][lane 32][i 16][tm 4]
    [tn 4], row og 4 + tm, column (32 sgn + lane) 4 + 1024 i + tn."""

    m = fn_bf16.reshape(6, 4, 16, 8, 32, 4)                     # [og, tm, i, sgn, lane, tn]
    return mx.contiguous(m.transpose(0, 3, 4, 2, 1, 5))
