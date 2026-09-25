"""Decode attention with one arithmetic for 1 to 32 queries: the lane attention.

The verify window's rows must attend exactly as serial decoding does. MLX's
vector kernel gives each query and head its own threadgroup, re-reads every key
per query and switches variant with the key count, so an 8-row window at 20k
keys cost ~1.4 ms a layer. Here the arithmetic is fixed by absolute key position
and never by how many queries share a call:

  keys are cut into chunks of 512 positions and each chunk into tiles of 64;
  for a 16-row tile of (query, head) rows the tensor units compute the tile's
  scores S = Q K^T (bf16 in, fp32 out), keys at or past a row's own limit n_t
  are masked, the row's running max, sum and output are updated (online
  softmax, fp32; P V on the tensor units, half P x bf16 V into fp32, in two
  128-column halves); chunks are merged in chunk order. A tile or chunk with no
  valid key for a row adds exactly nothing to it.

A row's result depends only on its own query and keys, so a query gives the
same bits alone (serial decoding) or as row t of a window, and each key tile is
read once for all rows of a 16-row tile. Serial decoding uses this kernel too:
in lane mode it is what "serial" means.

A draft tree's committed keys in whole tiles go through the shared kernel; each node's tail
kernel picks up the state it left in the last of those chunks and runs the remaining tiles
(committed, then the node's path), so a node's own work stays within ~2 tiles whatever the
chunk size.

Measured on the M5 Max (2026-09-23), per layer, 24 query heads over 4 key heads, head dim
256, 16 dependent layers (64-key tiles, half P, 512-key chunks), tree attention:
    20k keys:  1 query 0.29 ms, 16 nodes 0.49, 32-row chain 0.87
    40k keys:  1 query 0.46 ms, 16 nodes 0.82, 32-row chain 1.48
    80k keys:  1 query 0.84 ms, 16 nodes 1.57, 32-row chain 2.74
(256-key chunks and 32-key tiles: 16 nodes 0.59 / 1.08 / 2.02 ms.)
"""

from __future__ import annotations

import os
from typing import Any, Sequence

import mlx.core as mx

MAX_QUERIES = 128
TILES_PER_GROUP = int(os.environ.get("TF_ATTN_TILES", "16"))    # 16-row tiles per threadgroup; more tiles spread over grid.z. Each threadgroup
                        # reads its key chunk once for its tiles: a 128-row prompt chain (48 tiles) at 21k
                        # keys 55.4 -> 47.3 ms a forward with 16 (40k: 88.9 -> 86.5), bits unchanged; 12
                        # was slower than 8, and 24 overflows the P staging (2026-09-23). Decode trees
                        # (16 rows, 6 tiles) are unaffected.
CHUNK = 512            # keys per chunk (fixed: part of the arithmetic)
TILE = 64              # keys per tile (fixed: part of the arithmetic)

_HEADER = "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace mpp::tensor_ops;\n"

_PARTIAL = r"""
  const ushort lane = thread_index_in_simdgroup;
  const ushort sg = simdgroup_index_in_threadgroup;
  const int tile = int(threadgroup_position_in_grid.z) * SG + sg;   // 16-row tile (SGA in all, SG per threadgroup)
  const uint hk = threadgroup_position_in_grid.x;              // key head
  const uint c = threadgroup_position_in_grid.y;               // chunk of CK keys
  const int L = dims[0], NCH = dims[1], NQ = dims[2], SGA = dims[4];   // runtime: one variant per SG
  const int RP = 16 * SGA;
  const short qid = lane >> 2;
  const short fm = (qid & 4) | ((lane >> 1) & 3);
  const short fn = ((qid & 2) | (lane & 1)) * 4;
  const int r0 = tile * 16 + fm, r1 = r0 + 8;
  const bool causal = dims[3] != 0;
  const int n0 = (r0 < G * NQ) ? (causal ? L - NQ + r0 / G + 1 : L) : 0;
  const int n1 = (r1 < G * NQ) ? (causal ? L - NQ + r1 / G + 1 : L) : 0;
  // P in half (it lies in [0, 1]): the P x V op then runs ~2x the fp32 rate. With 16-bit operands the op's
  // destination interleaves rows fm and fm + 8 every 4 elements (fp32 P: elements 0-31 are row fm)
  threadgroup half Ps[SG * 16 * TK];
  threadgroup half* myP = Ps + sg * 16 * TK;
  if (tile >= SGA) return;                                      // the last threadgroup's spare simdgroups
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tQ((device bfloat*)Qp + (int64_t)hk * RP * D, dextents<int32_t, 2>(D, RP));
  // rows at their real stride (a cache buffer's rows are D apart; other layouts need not be)
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tK((device bfloat*)K + (int64_t)hk * K_strides[1], dextents<int32_t, 2>(D, L), array<int32_t, 2>({1, int32_t(K_strides[2])}));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tV((device bfloat*)V + (int64_t)hk * V_strides[1], dextents<int32_t, 2>(D, L), array<int32_t, 2>({1, int32_t(V_strides[2])}));
  tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tP(myP, dextents<int32_t, 2>(TK, 16));
  constexpr auto dS = matmul2d_descriptor(16, TK, D, false, true, false, matmul2d_descriptor::mode::multiply);
  constexpr auto dO = matmul2d_descriptor(16, 128, TK, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<dS, execution_simdgroup> opS;
  matmul2d<dO, execution_simdgroup> opO;
  auto aQ = tQ.slice(0, tile * 16);
  auto bV0 = tV.slice(0, 0);
  // the op is exact up to 128 output columns: the head dimension goes in two halves
  auto Olo = opO.template get_destination_cooperative_tensor<decltype(tP), decltype(bV0), float>();
  auto Ohi = opO.template get_destination_cooperative_tensor<decltype(tP), decltype(bV0), float>();
  for (int i = 0; i < 64; i++) { Olo[i] = 0.0f; Ohi[i] = 0.0f; }
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
  const int kbeg = int(c) * CK;
  const int kend = min(kbeg + CK, L);
  for (int kt = kbeg; kt < kend; kt += TK) {
    auto bK = tK.slice(0, kt);
    auto S = opS.template get_destination_cooperative_tensor<decltype(aQ), decltype(bK), float>();
    opS.run(aQ, bK, S);
    float s[TK / 2];
    for (int i = 0; i < TK / 2; i++) {                         // 8 elements per 16-key block: 4 of row fm, then 4 of fm + 8
      const int key = kt + (i >> 3) * 16 + fn + (i & 3);
      s[i] = key < ((i & 4) ? n1 : n0) ? S[i] * scale[0] : -INFINITY;
    }
    float x0 = -INFINITY, x1 = -INFINITY;
    for (int i = 0; i < TK / 2; i++) { if (i & 4) x1 = max(x1, s[i]); else x0 = max(x0, s[i]); }
    x0 = max(x0, simd_shuffle_xor(x0, 1)); x0 = max(x0, simd_shuffle_xor(x0, 8));
    x1 = max(x1, simd_shuffle_xor(x1, 1)); x1 = max(x1, simd_shuffle_xor(x1, 8));
    const float nm0 = max(m0, x0), nm1 = max(m1, x1);
    const float f0 = (x0 == -INFINITY) ? 1.0f : fast::exp(m0 - nm0);
    const float f1 = (x1 == -INFINITY) ? 1.0f : fast::exp(m1 - nm1);
    float p[TK / 2];
    for (int i = 0; i < TK / 2; i++) p[i] = (s[i] == -INFINITY) ? 0.0f : fast::exp(s[i] - ((i & 4) ? nm1 : nm0));
    float y0 = 0.0f, y1 = 0.0f;
    for (int b = 0; b < TK / 16; b++) {
      y0 += (p[b * 8] + p[b * 8 + 1]) + (p[b * 8 + 2] + p[b * 8 + 3]);
      y1 += (p[b * 8 + 4] + p[b * 8 + 5]) + (p[b * 8 + 6] + p[b * 8 + 7]);
    }
    y0 += simd_shuffle_xor(y0, 1); y0 += simd_shuffle_xor(y0, 8);
    y1 += simd_shuffle_xor(y1, 1); y1 += simd_shuffle_xor(y1, 8);
    if (x0 != -INFINITY) { l0 = l0 * f0 + y0; m0 = nm0; }
    if (x1 != -INFINITY) { l1 = l1 * f1 + y1; m1 = nm1; }
    for (int f = 0; f < TK / 16; f++)
      for (int i = 0; i < 4; i++) {
        myP[fm * TK + f * 16 + fn + i] = half(p[f * 8 + i]);
        myP[(fm + 8) * TK + f * 16 + fn + i] = half(p[f * 8 + 4 + i]);
      }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < 64; i++) { const float f = (i & 4) ? f1 : f0; Olo[i] *= f; Ohi[i] *= f; }
    auto bVlo = tV.slice(0, kt);
    auto bVhi = tV.slice(128, kt);
    opO.run(tP, bVlo, Olo);
    opO.run(tP, bVhi, Ohi);
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
  const int64_t base = ((int64_t)hk * NCH + c) * RP;
  // 16-bit operand layout: elements 4q..4q+3 are four consecutive columns of one row
  for (int q = 0; q < 16; q++) {
    device float* dst = PO + (base + tile * 16 + fm + (q & 1) * 8) * D + (q >> 1) * 16 + fn;
    *(device float4*)dst = float4(Olo[4 * q], Olo[4 * q + 1], Olo[4 * q + 2], Olo[4 * q + 3]);
    *(device float4*)(dst + 128) = float4(Ohi[4 * q], Ohi[4 * q + 1], Ohi[4 * q + 2], Ohi[4 * q + 3]);
  }
  if ((lane & 9) == 0) {
    PM[base + r0] = m0; PL[base + r0] = l0;
    PM[base + r1] = m1; PL[base + r1] = l1;
  }
"""

# P handed to the P V op in registers (``get_left_input_cooperative_tensor``) instead of through threadgroup
# memory and two simdgroup barriers. Same bits as ``_PARTIAL`` for 1-128 query rows (tested), which stays the
# reference: the lane decoder's version hash (and so every stored snapshot) covers it. Live, an agent client's 70,510-token
# turn replayed with blocks of rounds alternating: 78.5 -> 77.0 ms a round, same output sha (2026-09-24; an
# isolated benchmark had promised 26 -> 15 ms of attention a forward).
_P_STORE = """    for (int f = 0; f < TK / 16; f++)
      for (int i = 0; i < 4; i++) {
        myP[fm * TK + f * 16 + fn + i] = half(p[f * 8 + i]);
        myP[(fm + 8) * TK + f * 16 + fn + i] = half(p[f * 8 + 4 + i]);
      }
    simdgroup_barrier(mem_flags::mem_threadgroup);
"""
_P_REGISTERS = """    auto Pc = opS.template get_destination_cooperative_tensor<decltype(aQ), decltype(bK), half>();
    for (int i = 0; i < TK / 2; i++) Pc[i] = half(p[i]);
    auto Pin = opO.template get_left_input_cooperative_tensor<half, bfloat, float>(Pc);
"""
_PV_THREADGROUP = """    opO.run(tP, bVlo, Olo);
    opO.run(tP, bVhi, Ohi);
    simdgroup_barrier(mem_flags::mem_threadgroup);"""
_PV_REGISTERS = """    opO.run(Pin, bVlo, Olo);
    opO.run(Pin, bVhi, Ohi);"""
assert _P_STORE in _PARTIAL and _PV_THREADGROUP in _PARTIAL
_PARTIAL_DIRECT = _PARTIAL.replace(_P_STORE, _P_REGISTERS).replace(_PV_THREADGROUP, _PV_REGISTERS)
# TF_ATTN_DIRECT_P=0: the threadgroup-memory P of before (same bits)
DIRECT_P = os.environ.get("TF_ATTN_DIRECT_P", "1") != "0"

def _single_half(source: str) -> str:
    """Head dimension 128 (Nemotron-H): the output in one 128-column half; the 256 kernels run two."""

    out = source
    for old, new in (
        ("  auto Ohi = opO.template get_destination_cooperative_tensor<decltype(tP), decltype(bV0), float>();\n", ""),
        ("{ Olo[i] = 0.0f; Ohi[i] = 0.0f; }", "{ Olo[i] = 0.0f; }"),
        ("Olo[i] *= f; Ohi[i] *= f; }", "Olo[i] *= f; }"),
        ("    auto bVhi = tV.slice(128, kt);\n", ""),
        ("    opO.run(Pin, bVhi, Ohi);", ""),
        ("    opO.run(tP, bVhi, Ohi);\n", ""),
        ("    *(device float4*)(dst + 128) = float4(Ohi[4 * q], Ohi[4 * q + 1], Ohi[4 * q + 2], Ohi[4 * q + 3]);\n", ""),
    ):
        out = out.replace(old, new)
    if "Ohi" in out or "bVhi" in out:
        raise AssertionError("lane_attention: the 128 variant still refers to the second half")
    return out


_PARTIAL_128 = _single_half(_PARTIAL)
_PARTIAL_DIRECT_128 = _single_half(_PARTIAL_DIRECT)

# Tried and removed 2026-09-24: each 16-row tile's output split over two simdgroups (both compute the scores and
# softmax, each runs P V for half of D; same bits, tested through lane_sdpa and lane_tree_sdpa). Microbenchmark
# 16 rows 0.374 -> 0.361 ms a layer at 21k keys; live A/B in one server (TF_KERNEL_AB=8, agent turns at 21-28k,
# 1,775 tree rounds, same output shas): 62.2 -> 63.0 ms a forward, +1.05 ms a round over 114 paired blocks.

_MERGE = r"""
  const uint lane = thread_index_in_simdgroup;
  const uint hk = threadgroup_position_in_grid.x;
  const uint r = threadgroup_position_in_grid.y;              // row t * G + g
  const int NCH = dims[1], NQ = dims[2], SGA = dims[4];
  const int RP = 16 * SGA;
  constexpr int DP = D / 32;
  float m = -INFINITY, l = 0.0f, o[DP];
  for (int i = 0; i < DP; i++) o[i] = 0.0f;
  for (int c = 0; c < NCH; c++) {
    const int64_t row = ((int64_t)hk * NCH + c) * RP + r;
    const float mc = PM[row];
    if (mc == -INFINITY) continue;
    const float lc = PL[row];
    const float nm = max(m, mc);
    const float f1 = fast::exp(m - nm), f2 = fast::exp(mc - nm);
    l = l * f1 + lc * f2;
    for (int i = 0; i < DP; i++) o[i] = o[i] * f1 + PO[row * D + lane * DP + i] * f2;
    m = nm;
  }
  const int t = r / G, g = r % G, h = hk * G + g;
  for (int i = 0; i < DP; i++) OUT[((int64_t)h * NQ + t) * D + lane * DP + i] = static_cast<bfloat>(o[i] / l);
"""

_TAIL = r"""
  const ushort lane = thread_index_in_simdgroup;
  const uint hk = threadgroup_position_in_grid.x;              // key head
  const uint cb = threadgroup_position_in_grid.y;              // tail chunk (from the first chunk holding the window)
  const uint node = threadgroup_position_in_grid.z;            // tree node
  const int P = dims[1], PT = dims[2], NCB = dims[3], W = dims[4], RPA = dims[5], CA = dims[6];
  const int depth = depths[node];
  const int nmax = P + depth + 1;                              // logical keys 0 .. P + depth
  const short qid = lane >> 2;
  const short fm = (qid & 4) | ((lane >> 1) & 3);
  const short fn = ((qid & 2) | (lane & 1)) * 4;
  const int r0 = fm, r1 = fm + 8;                              // query rows: the node's heads (G of 16)
  const int n0 = r0 < G ? nmax : 0;
  const int n1 = r1 < G ? nmax : 0;
  threadgroup half myP[16 * TK];
  threadgroup bfloat KV[32 * D];                               // 32 keys at a time, or TK keys' half rows of values
  const device bfloat* kbase = (const device bfloat*)K + (int64_t)hk * K_strides[1];
  const device bfloat* vbase = (const device bfloat*)V + (int64_t)hk * V_strides[1];
  const int64_t kstep = K_strides[2], vstep = V_strides[2];
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tQ((device bfloat*)QB + ((int64_t)hk * W + node) * 16 * D, dextents<int32_t, 2>(D, 16));
  tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline> tK32(KV, dextents<int32_t, 2>(D, 32));
  tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline> tVh(KV, dextents<int32_t, 2>(128, TK));
  tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tP(myP, dextents<int32_t, 2>(TK, 16));
  // scores 32 keys at a time: each score equals the TK-key op's bit for bit (tested), and 32 keys
  // of K fit the 16 KB buffer that TK keys would overflow
  constexpr auto dS = matmul2d_descriptor(16, 32, D, false, true, false, matmul2d_descriptor::mode::multiply);
  constexpr auto dO = matmul2d_descriptor(16, 128, TK, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<dS, execution_simdgroup> opS;
  matmul2d<dO, execution_simdgroup> opO;
  auto Olo = opO.template get_destination_cooperative_tensor<decltype(tP), decltype(tVh), float>();
  auto Ohi = opO.template get_destination_cooperative_tensor<decltype(tP), decltype(tVh), float>();
  for (int i = 0; i < 64; i++) { Olo[i] = 0.0f; Ohi[i] = 0.0f; }
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
  // keys [0, PT) went through the shared kernel; the first chunk holding PT continues from the
  // state it left there (same tiles, same order, same arithmetic: the bits of one pass)
  const int c0 = PT / CK;
  const int c = c0 + int(cb);
  const int kbeg = max(c * CK, PT);
  const int kend = min((c + 1) * CK, nmax);
  if (cb == 0 && PT > c0 * CK) {
    const int64_t baseA = ((int64_t)hk * CA + c0) * RPA + node * G;
    for (int q = 0; q < 16; q++) {
      const int row = fm + (q & 1) * 8;
      if (row >= G) continue;
      const auto src = POA + (baseA + row) * D + (q >> 1) * 16 + fn;   // a placeholder is tiny: constant space
      for (int j = 0; j < 4; j++) { Olo[4 * q + j] = src[j]; Ohi[4 * q + j] = src[128 + j]; }
    }
    if (r0 < G) { m0 = PMA[baseA + r0]; l0 = PLA[baseA + r0]; }
    if (r1 < G) { m1 = PMA[baseA + r1]; l1 = PLA[baseA + r1]; }
  }
  for (int kt = kbeg; kt < kend; kt += TK) {
    float sraw[TK / 2];
    for (int h = 0; h < TK / 32; h++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint e = lane; e < 32 * D / 8; e += 32) {           // logical slot -> physical row
        const int row = int(e) / (D / 8), col = (int(e) % (D / 8)) * 8;
        const int q = kt + h * 32 + row;
        int phys = -1;
        if (q < P) phys = q;
        else if (q < nmax) phys = P + paths[node * MAXD + (q - P)];
        ((threadgroup vec<bfloat, 8>*)KV)[e] = phys >= 0 ? *(const device vec<bfloat, 8>*)(kbase + phys * kstep + col) : vec<bfloat, 8>(0);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      auto S = opS.template get_destination_cooperative_tensor<decltype(tQ), decltype(tK32), float>();
      opS.run(tQ, tK32, S);
      for (int i = 0; i < 16; i++) sraw[h * 16 + i] = S[i];
    }
    float s[TK / 2];
    for (int i = 0; i < TK / 2; i++) {                         // 8 elements per 16-key block: 4 of row fm, then 4 of fm + 8
      const int key = kt + (i >> 3) * 16 + fn + (i & 3);
      s[i] = key < ((i & 4) ? n1 : n0) ? sraw[i] * scale[0] : -INFINITY;
    }
    float x0 = -INFINITY, x1 = -INFINITY;
    for (int i = 0; i < TK / 2; i++) { if (i & 4) x1 = max(x1, s[i]); else x0 = max(x0, s[i]); }
    x0 = max(x0, simd_shuffle_xor(x0, 1)); x0 = max(x0, simd_shuffle_xor(x0, 8));
    x1 = max(x1, simd_shuffle_xor(x1, 1)); x1 = max(x1, simd_shuffle_xor(x1, 8));
    const float nm0 = max(m0, x0), nm1 = max(m1, x1);
    const float f0 = (x0 == -INFINITY) ? 1.0f : fast::exp(m0 - nm0);
    const float f1 = (x1 == -INFINITY) ? 1.0f : fast::exp(m1 - nm1);
    float p[TK / 2];
    for (int i = 0; i < TK / 2; i++) p[i] = (s[i] == -INFINITY) ? 0.0f : fast::exp(s[i] - ((i & 4) ? nm1 : nm0));
    float y0 = 0.0f, y1 = 0.0f;
    for (int b = 0; b < TK / 16; b++) {
      y0 += (p[b * 8] + p[b * 8 + 1]) + (p[b * 8 + 2] + p[b * 8 + 3]);
      y1 += (p[b * 8 + 4] + p[b * 8 + 5]) + (p[b * 8 + 6] + p[b * 8 + 7]);
    }
    y0 += simd_shuffle_xor(y0, 1); y0 += simd_shuffle_xor(y0, 8);
    y1 += simd_shuffle_xor(y1, 1); y1 += simd_shuffle_xor(y1, 8);
    if (x0 != -INFINITY) { l0 = l0 * f0 + y0; m0 = nm0; }
    if (x1 != -INFINITY) { l1 = l1 * f1 + y1; m1 = nm1; }
    for (int f = 0; f < TK / 16; f++)
      for (int i = 0; i < 4; i++) {
        myP[fm * TK + f * 16 + fn + i] = half(p[f * 8 + i]);
        myP[(fm + 8) * TK + f * 16 + fn + i] = half(p[f * 8 + 4 + i]);
      }
    for (int i = 0; i < 64; i++) { const float f = (i & 4) ? f1 : f0; Olo[i] *= f; Ohi[i] *= f; }
    for (int hv = 0; hv < 2; hv++) {                           // values: TK keys x 128 columns at a time
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint e = lane; e < TK * 128 / 8; e += 32) {
        const int row = int(e) / 16, col = hv * 128 + (int(e) % 16) * 8;
        const int q = kt + row;
        int phys = -1;
        if (q < P) phys = q;
        else if (q < nmax) phys = P + paths[node * MAXD + (q - P)];
        ((threadgroup vec<bfloat, 8>*)KV)[e] = phys >= 0 ? *(const device vec<bfloat, 8>*)(vbase + phys * vstep + col) : vec<bfloat, 8>(0);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (hv == 0) opO.run(tP, tVh, Olo);
      else opO.run(tP, tVh, Ohi);
    }
  }
  const int64_t base = (((int64_t)hk * NCB + cb) * W + node) * 16;
  for (int q = 0; q < 16; q++) {
    device float* dst = PO + (base + fm + (q & 1) * 8) * D + (q >> 1) * 16 + fn;
    *(device float4*)dst = float4(Olo[4 * q], Olo[4 * q + 1], Olo[4 * q + 2], Olo[4 * q + 3]);
    *(device float4*)(dst + 128) = float4(Ohi[4 * q], Ohi[4 * q + 1], Ohi[4 * q + 2], Ohi[4 * q + 3]);
  }
  if ((lane & 9) == 0) {
    PM[base + r0] = m0; PL[base + r0] = l0;
    PM[base + r1] = m1; PL[base + r1] = l1;
  }
"""

_TREE_MERGE = r"""
  const uint lane = thread_index_in_simdgroup;
  const uint hk = threadgroup_position_in_grid.x;
  const uint r = threadgroup_position_in_grid.y;               // node * G + g
  const int PT = dims[2], NCB = dims[3], W = dims[4], RPA = dims[5], CA = dims[6];
  const int CT = PT / CK;                                      // chunks the shared kernel finished
  constexpr int DP = D / 32;
  const int node = r / G, g = r % G;
  float m = -INFINITY, l = 0.0f, o[DP];
  for (int i = 0; i < DP; i++) o[i] = 0.0f;
  for (int c = 0; c < CT; c++) {                               // committed chunks, in order
    const int64_t row = ((int64_t)hk * CA + c) * RPA + r;
    const float mc = PMA[row];
    if (mc == -INFINITY) continue;
    const float lc = PLA[row];
    const float nm = max(m, mc);
    const float f1 = fast::exp(m - nm), f2 = fast::exp(mc - nm);
    l = l * f1 + lc * f2;
    for (int i = 0; i < DP; i++) o[i] = o[i] * f1 + POA[row * D + lane * DP + i] * f2;
    m = nm;
  }
  for (int c = 0; c < NCB; c++) {                              // then the window's chunks
    const int64_t row = (((int64_t)hk * NCB + c) * W + node) * 16 + g;
    const float mc = PMB[row];
    if (mc == -INFINITY) continue;
    const float lc = PLB[row];
    const float nm = max(m, mc);
    const float f1 = fast::exp(m - nm), f2 = fast::exp(mc - nm);
    l = l * f1 + lc * f2;
    for (int i = 0; i < DP; i++) o[i] = o[i] * f1 + POB[row * D + lane * DP + i] * f2;
    m = nm;
  }
  const int h = hk * G + g;
  for (int i = 0; i < DP; i++) OUT[((int64_t)h * W + node) * D + lane * DP + i] = static_cast<bfloat>(o[i] / l);
"""

# Tried and removed 2026-09-25: this merge with the layer's output gate and o_proj's group sums folded in
# (three kernels and a layout copy fewer a layer; bit-identical, sigmoid checked over all 65,536 bf16 values).
# Live A/B in one server, 125 paired blocks: forward -0.07 ms (SE 0.12), no gain: the gate's sigmoid already
# ran beside the attention kernels.
_kernels: dict[str, Any] = {}


def _named(base: str, source: str) -> str:
    """Kernel names carry a hash of their source: MLX caches compiled kernels by name."""

    import hashlib

    return f"{base}_{hashlib.sha256((_HEADER + source).encode()).hexdigest()[:16]}"


def _kernel(name: str) -> Any:
    if name not in _kernels:
        if name == "tail":
            _kernels[name] = mx.fast.metal_kernel(
                name=_named("lane_attention_tail", _TAIL), input_names=["QB", "K", "V", "scale", "dims", "paths", "depths",
                                                                        "POA", "PMA", "PLA"],
                output_names=["PO", "PM", "PL"], source=_TAIL, header=_HEADER, ensure_row_contiguous=False)
        elif name == "tree_merge":
            _kernels[name] = mx.fast.metal_kernel(
                name=_named("lane_attention_tree_merge", _TREE_MERGE),
                input_names=["POA", "PMA", "PLA", "POB", "PMB", "PLB", "dims"], output_names=["OUT"],
                source=_TREE_MERGE, header=_HEADER)
        elif name in ("partial", "partial_direct", "partial_128", "partial_direct_128"):
            source = {"partial": _PARTIAL, "partial_direct": _PARTIAL_DIRECT, "partial_128": _PARTIAL_128,
                      "partial_direct_128": _PARTIAL_DIRECT_128}[name]
            _kernels[name] = mx.fast.metal_kernel(
                name=_named("lane_attention_" + name, source), input_names=["Qp", "K", "V", "scale", "dims"],
                output_names=["PO", "PM", "PL"], source=source, header=_HEADER,
                ensure_row_contiguous=False)
        else:
            _kernels[name] = mx.fast.metal_kernel(
                name=_named("lane_attention_merge", _MERGE), input_names=["PO", "PM", "PL", "dims"], output_names=["OUT"],
                source=_MERGE, header=_HEADER)
    return _kernels[name]


def _partial(qp: mx.array, keys: mx.array, values: mx.array, sc: mx.array, dims: mx.array, *, G: int, D: int, SG: int,
             SGA: int, HKV: int, nch: int, RP: int) -> Any:
    return _kernel(("partial_direct" if DIRECT_P else "partial") + ("_128" if D == 128 else ""))(
        inputs=[qp, keys, values, sc, dims], template=[("G", G), ("D", D), ("SG", SG), ("CK", CHUNK), ("TK", TILE)],
        grid=(HKV * 32 * SG, nch, -(-SGA // SG)), threadgroup=(32 * SG, 1, 1),
        output_shapes=[(HKV * nch * RP * D,), (HKV * nch * RP,), (HKV * nch * RP,)],
        output_dtypes=[mx.float32, mx.float32, mx.float32])


def lane_sdpa(queries: mx.array, keys: mx.array, values: mx.array, scale: float) -> mx.array:
    """Causal attention of the last T positions: queries [1, H, T, D], keys/values [1, Hkv, L, D].

    ``keys``/``values`` may be views of a longer cache buffer (rows of D contiguous values).
    """

    _, H, T, D = (int(s) for s in queries.shape)
    HKV, L = int(keys.shape[1]), int(keys.shape[2])
    if D not in (128, 256) or H % HKV or T > MAX_QUERIES or T > L:
        raise ValueError(f"lane_sdpa: unsupported shape q={queries.shape} k={keys.shape}")
    if queries.dtype != mx.bfloat16 or keys.dtype != mx.bfloat16 or values.dtype != mx.bfloat16:
        raise ValueError("lane_sdpa: bf16 only")
    G = H // HKV
    R = G * T
    RP = 16 * ((R + 15) // 16)
    SGA = RP // 16
    SG = min(SGA, TILES_PER_GROUP)
    qp = queries.reshape(HKV, G, T, D).transpose(0, 2, 1, 3).reshape(HKV, R, D)
    if RP != R:
        qp = mx.concatenate([qp, mx.zeros((HKV, RP - R, D), dtype=queries.dtype)], axis=1)
    qp = mx.contiguous(qp)
    nch = -(-L // CHUNK)
    # runtime values: a template per key count compiled a kernel per token
    dims = mx.array([L, nch, T, 1, SGA], dtype=mx.int32)
    po, pm, pl = _partial(qp, keys, values, mx.array([float(scale)], dtype=mx.float32), dims, G=G, D=D, SG=SG, SGA=SGA,
                          HKV=HKV, nch=nch, RP=RP)
    return _kernel("merge")(
        inputs=[po, pm, pl, dims], template=[("G", G), ("D", D)],
        grid=(HKV * 32, R, 1), threadgroup=(32, 1, 1),
        output_shapes=[(1, H, T, D)], output_dtypes=[mx.bfloat16])[0]


def lane_tree_sdpa(queries: mx.array, keys: mx.array, values: mx.array, scale: float,
                   parents: Sequence[int]) -> mx.array:
    """Attention for a draft tree whose W nodes are the cache's last W key rows.

    Node v (query row v) attends to the committed keys [0, P) and its own path, with
    every key at its logical position P + depth: the bits of serial decoding along
    that path. Chunks wholly inside [0, P) run through the shared kernel; the chunks
    holding the window run per node with the node's keys gathered into their slots.
    """

    from tensorfold.kernels.qwen.dense.v1.lane_tree import MAX_DEPTH, tree_paths

    _, H, W, D = (int(s) for s in queries.shape)
    HKV, L = int(keys.shape[1]), int(keys.shape[2])
    if D != 256 or H % HKV or W > MAX_QUERIES or len(parents) != W:
        raise ValueError(f"lane_tree_sdpa: unsupported shape q={queries.shape} k={keys.shape}")
    G = H // HKV
    P = L - W
    depths, paths = tree_paths(parents)
    PT = (P // TILE) * TILE                           # committed keys in whole tiles: the shared kernel's
    CA = -(-PT // CHUNK)                              # its chunks (the last may continue in the tail kernel)
    last = P + max(depths)                            # deepest logical key
    NCB = last // CHUNK - PT // CHUNK + 1             # chunks the tail kernel works in
    sc = mx.array([float(scale)], dtype=mx.float32)
    # shared part (all rows see every key of the committed chunks)
    R = G * W
    RP = 16 * ((R + 15) // 16)
    SGA = RP // 16
    SG = min(SGA, TILES_PER_GROUP)
    qA = queries.reshape(HKV, G, W, D).transpose(0, 2, 1, 3).reshape(HKV, R, D)
    if RP != R:
        qA = mx.concatenate([qA, mx.zeros((HKV, RP - R, D), dtype=queries.dtype)], axis=1)
    qA = mx.contiguous(qA)
    if CA > 0:
        dimsA = mx.array([PT, CA, W, 0, SGA], dtype=mx.int32)
        poA, pmA, plA = _partial(qA, keys, values, sc, dimsA, G=G, D=D, SG=SG, SGA=SGA, HKV=HKV, nch=CA, RP=RP)
    else:
        poA = pmA = plA = mx.zeros((1,), dtype=mx.float32)
    # per-node part: [HKV, W, 16 rows (G valid), D] queries
    qB = queries.reshape(HKV, G, W, D).transpose(0, 2, 1, 3)
    qB = mx.contiguous(mx.concatenate([qB, mx.zeros((HKV, W, 16 - G, D), dtype=queries.dtype)], axis=2))
    flat = [0] * (W * MAX_DEPTH)
    for node, path in enumerate(paths):
        flat[node * MAX_DEPTH: node * MAX_DEPTH + len(path)] = [row for row in path]
    dimsB = mx.array([L, P, PT, NCB, W, RP, CA], dtype=mx.int32)
    paths_mx = mx.array(flat, dtype=mx.int32)
    depths_mx = mx.array(depths, dtype=mx.int32)
    poB, pmB, plB = _kernel("tail")(
        inputs=[qB, keys, values, sc, dimsB, paths_mx, depths_mx, poA, pmA, plA],
        template=[("G", G), ("D", D), ("CK", CHUNK), ("TK", TILE), ("MAXD", MAX_DEPTH)],
        grid=(HKV * 32, NCB, W), threadgroup=(32, 1, 1),
        output_shapes=[(HKV * NCB * W * 16 * D,), (HKV * NCB * W * 16,), (HKV * NCB * W * 16,)],
        output_dtypes=[mx.float32, mx.float32, mx.float32])
    return _kernel("tree_merge")(
        inputs=[poA, pmA, plA, poB, pmB, plB, dimsB], template=[("G", G), ("D", D), ("CK", CHUNK)],
        grid=(HKV * 32, R, 1), threadgroup=(32, 1, 1),
        output_shapes=[(1, H, W, D)], output_dtypes=[mx.bfloat16])[0]


def warm(heads: int = 24, kv_heads: int = 4, max_queries: int = 16) -> None:
    """Compile the kernel variants (one per simdgroups-per-threadgroup count) before the first request."""

    group = heads // kv_heads
    k = mx.zeros((1, kv_heads, max_queries + 64, 256), dtype=mx.bfloat16)
    widths = sorted({t for t in range(1, max_queries + 1) if (16 * ((group * t + 15) // 16)) // 16 <= TILES_PER_GROUP}
                    | {max_queries})
    seen, outs = set(), []
    for t in widths:
        sg = min((group * t + 15) // 16, TILES_PER_GROUP)
        if sg in seen:
            continue
        seen.add(sg)
        outs.append(lane_sdpa(mx.zeros((1, heads, t, 256), dtype=mx.bfloat16), k[:, :, :t + 64], k[:, :, :t + 64], 0.0625))
    mx.eval(outs)


# -- routing the model's decode attention ---------------------------------------------
_STOCK: Any = None
enabled = False


def lane_attention(queries: mx.array, keys: mx.array, values: mx.array, cache: Any, scale: float,
                   mask: Any, sinks: Any = None) -> mx.array:
    T = int(queries.shape[2])
    if (enabled and T <= MAX_QUERIES and sinks is None and not hasattr(cache, "bits")
            and int(queries.shape[0]) == 1 and not isinstance(mask, mx.array) and int(queries.shape[3]) == 256
            and queries.dtype == mx.bfloat16 and keys.dtype == mx.bfloat16):
        return lane_sdpa(queries, keys, values, scale)
    return _STOCK(queries, keys, values, cache, scale, mask, sinks)


def install() -> None:
    """Route one-stream decode attention (up to 32 queries, causal) through ``lane_sdpa``."""

    global _STOCK, enabled
    import mlx_lm.models.qwen3_next as qn

    current = qn.scaled_dot_product_attention
    if current is not lane_attention:
        _STOCK = current
        qn.scaled_dot_product_attention = lane_attention
    enabled = True


__all__ = ["CHUNK", "MAX_QUERIES", "install", "lane_attention", "lane_sdpa", "lane_tree_sdpa", "warm"]
