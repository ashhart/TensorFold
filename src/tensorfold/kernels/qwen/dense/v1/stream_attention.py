"""Tree attention for the windows of up to 8 streams in one launch per kernel; one stream is the same call.

Stream s's window rows sit last in its own KV buffers (logical keys P_s .. P_s + W_s - 1 after the cache's
update). Row v attends to keys [0, P_s) and its path's window keys at logical positions P_s + depth, with
``lane_attention``'s arithmetic, fixed by absolute key position: every stream's committed whole tiles in one
shared-part launch (16-row tiles of one stream's (query, head) rows), every node's remaining keys in one tail
launch, then one merge. The kernels are lane_attention's sources with only the stream lookups changed, so a row's
bits depend only on its query, its stream's keys and its path, alone or beside other streams.
"""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

import mlx.core as mx

from tensorfold.kernels.qwen.dense.v1 import lane_attention as la
from tensorfold.kernels.qwen.dense.v1.lane_tree import MAX_DEPTH, kernel_ints, tree_paths

MAX_STREAMS = 8          # K and V buffers per launch (Metal binds at most 31 buffers)
MG, MS = 8, 12           # meta: MG global ints, then MS ints per stream

# lane_attention's partial (direct P), tail and tree-merge kernels, each row's arithmetic unchanged; only where a
# row finds its stream's keys, key counts and first rows differs (every row == its one-stream call, tested)
_PARTIAL = r"""
  const ushort lane = thread_index_in_simdgroup;
  const ushort sg = simdgroup_index_in_threadgroup;
  const int tile = int(threadgroup_position_in_grid.z) * SG + sg;   // 16-row tile (SGA in all, SG per threadgroup)
  const uint hk = threadgroup_position_in_grid.x;              // key head
  const uint c = threadgroup_position_in_grid.y;               // chunk of CK keys
  const int NCH = meta[1], SGA = meta[2];
  const int st = tile < SGA ? tile_stream[tile] : 0;            // the tile's stream
  const int L = meta[MG + st * MS + 0], NQ = meta[MG + st * MS + 2];
  const int lt = tile - meta[MG + st * MS + 3];                 // the tile within its stream
  const int RP = 16 * SGA;
  const short qid = lane >> 2;
  const short fm = (qid & 4) | ((lane >> 1) & 3);
  const short fn = ((qid & 2) | (lane & 1)) * 4;
  const int r0 = tile * 16 + fm, r1 = r0 + 8;
  const int q0 = lt * 16 + fm, q1 = q0 + 8;
  const int n0 = (q0 < G * NQ) ? L : 0;
  const int n1 = (q1 < G * NQ) ? L : 0;
  // P in half (it lies in [0, 1]): the P x V op then runs ~2x the fp32 rate. With 16-bit operands the op's
  // destination interleaves rows fm and fm + 8 every 4 elements (fp32 P: elements 0-31 are row fm)
  threadgroup half Ps[SG * 16 * TK];
  threadgroup half* myP = Ps + sg * 16 * TK;
  if (tile >= SGA || int(c) >= meta[MG + st * MS + 1]) return;
  const device bfloat16_t* Kb = K0;
  const device bfloat16_t* Vb = V0;
  switch (st) {
    case 1: Kb = K1; Vb = V1; break;
    case 2: Kb = K2; Vb = V2; break;
    case 3: Kb = K3; Vb = V3; break;
    case 4: Kb = K4; Vb = V4; break;
    case 5: Kb = K5; Vb = V5; break;
    case 6: Kb = K6; Vb = V6; break;
    case 7: Kb = K7; Vb = V7; break;
  }
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tQ((device bfloat*)Qp + (int64_t)hk * RP * D, dextents<int32_t, 2>(D, RP));
  // rows at their real stride (a cache buffer's rows are D apart; other layouts need not be)
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tK((device bfloat*)Kb + (int64_t)hk * meta[MG + st * MS + 7], dextents<int32_t, 2>(D, L), array<int32_t, 2>({1, D}));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tV((device bfloat*)Vb + (int64_t)hk * meta[MG + st * MS + 8], dextents<int32_t, 2>(D, L), array<int32_t, 2>({1, D}));
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
    auto Pc = opS.template get_destination_cooperative_tensor<decltype(aQ), decltype(bK), half>();
    for (int i = 0; i < TK / 2; i++) Pc[i] = half(p[i]);
    auto Pin = opO.template get_left_input_cooperative_tensor<half, bfloat, float>(Pc);
    for (int i = 0; i < 64; i++) { const float f = (i & 4) ? f1 : f0; Olo[i] *= f; Ohi[i] *= f; }
    auto bVlo = tV.slice(0, kt);
    auto bVhi = tV.slice(128, kt);
    opO.run(Pin, bVlo, Olo);
    opO.run(Pin, bVhi, Ohi);
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

_TAIL = r"""
  const ushort lane = thread_index_in_simdgroup;
  const uint hk = threadgroup_position_in_grid.x;              // key head
  const uint cb = threadgroup_position_in_grid.y;              // tail chunk (from the first chunk holding the window)
  const uint node = threadgroup_position_in_grid.z;            // a node of any stream
  const int NCB = meta[3], W = meta[4], RPA = meta[5], CA = meta[6];
  const int st = nodes[2 * node + 1];
  const int P = meta[MG + st * MS + 4], PT = meta[MG + st * MS + 0];
  if (int(cb) >= meta[MG + st * MS + 5]) return;
  const int la = meta[MG + st * MS + 3] * 16 + (int(node) - meta[MG + st * MS + 6]) * G;
  const int depth = nodes[2 * node];
  const device bfloat16_t* Kb = K0;
  const device bfloat16_t* Vb = V0;
  switch (st) {
    case 1: Kb = K1; Vb = V1; break;
    case 2: Kb = K2; Vb = V2; break;
    case 3: Kb = K3; Vb = V3; break;
    case 4: Kb = K4; Vb = V4; break;
    case 5: Kb = K5; Vb = V5; break;
    case 6: Kb = K6; Vb = V6; break;
    case 7: Kb = K7; Vb = V7; break;
  }
  const int nmax = P + depth + 1;                              // logical keys 0 .. P + depth
  const short qid = lane >> 2;
  const short fm = (qid & 4) | ((lane >> 1) & 3);
  const short fn = ((qid & 2) | (lane & 1)) * 4;
  const int r0 = fm, r1 = fm + 8;                              // query rows: the node's heads (G of 16)
  const int n0 = r0 < G ? nmax : 0;
  const int n1 = r1 < G ? nmax : 0;
  threadgroup half myP[16 * TK];
  threadgroup bfloat KV[32 * D];                               // 32 keys at a time, or TK keys' half rows of values
  const device bfloat* kbase = (const device bfloat*)Kb + (int64_t)hk * meta[MG + st * MS + 7];
  const device bfloat* vbase = (const device bfloat*)Vb + (int64_t)hk * meta[MG + st * MS + 8];
  const int64_t kstep = D, vstep = D;
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
    const int64_t baseA = ((int64_t)hk * CA + c0) * RPA + la;
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

_MERGE = r"""
  const uint lane = thread_index_in_simdgroup;
  const uint hk = threadgroup_position_in_grid.x;
  const uint r = threadgroup_position_in_grid.y;               // node * G + g
  const int NCB = meta[3], W = meta[4], RPA = meta[5], CA = meta[6];
  const int st = nodes[2 * (int(r) / G) + 1];
  const int PT = meta[MG + st * MS + 0], NCBS = meta[MG + st * MS + 5];
  const int ra = meta[MG + st * MS + 3] * 16 + (int(r) / G - meta[MG + st * MS + 6]) * G + int(r) % G;
  const int CT = PT / CK;                                      // chunks the shared kernel finished
  constexpr int DP = D / 32;
  const int node = r / G, g = r % G;
  float m = -INFINITY, l = 0.0f, o[DP];
  for (int i = 0; i < DP; i++) o[i] = 0.0f;
  for (int c = 0; c < CT; c++) {                               // committed chunks, in order
    const int64_t row = ((int64_t)hk * CA + c) * RPA + ra;
    const float mc = PMA[row];
    if (mc == -INFINITY) continue;
    const float lc = PLA[row];
    const float nm = max(m, mc);
    const float f1 = fast::exp(m - nm), f2 = fast::exp(mc - nm);
    l = l * f1 + lc * f2;
    for (int i = 0; i < DP; i++) o[i] = o[i] * f1 + POA[row * D + lane * DP + i] * f2;
    m = nm;
  }
  for (int c = 0; c < NCBS; c++) {                              // then the window's chunks
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

_KV = [f"K{i}" for i in range(MAX_STREAMS)] + [f"V{i}" for i in range(MAX_STREAMS)]
_SPECS = {
    "partial": (_PARTIAL, ["Qp", *_KV, "scale", "meta", "tile_stream"], ["PO", "PM", "PL"]),
    "tail": (_TAIL, ["QB", *_KV, "scale", "meta", "paths", "nodes", "POA", "PMA", "PLA"], ["PO", "PM", "PL"]),
    "merge": (_MERGE, ["POA", "PMA", "PLA", "POB", "PMB", "PLB", "meta", "nodes"], ["OUT"]),
}
_kernels: dict[str, Any] = {}


def sources() -> dict[str, str]:
    return {name: spec[0] for name, spec in _SPECS.items()}


def _kernel(name: str) -> Any:
    if name not in _kernels:
        source, inputs, outputs = _SPECS[name]
        digest = hashlib.sha256((la._HEADER + source).encode()).hexdigest()[:16]
        _kernels[name] = mx.fast.metal_kernel(name=f"stream_attention_{name}_{digest}", input_names=inputs,
                                              output_names=outputs, source=source, header=la._HEADER,
                                              ensure_row_contiguous=False)
    return _kernels[name]


class Plan:
    """One round's layout for up to ``MAX_STREAMS`` streams, shared by every attention layer of the forward.

    ``parents[s]``: each window row's parent row (-1 for the root); ``starts[s]``: the stream's committed keys
    (its root's position). Rows are stream by stream; per-layer KV buffers come with each call.
    """

    def __init__(self, parents: Sequence[Sequence[int]], starts: Sequence[int], heads: int, kv_heads: int) -> None:
        if not 1 <= len(parents) <= MAX_STREAMS or len(parents) != len(starts):
            raise ValueError(f"stream_attention: 1 to {MAX_STREAMS} streams, one start each")
        G = heads // kv_heads
        self.G, self.heads, self.kv_heads, self.streams = G, heads, kv_heads, len(parents)
        base = [0] * (MG + MAX_STREAMS * MS)
        tile_stream: list[int] = []
        rows: list[int] = []
        nodes: list[int] = []
        paths: list[int] = []
        tiles = total = 0
        ca_max = ncb_max = 0
        for s, (rp, start) in enumerate(zip(parents, starts)):
            W, P = len(rp), int(start)
            depths, node_paths = tree_paths(rp)
            PT = (P // la.TILE) * la.TILE
            CA = -(-PT // la.CHUNK)
            NCB = (P + max(depths)) // la.CHUNK - PT // la.CHUNK + 1
            count = -(-(G * W) // 16)
            base[MG + s * MS: MG + s * MS + 7] = [PT, CA, W, tiles, P, NCB, total]
            tile_stream += [s] * count
            rows += [total * G + i if i < G * W else total * G for i in range(count * 16)]   # padding: masked
            for depth, path in zip(depths, node_paths):
                nodes += [depth, s]
                paths += list(path) + [0] * (MAX_DEPTH - len(path))
            tiles += count
            total += W
            ca_max, ncb_max = max(ca_max, CA), max(ncb_max, NCB)
        base[:7] = [len(parents), ca_max, tiles, ncb_max, total, 16 * tiles, ca_max]
        self.base, self.rows, self.tiles, self.ca, self.ncb = base, total, tiles, ca_max, ncb_max
        self.tile_stream = kernel_ints(tile_stream)
        self.q_rows = mx.array(rows, dtype=mx.int32)
        self.nodes = kernel_ints(nodes)
        self.paths = mx.array(paths, dtype=mx.int32)
        self._meta: dict[tuple[int, ...], mx.array] = {}
        self._spare: mx.array | None = None

    def meta(self, strides: tuple[int, ...]) -> mx.array:
        """The layout with each stream's K and V head strides (rows of D between key heads' buffers)."""

        if strides not in self._meta:
            meta = list(self.base)
            for s in range(len(strides) // 2):
                meta[MG + s * MS + 7], meta[MG + s * MS + 8] = strides[2 * s], strides[2 * s + 1]
            self._meta[strides] = mx.array(meta, dtype=mx.int32)
        return self._meta[strides]

    def spare(self) -> mx.array:
        """An unread K/V input for unused stream slots (8+ elements: one kernel source, see ``kernel_ints``)."""

        if self._spare is None:
            self._spare = mx.zeros((16,), dtype=mx.bfloat16)
        return self._spare


def tree_sdpa(queries: mx.array, kv: Sequence[tuple[mx.array, mx.array]], scale: float, plan: Plan) -> mx.array:
    """Attention [1, H, R, D] of every stream's window rows: ``queries`` [1, H, R, D] (rows stream by stream),
    ``kv[s]`` stream s's whole K and V buffers [1, HKV, capacity, D] (row-contiguous), its window rows last."""

    _, H, R, D = (int(v) for v in queries.shape)
    HKV = plan.kv_heads
    if D != 256 or H != plan.heads or R != plan.rows or len(kv) > MAX_STREAMS:
        raise ValueError(f"stream_attention: unsupported shape q={queries.shape} for {len(kv)} streams")
    G, SGA, NT = plan.G, plan.tiles, plan.rows
    RP = 16 * SGA
    strides: list[int] = []
    for keys, values in kv:
        strides += [int(keys.shape[2]) * D, int(values.shape[2]) * D]
    meta = plan.meta(tuple(strides))
    spare = plan.spare()
    ks = [k for k, _ in kv] + [spare] * (MAX_STREAMS - len(kv))
    vs = [v for _, v in kv] + [spare] * (MAX_STREAMS - len(kv))
    sc = mx.array([float(scale)], dtype=mx.float32)
    per_head = queries.reshape(HKV, G, R, D).transpose(0, 2, 1, 3)              # [HKV, R, G, D]
    if plan.ca > 0:
        # (a take's output need not be row-major, and these kernels read their inputs as laid out)
        qA = mx.contiguous(mx.take(per_head.reshape(HKV, R * G, D), plan.q_rows, axis=1))
        SG = min(SGA, la.TILES_PER_GROUP)
        poA, pmA, plA = _kernel("partial")(
            inputs=[qA, *ks, *vs, sc, meta, plan.tile_stream],
            template=[("G", G), ("D", D), ("SG", SG), ("CK", la.CHUNK), ("TK", la.TILE), ("MG", MG), ("MS", MS)],
            grid=(HKV * 32 * SG, plan.ca, -(-SGA // SG)), threadgroup=(32 * SG, 1, 1),
            output_shapes=[(HKV * plan.ca * RP * D,), (HKV * plan.ca * RP,), (HKV * plan.ca * RP,)],
            output_dtypes=[mx.float32, mx.float32, mx.float32])
    else:
        poA = pmA = plA = mx.zeros((16,), dtype=mx.float32)                     # unread placeholders
    qB = mx.contiguous(mx.concatenate([per_head, mx.zeros((HKV, R, 16 - G, D), dtype=queries.dtype)], axis=2))
    poB, pmB, plB = _kernel("tail")(
        inputs=[qB, *ks, *vs, sc, meta, plan.paths, plan.nodes, poA, pmA, plA],
        template=[("G", G), ("D", D), ("CK", la.CHUNK), ("TK", la.TILE), ("MAXD", MAX_DEPTH), ("MG", MG), ("MS", MS)],
        grid=(HKV * 32, plan.ncb, NT), threadgroup=(32, 1, 1),
        output_shapes=[(HKV * plan.ncb * NT * 16 * D,), (HKV * plan.ncb * NT * 16,), (HKV * plan.ncb * NT * 16,)],
        output_dtypes=[mx.float32, mx.float32, mx.float32])
    return _kernel("merge")(
        inputs=[poA, pmA, plA, poB, pmB, plB, meta, plan.nodes],
        template=[("G", G), ("D", D), ("CK", la.CHUNK), ("MG", MG), ("MS", MS)],
        grid=(HKV * 32, NT * G, 1), threadgroup=(32, 1, 1),
        output_shapes=[(1, H, NT, D)], output_dtypes=[mx.bfloat16])[0]


__all__ = ["MAX_STREAMS", "Plan", "sources", "tree_sdpa"]
