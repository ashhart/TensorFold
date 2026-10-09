// Own stream_attention arithmetic uses absolute key tiles, half probabilities and ordered FP32 chunk merge.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
constant constexpr int G = 6, D = 256, CK = 512, TK = 64, MAXD = 128, MG = 8, MS = 12;
struct Q27AttnReadPtrs { device const bfloat* keys; device const bfloat* values; };

template <int SG>
kernel void q27_attn_prefix(device const bfloat* Qp [[buffer(0)]], device const Q27AttnReadPtrs* caches [[buffer(1)]],
    constant float* scale [[buffer(2)]], device const int* meta [[buffer(3)]], device const int* tile_stream [[buffer(4)]],
    device float* PO [[buffer(5)]], device float* PM [[buffer(6)]], device float* PL [[buffer(7)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]], ushort thread_index_in_simdgroup [[thread_index_in_simdgroup]],
    ushort simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]) {
  const ushort lane = thread_index_in_simdgroup;
  const ushort sg = simdgroup_index_in_threadgroup;
  const int tile = int(threadgroup_position_in_grid.z) * SG + sg;
  const uint hk = threadgroup_position_in_grid.x;
  const uint c = threadgroup_position_in_grid.y;
  const int NCH = meta[1], SGA = meta[2];
  const int st = tile < SGA ? tile_stream[tile] : 0;
  const int L = meta[MG + st * MS + 0], NQ = meta[MG + st * MS + 2];
  const int lt = tile - meta[MG + st * MS + 3];
  const int RP = 16 * SGA;
  const short qid = lane >> 2;
  const short fm = (qid & 4) | ((lane >> 1) & 3);
  const short fn = ((qid & 2) | (lane & 1)) * 4;
  const int r0 = tile * 16 + fm, r1 = r0 + 8;
  const int q0 = lt * 16 + fm, q1 = q0 + 8;
  const int n0 = (q0 < G * NQ) ? L : 0;
  const int n1 = (q1 < G * NQ) ? L : 0;
  threadgroup half Ps[SG * 16 * TK];
  threadgroup half* myP = Ps + sg * 16 * TK;
  if (tile >= SGA || int(c) >= meta[MG + st * MS + 1]) return;
  const device bfloat* Kb = caches[st].keys;
  const device bfloat* Vb = caches[st].values;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tQ((device bfloat*)Qp + (int64_t)hk * RP * D, dextents<int32_t, 2>(D, RP));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tK((device bfloat*)Kb + (int64_t)hk * meta[MG + st * MS + 7], dextents<int32_t, 2>(D, L), array<int32_t, 2>({1, D}));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tV((device bfloat*)Vb + (int64_t)hk * meta[MG + st * MS + 8], dextents<int32_t, 2>(D, L), array<int32_t, 2>({1, D}));
  tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tP(myP, dextents<int32_t, 2>(TK, 16));
  constexpr auto dS = matmul2d_descriptor(16, TK, D, false, true, false, matmul2d_descriptor::mode::multiply);
  constexpr auto dO = matmul2d_descriptor(16, 128, TK, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<dS, execution_simdgroup> opS;
  matmul2d<dO, execution_simdgroup> opO;
  auto aQ = tQ.slice(0, tile * 16);
  auto bV0 = tV.slice(0, 0);
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
    for (int i = 0; i < TK / 2; i++) {
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
  for (int q = 0; q < 16; q++) {
    device float* dst = PO + (base + tile * 16 + fm + (q & 1) * 8) * D + (q >> 1) * 16 + fn;
    *(device float4*)dst = float4(Olo[4 * q], Olo[4 * q + 1], Olo[4 * q + 2], Olo[4 * q + 3]);
    *(device float4*)(dst + 128) = float4(Ohi[4 * q], Ohi[4 * q + 1], Ohi[4 * q + 2], Ohi[4 * q + 3]);
  }
  if ((lane & 9) == 0) {
    PM[base + r0] = m0; PL[base + r0] = l0;
    PM[base + r1] = m1; PL[base + r1] = l1;
  }
}
#define Q27_PREFIX(SG) template [[host_name("q27_attn_prefix_" #SG)]] kernel void q27_attn_prefix<SG>(device const bfloat*, device const Q27AttnReadPtrs*, constant float*, device const int*, device const int*, device float*, device float*, device float*, uint3, ushort, ushort);
Q27_PREFIX(1)
Q27_PREFIX(2)
Q27_PREFIX(3)
Q27_PREFIX(4)
Q27_PREFIX(5)
Q27_PREFIX(6)
Q27_PREFIX(7)
Q27_PREFIX(8)
Q27_PREFIX(9)
Q27_PREFIX(10)
Q27_PREFIX(11)
Q27_PREFIX(12)
Q27_PREFIX(13)
Q27_PREFIX(14)
Q27_PREFIX(15)
Q27_PREFIX(16)

kernel void q27_attn_tail(device const bfloat* QB [[buffer(0)]], device const Q27AttnReadPtrs* caches [[buffer(1)]],
    constant float* scale [[buffer(2)]], device const int* meta [[buffer(3)]], device const int* paths [[buffer(4)]], device const int* nodes [[buffer(5)]],
    device const float* POA [[buffer(6)]], device const float* PMA [[buffer(7)]], device const float* PLA [[buffer(8)]],
    device float* PO [[buffer(9)]], device float* PM [[buffer(10)]], device float* PL [[buffer(11)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]], ushort thread_index_in_simdgroup [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort tid [[thread_index_in_threadgroup]]) {
  // four simdgroups load each K and V tile; simdgroup 0 alone does the arithmetic, in the one-simdgroup order
  const ushort lane = thread_index_in_simdgroup;
  const uint hk = threadgroup_position_in_grid.x;
  const uint cb = threadgroup_position_in_grid.y;
  const uint node = threadgroup_position_in_grid.z;
  const int NCB = meta[3], W = meta[4], RPA = meta[5], CA = meta[6];
  const int st = nodes[2 * node + 1];
  const int P = meta[MG + st * MS + 4], PT = meta[MG + st * MS + 0];
  if (int(cb) >= meta[MG + st * MS + 5]) return;
  const int la = meta[MG + st * MS + 3] * 16 + (int(node) - meta[MG + st * MS + 6]) * G;
  const int depth = nodes[2 * node];
  const device bfloat* Kb = caches[st].keys;
  const device bfloat* Vb = caches[st].values;
  const int nmax = P + depth + 1;
  const short qid = lane >> 2;
  const short fm = (qid & 4) | ((lane >> 1) & 3);
  const short fn = ((qid & 2) | (lane & 1)) * 4;
  const int r0 = fm, r1 = fm + 8;
  const int n0 = r0 < G ? nmax : 0;
  const int n1 = r1 < G ? nmax : 0;
  threadgroup half myP[16 * TK];
  threadgroup bfloat KV[32 * D];
  const device bfloat* kbase = (const device bfloat*)Kb + (int64_t)hk * meta[MG + st * MS + 7];
  const device bfloat* vbase = (const device bfloat*)Vb + (int64_t)hk * meta[MG + st * MS + 8];
  const int64_t kstep = D, vstep = D;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tQ((device bfloat*)QB + ((int64_t)hk * W + node) * 16 * D, dextents<int32_t, 2>(D, 16));
  tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline> tK32(KV, dextents<int32_t, 2>(D, 32));
  tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline> tVh(KV, dextents<int32_t, 2>(128, TK));
  tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tP(myP, dextents<int32_t, 2>(TK, 16));
  constexpr auto dS = matmul2d_descriptor(16, 32, D, false, true, false, matmul2d_descriptor::mode::multiply);
  constexpr auto dO = matmul2d_descriptor(16, 128, TK, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<dS, execution_simdgroup> opS;
  matmul2d<dO, execution_simdgroup> opO;
  auto Olo = opO.template get_destination_cooperative_tensor<decltype(tP), decltype(tVh), float>();
  auto Ohi = opO.template get_destination_cooperative_tensor<decltype(tP), decltype(tVh), float>();
  for (int i = 0; i < 64; i++) { Olo[i] = 0.0f; Ohi[i] = 0.0f; }
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
  const int c0 = PT / CK;
  const int c = c0 + int(cb);
  const int kbeg = max(c * CK, PT);
  const int kend = min((c + 1) * CK, nmax);
  if (sg == 0 && cb == 0 && PT > c0 * CK) {
    const int64_t baseA = ((int64_t)hk * CA + c0) * RPA + la;
    for (int q = 0; q < 16; q++) {
      const int row = fm + (q & 1) * 8;
      if (row >= G) continue;
      const auto src = POA + (baseA + row) * D + (q >> 1) * 16 + fn;
      for (int j = 0; j < 4; j++) { Olo[4 * q + j] = src[j]; Ohi[4 * q + j] = src[128 + j]; }
    }
    if (r0 < G) { m0 = PMA[baseA + r0]; l0 = PLA[baseA + r0]; }
    if (r1 < G) { m1 = PMA[baseA + r1]; l1 = PLA[baseA + r1]; }
  }
  for (int kt = kbeg; kt < kend; kt += TK) {
    float sraw[TK / 2];
    for (int h = 0; h < TK / 32; h++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint e = tid; e < 32 * D / 8; e += 128) {
        const int row = int(e) / (D / 8), col = (int(e) % (D / 8)) * 8;
        const int q = kt + h * 32 + row;
        int phys = -1;
        if (q < P) phys = q;
        else if (q < nmax) phys = P + paths[node * MAXD + (q - P)];
        ((threadgroup vec<bfloat, 8>*)KV)[e] = phys >= 0 ? *(const device vec<bfloat, 8>*)(kbase + phys * kstep + col) : vec<bfloat, 8>(0);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg == 0) {
        auto S = opS.template get_destination_cooperative_tensor<decltype(tQ), decltype(tK32), float>();
        opS.run(tQ, tK32, S);
        for (int i = 0; i < 16; i++) sraw[h * 16 + i] = S[i];
      }
    }
    if (sg == 0) {
    float s[TK / 2];
    for (int i = 0; i < TK / 2; i++) {
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
    }
    for (int hv = 0; hv < 2; hv++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint e = tid; e < TK * 128 / 8; e += 128) {
        const int row = int(e) / 16, col = hv * 128 + (int(e) % 16) * 8;
        const int q = kt + row;
        int phys = -1;
        if (q < P) phys = q;
        else if (q < nmax) phys = P + paths[node * MAXD + (q - P)];
        ((threadgroup vec<bfloat, 8>*)KV)[e] = phys >= 0 ? *(const device vec<bfloat, 8>*)(vbase + phys * vstep + col) : vec<bfloat, 8>(0);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg == 0) {
        if (hv == 0) opO.run(tP, tVh, Olo);
        else opO.run(tP, tVh, Ohi);
      }
    }
  }
  if (sg != 0) return;
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
}

kernel void q27_attn_merge(device const float* POA [[buffer(0)]], device const float* PMA [[buffer(1)]], device const float* PLA [[buffer(2)]],
    device const float* POB [[buffer(3)]], device const float* PMB [[buffer(4)]], device const float* PLB [[buffer(5)]],
    device const int* meta [[buffer(6)]], device const int* nodes [[buffer(7)]], device bfloat* OUT [[buffer(8)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]], uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {
  const uint lane = thread_index_in_simdgroup;
  const uint hk = threadgroup_position_in_grid.x;
  const uint r = threadgroup_position_in_grid.y;
  const int NCB = meta[3], W = meta[4], RPA = meta[5], CA = meta[6];
  const int st = nodes[2 * (int(r) / G) + 1];
  const int PT = meta[MG + st * MS + 0], NCBS = meta[MG + st * MS + 5];
  const int ra = meta[MG + st * MS + 3] * 16 + (int(r) / G - meta[MG + st * MS + 6]) * G + int(r) % G;
  const int CT = PT / CK;
  constexpr int DP = D / 32;
  const int node = r / G, g = r % G;
  float m = -INFINITY, l = 0.0f, o[DP];
  for (int i = 0; i < DP; i++) o[i] = 0.0f;
  for (int c = 0; c < CT; c++) {
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
  for (int c = 0; c < NCBS; c++) {
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
  for (int i = 0; i < DP; i++) OUT[((int64_t)node * (G * 4) + h) * D + lane * DP + i] = static_cast<bfloat>(o[i] / l);
}
