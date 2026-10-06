// Tree windows for Nemotron: Mamba by parent, attention by logical key position, KV compaction, row gathers, draft top-k.
#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

// A segment's 8 ints: real first row, replayed rows, state slot in, replay id, parity in, save inputs, first virtual row.
#define SEG_B 0
#define SEG_K 1
#define SEG_IN 2
#define SEG_RID 3
#define SEG_PAR 4
#define SEG_SAVE 5
#define SEG_V0 6

// VA: segment, place, state slot to store, parent place (-1: slot state); VB: ancestors 3/2/1 back, replayed input row.
template <int XD, int NG, int DS, int KC, int PROJ, int XOFF>
[[kernel]] void tf_tree_conv(
  const device bfloat16_t* P [[buffer(0)]],
  const device bfloat16_t* CS_IN [[buffer(1)]],
  const device float* CW [[buffer(2)]],
  const device float* CB [[buffer(3)]],
  const device int4* VA [[buffer(4)]],
  const device int4* VB [[buffer(5)]],
  const device int32_t* SEGI [[buffer(6)]],
  const constant int32_t* dims [[buffer(7)]],
  device bfloat16_t* RAW [[buffer(8)]],
  device bfloat16_t* XBC [[buffer(9)]],
  device bfloat16_t* CS_OUT [[buffer(10)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
  constexpr int CD = XD + 2 * NG * DS;
  const int MW = dims[1];
  const int ch = int(thread_position_in_grid.x);
  const int v = int(thread_position_in_grid.y);
  const int4 a = VA[v];
  const int4 t = VB[v];
  const device int32_t* sg = SEGI + 8 * a.x;
  const int loc = a.y, b = sg[SEG_B], k = sg[SEG_K], slot = sg[SEG_IN], v0 = sg[SEG_V0];
  const int raw_in = (sg[SEG_RID] * 2 + sg[SEG_PAR]) * MW, raw_out = (sg[SEG_RID] * 2 + 1 - sg[SEG_PAR]) * MW;
  #define VAL(lp) ((lp) < 0 ? CS_IN[(slot * (KC - 1) + (lp) + KC - 1) * CD + ch] : (lp) < k ? RAW[size_t(raw_in + VB[v0 + (lp)].w) * CD + ch] : P[(b + (lp) - k) * PROJ + XOFF + ch])
  const int taps[KC] = {t.x, t.y, t.z, loc};
  float acc = float(CB[ch]);
  for (int i = 0; i < KC; i++) acc = fma(CW[i * CD + ch], float(VAL(taps[i])), acc);
  const float cv = float(bfloat(acc));
  XBC[v * CD + ch] = bfloat(cv / (1.0f + metal::exp(-cv)));
  if (loc >= k && sg[SEG_SAVE] != 0) RAW[size_t(raw_out + loc - k) * CD + ch] = P[(b + loc - k) * PROJ + XOFF + ch];
  if (a.z >= 0)
    for (int i = 0; i < KC - 1; i++) CS_OUT[(a.z * (KC - 1) + i) * CD + ch] = VAL(taps[i + 1]);
  #undef VAL
}

// VC: the tree-state slot a row's state is kept in for a later child (-1: none).
template <int H, int DH, int NG, int DS, int XD, int PROJ, int DTOFF, int SSZ>
[[kernel]] void tf_tree_scan(
  const device bfloat16_t* P [[buffer(0)]],
  const device bfloat16_t* XBC [[buffer(1)]],
  const device float* S_IN [[buffer(2)]],
  const device float* A_LOG [[buffer(3)]],
  const device float* DSKIP [[buffer(4)]],
  const device float* DT_BIAS [[buffer(5)]],
  const constant float* limits [[buffer(6)]],
  const constant int32_t* dims [[buffer(7)]],
  const device int4* VA [[buffer(8)]],
  const device int4* VB [[buffer(9)]],
  const device int32_t* VC [[buffer(10)]],
  const device int32_t* SEGI [[buffer(11)]],
  device bfloat16_t* DTRAW [[buffer(12)]],
  device bfloat16_t* Y [[buffer(13)]],
  device float* S_OUT [[buffer(14)]],
  device float* TREE [[buffer(15)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]],
  uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]]) {

  const uint lane = thread_position_in_threadgroup.x;
  const uint d = thread_position_in_grid.y;
  const uint h = thread_position_in_grid.z;
  const uint g = h / (H / NG);
  const int V = dims[0], MW = dims[1];
  constexpr int NS = DS / 32;
  constexpr int CD = XD + 2 * NG * DS;
  const int cx = int(h) * DH + int(d);
  const int cb = XD + int(g) * DS + int(lane) * NS;
  const int cc = XD + NG * DS + int(g) * DS + int(lane) * NS;
  const int sbase = cx * DS + int(lane) * NS;
  const float A = -metal::exp(float(A_LOG[h]));
  const float dskip = float(bfloat(float(DSKIP[h])));
  const float dtb = float(DT_BIAS[h]);
  float st[NS];
  int cur = -1, held = -2;
  for (int v = 0; v < V; v++) {
    const int4 a = VA[v];
    const device int32_t* sg = SEGI + 8 * a.x;
    if (a.x != cur) {
      cur = a.x;
      held = -2;
    }
    if (a.w != held) {
      if (a.w < 0)
        for (int i = 0; i < NS; i++) st[i] = float(S_IN[size_t(sg[SEG_IN]) * SSZ + sbase + i]);
      else
        for (int i = 0; i < NS; i++) st[i] = TREE[size_t(VC[sg[SEG_V0] + a.w]) * SSZ + sbase + i];
    }
    const int loc = a.y, k = sg[SEG_K], rr = sg[SEG_B] + loc - k;
    const float xv = float(XBC[v * CD + cx]);
    const bfloat raw = loc < k ? DTRAW[size_t((sg[SEG_RID] * 2 + sg[SEG_PAR]) * MW + VB[v].w) * H + h] : P[rr * PROJ + DTOFF + int(h)];
    float dt = float(raw) + dtb;
    dt = metal::max(dt, 0.0f) + metal::log(1.0f + metal::exp(-metal::abs(dt)));
    dt = metal::clamp(dt, limits[0], limits[1]);
    const float dA = metal::exp(A * dt);
    const float xdt = xv * dt;
    float acc = 0.0f;
    for (int i = 0; i < NS; i++) {
      const float sv = dA * st[i] + xdt * float(XBC[v * CD + cb + i]);
      st[i] = sv;
      acc += sv * float(XBC[v * CD + cc + i]);
    }
    acc = simd_sum(acc);
    held = loc;
    if (loc >= k && lane == 0) {
      const float y = float(bfloat(acc + xv * dskip));
      const float z = float(P[rr * PROJ + cx]);
      const float sz = float(bfloat(z / (1.0f + metal::exp(-z))));
      Y[rr * XD + cx] = bfloat(sz * y);
      if (d == 0 && sg[SEG_SAVE] != 0) DTRAW[size_t((sg[SEG_RID] * 2 + 1 - sg[SEG_PAR]) * MW + loc - k) * H + h] = raw;
    }
    if (VC[v] >= 0)
      for (int i = 0; i < NS; i++) TREE[size_t(VC[v]) * SSZ + sbase + i] = st[i];
    if (a.z >= 0)
      for (int i = 0; i < NS; i++) S_OUT[size_t(a.z) * SSZ + sbase + i] = st[i];
  }
}

// A node's keys past the shared pass, gathered by logical position (dims: P, PT, NCA, NCB, NT): its one-row step's sums.
template <int G, int D, int CK, int TK, int MAXD>
[[kernel]] void tf_tree_tail(
  const device bfloat16_t* QP [[buffer(0)]],
  const device bfloat16_t* K [[buffer(1)]],
  const constant int64_t* K_strides [[buffer(2)]],
  const device bfloat16_t* V [[buffer(3)]],
  const constant int64_t* V_strides [[buffer(4)]],
  const constant float* scale [[buffer(5)]],
  const constant int32_t* dims [[buffer(6)]],
  const device int32_t* paths [[buffer(7)]],
  const device int32_t* depths [[buffer(8)]],
  const device float* POA [[buffer(9)]],
  const device float* PMA [[buffer(10)]],
  const device float* PLA [[buffer(11)]],
  device float* PO [[buffer(12)]],
  device float* PM [[buffer(13)]],
  device float* PL [[buffer(14)]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  const ushort lane = thread_index_in_simdgroup;
  const uint hk = threadgroup_position_in_grid.x;
  const uint cb = threadgroup_position_in_grid.y;
  const uint node = threadgroup_position_in_grid.z;
  const int P = dims[0], PT = dims[1], NCA = dims[2], NCB = dims[3], NT = dims[4];
  if (int(cb) >= NCB) return;  // a grid past the dims' tail chunks: those would store over the next head's
  const int RPA = 16 * NT;
  const int nmax = P + depths[node] + 1;
  const short qid = lane >> 2;
  const short fm = (qid & 4) | ((lane >> 1) & 3);
  const short fn = ((qid & 2) | (lane & 1)) * 4;
  const int n0 = nmax, n1 = nmax;
  threadgroup half myP[16 * TK];
  threadgroup bfloat KV[TK * D];
  const device bfloat* kbase = (const device bfloat*)K + (int64_t)hk * K_strides[1];
  const device bfloat* vbase = (const device bfloat*)V + (int64_t)hk * V_strides[1];
  const int64_t kstep = K_strides[2], vstep = V_strides[2];
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tQ((device bfloat*)QP + ((int64_t)hk * RPA + node * 16) * D, dextents<int32_t, 2>(D, 16));
  tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline> tK32(KV, dextents<int32_t, 2>(D, 32));
  tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline> tVh(KV, dextents<int32_t, 2>(D, TK));
  tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tP(myP, dextents<int32_t, 2>(TK, 16));
  constexpr auto dS = matmul2d_descriptor(16, 32, D, false, true, false, matmul2d_descriptor::mode::multiply);
  constexpr auto dO = matmul2d_descriptor(16, D, TK, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<dS, execution_simdgroup> opS;
  matmul2d<dO, execution_simdgroup> opO;
  auto Olo = opO.template get_destination_cooperative_tensor<decltype(tP), decltype(tVh), float>();
  for (int i = 0; i < 64; i++) Olo[i] = 0.0f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
  const int c0 = PT / CK;
  const int c = c0 + int(cb);
  const int kbeg = max(c * CK, PT);
  const int kend = min((c + 1) * CK, nmax);
  if (cb == 0 && PT > c0 * CK) {
    const int64_t baseA = ((int64_t)hk * NCA + c0) * RPA + node * 16;
    for (int q = 0; q < 16; q++) {
      const int row = fm + (q & 1) * 8;
      const device float* src = POA + (baseA + row) * D + (q >> 1) * 16 + fn;
      for (int j = 0; j < 4; j++) Olo[4 * q + j] = src[j];
    }
    m0 = PMA[baseA + fm]; l0 = PLA[baseA + fm];
    m1 = PMA[baseA + fm + 8]; l1 = PLA[baseA + fm + 8];
  }
  for (int kt = kbeg; kt < kend; kt += TK) {
    float sraw[TK / 2];
    for (int hh = 0; hh < TK / 32; hh++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint e = lane; e < 32 * D / 8; e += 32) {
        const int row = int(e) / (D / 8), col = (int(e) % (D / 8)) * 8;
        const int q = kt + hh * 32 + row;
        const int phys = q < P ? q : q < nmax ? P + paths[node * MAXD + (q - P)] : -1;
        ((threadgroup vec<bfloat, 8>*)KV)[e] = phys >= 0 ? *(const device vec<bfloat, 8>*)(kbase + phys * kstep + col) : vec<bfloat, 8>(0);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      auto S = opS.template get_destination_cooperative_tensor<decltype(tQ), decltype(tK32), float>();
      opS.run(tQ, tK32, S);
      for (int i = 0; i < 16; i++) sraw[hh * 16 + i] = S[i];
    }
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
    for (int bb = 0; bb < TK / 16; bb++) {
      y0 += (p[bb * 8] + p[bb * 8 + 1]) + (p[bb * 8 + 2] + p[bb * 8 + 3]);
      y1 += (p[bb * 8 + 4] + p[bb * 8 + 5]) + (p[bb * 8 + 6] + p[bb * 8 + 7]);
    }
    y0 += simd_shuffle_xor(y0, 1); y0 += simd_shuffle_xor(y0, 8);
    y1 += simd_shuffle_xor(y1, 1); y1 += simd_shuffle_xor(y1, 8);
    if (x0 != -INFINITY) { l0 = l0 * f0 + y0; m0 = nm0; }
    if (x1 != -INFINITY) { l1 = l1 * f1 + y1; m1 = nm1; }
    for (int ff = 0; ff < TK / 16; ff++)
      for (int i = 0; i < 4; i++) {
        myP[fm * TK + ff * 16 + fn + i] = half(p[ff * 8 + i]);
        myP[(fm + 8) * TK + ff * 16 + fn + i] = half(p[ff * 8 + 4 + i]);
      }
    for (int i = 0; i < 64; i++) { const float fct = (i & 4) ? f1 : f0; Olo[i] *= fct; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = lane; e < TK * D / 8; e += 32) {
      const int row = int(e) / (D / 8), col = (int(e) % (D / 8)) * 8;
      const int q = kt + row;
      const int phys = q < P ? q : q < nmax ? P + paths[node * MAXD + (q - P)] : -1;
      ((threadgroup vec<bfloat, 8>*)KV)[e] = phys >= 0 ? *(const device vec<bfloat, 8>*)(vbase + phys * vstep + col) : vec<bfloat, 8>(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    opO.run(tP, tVh, Olo);
  }
  const int64_t base = (((int64_t)hk * NCB + cb) * NT + node) * 16;
  for (int q = 0; q < 16; q++) {
    device float* dst = PO + (base + fm + (q & 1) * 8) * D + (q >> 1) * 16 + fn;
    *(device float4*)dst = float4(Olo[4 * q], Olo[4 * q + 1], Olo[4 * q + 2], Olo[4 * q + 3]);
  }
  if ((lane & 9) == 0) {
    PM[base + fm] = m0; PL[base + fm] = l0;
    PM[base + fm + 8] = m1; PL[base + fm + 8] = l1;
  }
}

// A node's attention output: the shared pass's whole chunks in order, then its tail chunks (attn_merge's arithmetic).
template <int G, int D, int CK>
[[kernel]] void tf_tree_merge(
  const device float* POA [[buffer(0)]],
  const device float* PMA [[buffer(1)]],
  const device float* PLA [[buffer(2)]],
  const device float* POB [[buffer(3)]],
  const device float* PMB [[buffer(4)]],
  const device float* PLB [[buffer(5)]],
  const constant int32_t* dims [[buffer(6)]],
  device bfloat16_t* OUT [[buffer(7)]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  const uint lane = thread_index_in_simdgroup;
  const uint hk = threadgroup_position_in_grid.x;
  const uint r = threadgroup_position_in_grid.y;
  const int PT = dims[1], NCA = dims[2], NCB = dims[3], NT = dims[4];
  const int RPA = 16 * NT;
  const int CT = PT / CK;
  constexpr int DP = D / 32;
  const int node = int(r) / G, g = int(r) % G;
  float m = -INFINITY, l = 0.0f, o[DP];
  for (int i = 0; i < DP; i++) o[i] = 0.0f;
  for (int c = 0; c < CT; c++) {
    const int64_t row = ((int64_t)hk * NCA + c) * RPA + node * 16 + g;
    const float mc = PMA[row];
    if (mc == -INFINITY) continue;
    const float lc = PLA[row];
    const float nm = max(m, mc);
    const float f1 = fast::exp(m - nm), f2 = fast::exp(mc - nm);
    l = l * f1 + lc * f2;
    for (int i = 0; i < DP; i++) o[i] = o[i] * f1 + POA[row * D + lane * DP + i] * f2;
    m = nm;
  }
  for (int c = 0; c < NCB; c++) {
    const int64_t row = (((int64_t)hk * NCB + c) * NT + node) * 16 + g;
    const float mc = PMB[row];
    if (mc == -INFINITY) continue;
    const float lc = PLB[row];
    const float nm = max(m, mc);
    const float f1 = fast::exp(m - nm), f2 = fast::exp(mc - nm);
    l = l * f1 + lc * f2;
    for (int i = 0; i < DP; i++) o[i] = o[i] * f1 + POB[row * D + lane * DP + i] * f2;
    m = nm;
  }
  const int h = int(hk) * G + g;
  for (int i = 0; i < DP; i++) OUT[((int64_t)h * NT + node) * D + lane * DP + i] = static_cast<bfloat>(o[i] / l);
}

// The kept path's keys and values to the cache's next rows (len + path[j] -> len + j); rows in order, never read after written.
kernel void tf_kv_compact(device bfloat* K [[buffer(0)]],
                          device bfloat* V [[buffer(1)]],
                          const device int32_t* path [[buffer(2)]],
                          constant uint4& dims [[buffer(3)]],
                          uint2 pos [[thread_position_in_grid]]) {
  const uint d = pos.x, hk = pos.y, len = dims.x, n = dims.y, cap = dims.z, hd = dims.w;
  for (uint j = 0; j < n; j++) {
    const uint src = len + uint(path[j]), dst = len + j;
    if (src == dst) continue;
    K[(size_t(hk) * cap + dst) * hd + d] = K[(size_t(hk) * cap + src) * hd + d];
    V[(size_t(hk) * cap + dst) * hd + d] = V[(size_t(hk) * cap + src) * hd + d];
  }
}

// Rows of bf16 gathered by index: dst row j = src row rows[j]. grid (D, n).
kernel void tf_gather_rows(const device bfloat* src [[buffer(0)]],
                           device bfloat* dst [[buffer(1)]],
                           const device int32_t* rows [[buffer(2)]],
                           uint2 pos [[thread_position_in_grid]],
                           uint2 size [[threads_per_grid]]) {
  dst[size_t(pos.y) * size.x + pos.x] = src[size_t(rows[pos.y]) * size.x + pos.x];
}

// tf_argmax_bf16's order: a NaN first, then the larger value, then the lower id.
inline bool tf_before(float a, uint ia, float b, uint ib) {
  if (isnan(a)) return !isnan(b) || ia < ib;
  if (isnan(b)) return false;
  return a > b || (a == b && ia < ib);
}

// Each row's 4 best draft logits in argmax's order (one threadgroup a row), their ids through `map`, softmax chances.
kernel void tf_topk_probs(const device bfloat* logits [[buffer(0)]],
                          const device uint* map [[buffer(1)]],
                          device uint* ids [[buffer(2)]],
                          device float* probs [[buffer(3)]],
                          constant uint& vocab [[buffer(4)]],
                          uint row [[threadgroup_position_in_grid]],
                          uint t [[thread_position_in_threadgroup]],
                          uint sgi [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float bv[32];
  threadgroup uint bi[32];
  threadgroup uint chosen[4];
  threadgroup float cval[4];
  const device bfloat* x = logits + size_t(row) * vocab;
  for (int r = 0; r < 4; r++) {
    float best = -INFINITY;
    uint at = 0xffffffffu;
    for (uint i = t; i < vocab; i += 1024) {
      bool taken = false;
      for (int q = 0; q < r; q++) taken = taken || chosen[q] == i;
      const float v = float(x[i]);
      if (!taken && tf_before(v, i, best, at)) { best = v; at = i; }
    }
    for (uint o = 16; o > 0; o >>= 1) {
      const float ov = simd_shuffle_xor(best, o);
      const uint oi = simd_shuffle_xor(at, o);
      if (tf_before(ov, oi, best, at)) { best = ov; at = oi; }
    }
    if (lane == 0) { bv[sgi] = best; bi[sgi] = at; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
      float b0 = bv[0];
      uint i0 = bi[0];
      for (int q = 1; q < 32; q++) if (tf_before(bv[q], bi[q], b0, i0)) { b0 = bv[q]; i0 = bi[q]; }
      chosen[r] = i0;
      cval[r] = b0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float sum = 0.0f;
  const float top = cval[0];
  for (uint i = t; i < vocab; i += 1024) sum += metal::exp(float(x[i]) - top);
  sum = simd_sum(sum);
  if (lane == 0) bv[sgi] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    float total = 0.0f;
    for (int q = 0; q < 32; q++) total += bv[q];
    for (int r = 0; r < 4; r++) {
      ids[row * 4 + r] = map[chosen[r]];
      probs[row * 4 + r] = metal::exp(cval[r] - top) / total;
    }
  }
}

// A depth's lanes take their parent's ranked token: entry (parent's step slot, rank, window row, pass row or -1).
kernel void tf_lane_tokens(const device uint* topk [[buffer(0)]],
                           const device int4* entries [[buffer(1)]],
                           device uint* window [[buffer(2)]],
                           device uint* pass [[buffer(3)]],
                           uint i [[thread_position_in_grid]]) {
  const int4 x = entries[i];
  const uint t = topk[x.x * 4 + x.y];
  window[x.z] = t;
  if (x.w >= 0) pass[x.w] = t;
}

template [[host_name("tf_tree_conv")]] [[kernel]] decltype(tf_tree_conv<4096, 8, 128, 4, 10304, 4096>) tf_tree_conv<4096, 8, 128, 4, 10304, 4096>;
template [[host_name("tf_tree_scan")]] [[kernel]] decltype(tf_tree_scan<64, 64, 8, 128, 4096, 10304, 10240, 524288>) tf_tree_scan<64, 64, 8, 128, 4096, 10304, 10240, 524288>;
template [[host_name("tf_tree_tail")]] [[kernel]] decltype(tf_tree_tail<16, 128, 512, 64, 64>) tf_tree_tail<16, 128, 512, 64, 64>;
template [[host_name("tf_tree_merge")]] [[kernel]] decltype(tf_tree_merge<16, 128, 512>) tf_tree_merge<16, 128, 512>;
