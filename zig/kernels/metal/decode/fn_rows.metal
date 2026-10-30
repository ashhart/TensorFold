// Flash Next decode products without tensor units (0.6.6 rows.py): each input row alone in a simdgroup, row-invariant.
#include <metal_stdlib>
// Defines: FZ_K, FZ_N (outputs), FZ_LD (row stride), FZ_BITS, FZ_GS, FZ_RPS, FZ_SG, FZ_NIB (4-bit nibbles, same bits).
using namespace metal;

#define PRAGMA_UNROLL _Pragma("clang loop unroll(full)")

// MLX's 4-bit qmv inner loop: 16 inputs a lane pre-divided by 1, 16, 256, 4096; runs of 4 summed in bf16 first.
inline float load16(const device bfloat* x, thread float* xt) {
  float sum = 0.0f;
  for (int i = 0; i < 16; i += 4) {
    const bfloat a = x[i], b = x[i + 1], c = x[i + 2], d = x[i + 3];
    sum += float(bfloat(float(bfloat(float(bfloat(float(a) + float(b))) + float(c))) + float(d)));
    xt[i] = float(a); xt[i + 1] = float(b) / 16.0f; xt[i + 2] = float(c) / 256.0f; xt[i + 3] = float(d) / 4096.0f;
  }
  return sum;
}
inline float qdot16(const device uint8_t* w, const thread float* xt, float scale, float bias, float sum) {
  const device uint16_t* ws = (const device uint16_t*)w;
  float accum = 0.0f;
  for (int i = 0; i < 4; i++)
    accum += xt[4 * i] * float(ws[i] & 0x000f) + xt[4 * i + 1] * float(ws[i] & 0x00f0) +
             xt[4 * i + 2] * float(ws[i] & 0x0f00) + xt[4 * i + 3] * float(ws[i] & 0xf000);
  return scale * accum + sum * bias;
}
// float(v) for v < 2^23 by the exponent trick: the same value as a convert
inline float nib(uint v) { return as_type<float>(0x4B000000u | v) - 8388608.0f; }
inline float qdot16x(const device uint8_t* w, const thread float* xt, float scale, float bias, float sum) {
  const device uint16_t* ws = (const device uint16_t*)w;
  float accum = 0.0f;
  for (int i = 0; i < 4; i++)
    accum += xt[4 * i] * nib(ws[i] & 0x000fu) + xt[4 * i + 1] * nib(ws[i] & 0x00f0u) +
             xt[4 * i + 2] * nib(ws[i] & 0x0f00u) + xt[4 * i + 3] * nib(ws[i] & 0xf000u);
  return scale * accum + sum * bias;
}
// two 16-bit words' nibbles q (q < 1024 each half) as half with no convert
inline half2 nib2(uint q) { return as_type<half2>(0x64006400u | q) - half2(1024.0h); }
// load16 with the third and fourth inputs divided by 16 and 1: with qdot16h's nibbles, the same products as qdot16's
inline float load16h(const device bfloat* x, thread float* xt) {
  float sum = 0.0f;
  for (int i = 0; i < 16; i += 4) {
    const bfloat a = x[i], b = x[i + 1], c = x[i + 2], d = x[i + 3];
    sum += float(bfloat(float(bfloat(float(bfloat(float(a) + float(b))) + float(c))) + float(d)));
    xt[i] = float(a); xt[i + 1] = float(b) / 16.0f; xt[i + 2] = float(c) / 16.0f; xt[i + 3] = float(d);
  }
  return sum;
}
// qdot16 over load16h's inputs with half nibbles (q, 16 q, 16 q, q): the same products in the same order
inline float qdot16h(const device uint8_t* w, const thread float* xt, float scale, float bias, float sum) {
  const device uint32_t* ws = (const device uint32_t*)w;
  float accum = 0.0f;
  for (int i = 0; i < 2; i++) {
    const uint u = ws[i];
    const half2 a = nib2(u & 0x000F000Fu), b = nib2(u & 0x00F000F0u), c = nib2((u >> 4) & 0x00F000F0u),
                d = nib2((u >> 12) & 0x000F000Fu);
    accum += xt[8 * i] * float(a.x) + xt[8 * i + 1] * float(b.x) + xt[8 * i + 2] * float(c.x) +
             xt[8 * i + 3] * float(d.x);
    accum += xt[8 * i + 4] * float(a.y) + xt[8 * i + 5] * float(b.y) + xt[8 * i + 6] * float(c.y) +
             xt[8 * i + 7] * float(d.y);
  }
  return scale * accum + sum * bias;
}
#if FZ_NIB == 1
#define FZ_LOAD16 load16h
#define FZ_QDOT16 qdot16h
#elif FZ_NIB == 2
#define FZ_LOAD16 load16
#define FZ_QDOT16 qdot16x
#else
#define FZ_LOAD16 load16
#define FZ_QDOT16 qdot16
#endif

// A lane's step of MLX's qmv_fast loop for any width: VPT codes (16, or 8 at 6 and 8 bits) from one row.
constexpr int lane_values(int bits) { return bits == 6 || bits == 8 ? 8 : 16; }
// the VPT codes from value v0 of a row (v0 a multiple of VPT, so they start on a 2-byte boundary)
template <int BITS, int VPT>
inline void lane_codes(const device uint8_t* row, int v0, thread float* q) {
  constexpr int NH = VPT * BITS / 16;
  const device ushort* h = (const device ushort*)(row + v0 * BITS / 8);
  ushort u[NH + 1];
  for (int j = 0; j < NH; j++) u[j] = h[j];
  u[NH] = 0;
  for (int i = 0; i < VPT; i++) {
    const int bit = BITS * i, j = bit / 16, s = bit % 16;
    uint v = uint(u[j]) >> s;
    if (s + BITS > 16) v |= uint(u[j + 1]) << (16 - s);
    q[i] = float(v & ((1u << BITS) - 1u));
  }
}

// 4-bit group 32: simdgroup r runs MLX's qmv_fast for row r; rows from LIM[0] on (unverified) are left alone
[[max_total_threads_per_threadgroup(1024)]]
[[kernel]] void q4_qmv_rows(const device bfloat* X [[buffer(0)]], const device uint32_t* W [[buffer(1)]],
    const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]], device bfloat* OUT [[buffer(4)]],
    const constant int* LIM [[buffer(5)]], uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
    uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]) {
  constexpr int K = FZ_K, LD = FZ_LD, RPS = FZ_RPS;
  const uint lane = thread_index_in_simdgroup;
  const int r = int(simdgroup_index_in_threadgroup);
  if (r >= LIM[0]) return;
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
    const float sum = FZ_LOAD16(x, xt);
    for (int j = 0; j < RPS; j++)
      acc[j] += FZ_QDOT16(w + j * KB, xt, float(sc[j * KG]), float(bi[j * KG]), sum);
    w += 256; sc += 16; bi += 16; x += 512;
  }
  for (int j = 0; j < RPS; j++) {
    const float v = simd_sum(acc[j]);
    if (lane == 0) OUT[size_t(r) * LD + row0 + j] = bfloat(v);
  }
}

// Any width: simdgroup r runs the qmv_fast loop (VPT codes a lane a step) for input row r over outputs RPS b ..
[[max_total_threads_per_threadgroup(1024)]]
[[kernel]] void qa_qmv_rows(const device bfloat* X [[buffer(0)]], const device uint32_t* W [[buffer(1)]],
    const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]], device bfloat* OUT [[buffer(4)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
    uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]) {
  constexpr int K = FZ_K, LD = FZ_LD, BITS = FZ_BITS, GS = FZ_GS, RPS = FZ_RPS;
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
    if (lane == 0) OUT[size_t(r) * LD + row0 + j] = bfloat(v);
  }
}

// qa_qmv_rows' sum of each (row, block): block b's VPT inputs added in order from zero
[[kernel]] void qa_row_block_sums(const device bfloat* X [[buffer(0)]], device float* SUMS [[buffer(1)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]]) {
  constexpr int K = FZ_K, VPT = lane_values(FZ_BITS);
  const int b = int(thread_position_in_grid.x), r = int(thread_position_in_grid.y);
  if (b >= K / VPT) return;
  const device bfloat* xp = X + size_t(r) * K + b * VPT;
  float sum = 0.0f;
  for (int i = 0; i < VPT; i++) sum += float(xp[i]);
  SUMS[size_t(r) * (K / VPT) + b] = sum;
}

// qa_qmv_rows' bits for several rows on the matrix units: threadgroup (i, t) takes outputs 8 i .., rows 8 t ..
[[kernel]] void qa_qmv_rows_mma(const device bfloat* X [[buffer(0)]], const device float* SUMS [[buffer(1)]],
    const device uint* W [[buffer(2)]], const device bfloat* S [[buffer(3)]], const device bfloat* B [[buffer(4)]],
    device bfloat* OUT [[buffer(5)]], constant int& R [[buffer(6)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
    uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]) {
  constexpr int K = FZ_K, N = FZ_N, LD = FZ_LD, BITS = FZ_BITS, GS = FZ_GS, SG = FZ_SG;
  const uint lane = thread_index_in_simdgroup;
  const int sg = int(simdgroup_index_in_threadgroup);
  const int qid = int(lane) / 4;
  const int fm = (qid & 4) + ((int(lane) / 2) % 4);
  const int fn = (qid & 2) * 2 + (int(lane) % 2) * 2;
  constexpr int VPT = lane_values(BITS), NB = K / VPT, STEPS = NB / 32, KG = K / GS, WPR = K * BITS / 32;
  constexpr int L = 32 / SG;                            // simdgroup j keeps qa_qmv_rows' lanes j L .. j L + L - 1
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
      simdgroup_matrix<float, 8, 8> P = simdgroup_matrix<float, 8, 8>(0.0f);  // a block's dot per (output, row)
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
    if (lane == 0 && n < N && row < R) OUT[size_t(row) * LD + n] = bfloat(v);
  }
}

// qa_qmv_rows over four input rows a simdgroup: each code cut once, every row qa_qmv_rows' sums in its order
[[max_total_threads_per_threadgroup(1024)]]
[[kernel]] void qa_qmv_rows4(const device bfloat* X [[buffer(0)]], const device uint32_t* W [[buffer(1)]],
    const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]], device bfloat* OUT [[buffer(4)]],
    constant int& R [[buffer(5)]], uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
    uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]) {
  constexpr int K = FZ_K, LD = FZ_LD, BITS = FZ_BITS, GS = FZ_GS, RPS = FZ_RPS, M = 4;
  const uint lane = thread_index_in_simdgroup;
  const int r0 = M * int(simdgroup_index_in_threadgroup);
  const int row0 = int(threadgroup_position_in_grid.y) * RPS;
  constexpr int VPT = lane_values(BITS), RB = K * BITS / 8, KG = K / GS;
  const device uint8_t* w = (const device uint8_t*)W + size_t(row0) * RB;
  const device bfloat* x[M];
  for (int m = 0; m < M; m++) x[m] = X + min(r0 + m, R - 1) * K;
  float acc[M][RPS];
  for (int m = 0; m < M; m++)
    for (int j = 0; j < RPS; j++) acc[m][j] = 0.0f;
  for (int v0 = int(lane) * VPT; v0 < K; v0 += 32 * VPT) {
    float xv[M][VPT], sum[M];
    for (int m = 0; m < M; m++) {
      sum[m] = 0.0f;
      for (int i = 0; i < VPT; i++) { xv[m][i] = float(x[m][v0 + i]); sum[m] += xv[m][i]; }
    }
    for (int j = 0; j < RPS; j++) {
      float q[VPT];
      lane_codes<BITS, VPT>(w + j * RB, v0, q);
      const size_t at = size_t(row0 + j) * KG + v0 / GS;
      const float scale = float(S[at]), bias = float(B[at]);
      for (int m = 0; m < M; m++) {
        float d = 0.0f;
        for (int i = 0; i < VPT; i++) d = fma(q[i], xv[m][i], d);
        acc[m][j] += fma(scale, d, bias * sum[m]);
      }
    }
  }
  for (int m = 0; m < M; m++) {
    if (r0 + m >= R) break;
    for (int j = 0; j < RPS; j++) {
      const float v = simd_sum(acc[m][j]);
      if (lane == 0) OUT[size_t(r0 + m) * LD + row0 + j] = bfloat(v);
    }
  }
}
