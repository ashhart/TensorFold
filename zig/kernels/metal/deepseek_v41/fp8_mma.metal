// DeepSeek V4.1 dense FP8 prompt rows on simdgroup MMA: 32-wide K groups in fp32, one fixed group tree.
#include <metal_stdlib>
using namespace metal;

#ifndef TF_CTL
#define TF_CTL 0
#endif

// Release blockGroups at hidden 5120: K x N, world 1.
#define TF_DSV41_WQ_A_K 5120
#define TF_DSV41_WQ_A_N 1280
#define TF_DSV41_WKV_K 5120
#define TF_DSV41_WKV_N 512
#define TF_DSV41_WQ_B_K 1280
#define TF_DSV41_WQ_B_N 32768
#define TF_DSV41_WO_A_K 4096
#define TF_DSV41_WO_A_N 1024
#define TF_DSV41_WO_B_K 8192
#define TF_DSV41_WO_B_N 5120
#define TF_DSV41_COMP_WKV_K 5120
#define TF_DSV41_COMP_WKV_N 512
#define TF_DSV41_COMP_WGATE_K 5120
#define TF_DSV41_COMP_WGATE_N 512
#define TF_DSV41_IX_WK_K 512
#define TF_DSV41_IX_WK_N 128
#define TF_DSV41_IX_WQ_B_K 1280
#define TF_DSV41_IX_WQ_B_N 4096
#define TF_DSV41_SHARED_W1_K 5120
#define TF_DSV41_SHARED_W1_N 2304
#define TF_DSV41_SHARED_W2_K 2304
#define TF_DSV41_SHARED_W2_N 5120
#define TF_DSV41_SHARED_W3_K 5120
#define TF_DSV41_SHARED_W3_N 2304
#define TF_DSV41_ENGRAM_WKV_K 6144
#define TF_DSV41_ENGRAM_WKV_N 25600

struct TfDsv41Fp8MmaArgs {
  uint K, N, R;
};

#ifndef TF_RS
#define TF_RS 4   // simdgroups a threadgroup, 8 rows each: they share one decoded 32x32 weight block a K group
#endif
#define TF_T (TF_RS * 32)
#define TF_LV 9
#define TF_LD 36  // threadgroup row stride of a decoded weight block (floats)
#define TF_QW ((128 + TF_T - 1) / TF_T)  // 8-code weight quads a thread stages a group

inline float tf_dsv41_mma_e4m3(uint code) {
  const float v = float(as_type<half>(ushort((code & 0x7Fu) << 7))) * 256.0f;
  const float s = (code & 0x80u) ? -v : v;
  return (code & 0x7Fu) == 0x7Fu ? NAN : s;
}

inline float tf_dsv41_mma_e8m0(uint code) {
  if (code == 255u) return NAN;
  return as_type<float>(code == 0u ? 0x00400000u : code << 23);
}

inline float tf_dsv41_mma_x(uint bits) {
  const float v = as_type<float>(bits << 16);
  return (TF_CTL == 1 && !isfinite(v)) ? 0.0f : v;
}

// What a thread reads for one K group before it is needed: weight quads, the block scale, and two X values a step.
struct TfDsv41MmaLoad {
  uint2 w[TF_QW];
  uint sc;
  uint x[4];
};

// Group g's reads: the threadgroup's 32 weight rows (columns n0..) and the cell's X row r at k 8s + fn, 8s + fn + 1.
inline void tf_dsv41_mma_fetch(thread TfDsv41MmaLoad& l, const device uchar* Wn, const device uchar* Sn,
                               const device ushort* Xr, bool live, uint K, uint g, uint tid, uint fn) {
  for (uint i = 0; i < TF_QW; ++i) {
    const uint q = tid + i * TF_T;
    if (q < 128) l.w[i] = *(const device uint2*)(Wn + ulong(q / 4) * K + g * 32 + (q % 4) * 8);
  }
  l.sc = Sn[g];
  for (uint s = 0; s < 4; ++s) l.x[s] = live ? *(const device uint*)(Xr + g * 32 + 8 * s + fn) : 0u;
}

// The fetched weight quads, decoded and scaled, into the threadgroup block wt[column][k].
inline void tf_dsv41_mma_stage(threadgroup float* wt, thread const TfDsv41MmaLoad& l, uint tid) {
  const float sc = tf_dsv41_mma_e8m0(l.sc);
  for (uint i = 0; i < TF_QW; ++i) {
    const uint q = tid + i * TF_T;
    if (q >= 128) continue;
    float v[8];
    for (uint b = 0; b < 8; ++b) {
      const uint code = ((b < 4 ? l.w[i].x : l.w[i].y) >> (8 * (b % 4))) & 0xFFu;
      v[b] = tf_dsv41_mma_e4m3(code) * sc;
    }
    threadgroup float4* dst = (threadgroup float4*)(wt + (q / 4) * TF_LD + (q % 4) * 8);
    dst[0] = float4(v[0], v[1], v[2], v[3]);
    dst[1] = float4(v[4], v[5], v[6], v[7]);
  }
}

// One group root a cell: four 8-wide MMA steps in k order from zero, X rows (left) times the staged weights (right).
inline void tf_dsv41_mma_root(thread float (&cur)[8], const threadgroup float* wt, thread const TfDsv41MmaLoad& l) {
  simdgroup_matrix<float, 8, 8> acc[4];
  for (uint j = 0; j < 4; ++j) acc[j] = simdgroup_matrix<float, 8, 8>(0.0f);
  for (uint s = 0; s < 4; ++s) {
    simdgroup_matrix<float, 8, 8> xa;
    xa.thread_elements()[0] = tf_dsv41_mma_x(l.x[s] & 0xFFFFu);
    xa.thread_elements()[1] = tf_dsv41_mma_x(l.x[s] >> 16);
    for (uint j = 0; j < 4; ++j) {
      simdgroup_matrix<float, 8, 8> wb;
      simdgroup_load(wb, wt + 8 * j * TF_LD + 8 * s, TF_LD, ulong2(0, 0), true);
      simdgroup_multiply_accumulate(acc[j], xa, wb, acc[j]);
    }
  }
  for (uint j = 0; j < 4; ++j)
    for (uint e = 0; e < 2; ++e) {
      const float v = acc[j].thread_elements()[e];
      cur[j * 2 + e] = TF_CTL == 3 ? float(half(v)) : v;
    }
}

// TF_RS simdgroups of 8 rows x 32 columns; group g+1 is read while g multiplies; roots enter the zero-padded pair tree.
template <typename T>
[[kernel, max_total_threads_per_threadgroup(TF_T)]] void tf_dsv41_fp8_mma(
    const device ushort* X [[buffer(0)]], const device uchar* W [[buffer(1)]],
    const device uchar* S [[buffer(2)]], device T* Y [[buffer(3)]],
    constant TfDsv41Fp8MmaArgs& a [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float wt[2][32 * TF_LD];
  const uint qid = lane / 4;
  const uint fm = (qid & 4) + ((lane / 2) % 4);
  const uint fn = (qid & 2) * 2 + (lane % 2) * 2;
  const uint n0 = tg.x * 32, r = (tg.y * TF_RS + sg) * 8 + fm;
  const uint groups = a.K / 32;
  uint padded = 1, levels = 0;
  while (padded < groups) { padded *= 2; ++levels; }
  const device uchar* Wn = W + ulong(n0) * a.K;
  const device uchar* Sn = S + ulong(tg.x) * groups;
  const device ushort* Xr = X + ulong(min(r, a.R - 1)) * a.K;
  const bool live = r < a.R;
  TfDsv41MmaLoad ld, nx;
  tf_dsv41_mma_fetch(ld, Wn, Sn, Xr, live, a.K, 0, tid, fn);
  tf_dsv41_mma_stage(wt[0], ld, tid);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float stack[TF_LV][8];
  float chain[8];
  for (uint e = 0; e < 8; ++e) chain[e] = 0.0f;
  for (uint g = 0; g < padded; ++g) {
    float cur[8];
    for (uint e = 0; e < 8; ++e) cur[e] = 0.0f;
    if (g < groups) {
      if (g + 1 < groups) tf_dsv41_mma_fetch(nx, Wn, Sn, Xr, live, a.K, g + 1, tid, fn);
      tf_dsv41_mma_root(cur, wt[g & 1], ld);
      if (g + 1 < groups) tf_dsv41_mma_stage(wt[(g + 1) & 1], nx, tid);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      ld = nx;
    }
    if (TF_CTL == 2) {
      for (uint e = 0; e < 8; ++e) chain[e] += cur[e];
      continue;
    }
    uint lv = 0;
    while (lv < levels && ((g >> lv) & 1u)) {
      for (uint e = 0; e < 8; ++e) cur[e] = stack[lv][e] + cur[e];
      ++lv;
    }
    for (uint e = 0; e < 8; ++e) stack[lv][e] = cur[e];
  }
  for (uint j = 0; j < 4; ++j)
    for (uint d = 0; d < 2; ++d)
      if (live) Y[ulong(r) * a.N + n0 + 8 * j + fn + d] = T(TF_CTL == 2 ? chain[j * 2 + d] : stack[levels][j * 2 + d]);
  if (TF_CTL == 4 && tg.x == 0 && tg.y == 0 && tid == 0) Y[ulong(a.R) * a.N] = T(1.0f);
}

template [[host_name("tf_dsv41_fp8_mma_f32")]] [[kernel]] decltype(tf_dsv41_fp8_mma<float>) tf_dsv41_fp8_mma<float>;
template [[host_name("tf_dsv41_fp8_mma_bf16")]] [[kernel]] decltype(tf_dsv41_fp8_mma<bfloat>) tf_dsv41_fp8_mma<bfloat>;
