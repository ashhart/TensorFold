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
#define TF_RS 4   // simdgroups a threadgroup, 8 rows each: they share the decoded 32x32 weight blocks of two K groups
#endif
#define TF_T (TF_RS * 32)
#define TF_CPT (1024 / TF_T)  // weight codes a thread stages a group
#define TF_LV 9
#define TF_LD 36  // threadgroup row stride of a decoded weight block (floats)
#define TF_BLK (2 * 32 * TF_LD)  // one buffer: two groups' decoded blocks

inline float tf_dsv41_mma_e4m3(uint code) {
  const float v = float(as_type<half>(ushort((code & 0x7Fu) << 7))) * 256.0f;
  const float s = (code & 0x80u) ? -v : v;
  return (code & 0x7Fu) == 0x7Fu ? NAN : s;
}

inline float tf_dsv41_mma_e8m0(uint code) {
  if (code == 255u) return NAN;
  return as_type<float>(code == 0u ? 0x00400000u : code << 23);
}

// A bf16 already in a float's high half.
inline float tf_dsv41_mma_x(uint bits) {
  const float v = as_type<float>(bits);
  return (TF_CTL == 1 && !isfinite(v)) ? 0.0f : v;
}

// What a thread reads for two K groups before they are needed: weight codes, block scales, two X values a step.
struct TfDsv41MmaLoad {
  uint2 w[2];
  uint sc[2];
  uint x[2][4];
};

// The thread's fixed places: its weight codes (Wt), the column scales (Sn), its X row (Xt) and its staging slot.
struct TfDsv41MmaAt {
  const device uchar* Wt;
  const device uchar* Sn;
  const device ushort* Xt;
  bool live;
  uint groups, slot;
};

// Groups g0, g0 + 1 (those that exist): weight codes of the 32x32 blocks and the X row at k 8s + fn, 8s + fn + 1.
inline void tf_dsv41_mma_fetch(thread TfDsv41MmaLoad& l, thread const TfDsv41MmaAt& p, uint g0) {
  for (uint i = 0; i < 2; ++i) {
    const uint g = g0 + i;
    if (g >= p.groups) break;
#if TF_CPT == 8
    l.w[i] = *(const device uint2*)(p.Wt + g * 32);
#else
    l.w[i] = uint2(*(const device uint*)(p.Wt + g * 32), 0u);
#endif
    l.sc[i] = p.Sn[g];
    for (uint s = 0; s < 4; ++s) l.x[i][s] = p.live ? *(const device uint*)(p.Xt + g * 32 + 8 * s) : 0u;
  }
}

// Two codes of a word (bytes b, b + 2) as a half2 holding value / 256, sign included.
inline half2 tf_dsv41_mma_h2(uint word) {
  return as_type<half2>(((word & 0x007F007Fu) << 7) | ((word & 0x00800080u) << 8));
}

// One group's weight codes, decoded and scaled, into its block wt[column][k] at this thread's slot.
inline void tf_dsv41_mma_stage1(threadgroup float* blk, uint2 w, uint scode) {
  float v[8];
  const uint2 mag = w & 0x7F7F7F7Fu;
  // a NaN code or a scale whose 256 multiple overflows takes the per-code path
  if (scode > 246u || any(((mag + 0x01010101u) & 0x80808080u) != 0u)) {
    const float sc = tf_dsv41_mma_e8m0(scode);
    for (uint b = 0; b < TF_CPT; ++b) v[b] = tf_dsv41_mma_e4m3(((b < 4 ? w.x : w.y) >> (8 * (b % 4))) & 0xFFu) * sc;
  } else {
    // no NaN code and a finite 256 multiple of the scale: the same values two codes at a time
    const float x256 = tf_dsv41_mma_e8m0(scode) * 256.0f;
    for (uint j = 0; j < TF_CPT / 4; ++j) {
      const uint word = j == 0 ? w.x : w.y;
      const half2 p = tf_dsv41_mma_h2(word), q = tf_dsv41_mma_h2(word >> 8);
      v[4 * j] = float(p.x) * x256;
      v[4 * j + 1] = float(q.x) * x256;
      v[4 * j + 2] = float(p.y) * x256;
      v[4 * j + 3] = float(q.y) * x256;
    }
  }
  threadgroup float4* dst = (threadgroup float4*)blk;
  for (uint i = 0; i < TF_CPT / 4; ++i) dst[i] = float4(v[4 * i], v[4 * i + 1], v[4 * i + 2], v[4 * i + 3]);
}

// One group root a cell: four 8-wide MMA steps in k order from zero, X rows (left) times the staged weights (right).
inline void tf_dsv41_mma_root(thread float4 (&cur)[2], const threadgroup float* wt, thread const uint (&x)[4]) {
  simdgroup_matrix<float, 8, 8> acc[4];
  for (uint j = 0; j < 4; ++j) acc[j] = simdgroup_matrix<float, 8, 8>(0.0f);
  for (uint s = 0; s < 4; ++s) {
    simdgroup_matrix<float, 8, 8> xa;
    xa.thread_elements()[0] = tf_dsv41_mma_x(x[s] << 16);
    xa.thread_elements()[1] = tf_dsv41_mma_x(x[s] & 0xFFFF0000u);
    for (uint j = 0; j < 4; ++j) {
      simdgroup_matrix<float, 8, 8> wb;
      simdgroup_load(wb, wt + 8 * j * TF_LD + 8 * s, TF_LD, ulong2(0, 0), true);
      simdgroup_multiply_accumulate(acc[j], xa, wb, acc[j]);
    }
  }
  for (uint h = 0; h < 2; ++h) {
    const float4 v = float4(acc[2 * h].thread_elements()[0], acc[2 * h].thread_elements()[1],
                            acc[2 * h + 1].thread_elements()[0], acc[2 * h + 1].thread_elements()[1]);
    cur[h] = TF_CTL == 3 ? float4(half4(v)) : v;
  }
}

// Groups g0, g0 + 1 from buffer b & 1 into r0, r1, while the next two are read and then staged in the other buffer.
inline void tf_dsv41_mma_pair(thread float4 (&r0)[2], thread float4 (&r1)[2], thread const TfDsv41MmaLoad& ld,
                              thread TfDsv41MmaLoad& nx, threadgroup float (&wt)[2][TF_BLK],
                              thread const TfDsv41MmaAt& p, uint g0, uint b) {
  const bool more = g0 + 2 < p.groups;
  if (more) tf_dsv41_mma_fetch(nx, p, g0 + 2);
  tf_dsv41_mma_root(r0, wt[b & 1], ld.x[0]);
  if (g0 + 1 < p.groups) tf_dsv41_mma_root(r1, wt[b & 1] + 32 * TF_LD, ld.x[1]);
  if (more) {
    for (uint i = 0; i < 2; ++i)
      if (g0 + 2 + i < p.groups) tf_dsv41_mma_stage1(wt[(b + 1) & 1] + i * 32 * TF_LD + p.slot, nx.w[i], nx.sc[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// Pending left nodes of levels 0-2; higher levels sit in a separate indexed array, so these three stay in registers.
struct TfDsv41MmaTree {
  float4 s0[2], s1[2], s2[2];
};

// A level-2 node with index q (groups 4q to 4q + 3) joins the tree.
inline void tf_dsv41_mma_push2(thread TfDsv41MmaTree& t, thread float4 (&up)[TF_LV][2], thread float4 (&cur)[2],
                               uint q) {
  if (!(q & 1u)) {
    t.s2[0] = cur[0], t.s2[1] = cur[1];
    return;
  }
  cur[0] = t.s2[0] + cur[0], cur[1] = t.s2[1] + cur[1];
  uint lv = 3;
  for (; (q >> (lv - 2)) & 1u; ++lv) cur[0] = up[lv][0] + cur[0], cur[1] = up[lv][1] + cur[1];
  up[lv][0] = cur[0], up[lv][1] = cur[1];
}

inline float4 tf_dsv41_mma_at(thread const TfDsv41MmaTree& t, thread const float4 (&up)[TF_LV][2], uint lv, uint h) {
  return lv == 0 ? t.s0[h] : lv == 1 ? t.s1[h] : lv == 2 ? t.s2[h] : up[lv][h];
}

// The root once the real groups are in: the zero roots up to 2^levels add +0 wherever the full tree adds them.
inline float4 tf_dsv41_mma_close(thread const TfDsv41MmaTree& t, thread const float4 (&up)[TF_LV][2], uint groups,
                                 uint levels, uint h) {
  if (groups == (1u << levels)) return tf_dsv41_mma_at(t, up, levels, h);
  float4 z = float4(0.0f);
  bool open = false;
  for (uint lv = 0; lv < levels; ++lv) {
    if ((groups >> lv) & 1u) {
      z = tf_dsv41_mma_at(t, up, lv, h) + z;
      open = true;
    } else if (open) {
      z = z + float4(0.0f);
    }
  }
  return z;
}

// TF_RS simdgroups of 8 rows x 32 columns; two groups are read and staged while the previous two multiply.
template <typename T>
[[kernel, max_total_threads_per_threadgroup(TF_T)]] void tf_dsv41_fp8_mma(
    const device ushort* X [[buffer(0)]], const device uchar* W [[buffer(1)]],
    const device uchar* S [[buffer(2)]], device T* Y [[buffer(3)]],
    constant TfDsv41Fp8MmaArgs& a [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float wt[2][TF_BLK];
  const uint qid = lane / 4;
  const uint fm = (qid & 4) + ((lane / 2) % 4);
  const uint fn = (qid & 2) * 2 + (lane % 2) * 2;
  const uint n0 = tg.x * 32, r = (tg.y * TF_RS + sg) * 8 + fm;
  const uint groups = a.K / 32, e0 = tid * TF_CPT;
  uint levels = 0;
  while ((1u << levels) < groups) ++levels;
  const TfDsv41MmaAt p = {W + ulong(n0 + e0 / 32) * a.K + e0 % 32, S + ulong(tg.x) * groups,
                          X + ulong(min(r, a.R - 1)) * a.K + fn, r < a.R, groups, (e0 / 32) * TF_LD + e0 % 32};
  TfDsv41MmaLoad ld, nx;
  tf_dsv41_mma_fetch(ld, p, 0);
  for (uint i = 0; i < 2 && i < groups; ++i) tf_dsv41_mma_stage1(wt[0] + i * 32 * TF_LD + p.slot, ld.w[i], ld.sc[i]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  TfDsv41MmaTree t;
  float4 up[TF_LV][2];
  float4 chain[2] = {float4(0.0f), float4(0.0f)};
  // four groups a pass, the two pairs swapping read registers; their roots close levels 0 and 1 in place
  for (uint g0 = 0, q = 0; g0 < groups; g0 += 4, ++q) {
    float4 cur[2];
    tf_dsv41_mma_pair(t.s0, cur, ld, nx, wt, p, g0, 2 * q);
    if (TF_CTL == 2) {
      chain[0] += t.s0[0], chain[1] += t.s0[1];
      if (g0 + 1 < groups) chain[0] += cur[0], chain[1] += cur[1];
    } else if (g0 + 1 < groups) {
      t.s1[0] = t.s0[0] + cur[0], t.s1[1] = t.s0[1] + cur[1];
    }
    if (g0 + 2 >= groups) break;
    tf_dsv41_mma_pair(t.s0, cur, nx, ld, wt, p, g0 + 2, 2 * q + 1);
    if (TF_CTL == 2) {
      chain[0] += t.s0[0], chain[1] += t.s0[1];
      if (g0 + 3 < groups) chain[0] += cur[0], chain[1] += cur[1];
    } else if (g0 + 3 < groups) {
      cur[0] = t.s0[0] + cur[0], cur[1] = t.s0[1] + cur[1];
      cur[0] = t.s1[0] + cur[0], cur[1] = t.s1[1] + cur[1];
      tf_dsv41_mma_push2(t, up, cur, q);
    }
  }
  float4 out[2];
  for (uint h = 0; h < 2; ++h) out[h] = TF_CTL == 2 ? chain[h] : tf_dsv41_mma_close(t, up, groups, levels, h);
  if (p.live)
    for (uint j = 0; j < 4; ++j)
      for (uint d = 0; d < 2; ++d) Y[ulong(r) * a.N + n0 + 8 * j + fn + d] = T(out[j / 2][(j % 2) * 2 + d]);
  if (TF_CTL == 4 && tg.x == 0 && tg.y == 0 && tid == 0) Y[ulong(a.R) * a.N] = T(1.0f);
}

template [[host_name("tf_dsv41_fp8_mma_f32")]] [[kernel]] decltype(tf_dsv41_fp8_mma<float>) tf_dsv41_fp8_mma<float>;
template [[host_name("tf_dsv41_fp8_mma_bf16")]] [[kernel]] decltype(tf_dsv41_fp8_mma<bfloat>) tf_dsv41_fp8_mma<bfloat>;
