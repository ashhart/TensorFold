// DeepSeek V4.1 dense FP8, 2-16 BF16 rows on simdgroup MMA: 8 columns a group, K split over simdgroups, one tree.
#include <metal_stdlib>
using namespace metal;

#ifndef TF_CTL
#define TF_CTL 0
#endif

struct TfDsv41Fp8TileArgs {
  uint K, N, R, groups, per, sgs;
  long xs, ws, ss, ys; // byte strides between the launches of a batch (grid y)
};

#define TF_TLD 36  // staged row stride of one column's 32 decoded K values (floats)
#define TF_TLV 10  // pair-tree levels a simdgroup's slice holds (up to 512 groups)

inline float tf_dsv41_tile_e4m3(uint code) {
  const float v = float(as_type<half>(ushort((code & 0x7Fu) << 7))) * 256.0f;
  const float s = (code & 0x80u) ? -v : v;
  return (code & 0x7Fu) == 0x7Fu ? NAN : s;
}

inline float tf_dsv41_tile_e8m0(uint code) {
  if (code == 255u) return NAN;
  return as_type<float>(code == 0u ? 0x00400000u : code << 23);
}

inline float tf_dsv41_tile_x(uint bits) {
  const float v = as_type<float>(bits << 16);
  return (TF_CTL == 1 && !isfinite(v)) ? 0.0f : v;
}

// Two codes of a word (bytes b, b + 2) as a half2 holding value / 256, sign included (the row kernel's fast path).
inline half2 tf_dsv41_tile_h2(uint word) {
  return as_type<half2>(((word & 0x007F007Fu) << 7) | ((word & 0x00800080u) << 8));
}

// Eight weights of one column; a NaN code (0x7F, 0xFF) or a scale past 2^120 takes the per-code path, the same values.
inline void tf_dsv41_tile_w8(uint2 q, uint code, thread float (&v)[8]) {
  const uint2 mag = q & 0x7F7F7F7Fu;
  if (code > 247u || any(((mag + 0x01010101u) & 0x80808080u) != 0u)) {
    const float sc = tf_dsv41_tile_e8m0(code);
    for (uint b = 0; b < 8; ++b) v[b] = tf_dsv41_tile_e4m3(((b < 4 ? q.x : q.y) >> (8 * (b % 4))) & 0xFFu) * sc;
    return;
  }
  const float x256 = tf_dsv41_tile_e8m0(code) * 256.0f;
  for (uint j = 0; j < 2; ++j) {
    const half2 a = tf_dsv41_tile_h2(q[j]), b = tf_dsv41_tile_h2(q[j] >> 8);
    v[4 * j] = float(a.x) * x256;
    v[4 * j + 1] = float(b.x) * x256;
    v[4 * j + 2] = float(a.y) * x256;
    v[4 * j + 3] = float(b.y) * x256;
  }
}

// Simdgroup sg: groups [sg * per, +per) of 8 * NJ columns, each from zero in 4 MMA steps, roots in one pair tree.
template <uint RT, uint NJ, typename T>
[[kernel, max_total_threads_per_threadgroup(512)]] void tf_dsv41_fp8_tile(
    const device uchar* X0 [[buffer(0)]], const device uchar* W0 [[buffer(1)]],
    const device uchar* S0 [[buffer(2)]], device uchar* Y0 [[buffer(3)]],
    constant TfDsv41Fp8TileArgs& a [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
  constexpr uint CB = 8 * NJ;
  constexpr uint LQ = 32 / CB;  // lanes a column's 32 bytes are read by
  threadgroup float tgm[128 * TF_TLD];
  const device ushort* X = (const device ushort*)(X0 + long(tg.y) * a.xs);
  const device uchar* W = W0 + long(tg.y) * a.ws;
  const device uchar* S = S0 + long(tg.y) * a.ss;
  device T* Y = (device T*)(Y0 + long(tg.y) * a.ys);
  const uint qid = lane / 4;
  const uint fm = (qid & 4) + ((lane / 2) % 4);
  const uint fn = (qid & 2) * 2 + (lane % 2) * 2;
  const uint n0 = tg.x * CB;
  threadgroup float* wt = tgm + sg * CB * TF_TLD;
  threadgroup float4* dst = (threadgroup float4*)(wt + (lane / LQ) * TF_TLD + (lane % LQ) * (32 / LQ));
  const device uchar* Wl = W + ulong(n0 + lane / LQ) * a.K + (lane % LQ) * (32 / LQ);
  const device uchar* Sn = S + ulong(n0 / 32) * a.groups;
  const device ushort* xr[RT];
  bool live[RT];
  for (uint t = 0; t < RT; ++t) {
    const uint r = t * 8 + fm;
    live[t] = r < a.R;
    xr[t] = X + ulong(min(r, a.R - 1)) * a.K + fn;
  }
  uint levels = 0;
  while ((1u << levels) < a.per) ++levels;
  const uint g0 = sg * a.per;
  const uint end = min(g0 + a.per, a.groups);
  float stack[TF_TLV][RT][NJ][2];
  float chain[RT][NJ][2];
  for (uint t = 0; t < RT; ++t)
    for (uint j = 0; j < NJ; ++j) chain[t][j][0] = chain[t][j][1] = 0.0f;
  uint2 q[NJ];
  uint code = 0;
  for (uint h = 0; h < NJ; ++h) q[h] = g0 < end ? *(const device uint2*)(Wl + g0 * 32 + 8 * h) : uint2(0);
  if (g0 < end) code = Sn[g0];
  for (uint i = 0; i < a.per; ++i) {
    const uint g = g0 + i;
    float cur[RT][NJ][2];
    for (uint t = 0; t < RT; ++t)
      for (uint j = 0; j < NJ; ++j) cur[t][j][0] = cur[t][j][1] = 0.0f;
    if (g < end) {
      for (uint h = 0; h < NJ; ++h) {
        float v[8];
        tf_dsv41_tile_w8(q[h], code, v);
        dst[2 * h] = float4(v[0], v[1], v[2], v[3]);
        dst[2 * h + 1] = float4(v[4], v[5], v[6], v[7]);
      }
      if (g + 1 < end) {
        for (uint h = 0; h < NJ; ++h) q[h] = *(const device uint2*)(Wl + (g + 1) * 32 + 8 * h);
        code = Sn[g + 1];
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
      simdgroup_matrix<float, 8, 8> acc[RT][NJ];
      for (uint t = 0; t < RT; ++t)
        for (uint j = 0; j < NJ; ++j) acc[t][j] = simdgroup_matrix<float, 8, 8>(0.0f);
      for (uint s = 0; s < 4; ++s) {
        simdgroup_matrix<float, 8, 8> wb[NJ];
        for (uint j = 0; j < NJ; ++j) simdgroup_load(wb[j], wt + 8 * j * TF_TLD + 8 * s, TF_TLD, ulong2(0, 0), true);
        for (uint t = 0; t < RT; ++t) {
          const uint xb = live[t] ? *(const device uint*)(xr[t] + g * 32 + 8 * s) : 0u;
          simdgroup_matrix<float, 8, 8> xa;
          xa.thread_elements()[0] = tf_dsv41_tile_x(xb & 0xFFFFu);
          xa.thread_elements()[1] = tf_dsv41_tile_x(xb >> 16);
          for (uint j = 0; j < NJ; ++j) simdgroup_multiply_accumulate(acc[t][j], xa, wb[j], acc[t][j]);
        }
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
      for (uint t = 0; t < RT; ++t)
        for (uint j = 0; j < NJ; ++j)
          for (uint e = 0; e < 2; ++e) {
            const float r = acc[t][j].thread_elements()[e];
            cur[t][j][e] = TF_CTL == 3 ? float(half(r)) : r;
          }
    }
    if (TF_CTL == 2) {
      for (uint t = 0; t < RT; ++t)
        for (uint j = 0; j < NJ; ++j) chain[t][j][0] += cur[t][j][0], chain[t][j][1] += cur[t][j][1];
      continue;
    }
    uint lv = 0;
    while (lv < levels && ((i >> lv) & 1u)) {
      for (uint t = 0; t < RT; ++t)
        for (uint j = 0; j < NJ; ++j)
          for (uint e = 0; e < 2; ++e) cur[t][j][e] = stack[lv][t][j][e] + cur[t][j][e];
      ++lv;
    }
    for (uint t = 0; t < RT; ++t)
      for (uint j = 0; j < NJ; ++j) stack[lv][t][j][0] = cur[t][j][0], stack[lv][t][j][1] = cur[t][j][1];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // roots [sg][column][row]: the staging blocks are free once every simdgroup has passed the barrier
  threadgroup float* roots = tgm;
  for (uint t = 0; t < RT; ++t)
    for (uint j = 0; j < NJ; ++j)
      for (uint e = 0; e < 2; ++e)
        roots[(sg * CB + 8 * j + fn + e) * (RT * 8) + t * 8 + fm] = TF_CTL == 2 ? chain[t][j][e] : stack[levels][t][j][e];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint o = tid; o < CB * RT * 8; o += 32 * a.sgs) {
    const uint col = o / (RT * 8), r = o % (RT * 8);
    if (r >= a.R) continue;
    float v[16];
    for (uint k = 0; k < 16; ++k) v[k] = k < a.sgs ? roots[(k * CB + col) * (RT * 8) + r] : 0.0f;
    if (TF_CTL == 2) {
      for (uint k = 1; k < a.sgs; ++k) v[0] += v[k];
    } else {
      for (uint s = 1; s < 16; s *= 2)
        for (uint i = 0; i < 16; i += 2 * s) v[i] += v[i + s];
    }
    Y[ulong(r) * a.N + n0 + col] = T(v[0]);
  }
  if (TF_CTL == 4 && tg.x == 0 && tg.y == 0 && tid == 0) Y[ulong(a.R) * a.N] = T(1.0f);
}

#define TF_TILE(RT, NJ, T, NAME) \
  template [[host_name(NAME)]] [[kernel]] decltype(tf_dsv41_fp8_tile<RT, NJ, T>) tf_dsv41_fp8_tile<RT, NJ, T>;
TF_TILE(1, 1, float, "tf_dsv41_fp8_tile_r8_c8_f32")
TF_TILE(1, 1, bfloat, "tf_dsv41_fp8_tile_r8_c8_bf16")
TF_TILE(2, 1, float, "tf_dsv41_fp8_tile_r16_c8_f32")
TF_TILE(2, 1, bfloat, "tf_dsv41_fp8_tile_r16_c8_bf16")
TF_TILE(1, 2, float, "tf_dsv41_fp8_tile_r8_c16_f32")
TF_TILE(1, 2, bfloat, "tf_dsv41_fp8_tile_r8_c16_bf16")
TF_TILE(2, 2, float, "tf_dsv41_fp8_tile_r16_c16_f32")
TF_TILE(2, 2, bfloat, "tf_dsv41_fp8_tile_r16_c16_bf16")
