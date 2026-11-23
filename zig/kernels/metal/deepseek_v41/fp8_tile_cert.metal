// DeepSeek V4.1 opt-in FP64-certified BF16 dense FP8 tile (1-16 verify/decode rows): exact double-float sums, BF16 RNE.
#include <metal_stdlib>
using namespace metal;

#ifndef TF_CTL
#define TF_CTL 0
#endif

struct TfDsv41Fp8TileArgs {
  uint K, N, R, groups, per, sgs;
  long xs, ws, ss, ys;
};

#define TF_TLD 36

struct Fp8TileDD {
  float hi;
  float lo;
};

inline void fp8_tile_neumaier(thread float& s, thread float& c, float v) {
  if (!isfinite(v) || !isfinite(s)) {
    s = s + v;
    c = 0.0f;
    return;
  }
  const float t = s + v;
  const float err = (fabs(s) >= fabs(v)) ? ((s - t) + v) : ((v - t) + s);
  c = c + err;
  s = t;
}

inline Fp8TileDD fp8_tile_two_sum(float a, float b) {
  const float s = a + b;
  if (!isfinite(s)) return {s, 0.0f};
  const float bb = s - a;
  return {s, (a - (s - bb)) + (b - bb)};
}

inline Fp8TileDD fp8_tile_dd_add(Fp8TileDD a, Fp8TileDD b) {
  if (!isfinite(a.hi) || !isfinite(b.hi)) return {a.hi + b.hi, 0.0f};
  Fp8TileDD s = fp8_tile_two_sum(a.hi, b.hi);
  if (!isfinite(s.hi)) return {s.hi, 0.0f};
  const Fp8TileDD t = fp8_tile_two_sum(a.lo, b.lo);
  s.lo = s.lo + t.hi;
  s = fp8_tile_two_sum(s.hi, s.lo);
  s.lo = s.lo + t.lo;
  return fp8_tile_two_sum(s.hi, s.lo);
}

template <typename T>
inline T fp8_tile_dd_out(Fp8TileDD x);
template <>
inline float fp8_tile_dd_out<float>(Fp8TileDD x) {
  return x.hi + x.lo;
}
template <>
inline bfloat fp8_tile_dd_out<bfloat>(Fp8TileDD x) {
  if (!isfinite(x.hi)) return bfloat(x.hi);
  const Fp8TileDD s = fp8_tile_two_sum(x.hi, x.lo);
  if (!isfinite(s.hi)) return bfloat(s.hi);
  const uint u = as_type<uint>(s.hi);
  if ((u & 0xFFFFu) == 0x8000u && s.lo != 0.0f) {
    const bool same_sign = ((u >> 31) == (as_type<uint>(s.lo) >> 31));
    const ushort trunc_bf = ushort(u >> 16);
    return as_type<bfloat>(same_sign ? ushort(trunc_bf + 1u) : trunc_bf);
  }
  const uint lsb = (u >> 16) & 1u;
  return as_type<bfloat>(ushort((u + 0x7FFFu + lsb) >> 16));
}

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

inline half2 tf_dsv41_tile_h2(uint word) {
  return as_type<half2>(((word & 0x007F007Fu) << 7) | ((word & 0x00800080u) << 8));
}

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

// Reuses decoded 32-element weight groups across 1-16 rows in exact 48-bit dd; fits the 18432-byte TG budget.
template <uint RT, uint NJ, typename T>
[[kernel, max_total_threads_per_threadgroup(512)]] void tf_dsv41_fp8_tile(
    const device uchar* X0 [[buffer(0)]], const device uchar* W0 [[buffer(1)]],
    const device uchar* S0 [[buffer(2)]], device uchar* Y0 [[buffer(3)]],
    constant TfDsv41Fp8TileArgs& a [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
  constexpr uint CB = 8 * NJ;
  constexpr uint LQ = 32 / CB;
  constexpr uint R_PER = (RT * 8 * CB) / 32;
  threadgroup float tgm[128 * TF_TLD];
  const device uint2* X2 = (const device uint2*)(X0 + long(tg.y) * a.xs);
  const device uchar* W = W0 + long(tg.y) * a.ws;
  const device uchar* S = S0 + long(tg.y) * a.ss;
  device T* Y = (device T*)(Y0 + long(tg.y) * a.ys);
  const uint n0 = tg.x * CB;
  threadgroup float* wt = tgm + sg * CB * TF_TLD;
  threadgroup float4* dst = (threadgroup float4*)(wt + (lane / LQ) * TF_TLD + (lane % LQ) * (32 / LQ));
  const device uchar* Wl = W + ulong(n0 + lane / LQ) * a.K + (lane % LQ) * (32 / LQ);
  const device uchar* Sn = S + ulong(n0 / 32) * a.groups;
  const uint col_l = lane % CB;
  const uint r_base = (lane / CB) * R_PER;
  const uint Ge = (TF_CTL == 2 && a.groups > 1u) ? (a.groups - 1u) : a.groups;
  const uint g0 = sg * a.per;
  const uint end = min(g0 + a.per, Ge);
  Fp8TileDD acc[R_PER];
  for (uint dr = 0; dr < R_PER; ++dr) acc[dr] = {0.0f, 0.0f};
  uint2 q[NJ];
  uint code = 0;
  for (uint h = 0; h < NJ; ++h) q[h] = g0 < end ? *(const device uint2*)(Wl + g0 * 32 + 8 * h) : uint2(0);
  if (g0 < end) code = Sn[g0];
  for (uint g = g0; g < end; ++g) {
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
    threadgroup const float* wc = wt + col_l * TF_TLD;
    float ghi[R_PER], glo[R_PER];
    for (uint dr = 0; dr < R_PER; ++dr) {
      ghi[dr] = 0.0f;
      glo[dr] = 0.0f;
    }
    for (uint k4 = 0; k4 < 8u; ++k4) {
      const float w0 = wc[4u * k4 + 0u];
      const float w1 = wc[4u * k4 + 1u];
      const float w2 = wc[4u * k4 + 2u];
      const float w3 = wc[4u * k4 + 3u];
      for (uint dr = 0; dr < R_PER; ++dr) {
        const uint r = r_base + dr;
        if (r >= a.R) break;
        const uint2 xq = X2[ulong(r) * (a.K / 4u) + 8u * g + k4];
        fp8_tile_neumaier(ghi[dr], glo[dr], tf_dsv41_tile_x(xq.x & 0xFFFFu) * w0);
        fp8_tile_neumaier(ghi[dr], glo[dr], tf_dsv41_tile_x(xq.x >> 16) * w1);
        fp8_tile_neumaier(ghi[dr], glo[dr], tf_dsv41_tile_x(xq.y & 0xFFFFu) * w2);
        fp8_tile_neumaier(ghi[dr], glo[dr], tf_dsv41_tile_x(xq.y >> 16) * w3);
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint dr = 0; dr < R_PER; ++dr) {
      if (r_base + dr >= a.R) break;
      if (TF_CTL == 3) {
        ghi[dr] = float(half(ghi[dr]));
        glo[dr] = 0.0f;
      }
      acc[dr] = fp8_tile_dd_add(acc[dr], fp8_tile_two_sum(ghi[dr], glo[dr]));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Reuse tgm (18432 bytes >= 128 * 16 * 8 = 16384 bytes) for float2 dd roots [sg][col][row].
  threadgroup float2* roots = (threadgroup float2*)tgm;
  for (uint dr = 0; dr < R_PER; ++dr) {
    roots[(sg * CB + col_l) * (RT * 8u) + r_base + dr] = float2(acc[dr].hi, acc[dr].lo);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint o = tid; o < CB * RT * 8u; o += 32u * a.sgs) {
    const uint col = o / (RT * 8u), r = o % (RT * 8u);
    if (r >= a.R) continue;
    Fp8TileDD tot = {0.0f, 0.0f};
    for (uint k = 0; k < a.sgs; ++k) {
      const float2 rv = roots[(k * CB + col) * (RT * 8u) + r];
      tot = fp8_tile_dd_add(tot, {rv.x, rv.y});
    }
    Y[ulong(r) * a.N + n0 + col] = fp8_tile_dd_out<T>(tot);
  }
  if (TF_CTL == 4 && tg.x == 0u && tg.y == 0u && tid == 0u) Y[ulong(a.R) * a.N] = T(1.0f);
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
