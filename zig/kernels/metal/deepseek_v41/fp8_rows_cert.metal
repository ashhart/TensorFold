// DeepSeek V4.1 opt-in FP64-certified BF16 dense FP8 decode rows (1-16 rows): exact double-float sums and BF16 RNE.
#include <metal_stdlib>
using namespace metal;

#ifndef TF_CTL
#define TF_CTL 0
#endif

struct TfDsv41Fp8Args {
  uint K, N, R, groups, tiles, blocks;
};

struct Fp8DD {
  float hi;
  float lo;
};

inline void fp8_neumaier(thread float& s, thread float& c, float v) {
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

inline Fp8DD fp8_two_sum(float a, float b) {
  const float s = a + b;
  if (!isfinite(s)) return {s, 0.0f};
  const float bb = s - a;
  return {s, (a - (s - bb)) + (b - bb)};
}

inline Fp8DD fp8_dd_add(Fp8DD a, Fp8DD b) {
  if (!isfinite(a.hi) || !isfinite(b.hi)) return {a.hi + b.hi, 0.0f};
  Fp8DD s = fp8_two_sum(a.hi, b.hi);
  if (!isfinite(s.hi)) return {s.hi, 0.0f};
  const Fp8DD t = fp8_two_sum(a.lo, b.lo);
  s.lo = s.lo + t.hi;
  s = fp8_two_sum(s.hi, s.lo);
  s.lo = s.lo + t.lo;
  return fp8_two_sum(s.hi, s.lo);
}

template <typename T>
inline T fp8_dd_out(Fp8DD x);
template <>
inline float fp8_dd_out<float>(Fp8DD x) {
  return x.hi + x.lo;
}
template <>
inline bfloat fp8_dd_out<bfloat>(Fp8DD x) {
  if (!isfinite(x.hi)) return bfloat(x.hi);
  const Fp8DD s = fp8_two_sum(x.hi, x.lo);
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

inline float tf_dsv41_e4m3(uint code) {
  const float v = float(as_type<half>(ushort((code & 0x7Fu) << 7))) * 256.0f;
  const float s = (code & 0x80u) ? -v : v;
  return (code & 0x7Fu) == 0x7Fu ? NAN : s;
}

inline float tf_dsv41_e8m0(uint code) {
  if (code == 255u) return NAN;
  return as_type<float>(code == 0u ? 0x00400000u : code << 23);
}

inline float tf_dsv41_bf16(uint bits) {
  const float v = as_type<float>(bits << 16);
  return (TF_CTL == 1 && !isfinite(v)) ? 0.0f : v;
}

struct TfDsv41Scale {
  uint code;
  float x256;
};

inline TfDsv41Scale tf_dsv41_scale(uint code) {
  return TfDsv41Scale{code, tf_dsv41_e8m0(code) * 256.0f};
}

inline half2 tf_dsv41_h2(uint word) {
  return as_type<half2>(((word & 0x007F007Fu) << 7) | ((word & 0x00800080u) << 8));
}

inline void tf_dsv41_w16(uint4 raw, TfDsv41Scale s, thread float* w) {
  const uint4 mag = raw & 0x7F7F7F7Fu;
  if (s.code > 247u || any(((mag + 0x01010101u) & 0x80808080u) != 0u)) {
    const float scale = tf_dsv41_e8m0(s.code);
    for (uint j = 0; j < 4; ++j)
      for (uint b = 0; b < 4; ++b) w[4 * j + b] = tf_dsv41_e4m3((raw[j] >> (8 * b)) & 0xFFu) * scale;
    return;
  }
  for (uint j = 0; j < 4; ++j) {
    const half2 a = tf_dsv41_h2(raw[j]), b = tf_dsv41_h2(raw[j] >> 8);
    w[4 * j] = float(a.x) * s.x256;
    w[4 * j + 1] = float(b.x) * s.x256;
    w[4 * j + 2] = float(a.y) * s.x256;
    w[4 * j + 3] = float(b.y) * s.x256;
  }
}

inline void tf_dsv41_x16(uint4 lo, uint4 hi, thread float* x) {
  for (uint j = 0; j < 4; ++j) {
    x[2 * j] = tf_dsv41_bf16(lo[j] & 0xFFFFu);
    x[2 * j + 1] = tf_dsv41_bf16(lo[j] >> 16);
    x[8 + 2 * j] = tf_dsv41_bf16(hi[j] & 0xFFFFu);
    x[8 + 2 * j + 1] = tf_dsv41_bf16(hi[j] >> 16);
  }
}

// Two-launch part kernel kept for API symmetry; whole kernels carry full 48-bit dd roots in threadgroup memory.
template <uint RT, uint NB, uint CL>
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_dsv41_fp8_part(
    const device uint4* X [[buffer(0)]], const device uint4* W [[buffer(1)]],
    const device uchar* S [[buffer(2)]], device float* P [[buffer(3)]],
    constant TfDsv41Fp8Args& a [[buffer(4)]],
    uint tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint GL = 32 / CL, blk = tg % a.blocks, tile = tg / a.blocks, c = lane / GL, gl = lane % GL, g = tile * GL + gl;
  const uint Ge = (TF_CTL == 2 && a.groups > 1u) ? (a.groups - 1u) : a.groups;
  float hi[NB][RT], lo[NB][RT];
  for (uint nb = 0; nb < NB; ++nb)
    for (uint r = 0; r < RT; ++r) {
      hi[nb][r] = 0.0f;
      lo[nb][r] = 0.0f;
    }
  if (g < Ge) {
    TfDsv41Scale sc[NB];
    for (uint nb = 0; nb < NB; ++nb) sc[nb] = tf_dsv41_scale(S[ulong((blk * 4 + c * NB + nb) / 32) * a.groups + g]);
    for (uint h = 0; h < 2; ++h) {
      float w[NB][16];
      for (uint nb = 0; nb < NB; ++nb) tf_dsv41_w16(W[ulong(blk * 4 + c * NB + nb) * (a.K / 16) + 2 * g + h], sc[nb], w[nb]);
      for (uint r = 0; r < RT; ++r) {
        if (r >= a.R) break;
        const ulong at = ulong(r) * (a.K / 8) + 4 * g + 2 * h;
        float x[16];
        tf_dsv41_x16(X[at], X[at + 1], x);
        for (uint i = 0; i < 16; ++i)
          for (uint nb = 0; nb < NB; ++nb) fp8_neumaier(hi[nb][r], lo[nb][r], x[i] * w[nb][i]);
      }
    }
  }
  for (uint nb = 0; nb < NB; ++nb) {
    for (uint r = 0; r < RT; ++r) {
      if (TF_CTL == 3) {
        hi[nb][r] = float(half(hi[nb][r]));
        lo[nb][r] = 0.0f;
      }
      Fp8DD v = fp8_two_sum(hi[nb][r], lo[nb][r]);
      for (uint m = 1; m < GL; m *= 2) v = fp8_dd_add(v, {simd_shuffle_xor(v.hi, m), simd_shuffle_xor(v.lo, m)});
      if (gl == 0 && r < a.R) P[(ulong(tile) * a.N + blk * 4 + c * NB + nb) * a.R + r] = v.hi + v.lo;
    }
  }
}

#define TF_PART(RT, NB, CL, NAME) \
  template [[host_name(NAME)]] [[kernel]] decltype(tf_dsv41_fp8_part<RT, NB, CL>) tf_dsv41_fp8_part<RT, NB, CL>;
TF_PART(1, 4, 1, "tf_dsv41_fp8_a_r1")
TF_PART(2, 4, 1, "tf_dsv41_fp8_a_r2")
TF_PART(4, 4, 1, "tf_dsv41_fp8_a_r4")
TF_PART(8, 4, 1, "tf_dsv41_fp8_a_r8")
TF_PART(16, 4, 1, "tf_dsv41_fp8_a_r16")
TF_PART(1, 1, 4, "tf_dsv41_fp8_b_r1")
TF_PART(2, 1, 4, "tf_dsv41_fp8_b_r2")
TF_PART(4, 1, 4, "tf_dsv41_fp8_b_r4")
TF_PART(8, 1, 4, "tf_dsv41_fp8_b_r8")
TF_PART(16, 1, 4, "tf_dsv41_fp8_b_r16")

template <typename T>
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_dsv41_fp8_finish(
    const device float* P [[buffer(0)]], device T* Y [[buffer(1)]], constant TfDsv41Fp8Args& a [[buffer(2)]],
    uint2 at [[thread_position_in_grid]]) {
  if (at.x >= a.N || at.y >= a.R) return;
  Fp8DD total = {0.0f, 0.0f};
  for (uint t = 0; t < a.tiles; ++t) total = fp8_dd_add(total, {P[(ulong(t) * a.N + at.x) * a.R + at.y], 0.0f});
  Y[ulong(at.y) * a.N + at.x] = fp8_dd_out<T>(total);
  if (TF_CTL == 4 && at.x == 0u && at.y == 0u) Y[ulong(a.R) * a.N] = T(1.0f);
}
template [[host_name("tf_dsv41_fp8_finish_f32")]] [[kernel]] decltype(tf_dsv41_fp8_finish<float>) tf_dsv41_fp8_finish<float>;
template [[host_name("tf_dsv41_fp8_finish_bf16")]] [[kernel]] decltype(tf_dsv41_fp8_finish<bfloat>) tf_dsv41_fp8_finish<bfloat>;

struct TfDsv41Fp8WArgs {
  uint K, N, R, groups, tiles, blocks;
  long xs, ws, ss, ys;
};

// One-launch exact double-float decode kernel: reduces rows in chunks of <= 4 so TG memory stays <= 8 KiB.
template <uint RT, uint NB, uint CL, typename T, uint TAG = 0>
[[kernel, max_total_threads_per_threadgroup(1024)]] void tf_dsv41_fp8_whole(
    const device uchar* X0 [[buffer(0)]], const device uchar* W0 [[buffer(1)]],
    const device uchar* S0 [[buffer(2)]], device uchar* Y0 [[buffer(3)]],
    constant TfDsv41Fp8WArgs& a [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint tile [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]]) {
  constexpr uint BW = CL * NB;
  constexpr uint RB = RT < 4 ? RT : 4;
  threadgroup float2 roots[32 * BW * RB];
  const device uint4* X = (const device uint4*)(X0 + long(tg.y) * a.xs);
  const device uint4* W = (const device uint4*)(W0 + long(tg.y) * a.ws);
  const device uchar* S = S0 + long(tg.y) * a.ss;
  device T* Y = (device T*)(Y0 + long(tg.y) * a.ys);
  const uint GL = 32 / CL, c = lane / GL, gl = lane % GL, g = tile * GL + gl, blk = tg.x;
  const uint Ge = (TF_CTL == 2 && a.groups > 1u) ? (a.groups - 1u) : a.groups;
  float hi[NB][RT], lo[NB][RT];
  for (uint nb = 0; nb < NB; ++nb)
    for (uint r = 0; r < RT; ++r) {
      hi[nb][r] = 0.0f;
      lo[nb][r] = 0.0f;
    }
  const uint gs = min(g, a.groups - 1);
  uint4 raw[NB][2];
  uint code[NB];
  for (uint nb = 0; nb < NB; ++nb) {
    const uint n = blk * BW + c * NB + nb;
    raw[nb][0] = W[ulong(n) * (a.K / 16) + 2 * gs];
    raw[nb][1] = W[ulong(n) * (a.K / 16) + 2 * gs + 1];
    code[nb] = S[ulong(n / 32) * a.groups + gs];
  }
  if (g < Ge) {
    for (uint h = 0; h < 2; ++h) {
      float w[NB][16];
      for (uint nb = 0; nb < NB; ++nb) tf_dsv41_w16(raw[nb][h], tf_dsv41_scale(code[nb]), w[nb]);
      for (uint r = 0; r < RT; ++r) {
        if (r >= a.R) break;
        const ulong at = ulong(r) * (a.K / 8) + 4 * g + 2 * h;
        float x[16];
        tf_dsv41_x16(X[at], X[at + 1], x);
        for (uint i = 0; i < 16; ++i)
          for (uint nb = 0; nb < NB; ++nb) fp8_neumaier(hi[nb][r], lo[nb][r], x[i] * w[nb][i]);
      }
    }
  }
  for (uint r0 = 0; r0 < RT; r0 += RB) {
    for (uint nb = 0; nb < NB; ++nb) {
      for (uint dr = 0; dr < RB && r0 + dr < RT; ++dr) {
        float h = hi[nb][r0 + dr], l = lo[nb][r0 + dr];
        if (TF_CTL == 3) {
          h = float(half(h));
          l = 0.0f;
        }
        Fp8DD v = fp8_two_sum(h, l);
        for (uint m = 1; m < GL; m *= 2) v = fp8_dd_add(v, {simd_shuffle_xor(v.hi, m), simd_shuffle_xor(v.lo, m)});
        if (gl == 0) roots[(tile * BW + c * NB + nb) * RB + dr] = float2(v.hi, v.lo);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint o = tile; o < BW * RB; o += nsg) {
      const uint col = o / RB, dr = o % RB, r = r0 + dr;
      if (r >= a.R || r >= RT) continue;
      const float2 rv = lane < a.tiles ? roots[(lane * BW + col) * RB + dr] : float2(0.0f);
      Fp8DD v = {rv.x, rv.y};
      for (uint m = 1; m < 32; m *= 2) v = fp8_dd_add(v, {simd_shuffle_xor(v.hi, m), simd_shuffle_xor(v.lo, m)});
      if (lane == 0) Y[ulong(r) * a.N + blk * BW + col] = fp8_dd_out<T>(v);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (TF_CTL == 4 && tg.x == 0u && tg.y == 0u && tile == 0u && lane == 0u) Y[ulong(a.R) * a.N] = T(1.0f);
}

#define TF_WHOLE(RT, NB, CL, T, TAG, NAME) \
  template [[host_name(NAME)]] [[kernel]] decltype(tf_dsv41_fp8_whole<RT, NB, CL, T, TAG>) tf_dsv41_fp8_whole<RT, NB, CL, T, TAG>;
#define TF_WHOLE_TIER(RT) \
  TF_WHOLE(RT, 4, 1, float, 0, "tf_dsv41_fp8_wa_r" #RT "_f32") \
  TF_WHOLE(RT, 4, 1, bfloat, 0, "tf_dsv41_fp8_wa_r" #RT "_bf16") \
  TF_WHOLE(RT, 1, 4, float, 0, "tf_dsv41_fp8_wb_r" #RT "_f32") \
  TF_WHOLE(RT, 1, 4, bfloat, 0, "tf_dsv41_fp8_wb_r" #RT "_bf16") \
  TF_WHOLE(RT, 2, 4, float, 0, "tf_dsv41_fp8_wc_r" #RT "_f32") \
  TF_WHOLE(RT, 2, 4, bfloat, 0, "tf_dsv41_fp8_wc_r" #RT "_bf16") \
  TF_WHOLE(RT, 1, 1, float, 0, "tf_dsv41_fp8_wd_r" #RT "_f32") \
  TF_WHOLE(RT, 1, 1, bfloat, 0, "tf_dsv41_fp8_wd_r" #RT "_bf16") \
  TF_WHOLE(RT, 2, 1, float, 0, "tf_dsv41_fp8_we_r" #RT "_f32") \
  TF_WHOLE(RT, 2, 1, bfloat, 0, "tf_dsv41_fp8_we_r" #RT "_bf16") \
  TF_WHOLE(RT, 1, 4, float, 1, "tf_dsv41_fp8_wf_r" #RT "_f32") \
  TF_WHOLE(RT, 1, 4, bfloat, 1, "tf_dsv41_fp8_wf_r" #RT "_bf16")
TF_WHOLE_TIER(1)
TF_WHOLE_TIER(2)
TF_WHOLE_TIER(3)
TF_WHOLE_TIER(4)
TF_WHOLE_TIER(5)
TF_WHOLE_TIER(6)
TF_WHOLE_TIER(8)
TF_WHOLE_TIER(16)
