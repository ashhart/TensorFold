// DeepSeek V4.1 dense FP8, 1-16 BF16 rows: a 32-wide K group is one FMA chain, the groups meet in one fixed tree.
#include <metal_stdlib>
using namespace metal;

#ifndef TF_CTL
#define TF_CTL 0
#endif

// Literal K x N shapes of names.blockGroups at hidden 5120, world 1.
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

struct TfDsv41Fp8Args {
  uint K, N, R, groups, tiles, blocks;
};

// Exact: the seven magnitude bits of E4M3 placed as a half's exponent and mantissa, then scaled by 2^8.
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

inline float tf_dsv41_step(float acc, float x, float w) {
  const float r = fma(x, w, acc);
  return TF_CTL == 3 ? float(half(r)) : r;
}

// A group's scale as its E8M0 code and the product 256 * scale, exact for codes up to 247.
struct TfDsv41Scale {
  uint code;
  float x256;
};

inline TfDsv41Scale tf_dsv41_scale(uint code) {
  return TfDsv41Scale{code, tf_dsv41_e8m0(code) * 256.0f};
}

// Two codes of a word (bytes b, b + 2) as a half2 holding value / 256, sign included.
inline half2 tf_dsv41_h2(uint word) {
  return as_type<half2>(((word & 0x007F007Fu) << 7) | ((word & 0x00800080u) << 8));
}

// Sixteen weights; a code 0x7F or 0xFF (NaN) or a scale past 2^120 takes the per-code path, the same values.
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

// One simdgroup: four columns (NB in registers, CL across lanes), 32/CL K groups a column; roots to P[tile][col][row].
template <uint RT, uint NB, uint CL>
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_dsv41_fp8_part(
    const device uint4* X [[buffer(0)]], const device uint4* W [[buffer(1)]],
    const device uchar* S [[buffer(2)]], device float* P [[buffer(3)]],
    constant TfDsv41Fp8Args& a [[buffer(4)]],
    uint tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint GL = 32 / CL;
  const uint blk = tg % a.blocks, tile = tg / a.blocks;
  const uint c = lane / GL, gl = lane % GL;
  const uint g = tile * GL + gl;
  float acc[NB][RT];
  for (uint nb = 0; nb < NB; ++nb)
    for (uint r = 0; r < RT; ++r) acc[nb][r] = 0.0f;
  if (g < a.groups) {
    TfDsv41Scale sc[NB];
    for (uint nb = 0; nb < NB; ++nb) {
      const uint n = blk * 4 + c * NB + nb;
      sc[nb] = tf_dsv41_scale(S[ulong(n / 32) * a.groups + g]);
    }
    for (uint h = 0; h < 2; ++h) {
      float w[NB][16];
      for (uint nb = 0; nb < NB; ++nb) {
        const uint n = blk * 4 + c * NB + nb;
        tf_dsv41_w16(W[ulong(n) * (a.K / 16) + 2 * g + h], sc[nb], w[nb]);
      }
      for (uint r = 0; r < RT; ++r) {
        if (r >= a.R) break;
        const ulong at = ulong(r) * (a.K / 8) + 4 * g + 2 * h;
        float x[16];
        tf_dsv41_x16(X[at], X[at + 1], x);
        for (uint i = 0; i < 16; ++i)
          for (uint nb = 0; nb < NB; ++nb) acc[nb][r] = tf_dsv41_step(acc[nb][r], x[i], w[nb][i]);
      }
    }
  }
  for (uint nb = 0; nb < NB; ++nb) {
    for (uint r = 0; r < RT; ++r) {
      float v = acc[nb][r];
      for (uint m = 1; m < GL; m *= 2) v += simd_shuffle_xor(v, m);
      if (gl == 0 && r < a.R) P[(ulong(tile) * a.N + blk * 4 + c * NB + nb) * a.R + r] = v;
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

// The tile roots, zero-padded to 32, through the adjacent-pair tree; one thread an output.
template <typename T>
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_dsv41_fp8_finish(
    const device float* P [[buffer(0)]], device T* Y [[buffer(1)]], constant TfDsv41Fp8Args& a [[buffer(2)]],
    uint2 at [[thread_position_in_grid]]) {
  if (at.x >= a.N || at.y >= a.R) return;
  float v[32];
  for (uint t = 0; t < 32; ++t) v[t] = t < a.tiles ? P[(ulong(t) * a.N + at.x) * a.R + at.y] : 0.0f;
  float total = v[0];
  if (TF_CTL == 2) {
    for (uint t = 1; t < a.tiles; ++t) total += v[t];
  } else {
    for (uint s = 1; s < 32; s *= 2)
      for (uint i = 0; i < 32; i += 2 * s) v[i] += v[i + s];
    total = v[0];
  }
  Y[ulong(at.y) * a.N + at.x] = T(total);
  if (TF_CTL == 4 && at.x == 0 && at.y == 0) Y[ulong(a.R) * a.N] = T(1.0f);
}

template [[host_name("tf_dsv41_fp8_finish_f32")]] [[kernel]] decltype(tf_dsv41_fp8_finish<float>) tf_dsv41_fp8_finish<float>;
template [[host_name("tf_dsv41_fp8_finish_bf16")]] [[kernel]] decltype(tf_dsv41_fp8_finish<bfloat>) tf_dsv41_fp8_finish<bfloat>;

struct TfDsv41Fp8WArgs {
  uint K, N, R, groups, tiles, blocks;
  long xs, ws, ss, ys; // byte strides between the launches of a batch (grid y)
};

// Part and finish in one group: simdgroup t is tile t of CL * NB columns; the roots meet in the same zero-padded tree.
template <uint RT, uint NB, uint CL, typename T>
[[kernel, max_total_threads_per_threadgroup(1024)]] void tf_dsv41_fp8_whole(
    const device uchar* X0 [[buffer(0)]], const device uchar* W0 [[buffer(1)]],
    const device uchar* S0 [[buffer(2)]], device uchar* Y0 [[buffer(3)]],
    constant TfDsv41Fp8WArgs& a [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint tile [[simdgroup_index_in_threadgroup]],
    uint t [[thread_index_in_threadgroup]]) {
  constexpr uint BW = CL * NB;
  threadgroup float roots[32 * BW * RT];
  const device uint4* X = (const device uint4*)(X0 + long(tg.y) * a.xs);
  const device uint4* W = (const device uint4*)(W0 + long(tg.y) * a.ws);
  const device uchar* S = S0 + long(tg.y) * a.ss;
  device T* Y = (device T*)(Y0 + long(tg.y) * a.ys);
  const uint GL = 32 / CL;
  const uint c = lane / GL, gl = lane % GL;
  const uint g = tile * GL + gl;
  const uint blk = tg.x;
  float acc[NB][RT];
  for (uint nb = 0; nb < NB; ++nb)
    for (uint r = 0; r < RT; ++r) acc[nb][r] = 0.0f;
  if (g < a.groups) {
    TfDsv41Scale sc[NB];
    for (uint nb = 0; nb < NB; ++nb) {
      const uint n = blk * BW + c * NB + nb;
      sc[nb] = tf_dsv41_scale(S[ulong(n / 32) * a.groups + g]);
    }
    for (uint h = 0; h < 2; ++h) {
      float w[NB][16];
      for (uint nb = 0; nb < NB; ++nb) {
        const uint n = blk * BW + c * NB + nb;
        tf_dsv41_w16(W[ulong(n) * (a.K / 16) + 2 * g + h], sc[nb], w[nb]);
      }
      for (uint r = 0; r < RT; ++r) {
        if (r >= a.R) break;
        const ulong at = ulong(r) * (a.K / 8) + 4 * g + 2 * h;
        float x[16];
        tf_dsv41_x16(X[at], X[at + 1], x);
        for (uint i = 0; i < 16; ++i)
          for (uint nb = 0; nb < NB; ++nb) acc[nb][r] = tf_dsv41_step(acc[nb][r], x[i], w[nb][i]);
      }
    }
  }
  for (uint nb = 0; nb < NB; ++nb) {
    for (uint r = 0; r < RT; ++r) {
      float v = acc[nb][r];
      for (uint m = 1; m < GL; m *= 2) v += simd_shuffle_xor(v, m);
      if (gl == 0) roots[(tile * BW + c * NB + nb) * RT + r] = v;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint o = t; o < BW * RT; o += 32 * a.tiles) {
    const uint col = o / RT, r = o % RT;
    if (r >= a.R) continue;
    float v[32];
    for (uint k = 0; k < 32; ++k) v[k] = k < a.tiles ? roots[(k * BW + col) * RT + r] : 0.0f;
    for (uint s = 1; s < 32; s *= 2)
      for (uint i = 0; i < 32; i += 2 * s) v[i] += v[i + s];
    Y[ulong(r) * a.N + blk * BW + col] = T(v[0]);
  }
}

#define TF_WHOLE(RT, NB, CL, T, NAME) \
  template [[host_name(NAME)]] [[kernel]] decltype(tf_dsv41_fp8_whole<RT, NB, CL, T>) tf_dsv41_fp8_whole<RT, NB, CL, T>;
#define TF_WHOLE_TIER(RT) \
  TF_WHOLE(RT, 4, 1, float, "tf_dsv41_fp8_wa_r" #RT "_f32") \
  TF_WHOLE(RT, 4, 1, bfloat, "tf_dsv41_fp8_wa_r" #RT "_bf16") \
  TF_WHOLE(RT, 1, 4, float, "tf_dsv41_fp8_wb_r" #RT "_f32") \
  TF_WHOLE(RT, 1, 4, bfloat, "tf_dsv41_fp8_wb_r" #RT "_bf16") \
  TF_WHOLE(RT, 2, 4, float, "tf_dsv41_fp8_wc_r" #RT "_f32") \
  TF_WHOLE(RT, 2, 4, bfloat, "tf_dsv41_fp8_wc_r" #RT "_bf16")
TF_WHOLE_TIER(1)
TF_WHOLE_TIER(2)
TF_WHOLE_TIER(3)
TF_WHOLE_TIER(4)
TF_WHOLE_TIER(5)
TF_WHOLE_TIER(6)
TF_WHOLE_TIER(8)
TF_WHOLE_TIER(16)
