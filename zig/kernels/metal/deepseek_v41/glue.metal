// DeepSeek V4.1 glue: RMS norm rows, embedding gather, joins, RoPE and act-quant round trips, one fixed formula each.
#include <metal_stdlib>
using namespace metal;
#pragma METAL fp contract(off)

#ifndef TF_CTL
#define TF_CTL 0
#endif

struct TfRmsArgs { uint D, R; float eps; };
struct TfEmbedArgs { uint D, R, V, reps; int lo; };
struct TfMeanArgs { uint D, R, hc; };
struct TfRopeArgs { uint R, H, pairs, ldx, hx, ox, ldy, hy, oy; };
struct TfQuantArgs { uint N, R; };

// Control 1 of every kernel: a non-finite input is read as 0, so the output class changes.
inline float tf_g_in(float v) { return (TF_CTL == 1 && !isfinite(v)) ? 0.0f : v; }

// RMSNorm.forward (model.py:288), a row a threadgroup: fma chains t, t+256, ..; xor butterfly 16..1; p0..p7 in order.
template <typename W>
[[kernel, max_total_threads_per_threadgroup(256)]] void tf_dsv41_rms(
    const device bfloat* X [[buffer(0)]], const device W* Wt [[buffer(1)]], device bfloat* Y [[buffer(2)]],
    constant TfRmsArgs& a [[buffer(3)]], uint row [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[8];
  const device bfloat* x = X + ulong(row) * a.D;
  const uint n = (TF_CTL == 2 && a.D % 256 != 0) ? a.D - 1 : a.D;
  float s = 0.0f;
  for (uint i = t; i < n; i += 256) {
    const float v = tf_g_in(float(x[i]));
    s = fma(v, v, s);
    if (TF_CTL == 3) s = float(half(s));
  }
  for (uint m = 16; m > 0; m >>= 1) s += simd_shuffle_xor(s, m);
  if (lane == 0) part[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = part[0];
  for (uint j = 1; j < 8; ++j) total += part[j];
  const float r = precise::rsqrt(precise::divide(total, float(a.D)) + a.eps);
  for (uint i = t; i < a.D; i += 256) Y[ulong(row) * a.D + i] = bfloat(float(Wt[i]) * (float(x[i]) * r));
  if (TF_CTL == 4 && row == 0 && t == 0) Y[ulong(a.R) * a.D] = bfloat(1.0f);
}

#define TF_RMS(W, NAME) template [[host_name(NAME)]] [[kernel]] decltype(tf_dsv41_rms<W>) tf_dsv41_rms<W>;
TF_RMS(bfloat, "tf_dsv41_rms_bf16w")
TF_RMS(float, "tf_dsv41_rms_f32w")

// ParallelEmbedding.forward (model.py:168), one rank: slice row id - lo copied, zeros off it, `reps` times.
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_dsv41_embed(
    const device int* ids [[buffer(0)]], const device uint* T [[buffer(1)]], device uint* Y [[buffer(2)]],
    constant TfEmbedArgs& a [[buffer(3)]], uint2 at [[thread_position_in_grid]]) {
  const uint half_d = a.D / 2;
  if (at.x >= half_d || at.y >= a.R) return;
  const int id = ids[at.y] - a.lo;
  uint v = (id >= 0 && uint(id) < a.V) ? T[ulong(id) * half_d + at.x] : 0u;
  if (TF_CTL == 1 && !all(isfinite(float2(as_type<bfloat2>(v))))) v = 0u;
  if (TF_CTL == 2 && at.x + 1 == half_d && a.D % 64 != 0) return;
  for (uint j = 0; j < a.reps; ++j) Y[(ulong(at.y) * a.reps + j) * half_d + at.x] = v;
  if (TF_CTL == 4 && at.x == 0 && at.y == 0) Y[ulong(a.R) * a.reps * half_d] = 0x3F803F80u;
}

// MoE.forward's tail (model.py:905-906): y += shared (fp32 += bf16), then y.type_as(x): out = bf16(routed + shared).
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_dsv41_moe_join(
    const device float* routed [[buffer(0)]], const device bfloat* shared [[buffer(1)]], device bfloat* Y [[buffer(2)]],
    constant uint& n [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  const float r = tf_g_in(routed[i]);
  Y[i] = bfloat((TF_CTL == 2 ? float(bfloat(r)) : r) + float(shared[i]));
  if (TF_CTL == 4 && i == 0) Y[n] = bfloat(1.0f);
}

// h.mean(dim=2) over the hc copies (model.py:1269): out = bf16(((c0 + c1) + c2) + ... / hc), fp32 adds in copy order.
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_dsv41_hc_mean(
    const device bfloat* X [[buffer(0)]], device bfloat* Y [[buffer(1)]], constant TfMeanArgs& a [[buffer(2)]],
    uint2 at [[thread_position_in_grid]]) {
  if (at.x >= a.D || at.y >= a.R) return;
  const device bfloat* x = X + ulong(at.y) * a.hc * a.D + at.x;
  float s = tf_g_in(float(x[0]));
  for (uint j = 1; j < (TF_CTL == 2 ? a.hc - 1 : a.hc); ++j) s += tf_g_in(float(x[ulong(j) * a.D]));
  Y[ulong(at.y) * a.D + at.x] = bfloat(precise::divide(s, float(a.hc)));
  if (TF_CTL == 4 && at.x == 0 && at.y == 0) Y[ulong(a.R) * a.D] = bfloat(1.0f);
}

// apply_rotary_emb (model.py:392): (xr, xi) * (c, s) of the row's position, each product and sum in fp32; x may be y.
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_dsv41_rope(
    const device uint* X [[buffer(0)]], device uint* Y [[buffer(1)]], const device int* pos [[buffer(2)]],
    const device float2* tab [[buffer(3)]], constant TfRopeArgs& a [[buffer(4)]],
    uint3 at [[thread_position_in_grid]]) {
  if (at.x >= a.pairs || at.y >= a.H || at.z >= a.R) return;
  const uint i = TF_CTL == 2 ? (at.x < a.pairs / 2 ? 2 * at.x : 2 * (at.x - a.pairs / 2) + 1) : at.x;
  const float2 cs = tab[ulong(pos[at.z]) * a.pairs + at.x];
  const ulong from = (ulong(at.z) * a.ldx + at.y * a.hx + a.ox) / 2 + i;
  float2 v = float2(as_type<bfloat2>(X[from]));
  v = float2(tf_g_in(v.x), tf_g_in(v.y));
  const float re = v.x * cs.x - v.y * cs.y;
  const float im = v.x * cs.y + v.y * cs.x;
  Y[(ulong(at.z) * a.ldy + at.y * a.hy + a.oy) / 2 + at.x] = as_type<uint>(bfloat2(bfloat(re), bfloat(im)));
  if (TF_CTL == 4 && at.x == 0 && at.y == 0 && at.z == 0) Y[ulong(a.R) * a.H * a.pairs] = 0x3F803F80u;
}

// The reference's act_quant scale exponent for an amax: ceil(log2(max(amax, lo) * inv)) from the product's bits.
inline int tf_g_exp(float amax, float lo, float inv) {
  const uint b = as_type<uint>(max(amax, lo) * inv);
  return int((b >> 23) & 0xFFu) - 127 + ((b & 0x7FFFFFu) != 0u && TF_CTL != 2 ? 1 : 0);
}

inline float tf_g_pow2(int e) { return as_type<float>(uint(e + 127) << 23); }

// A magnitude of at most 448 as its nearest E4M3 code, ties to even (codes 0..8: subnormals and the first normal).
inline uint tf_g_code(float a) {
  uint b = as_type<uint>(a);
  b += TF_CTL == 3 ? 0x80000u : 0x7FFFFu + ((b >> 20) & 1u);
  const uint sub = uint(TF_CTL == 3 ? round(a * 512.0f) : rint(a * 512.0f));
  return a < 0.015625f ? sub : (((b >> 23) - 120u) << 3) | ((b >> 20) & 7u);
}

// An E4M3 magnitude code exactly: its bits placed as a half's exponent and mantissa, then scaled by 2^8.
inline float tf_g_val(uint c) { return float(as_type<half>(ushort(c << 7))) * 256.0f; }

// An fp32 magnitude of at most 6 as its nearest E2M1 value, ties to the even code.
inline float tf_g_fp4(float a) {
  const bool t = TF_CTL == 3;
  if (a < 0.25f || (!t && a == 0.25f)) return 0.0f;
  if (a < 0.75f) return 0.5f;
  if (a < 1.25f || (!t && a == 1.25f)) return 1.0f;
  if (a < 1.75f) return 1.5f;
  if (a < 2.5f || (!t && a == 2.5f)) return 2.0f;
  if (a < 3.5f) return 3.0f;
  return a < 5.0f || (!t && a == 5.0f) ? 4.0f : 6.0f;
}

// Lane i of an amax over the block's `blk` lanes: a xor butterfly; max is order free, so any tree gives the same bits.
inline float tf_g_amax(float v, uint blk) {
  for (uint m = blk / 2; m > 0; m >>= 1) v = max(v, simd_shuffle_xor(v, m));
  return v;
}

// act_quant inplace (kernel.py:98): s = 2^ceil(log2(max(amax, 1e-4) / 448)), e4m3(x / s) * s; a bad block is NaN.
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_dsv41_fp8_rt(
    const device bfloat* X [[buffer(0)]], device bfloat* Y [[buffer(1)]], constant TfQuantArgs& a [[buffer(2)]],
    uint2 at [[thread_position_in_grid]]) {
  if (at.y >= a.R) return;
  const ulong i = ulong(at.y) * a.N + at.x;
  const float v = float(X[i]);
  const bool bad = TF_CTL == 1 ? false : tf_g_amax(isfinite(v) ? 0.0f : 1.0f, 32) > 0.0f;
  const int e = tf_g_exp(tf_g_amax(fabs(v), 32), 1e-4f, 1.0f / 448.0f);
  const float q = tf_g_val(tf_g_code(min(fabs(v) * tf_g_pow2(-e), 448.0f))) * tf_g_pow2(e);
  Y[i] = bfloat(bad ? NAN : (signbit(v) ? -q : q));
  if (TF_CTL == 4 && at.x == 0 && at.y == 0) Y[ulong(a.R) * a.N] = bfloat(1.0f);
}

// fp4_act_quant inplace (kernel.py:184): E8M0 blocks of 32 (s = 2^ceil(log2(max(amax, 6 2^-126) / 6))); E4: E4M3 of 16.
template <uint BLK, bool E4>
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_dsv41_fp4_rt(
    const device bfloat* X [[buffer(0)]], device bfloat* Y [[buffer(1)]], constant TfQuantArgs& a [[buffer(2)]],
    uint2 at [[thread_position_in_grid]]) {
  if (at.y >= a.R) return;
  const ulong i = ulong(at.y) * a.N + at.x;
  const float v = float(X[i]);
  const float amax = tf_g_amax(fabs(v), BLK);
  const bool bad = TF_CTL == 1 ? false : tf_g_amax(isfinite(v) ? 0.0f : 1.0f, BLK) > 0.0f;
  float s, a_q;
  if (E4) {
    const float raw = precise::divide(max(amax, 0.01171875f), 6.0f);
    s = TF_CTL == 2 ? raw : tf_g_val(tf_g_code(min(raw, 448.0f)));
    a_q = min(precise::divide(fabs(v), s), 6.0f);
  } else {
    const int e = tf_g_exp(amax, 0x1.8p-124f, 1.0f / 6.0f);
    s = tf_g_pow2(e);
    a_q = min(fabs(v) * tf_g_pow2(-e), 6.0f);
  }
  const float f = tf_g_fp4(a_q) * s;
  Y[i] = bfloat(bad ? NAN : (signbit(v) ? -f : f));
  if (TF_CTL == 4 && at.x == 0 && at.y == 0) Y[ulong(a.R) * a.N] = bfloat(1.0f);
}

#define TF_FP4(BLK, E4, NAME) \
  template [[host_name(NAME)]] [[kernel]] decltype(tf_dsv41_fp4_rt<BLK, E4>) tf_dsv41_fp4_rt<BLK, E4>;
TF_FP4(32, false, "tf_dsv41_fp4_rt32")
TF_FP4(16, true, "tf_dsv41_fp4_rt16")
