// DeepSeek V4.1 Engram (model.py Engram.forward, engram.py rows): dequantize, act_quant, gate, add; wkv is fp8_rows.
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

#ifndef TF_CTL
#define TF_CTL 0
#endif

#define TF_ENGRAM_HEAD 256
#define TF_ENGRAM_HC 4

// R rows, cols hash columns (K = cols * 256), dim hidden; dscale = float(dim ^ -0.5).
struct TfEngramArgs {
  uint R, cols, dim;
  float eps, dscale;
};

// Exact: the seven magnitude bits of E4M3 placed as a half's exponent and mantissa, then scaled by 2^8.
inline float tf_engram_e4m3(uint code) {
  const float v = float(as_type<half>(ushort((code & 0x7Fu) << 7))) * 256.0f;
  const float s = (code & 0x80u) ? -v : v;
  return (code & 0x7Fu) == 0x7Fu ? NAN : s;
}

inline float tf_engram_e8m0(uint code) {
  if (code == 255u) return NAN;
  return as_type<float>(code == 0u ? 0x00400000u : code << 23);
}

// values = (e4m3 * e8m0).to(bfloat16); codes [R][cols][256], one scale per 32 codes; emb [R][cols * 256].
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_dsv41_engram_embed(
    const device uchar* C [[buffer(0)]], const device uchar* S [[buffer(1)]], device bfloat* E [[buffer(2)]],
    constant TfEngramArgs& a [[buffer(3)]], uint2 at [[thread_position_in_grid]]) {
  const uint width = a.cols * TF_ENGRAM_HEAD;
  if (at.x >= width || at.y >= a.R) return;
  const ulong i = ulong(at.y) * width + at.x;
  E[i] = bfloat(tf_engram_e4m3(C[i]) * tf_engram_e8m0(S[i / (TF_CTL == 1 ? 64 : 32)]));
}

// The reference's act_quant scale exponent for a block's amax: ceil(log2(max(amax, 1e-4) / 448)) from the float's bits.
inline int tf_engram_exp(float amax) {
  const uint b = as_type<uint>(max(amax, 1e-4f) * (1.0f / 448.0f));
  return int((b >> 23) & 0xFFu) - 127 + ((b & 0x7FFFFFu) != 0u ? 1 : 0);
}

// A magnitude of at most 448 as its nearest E4M3 code, ties to even (codes 0..8: subnormals and the first normal).
inline uint tf_engram_code(float a, bool neg) {
  uint b = as_type<uint>(a);
  b += TF_CTL == 7 ? 0u : 0x7FFFFu + ((b >> 20) & 1u);
  const uint c = a < 0.015625f ? uint(rint(a * 512.0f)) : (((b >> 23) - 120u) << 3) | ((b >> 20) & 7u);
  return neg ? (c | 0x80u) : c;
}

// emb as linear() leaves it for an FP8 weight: act_quant per 32 values to bf16; a thread a block, NaN if non-finite.
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_dsv41_engram_aq(
    const device bfloat* E [[buffer(0)]], device bfloat* Y [[buffer(1)]], constant TfEngramArgs& a [[buffer(2)]],
    uint2 at [[thread_position_in_grid]]) {
  if (at.x >= a.cols * 8 || at.y >= a.R) return;
  const ulong base = ulong(at.y) * a.cols * TF_ENGRAM_HEAD + at.x * 32;
  float v[32], amax = 0.0f;
  bool bad = false;
  for (uint j = 0; j < 32; ++j) {
    v[j] = float(E[base + j]);
    bad |= !isfinite(v[j]);
    amax = max(amax, fabs(v[j]));
  }
  const int e = tf_engram_exp(amax);
  const float up = as_type<float>(uint(127 - e) << 23), down = as_type<float>(uint(e + 127) << 23);
  for (uint j = 0; j < 32; ++j) {
    const uint c = tf_engram_code(min(fabs(v[j]) * up, 448.0f), signbit(v[j]));
    const float val = float(as_type<half>(ushort(((c & 0x7Fu) << 7) | ((c & 0x80u) << 8)))) * 256.0f * down;
    Y[base + j] = bad ? bfloat(NAN) : bfloat(val);
  }
}

// Fixed order: the gate's lane chains (i = 32 t + lane, ascending) are met by the xor tree 16, 8, 4, 2, 1.
inline float tf_engram_tree(float v) {
  for (uint m = 16; m > 0; m /= 2) v += simd_shuffle_xor(v, m);
  return v;
}

inline float tf_engram_add(float acc, float t) {
  const float r = acc + t;
  return TF_CTL == 5 ? float(half(r)) : r;
}

// gate [R][hc] from x, kv [R][hc + 1][dim] (keys, value), w = q * k from bf16 q, k (exact in fp32); mask 0: gate 0.
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_dsv41_engram_gate(
    const device bfloat* X [[buffer(0)]], const device bfloat* KV [[buffer(1)]], const device bfloat* Qw [[buffer(2)]],
    const device bfloat* Kw [[buffer(3)]], const device uchar* M [[buffer(4)]], device float* G [[buffer(5)]],
    constant TfEngramArgs& a [[buffer(6)]],
    uint tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint r = tg / TF_ENGRAM_HC, hc = tg % TF_ENGRAM_HC;
  const device bfloat* h = X + ulong(tg) * a.dim;
  const device bfloat* key = KV + ulong(r) * (TF_ENGRAM_HC + 1) * a.dim + ulong(hc) * a.dim;
  const ulong wat = ulong(hc) * a.dim;
  float sh = 0.0f, sk = 0.0f, sd = 0.0f;
  for (uint i = lane; i < a.dim; i += 32) {
    const float x = float(h[i]), k = float(key[i]);
    const float xx = x * x, kk = k * k, xwk = (x * (float(Qw[wat + i]) * float(Kw[wat + i]))) * k;
    sh = tf_engram_add(sh, xx);
    sk = tf_engram_add(sk, kk);
    sd = tf_engram_add(sd, xwk);
  }
  sh = tf_engram_tree(sh);
  sk = tf_engram_tree(sk);
  sd = tf_engram_tree(sd);
  const float rstd = precise::rsqrt(sh / float(a.dim) + a.eps) * precise::rsqrt(sk / float(a.dim) + a.eps);
  const float dot = (sd * rstd) * a.dscale;
  float mag = TF_CTL == 4 ? fmax(fabs(dot), 1e-6f) : fabs(dot);
  if (mag < 1e-6f) mag = 1e-6f;
  const float z = copysign(precise::sqrt(mag), dot);
  const float g = 1.0f / (1.0f + precise::exp(-z));
  if (lane == 0) G[tg] = M[r] != 0 ? g : 0.0f;
}

// y [R][hc][dim] = bfloat16(x + gate * value) in two roundings; y may be x. One thread an element.
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_dsv41_engram_add(
    const device bfloat* X [[buffer(0)]], const device bfloat* KV [[buffer(1)]], const device float* G [[buffer(2)]],
    device bfloat* Y [[buffer(3)]], constant TfEngramArgs& a [[buffer(4)]], uint2 at [[thread_position_in_grid]]) {
  const uint width = TF_ENGRAM_HC * a.dim;
  if (at.x >= width || at.y >= a.R) return;
  const float v = float(KV[ulong(at.y) * (TF_ENGRAM_HC + 1) * a.dim + width + at.x % a.dim]);
  const float p = G[at.y * TF_ENGRAM_HC + at.x / a.dim] * v;
  const ulong i = ulong(at.y) * width + at.x;
  Y[i] = bfloat(float(X[i]) + p);
  if (TF_CTL == 6 && at.x == 0 && at.y == 0) Y[ulong(a.R) * width] = bfloat(1.0f);
}
