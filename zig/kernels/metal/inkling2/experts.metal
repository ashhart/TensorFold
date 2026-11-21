// Inkling 2 Mini M3: packed weight decode in registers, BF16 activations, FP32 reductions.
#include <metal_stdlib>
using namespace metal;
#pragma METAL fp contract(off)

struct Ink2MatrixArgs { uint k, n, topk, experts; };
struct Ink2DecodeArgs { uint rows, k; };
struct Ink2GroupArgs { uint rows, topk, experts, tile_m; };
struct Ink2GroupedMatrixArgs { uint rows, k, n, topk, experts; };

inline float ink2_e2m1(uint code) {
  const float magnitude[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float v = magnitude[code & 7u];
  return (code & 8u) ? -v : v;
}

inline float ink2_e4m3(uint code) {
  const uint exp = (code >> 3u) & 15u, mant = code & 7u;
  if (exp == 15u && mant == 7u) return NAN;
  const float v = exp ? ldexp(1.0f + float(mant) * 0.125f, int(exp) - 7)
                      : float(mant) * (1.0f / 512.0f);
  return (code & 0x80u) ? -v : v;
}

inline float ink2_phi(device const float* amax, device const float* denominator, uint expert) {
  // Loader casts both globals to BF16 before the dequantization operation.
  return precise::divide(float(bfloat(amax[expert])), float(bfloat(denominator[0])));
}

inline float ink2_nv_weight(uchar packed, uint column, uchar scale, float phi) {
  const uint code = (column & 1u) ? uint(packed >> 4u) : uint(packed & 15u);
  const float inner = ink2_e2m1(code) * ink2_e4m3(scale);
  return float(bfloat(inner * phi));
}

inline uint ink2_pack_item(uint expert, uint offset, uint len) {
  return (expert & 0xffffu) | ((offset & 0xfffu) << 16u) | (((len - 1u) & 0xfu) << 28u);
}

inline float ink2_mx_weight(uchar value, uchar scale) {
  const float w = ldexp(ink2_e4m3(value), int(scale) - 127);
  // E4M3 times a power of two is exactly representable in BF16, still round explicitly.
  return float(bfloat(w));
}

inline float2 ink2_bf2(uint pair) {
  return float2(as_type<float>(pair << 16u), as_type<float>(pair & 0xffff0000u));
}

// A byte of two E2M1 codes as two FP16 values scaled by 2^-14, low nibble first; the byte must stand alone.
inline float2 ink2_e2m1_pair(uint code) {
  return float2(as_type<half2>(((code * 0x00200200u) & 0x0e000e00u) | ((code * 0x01001000u) & 0x80008000u)));
}

// ink2_e4m3(code) * 2^14 from the FP16 bits of the same layout; 0x7f and 0xff stay NaN.
inline float ink2_e4m3_scale(uint code) {
  const float v = float(as_type<half>(ushort(((code & 0x7fu) << 7u) | ((code & 0x80u) << 8u)))) * 4194304.0f;
  return (code & 0x7fu) == 0x7fu ? as_type<float>(((code & 0x80u) << 24u) | 0x7fc00000u) : v;
}

// ink2_nv_weight twice: (e2m1 * 2^-14) * (e4m3 * 2^14) is the same exact product, then phi and one BF16 round.
inline float ink2_nv_byte(float sum, uint code, float scale, float phi, float2 x) {
  const float2 m = ink2_e2m1_pair(code);
  sum += float(bfloat((m.x * scale) * phi)) * x.x;
  return sum + float(bfloat((m.y * scale) * phi)) * x.y;
}

// Columns in order, 16 to a scale block; k is a multiple of 16 and w, x are 16-byte aligned.
inline float ink2_dot_nv(device const uchar* w, device const uchar* scales,
                         device const bfloat* x, float phi, uint k) {
  float sum = 0.0f;
  for (uint b = 0; b < k / 16u; ++b) {
    const uint2 codes = *(device const uint2*)(w + 8u * b);
    const float scale = ink2_e4m3_scale(scales[b]);
    const uint4 xa = *(device const uint4*)(x + 16u * b), xb = *(device const uint4*)(x + 16u * b + 8u);
    sum = ink2_nv_byte(sum, codes.x & 255u, scale, phi, ink2_bf2(xa.x));
    sum = ink2_nv_byte(sum, (codes.x >> 8u) & 255u, scale, phi, ink2_bf2(xa.y));
    sum = ink2_nv_byte(sum, (codes.x >> 16u) & 255u, scale, phi, ink2_bf2(xa.z));
    sum = ink2_nv_byte(sum, codes.x >> 24u, scale, phi, ink2_bf2(xa.w));
    sum = ink2_nv_byte(sum, codes.y & 255u, scale, phi, ink2_bf2(xb.x));
    sum = ink2_nv_byte(sum, (codes.y >> 8u) & 255u, scale, phi, ink2_bf2(xb.y));
    sum = ink2_nv_byte(sum, (codes.y >> 16u) & 255u, scale, phi, ink2_bf2(xb.z));
    sum = ink2_nv_byte(sum, codes.y >> 24u, scale, phi, ink2_bf2(xb.w));
  }
  return sum;
}

// Four E4M3 bytes, columns 4i to 4i + 3, added in order; the FP16 route gives e4m3 / 256, 2^(scale - 119) restores it.
inline float ink2_mx_word(float sum, uint word, float factor, float2 x01, float2 x23) {
  const uint even = word & 0x00ff00ffu, odd = (word >> 8u) & 0x00ff00ffu;
  const float2 w02 = float2(as_type<half2>(((even << 7u) & 0x3f803f80u) | ((even << 8u) & 0x80008000u)));
  const float2 w13 = float2(as_type<half2>(((odd << 7u) & 0x3f803f80u) | ((odd << 8u) & 0x80008000u)));
  sum += (w02.x * factor) * x01.x;
  sum += (w13.x * factor) * x01.y;
  sum += (w02.y * factor) * x23.x;
  return sum + (w13.y * factor) * x23.y;
}

// A block of 32 columns: scales 24 to 230 with no NaN code take the word route, the rest decode as ink2_mx_weight.
inline float ink2_mx_block(float sum, device const uchar* w, uint scale, device const bfloat* x) {
  const uint4 q0 = *(device const uint4*)w, q1 = *(device const uint4*)(w + 16);
  uint nan = 0u;
  for (uint i = 0; i < 4u; ++i) nan |= ((q0[i] & 0x7f7f7f7fu) + 0x01010101u) | ((q1[i] & 0x7f7f7f7fu) + 0x01010101u);
  if (scale - 24u > 206u || (nan & 0x80808080u) != 0u) {
    for (uint j = 0; j < 32u; ++j) sum += ink2_mx_weight(w[j], uchar(scale)) * float(x[j]);
    return sum;
  }
  const float factor = as_type<float>((scale + 8u) << 23u);
  const uint4 xa = *(device const uint4*)x, xb = *(device const uint4*)(x + 8), xc = *(device const uint4*)(x + 16);
  const uint4 xd = *(device const uint4*)(x + 24);
  sum = ink2_mx_word(sum, q0.x, factor, ink2_bf2(xa.x), ink2_bf2(xa.y));
  sum = ink2_mx_word(sum, q0.y, factor, ink2_bf2(xa.z), ink2_bf2(xa.w));
  sum = ink2_mx_word(sum, q0.z, factor, ink2_bf2(xb.x), ink2_bf2(xb.y));
  sum = ink2_mx_word(sum, q0.w, factor, ink2_bf2(xb.z), ink2_bf2(xb.w));
  sum = ink2_mx_word(sum, q1.x, factor, ink2_bf2(xc.x), ink2_bf2(xc.y));
  sum = ink2_mx_word(sum, q1.y, factor, ink2_bf2(xc.z), ink2_bf2(xc.w));
  sum = ink2_mx_word(sum, q1.z, factor, ink2_bf2(xd.x), ink2_bf2(xd.y));
  return ink2_mx_word(sum, q1.w, factor, ink2_bf2(xd.z), ink2_bf2(xd.w));
}

inline float ink2_dot_mx(device const uchar* w, device const uchar* scales, device const bfloat* x, uint k) {
  float sum = 0.0f;
  for (uint b = 0; b < k / 32u; ++b) sum = ink2_mx_block(sum, w + 32u * b, scales[b], x + 32u * b);
  return sum;
}

inline float ink2_cap(float x, float cap) {
  return cap * precise::tanh(precise::divide(x, cap));
}

inline bfloat ink2_act(float gate_dot, float up_dot) {
  const float gate = float(bfloat(gate_dot)), up = float(bfloat(up_dot));
  const float left = ink2_cap(gate, 4.0f) * precise::divide(1.0f, 1.0f + precise::exp(-gate));
  // Both GEMM results round before the softcapped SwiGLU; sigmoid uses the rounded gate.
  return bfloat(left * ink2_cap(up, 25.0f));
}

[[kernel, max_total_threads_per_threadgroup(32)]] void tf_ink2_nvfp4_gu(
    device const bfloat* X [[buffer(0)]], device const uchar* W [[buffer(1)]],
    device const uchar* S [[buffer(2)]], device const float* amax [[buffer(3)]],
    device const float* denominator [[buffer(4)]], device const int* ids [[buffer(5)]],
    device bfloat* Y [[buffer(6)]], constant Ink2MatrixArgs& a [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
  // One thread an output: threadgroup x is 32 outputs, y the route pair; each dot is the serial chain above.
  const uint n = tg.x * 32u + lane, group = tg.y;
  if (n >= a.n) return;
  const uint token = group / a.topk, expert = uint(ids[group]);
  const ulong expert_w = ulong(expert) * 2u * a.n * (a.k / 2u);
  const ulong expert_s = ulong(expert) * 2u * a.n * (a.k / 16u);
  const float phi = ink2_phi(amax, denominator, expert);
  const uint gr = 2u * n, ur = gr + 1u;
  const float g = ink2_dot_nv(W + expert_w + ulong(gr) * (a.k / 2u),
                              S + expert_s + ulong(gr) * (a.k / 16u), X + ulong(token) * a.k, phi, a.k);
  const float u = ink2_dot_nv(W + expert_w + ulong(ur) * (a.k / 2u),
                              S + expert_s + ulong(ur) * (a.k / 16u), X + ulong(token) * a.k, phi, a.k);
  Y[group * a.n + n] = ink2_act(g, u);
}

[[kernel, max_total_threads_per_threadgroup(32)]] void tf_ink2_mxfp8_gu(
    device const bfloat* X [[buffer(0)]], device const uchar* W [[buffer(1)]],
    device const uchar* S [[buffer(2)]], device const int* ids [[buffer(3)]],
    device bfloat* Y [[buffer(4)]], constant Ink2MatrixArgs& a [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
  // One thread an output: threadgroup x is 32 outputs, y the route pair; each dot is the serial chain above.
  const uint n = tg.x * 32u + lane, group = tg.y;
  if (n >= a.n) return;
  const uint token = group / a.topk, expert = uint(ids[group]);
  const ulong expert_w = ulong(expert) * 2u * a.n * a.k;
  const ulong expert_s = ulong(expert) * 2u * a.n * (a.k / 32u);
  const uint gr = 2u * n, ur = gr + 1u;
  const float g = ink2_dot_mx(W + expert_w + ulong(gr) * a.k,
                              S + expert_s + ulong(gr) * (a.k / 32u), X + ulong(token) * a.k, a.k);
  const float u = ink2_dot_mx(W + expert_w + ulong(ur) * a.k,
                              S + expert_s + ulong(ur) * (a.k / 32u), X + ulong(token) * a.k, a.k);
  Y[group * a.n + n] = ink2_act(g, u);
}

[[kernel, max_total_threads_per_threadgroup(32)]] void tf_ink2_nvfp4_down(
    device const bfloat* X [[buffer(0)]], device const uchar* W [[buffer(1)]],
    device const uchar* S [[buffer(2)]], device const float* amax [[buffer(3)]],
    device const float* denominator [[buffer(4)]], device const int* ids [[buffer(5)]],
    device bfloat* Y [[buffer(6)]], constant Ink2MatrixArgs& a [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
  // One thread an output: threadgroup x is 32 outputs, y the route pair; each dot is the serial chain above.
  const uint n = tg.x * 32u + lane, group = tg.y;
  if (n >= a.n) return;
  const uint expert = uint(ids[group]);
  const ulong expert_w = ulong(expert) * a.n * (a.k / 2u);
  const ulong expert_s = ulong(expert) * a.n * (a.k / 16u);
  const float phi = ink2_phi(amax, denominator, expert);
  Y[group * a.n + n] = bfloat(ink2_dot_nv(W + expert_w + ulong(n) * (a.k / 2u),
      S + expert_s + ulong(n) * (a.k / 16u), X + ulong(group) * a.k, phi, a.k));
}

[[kernel, max_total_threads_per_threadgroup(32)]] void tf_ink2_mxfp8_down(
    device const bfloat* X [[buffer(0)]], device const uchar* W [[buffer(1)]],
    device const uchar* S [[buffer(2)]], device const int* ids [[buffer(3)]],
    device bfloat* Y [[buffer(4)]], constant Ink2MatrixArgs& a [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
  // One thread an output: threadgroup x is 32 outputs, y the route pair; each dot is the serial chain above.
  const uint n = tg.x * 32u + lane, group = tg.y;
  if (n >= a.n) return;
  const uint expert = uint(ids[group]);
  const ulong expert_w = ulong(expert) * a.n * a.k;
  const ulong expert_s = ulong(expert) * a.n * (a.k / 32u);
  Y[group * a.n + n] = bfloat(ink2_dot_mx(W + expert_w + ulong(n) * a.k,
      S + expert_s + ulong(n) * (a.k / 32u), X + ulong(group) * a.k, a.k));
}

// ---- Grouped by expert: the same sums a row as the kernels above, with each weight block decoded once a pass. ----
struct Ink2GroupArgs { uint pairs, experts, cap; };
struct Ink2GroupedArgs { uint k, n, topk, cap; };
#define INK2_PASS 4u

// Expert e's route pairs in pair order: counts[e], then members[e][0 .. counts[e]).
kernel void tf_ink2_group(
    device const int* ids [[buffer(0)]], device uint* counts [[buffer(1)]],
    device uint* members [[buffer(2)]], constant Ink2GroupArgs& a [[buffer(3)]],
    uint e [[thread_position_in_grid]]) {
  if (e >= a.experts) return;
  uint n = 0u;
  for (uint p = 0u; p < a.pairs; ++p)
    if (uint(ids[p]) == e) {
      if (n < a.cap) members[e * a.cap + n] = p;
      ++n;
    }
  counts[e] = min(n, a.cap);
}

// Block b of a packed NVFP4 row as its 16 BF16 weights, each as ink2_dot_nv decodes it.
inline void ink2_nv_block(device const uchar* w, device const uchar* scales, float phi, uint b, thread float* out) {
  const float half_scale = 0.5f * ink2_e4m3(scales[b]);
  const uint lo = ink2_word(w + 8u * b), hi = ink2_word(w + 8u * b + 4u);
  for (uint j = 0; j < 16u; ++j) {
    const uint code = ((j < 8u ? lo : hi) >> (4u * (j & 7u))) & 15u;
    out[j] = float(bfloat(ink2_e2m1_twice(code) * half_scale * phi));
  }
}

// Block b of an MXFP8 row as its 32 BF16 weights, each as ink2_dot_mx decodes it.
inline void ink2_mx_block(device const uchar* w, device const uchar* scales, uint b, thread float* out) {
  const int shift = int(scales[b]) - 127;
  for (uint j = 0; j < 32u; ++j) out[j] = float(bfloat(ldexp(ink2_e4m3_bits(w[32u * b + j]), shift)));
}

// Threadgroup (x, e): outputs 8x to 8x + 7 of expert e for each of its route pairs, four pairs a pass.
[[kernel, max_total_threads_per_threadgroup(256)]] void tf_ink2_nvfp4_gu_grouped(
    device const bfloat* X [[buffer(0)]], device const uchar* W [[buffer(1)]],
    device const uchar* S [[buffer(2)]], device const float* amax [[buffer(3)]],
    device const float* denominator [[buffer(4)]], device const uint* counts [[buffer(5)]],
    device const uint* members [[buffer(6)]], device bfloat* Y [[buffer(7)]],
    constant Ink2GroupedArgs& a [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint n = tg.x * 8u + sg, expert = tg.y;
  if (n >= a.n) return;
  const uint count = counts[expert];
  if (count == 0u) return;
  const ulong expert_w = ulong(expert) * 2u * a.n * (a.k / 2u);
  const ulong expert_s = ulong(expert) * 2u * a.n * (a.k / 16u);
  const float phi = precise::divide(float(bfloat(amax[expert])), float(bfloat(denominator[0])));
  device const uchar* gate_w = W + expert_w + ulong(2u * n) * (a.k / 2u);
  device const uchar* gate_s = S + expert_s + ulong(2u * n) * (a.k / 16u);
  device const uchar* up_w = gate_w + (a.k / 2u);
  device const uchar* up_s = gate_s + (a.k / 16u);
  for (uint first = 0u; first < count; first += INK2_PASS) {
    uint pair[INK2_PASS];
    device const bfloat* x[INK2_PASS];
    float g[INK2_PASS], u[INK2_PASS];
    for (uint j = 0u; j < INK2_PASS; ++j) {
      pair[j] = members[expert * a.cap + min(first + j, count - 1u)];
      x[j] = X + ulong(pair[j] / a.topk) * a.k;
      g[j] = 0.0f;
      u[j] = 0.0f;
    }
    for (uint b = lane; b < a.k / 16u; b += 32u) {
      float gate[16], up[16];
      ink2_nv_block(gate_w, gate_s, phi, b, gate);
      ink2_nv_block(up_w, up_s, phi, b, up);
      for (uint j = 0u; j < INK2_PASS; ++j)
        for (uint i = 0u; i < 16u; ++i) {
          const float v = float(x[j][16u * b + i]);
          g[j] += gate[i] * v;
          u[j] += up[i] * v;
        }
    }
    for (uint j = 0u; j < INK2_PASS; ++j) {
      const float gs = simd_sum(g[j]), us = simd_sum(u[j]);
      if (lane == 0u && first + j < count) Y[ulong(pair[j]) * a.n + n] = ink2_act(gs, us);
    }
  }
}

[[kernel, max_total_threads_per_threadgroup(256)]] void tf_ink2_mxfp8_gu_grouped(
    device const bfloat* X [[buffer(0)]], device const uchar* W [[buffer(1)]],
    device const uchar* S [[buffer(2)]], device const uint* counts [[buffer(3)]],
    device const uint* members [[buffer(4)]], device bfloat* Y [[buffer(5)]],
    constant Ink2GroupedArgs& a [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint n = tg.x * 8u + sg, expert = tg.y;
  if (n >= a.n) return;
  const uint count = counts[expert];
  if (count == 0u) return;
  const ulong expert_w = ulong(expert) * 2u * a.n * a.k;
  const ulong expert_s = ulong(expert) * 2u * a.n * (a.k / 32u);
  device const uchar* gate_w = W + expert_w + ulong(2u * n) * a.k;
  device const uchar* gate_s = S + expert_s + ulong(2u * n) * (a.k / 32u);
  device const uchar* up_w = gate_w + a.k;
  device const uchar* up_s = gate_s + (a.k / 32u);
  for (uint first = 0u; first < count; first += INK2_PASS) {
    uint pair[INK2_PASS];
    device const bfloat* x[INK2_PASS];
    float g[INK2_PASS], u[INK2_PASS];
    for (uint j = 0u; j < INK2_PASS; ++j) {
      pair[j] = members[expert * a.cap + min(first + j, count - 1u)];
      x[j] = X + ulong(pair[j] / a.topk) * a.k;
      g[j] = 0.0f;
      u[j] = 0.0f;
    }
    for (uint b = lane; b < a.k / 32u; b += 32u) {
      float gate[32], up[32];
      ink2_mx_block(gate_w, gate_s, b, gate);
      ink2_mx_block(up_w, up_s, b, up);
      for (uint j = 0u; j < INK2_PASS; ++j)
        for (uint i = 0u; i < 32u; ++i) {
          const float v = float(x[j][32u * b + i]);
          g[j] += gate[i] * v;
          u[j] += up[i] * v;
        }
    }
    for (uint j = 0u; j < INK2_PASS; ++j) {
      const float gs = simd_sum(g[j]), us = simd_sum(u[j]);
      if (lane == 0u && first + j < count) Y[ulong(pair[j]) * a.n + n] = ink2_act(gs, us);
    }
  }
}

[[kernel, max_total_threads_per_threadgroup(256)]] void tf_ink2_nvfp4_down_grouped(
    device const bfloat* X [[buffer(0)]], device const uchar* W [[buffer(1)]],
    device const uchar* S [[buffer(2)]], device const float* amax [[buffer(3)]],
    device const float* denominator [[buffer(4)]], device const uint* counts [[buffer(5)]],
    device const uint* members [[buffer(6)]], device bfloat* Y [[buffer(7)]],
    constant Ink2GroupedArgs& a [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint n = tg.x * 8u + sg, expert = tg.y;
  if (n >= a.n) return;
  const uint count = counts[expert];
  if (count == 0u) return;
  const float phi = precise::divide(float(bfloat(amax[expert])), float(bfloat(denominator[0])));
  device const uchar* row_w = W + ulong(expert) * a.n * (a.k / 2u) + ulong(n) * (a.k / 2u);
  device const uchar* row_s = S + ulong(expert) * a.n * (a.k / 16u) + ulong(n) * (a.k / 16u);
  for (uint first = 0u; first < count; first += INK2_PASS) {
    uint pair[INK2_PASS];
    device const bfloat* x[INK2_PASS];
    float d[INK2_PASS];
    for (uint j = 0u; j < INK2_PASS; ++j) {
      pair[j] = members[expert * a.cap + min(first + j, count - 1u)];
      x[j] = X + ulong(pair[j]) * a.k;
      d[j] = 0.0f;
    }
    for (uint b = lane; b < a.k / 16u; b += 32u) {
      float weight[16];
      ink2_nv_block(row_w, row_s, phi, b, weight);
      for (uint j = 0u; j < INK2_PASS; ++j)
        for (uint i = 0u; i < 16u; ++i) d[j] += weight[i] * float(x[j][16u * b + i]);
    }
    for (uint j = 0u; j < INK2_PASS; ++j) {
      const float ds = simd_sum(d[j]);
      if (lane == 0u && first + j < count) Y[ulong(pair[j]) * a.n + n] = bfloat(ds);
    }
  }
}

[[kernel, max_total_threads_per_threadgroup(256)]] void tf_ink2_mxfp8_down_grouped(
    device const bfloat* X [[buffer(0)]], device const uchar* W [[buffer(1)]],
    device const uchar* S [[buffer(2)]], device const uint* counts [[buffer(3)]],
    device const uint* members [[buffer(4)]], device bfloat* Y [[buffer(5)]],
    constant Ink2GroupedArgs& a [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint n = tg.x * 8u + sg, expert = tg.y;
  if (n >= a.n) return;
  const uint count = counts[expert];
  if (count == 0u) return;
  device const uchar* row_w = W + ulong(expert) * a.n * a.k + ulong(n) * a.k;
  device const uchar* row_s = S + ulong(expert) * a.n * (a.k / 32u) + ulong(n) * (a.k / 32u);
  for (uint first = 0u; first < count; first += INK2_PASS) {
    uint pair[INK2_PASS];
    device const bfloat* x[INK2_PASS];
    float d[INK2_PASS];
    for (uint j = 0u; j < INK2_PASS; ++j) {
      pair[j] = members[expert * a.cap + min(first + j, count - 1u)];
      x[j] = X + ulong(pair[j]) * a.k;
      d[j] = 0.0f;
    }
    for (uint b = lane; b < a.k / 32u; b += 32u) {
      float weight[32];
      ink2_mx_block(row_w, row_s, b, weight);
      for (uint j = 0u; j < INK2_PASS; ++j)
        for (uint i = 0u; i < 32u; ++i) d[j] += weight[i] * float(x[j][32u * b + i]);
    }
    for (uint j = 0u; j < INK2_PASS; ++j) {
      const float ds = simd_sum(d[j]);
      if (lane == 0u && first + j < count) Y[ulong(pair[j]) * a.n + n] = bfloat(ds);
    }
  }
}

// Weight decode helpers unpack checkpoint bytes and never quantize an activation.
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_ink2_nvfp4_quantize(
    device const uchar* W [[buffer(0)]], device const uchar* S [[buffer(1)]],
    device const float* amax [[buffer(2)]], device const float* denominator [[buffer(3)]],
    device bfloat* Y [[buffer(4)]], constant Ink2DecodeArgs& a [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
  const uint at = group * 32u + lane;
  if (at >= a.rows * a.k) return;
  const uint row = at / a.k, col = at % a.k;
  const float phi = ink2_phi(amax, denominator, 0);
  Y[at] = bfloat(ink2_nv_weight(W[row * (a.k / 2u) + col / 2u], col,
                                S[row * (a.k / 16u) + col / 16u], phi));
}

[[kernel, max_total_threads_per_threadgroup(32)]] void tf_ink2_mxfp8_quantize(
    device const uchar* W [[buffer(0)]], device const uchar* S [[buffer(1)]],
    device bfloat* Y [[buffer(2)]], constant Ink2DecodeArgs& a [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
  const uint at = group * 32u + lane;
  if (at >= a.rows * a.k) return;
  const uint row = at / a.k, col = at % a.k;
  Y[at] = bfloat(ink2_mx_weight(W[at], S[row * (a.k / 32u) + col / 32u]));
}

[[kernel, max_total_threads_per_threadgroup(128)]] void tf_ink2_combine(
    device const bfloat* down [[buffer(0)]], device const int* ids [[buffer(1)]],
    device const float* weights [[buffer(2)]], device bfloat* Y [[buffer(3)]],
    constant Ink2MatrixArgs& a [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  // One thread an output: threadgroup x is 128 outputs, y the token; slots sort once per group.
  threadgroup uint tg_slots[8];
  threadgroup float tg_weights[8];
  threadgroup uint tg_used;
  const uint group = tg.y;
  if (tid < 32u) {
    const bool valid = tid < a.topk;
    const uint my_id = valid ? uint(ids[group * a.topk + tid]) : a.experts;
    const bool ok = valid && (my_id < a.experts);
    const float my_w = ok ? weights[group * a.topk + tid] : 0.0f;
    const uint prev_id = simd_shuffle_up(my_id, 1u);
    const bool needs_sort = simd_any(valid && (!ok || (tid > 0u && my_id < prev_id)));
    if (!needs_sort) {
      if (valid) { tg_slots[tid] = tid; tg_weights[tid] = my_w; }
      if (tid == 0u) tg_used = a.topk;
    } else {
      const uint key = ok ? my_id : a.experts;
      uint rank = 0u;
      for (uint s = 0u; s < a.topk; ++s) {
        const uint other = simd_shuffle(key, s);
        rank += (other < my_id || (other == my_id && s < tid)) ? 1u : 0u;
      }
      if (ok) { tg_slots[rank] = tid; tg_weights[rank] = my_w; }
      const uint used = popcount(uint(ulong(simd_vote::vote_t(simd_ballot(ok)))));
      if (tid == 0u) tg_used = used;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint n = tg.x * 128u + tid;
  if (n >= a.n) return;
  const uint used = tg_used;
  float sum = 0.0f;
  for (uint i = 0u; i < used; ++i)
    sum += tg_weights[i] * float(down[ulong(group * a.topk + tg_slots[i]) * a.n + n]);
  Y[ulong(group) * a.n + n] = bfloat(sum);
}

// Bucket route pairs by ascending expert ID into <= tile_m slices, preserving pair order within each expert.
[[kernel, max_total_threads_per_threadgroup(128)]] void tf_ink2_expert_group(
    device const int* ids [[buffer(0)]], device uint* group_buf [[buffer(1)]],
    constant Ink2GroupArgs& a [[buffer(2)]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint pairs = a.rows * a.topk;
  if (pairs == 0u || pairs > 512u || a.experts == 0u || a.tile_m == 0u || a.tile_m > 16u) {
    if (tid == 0u) group_buf[0] = 0u;
    return;
  }
  threadgroup int tg_ids[512];
  threadgroup uint tg_scan[8];
  threadgroup uint base_offsets[2];
  for (uint p = tid; p < pairs; p += 128u) tg_ids[p] = ids[p];
  if (tid == 0u) { base_offsets[0] = 0u; base_offsets[1] = 0u; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device uint* work_items = group_buf + 1u;
  device uint* pair_map = group_buf + 1u + pairs;
  for (uint e_base = 0u; e_base < a.experts; e_base += 128u) {
    const uint e = e_base + tid;
    uint cnt = 0u;
    if (e < a.experts)
      for (uint p = 0u; p < pairs; ++p) cnt += (tg_ids[p] == int(e)) ? 1u : 0u;
    const uint tiles = cnt ? ((cnt + a.tile_m - 1u) / a.tile_m) : 0u;
    const uint p_ex = simd_prefix_exclusive_sum(cnt);
    const uint t_ex = simd_prefix_exclusive_sum(tiles);
    if (lane == 31u) {
      tg_scan[sg] = p_ex + cnt;
      tg_scan[4u + sg] = t_ex + tiles;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint p_sg = 0u, t_sg = 0u;
    for (uint s = 0u; s < sg; ++s) { p_sg += tg_scan[s]; t_sg += tg_scan[4u + s]; }
    const uint pair_base = base_offsets[0];
    const uint tile_base = base_offsets[1];
    const uint p_off = pair_base + p_ex + p_sg;
    const uint t_off = tile_base + t_ex + t_sg;
    if (cnt > 0u) {
      for (uint t = 0u; t < tiles; ++t) {
        const uint rem = cnt - t * a.tile_m;
        const uint len = rem < a.tile_m ? rem : a.tile_m;
        work_items[t_off + t] = ink2_pack_item(e, p_off + t * a.tile_m, len);
      }
      uint at = p_off;
      for (uint p = 0u; p < pairs; ++p) if (tg_ids[p] == int(e)) pair_map[at++] = p;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 127u) {
      base_offsets[0] = pair_base + p_ex + p_sg + cnt;
      base_offsets[1] = tile_base + t_ex + t_sg + tiles;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0u) group_buf[0] = base_offsets[1];
}
