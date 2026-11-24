// Inkling 2 Mini M3: packed weight decode in registers, BF16 activations, FP32 reductions.
#include <metal_stdlib>
using namespace metal;
#pragma METAL fp contract(off)

struct Ink2MatrixArgs { uint k, n, topk, experts; };
struct Ink2DecodeArgs { uint rows, k; };

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

inline float ink2_mx_weight(uchar value, uchar scale) {
  const float w = ldexp(ink2_e4m3(value), int(scale) - 127);
  // E4M3 times a power of two is exactly representable in BF16, still round explicitly.
  return float(bfloat(w));
}

inline float ink2_dot_nv(device const uchar* w, device const uchar* scales,
                         device const bfloat* x, float phi, uint k) {
  float sum = 0.0f;
  for (uint c = 0; c < k; ++c)
    sum += ink2_nv_weight(w[c / 2u], c, scales[c / 16u], phi) * float(x[c]);
  return sum;
}

inline float ink2_dot_mx(device const uchar* w, device const uchar* scales,
                         device const bfloat* x, uint k) {
  float sum = 0.0f;
  for (uint c = 0; c < k; ++c)
    sum += ink2_mx_weight(w[c], scales[c / 32u]) * float(x[c]);
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

[[kernel, max_total_threads_per_threadgroup(32)]] void tf_ink2_combine(
    device const bfloat* down [[buffer(0)]], device const int* ids [[buffer(1)]],
    device const float* weights [[buffer(2)]], device bfloat* Y [[buffer(3)]],
    constant Ink2MatrixArgs& a [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
  // One thread an output: threadgroup x is 32 outputs, y the token.
  const uint n = tg.x * 32u + lane, group = tg.y;
  if (n >= a.n) return;
  // Outer expert order, not route-slot order, fixes the reference's accumulation order; equal ids keep slot order.
  uint slots[8], used = 0u;
  for (uint slot = 0; slot < a.topk; ++slot) {
    const uint id = uint(ids[group * a.topk + slot]);
    if (id >= a.experts) continue;
    uint at = used++;
    while (at > 0u && uint(ids[group * a.topk + slots[at - 1u]]) > id) {
      slots[at] = slots[at - 1u];
      --at;
    }
    slots[at] = slot;
  }
  float sum = 0.0f;
  for (uint i = 0; i < used; ++i)
    sum += weights[group * a.topk + slots[i]] * float(down[(group * a.topk + slots[i]) * a.n + n]);
  Y[group * a.n + n] = bfloat(sum);
}
