// Kimi K3 kernel helpers: every sum is an explicit fma chain or simd_sum, so a row's bits never depend on its round.
#pragma once
#include <metal_stdlib>
using namespace metal;

inline float k3_sigmoid(float x) {
  return 1.0f / (1.0f + precise::exp(-x));
}

// Four bf16 values packed in 8 bytes (checkpoint tensors are only 8-byte aligned), as floats.
inline float4 k3_bf16x4(uint2 w) {
  return float4(float2(as_type<bfloat2>(w.x)), float2(as_type<bfloat2>(w.y)));
}

// Two E2M1 codes of a byte as half bits scaled by 2^-14 (0.5 is a half subnormal), low nibble first.
inline float2 k3_fp4_pair(uint b) {
  const uint mags = ((b & 0x77u) * 0x00200200u) & 0x0E000E00u;
  const uint signs = ((b & 0x88u) * 0x01001000u) & 0x80008000u;
  return float2(as_type<half2>(mags | signs));
}

// E8M0 scale 2^(e-127) times 2^14 (undoing k3_fp4_pair's scale): exact for e in 0..240.
inline float k3_e8m0(uchar e) {
  return as_type<float>((uint(e) + 14u) << 23);
}

// fma chain over the 16 E2M1 codes of 8 bytes against 16 activations, in code order.
inline float k3_fp4_dot16(float p, const thread float* x, uint2 w) {
  for (int b = 0; b < 8; ++b) {
    const float2 v = k3_fp4_pair(((b < 4 ? w.x : w.y) >> (8 * (b & 3))) & 0xFFu);
    p = fma(x[2 * b], v.x, p);
    p = fma(x[2 * b + 1], v.y, p);
  }
  return p;
}

// Kimi's SiTU: beta tanh(g/beta) sigmoid(g) times lin tanh(u/lin), on the bf16-rounded projections.
inline float k3_situ(float g, float u, float beta, float lin) {
  return beta * precise::tanh(g / beta) * k3_sigmoid(g) * (lin * precise::tanh(u / lin));
}
