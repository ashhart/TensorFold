// Packed experts share the reference's BF16 materialization boundaries.
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

// src_div is topk for gate/up, one for down; n counts the stored weight rows per expert.
struct Inkling2ExpDotArgs {
  uint pairs, src_div, k, n;
  uint experts, out_width;
  float gate_cap, up_cap;
};

struct Inkling2ExpCombineArgs {
  uint rows, topk, hidden, experts;
};

constant float kI2xE2m1[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

// E4M3FN (bias 7, no infinities, 0x7f / 0xff are NaN): (8 + m) 2^(e - 10), or m 2^-9 below the least normal.
inline float i2x_e4m3(uint b) {
  const uint e = (b >> 3) & 15u, m = b & 7u;
  float v;
  if (e == 0u) v = float(m) * 0.001953125f;
  else v = as_type<float>(((e + 120u) << 23) | (m << 20));
  if ((b & 0x7fu) == 0x7fu) v = NAN;
  return (b & 0x80u) ? -v : v;
}

// Keep 2^-127 out of FP32 scale operands on Metal; the MXFP8 dot halves the code for s == 0.
inline float i2x_mx_scale(uint s) {
  return as_type<float>(max(s, 1u) << 23);
}
