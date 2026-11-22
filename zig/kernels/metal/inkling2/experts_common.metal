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

// E4M3FN as FP16 bits (same layout, bias 15) times 2^8; the sign sits in bit 15, and 0x7f / 0xff stay NaN.
inline float i2x_e4m3(uint b) {
  const float v = float(as_type<half>(ushort(((b & 0x7fu) << 7) | ((b & 0x80u) << 8)))) * 256.0f;
  return (b & 0x7fu) == 0x7fu ? as_type<float>(((b & 0x80u) << 24) | 0x7fc00000u) : v;
}

// Keep 2^-127 out of FP32 scale operands on Metal; the MXFP8 dot halves the code for s == 0.
inline float i2x_mx_scale(uint s) {
  return as_type<float>(max(s, 1u) << 23);
}
