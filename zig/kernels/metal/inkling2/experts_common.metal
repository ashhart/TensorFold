// Inkling 2 Mini routed experts: shared definitions. The checkpoint stores each layer's experts as
//   NVFP4  (layers 0-19): U8 [E, N, K/2] E2M1 pairs (even element in the low nibble), F8_E4M3 [E, N, K/16] block scales
//                         and F32 amax and outer_scale_denominator metadata, cast separately to BF16 before dividing,
//   MXFP8  (layers 20-23): F8_E4M3 [E, N, K] codes with U8 [E, N, K/32] UE8M0 scales (2^(s - 127)).
// Match the BF16 materialization boundaries in Transformers PR #51050 (6173a2f6), without expanding the weights.
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)


// Both dot stages share one argument block. `src_div` maps a (row, slot) pair to its input row (topk for the gate/up
// stage, whose input is the layer's rows; 1 for the down stage, whose input is one row a pair). `n` is the stage's
// weight rows per expert: 2 * inter for gate/up (the output is inter wide), hidden for down.
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
