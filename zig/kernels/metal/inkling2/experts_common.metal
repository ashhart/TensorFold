// Inkling 2 Mini routed experts: shared definitions. The checkpoint stores each layer's experts as
//   NVFP4  (layers 0-19): U8 [E, N, K/2] E2M1 pairs (even element in the low nibble), F8_E4M3 [E, N, K/16] block scales
//                         and a per-expert factor amax[e] / outer_scale_denominator (stored F32),
//   MXFP8  (layers 20-23): F8_E4M3 [E, N, K] codes with U8 [E, N, K/32] UE8M0 scales (2^(s - 127)).
// Every kernel here follows the arithmetic its comments name rounding point for rounding point; fp32 contraction is off
// so a product and a sum round separately, and the divisions that decide a code are IEEE (precise::divide).
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

constant float kI2xF32Max = 3.4028234663852886e38f;

struct Inkling2ExpQuantArgs {
  uint rows, k;
};

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

// UE8M0: exponent s - 127 (including s=0, the 2^-127 scale); 255 is NaN.
// Apply both power-of-two factors together, avoiding a separately rounded scale product.
inline float i2x_mx_scale(float dot, uint xs, uint ws) {
  return (xs == 255u || ws == 255u) ? NAN : ldexp(dot, int(xs) + int(ws) - 254);
}

// Non-negative `v` to E4M3 by round-to-nearest-even, saturating at 448 (code 0x7e). Below 2^-6 the grid is m 2^-9.
inline uint i2x_e4m3_encode(float v) {
  if (!(v > 0.0f)) return 0u;
  if (v >= 448.0f) return 0x7eu;
  if (v < 0.015625f) return uint(rint(v * 512.0f));
  const uint bits = as_type<uint>(v);
  uint e = (bits >> 23) - 127u + 7u;
  uint q = (bits >> 20) & 7u;
  const uint rem = bits & 0xfffffu;
  if (rem > 0x80000u || (rem == 0x80000u && (q & 1u))) q += 1u;
  if (q == 8u) {
    q = 0u;
    e += 1u;
  }
  const uint code = (e << 3) | q;
  return code > 0x7eu ? 0x7eu : code;
}

// The value of the E2M1 code (magnitude 0..7) a grid value names; `v` is already on the grid.
inline uint i2x_e2m1_code(float v) {
  const float a = abs(v);
  uint c = uint(a >= 0.5f) + uint(a >= 1.0f) + uint(a >= 1.5f) + uint(a >= 2.0f) + uint(a >= 3.0f) + uint(a >= 4.0f) + uint(a >= 6.0f);
  return c | ((as_type<uint>(v) >> 28) & 8u);
}

// Round-to-nearest-even onto the E2M1 grid (three steps by magnitude), saturating at 6.
inline float i2x_e2m1_round(float x) {
  const float a = abs(x);
  float r;
  if (a < 2.0f) r = rint(2.0f * a) / 2.0f;
  else if (a < 4.0f) r = rint(a);
  else r = 2.0f * rint(a / 2.0f);
  r = min(r, 6.0f);
  return copysign(r, x);
}
