// Ripple's NVFP4 MLPs (ModelOpt): E2M1 nibbles low first, E4M3 scale per 16, fp32 global; fp4 * scale exact in bf16.
#include <metal_stdlib>
using namespace metal;

#define TF_UNROLL_4 _Pragma("clang loop unroll(full)")

struct rc_fp4_args { uint rows, n, k, ldy; float global; };

// E2M1: sign, two exponent bits, one mantissa bit.
inline float rc_e2m1(uint code) {
  const uint m = code & 7u;
  const float v = m < 4u ? float(m) * 0.5f : (m == 4u ? 2.0f : m == 5u ? 3.0f : m == 6u ? 4.0f : 6.0f);
  return (code & 8u) ? -v : v;
}

// Unsigned E4M3 (ModelOpt's scales are positive): exponent bias 7, three mantissa bits, 0x7f NaN as torch reads it.
inline float rc_e4m3(uint code) {
  const uint e = (code >> 3) & 15u, m = code & 7u;
  if ((code & 0x7fu) == 0x7fu) return NAN;
  return e ? ldexp(1.0f + float(m) * 0.125f, int(e) - 7) : float(m) * 0.001953125f;
}

// The 16 weights of block `b` in row j: two words of nibbles, each times the block's scale, exact in fp32 and bf16.
inline void rc_fp4_block(const device uint2* W, const device uchar* S, size_t at, thread float4 (&w)[4]) {
  const uint2 q = W[at];
  const float s = rc_e4m3(S[at]);
  TF_UNROLL_4
  for (uint c = 0; c < 4; ++c) {
    const uint word = c < 2 ? q.x : q.y, sh = 16u * (c & 1u);
    w[c] = float4(rc_e2m1(word >> sh), rc_e2m1(word >> (sh + 4u)), rc_e2m1(word >> (sh + 8u)), rc_e2m1(word >> (sh + 12u))) * s;
  }
}

// Y[r, j] = bf16(global * sum_i X[r, i] * W[j, i]) in fp32: a simdgroup an output j, a lane a block of 16 at a time.
template <uint ROWS>
inline void rc_fp4_matvec_t(const device bfloat* X, const device uint2* W, const device uchar* S,
    constant rc_fp4_args& a, device bfloat* Y, uint j, uint lane) {
  const uint blocks = a.k / 16;
  float acc[ROWS];
  TF_UNROLL_4
  for (uint r = 0; r < ROWS; ++r) acc[r] = 0.0f;
  for (uint b = lane; b < blocks; b += 32) {
    float4 w[4];
    rc_fp4_block(W, S, (size_t)j * blocks + b, w);
    TF_UNROLL_4
    for (uint r = 0; r < ROWS; ++r) {
      const device bfloat4* x = (const device bfloat4*)(X + (size_t)r * a.k + b * 16);
      float t = 0.0f;
      TF_UNROLL_4
      for (uint c = 0; c < 4; ++c) t += dot(w[c], float4(x[c]));
      acc[r] += t;
    }
  }
  TF_UNROLL_4
  for (uint r = 0; r < ROWS; ++r) {
    const float sum = simd_sum(acc[r]);
    if (lane == 0 && r < a.rows) Y[(size_t)r * a.ldy + j] = bfloat(sum * a.global);
  }
}

#define RC_FP4_MATVEC(ROWS)                                                                                        \
  [[kernel]] void rc_fp4_matvec_##ROWS(const device bfloat* X [[buffer(0)]], const device uint2* W [[buffer(1)]],  \
      const device uchar* S [[buffer(2)]], constant rc_fp4_args& a [[buffer(3)]], device bfloat* Y [[buffer(4)]],  \
      uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],                       \
      uint lane [[thread_index_in_simdgroup]]) {                                                                   \
    const uint j = tg.x * 8 + sg;                                                                                  \
    if (j < a.n) rc_fp4_matvec_t<ROWS>(X, W, S, a, Y, j, lane);                                                    \
  }
RC_FP4_MATVEC(1)
RC_FP4_MATVEC(2)
RC_FP4_MATVEC(4)
RC_FP4_MATVEC(8)
RC_FP4_MATVEC(16)
RC_FP4_MATVEC(32)

// W as bf16 [n, k] without its global, for a prompt chunk's GEMM: a thread a block of 16, each value exact.
[[kernel]] void rc_fp4_dequant(const device uint2* W [[buffer(0)]], const device uchar* S [[buffer(1)]],
    constant rc_fp4_args& a [[buffer(2)]], device bfloat4* D [[buffer(3)]], uint2 p [[thread_position_in_grid]]) {
  const uint blocks = a.k / 16;
  if (p.x >= blocks || p.y >= a.n) return;
  const size_t at = (size_t)p.y * blocks + p.x;
  float4 w[4];
  rc_fp4_block(W, S, at, w);
  TF_UNROLL_4
  for (uint c = 0; c < 4; ++c) D[at * 4 + c] = bfloat4(w[c]);
}
