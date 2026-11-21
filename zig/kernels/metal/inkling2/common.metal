// Inkling 2 helpers: fixed-order fp32 trees (lane partials, simd_sum, simdgroup partials) keep rows deterministic.
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

constant constexpr uint I2_TG = 256;       // threads of a row kernel: eight simdgroups
constant constexpr uint I2_SGS = 8;        // simdgroups of a row kernel
constant constexpr uint I2_MAX_ELEMS = 24; // elements a row-kernel thread holds: hidden <= 24 * 256

// Sum of v over the 256 threads, the same bits in every thread. `scratch` holds I2_SGS floats; safe to reuse at once.
inline float i2_tg_sum(float v, threadgroup float* scratch, uint sg, uint lane) {
  v = simd_sum(v);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) scratch[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float t = scratch[0];
  for (uint i = 1; i < I2_SGS; ++i) t += scratch[i];
  return t;
}

// max with NaN propagating (a NaN on either side wins), as float64 arithmetic on the formula does.
inline float i2_max(float a, float b) {
  if (isnan(a) || isnan(b)) return NAN;
  return a > b ? a : b;
}

inline float i2_sigmoid(float x) { return 1.0f / (1.0f + metal::precise::exp(-x)); }

// RMS norm scale 1 / sqrt(mean + eps), the torch rsqrt in fp32.
inline float i2_rsqrt_mean(float sum, uint n, float eps) { return metal::precise::rsqrt(sum / float(n) + eps); }

// Scale of a clipped head vector: min(threshold / (norm + 1e-6), 1), a NaN staying a NaN as torch's clamp keeps it.
inline float i2_clip_factor(float sum_squares, float threshold) {
  const float f = threshold / (metal::precise::sqrt(sum_squares) + 1e-6f);
  return isnan(f) ? f : min(f, 1.0f);
}
