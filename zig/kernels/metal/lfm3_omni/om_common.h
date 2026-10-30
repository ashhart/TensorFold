// LFM3-Omni's Metal helpers: bf16 as ushort bits (round to nearest even), erf for the exact GELU, group reductions.
#pragma once
#include <metal_stdlib>
using namespace metal;

typedef ushort bf;

inline float f(bf x) { return as_type<float>(uint(x) << 16); }

inline bf b(float x) {
  const uint u = as_type<uint>(x);
  if ((u & 0x7FFFFFFFu) > 0x7F800000u) return bf((u >> 16) | 0x40u);
  return bf((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

inline float rb(float x) { return f(b(x)); }

inline float sigm(float x) { return 1.0f / (1.0f + precise::exp(-x)); }

// erf for the exact GELU: Numerical Recipes' erfcc Chebyshev fit, and the Taylor series near zero where it cancels.
inline float erf_f(float xf) {
  const float x = fabs(xf);
  float r;
  if (x < 0.25f) {
    const float t = x * x;
    r = 1.1283791670955126f * x * (1.0f - t * (1.0f / 3.0f - t * (0.1f - t * (1.0f / 42.0f - t * (1.0f / 216.0f)))));
  } else {
    const float t = 1.0f / (1.0f + 0.5f * x);
    const float e = t * precise::exp(-x * x - 1.26551223f + t * (1.00002368f + t * (0.37409196f + t * (0.09678418f + t * (-0.18628806f + t * (0.27886807f + t * (-1.13520398f + t * (1.48851587f + t * (-0.82215223f + t * 0.17087277f)))))))));
    r = 1.0f - e;
  }
  return xf < 0.0f ? -r : r;
}

template <uint NT>
inline float group_sum(float v, threadgroup float* red, uint tid) {
  v = simd_sum(v);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if ((tid & 31) == 0) red[tid >> 5] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float t = 0.0f;
  for (uint i = 0; i < NT / 32; ++i) t += red[i];
  return t;
}

template <uint NT>
inline float group_max(float v, threadgroup float* red, uint tid) {
  v = simd_max(v);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if ((tid & 31) == 0) red[tid >> 5] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float t = -INFINITY;
  for (uint i = 0; i < NT / 32; ++i) t = max(t, red[i]);
  return t;
}

