// Drama 3 kernel helpers: bf16 as ushort bits (round to nearest even), group reductions, e4m3 in both directions.
#pragma once
#include <metal_stdlib>
using namespace metal;

typedef ushort bf16;

inline float b2f(bf16 x) { return as_type<float>(uint(x) << 16); }

inline bf16 f2b(float x) {
  const uint u = as_type<uint>(x);
  if ((u & 0x7FFFFFFFu) > 0x7F800000u) return bf16((u >> 16) | 0x40u);
  return bf16((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

// Round to bf16 and back (the reference's bf16 intermediates).
inline float rb(float x) { return b2f(f2b(x)); }

inline float d3_sigmoid(float x) { return 1.0f / (1.0f + precise::exp(-x)); }
inline float d3_silu(float x) { return x / (1.0f + precise::exp(-x)); }
inline float d3_softplus(float x) { return x > 20.0f ? x : precise::log(1.0f + precise::exp(x)); }

// Four bf16 values of 8 bytes as floats.
inline float4 bf4(uint2 w) {
  return float4(as_type<float>(w.x << 16), as_type<float>(w.x & 0xFFFF0000u), as_type<float>(w.y << 16),
                as_type<float>(w.y & 0xFFFF0000u));
}

// Sum over a threadgroup of `n` simdgroups (scratch holds 32 floats); every thread gets the total.
inline float group_sum(float v, threadgroup float* scratch, uint sg, uint lane, uint n) {
  v = simd_sum(v);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) scratch[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float s = 0.0f;
  for (uint i = 0; i < n; ++i) s += scratch[i];
  return s;
}

inline float group_max(float v, threadgroup float* scratch, uint sg, uint lane, uint n) {
  v = simd_max(v);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) scratch[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float s = -INFINITY;
  for (uint i = 0; i < n; ++i) s = max(s, scratch[i]);
  return s;
}

// float to the nearest E4M3 value (ties to even), |x| <= 448 assumed (the caller clamps), returned as a float.
inline float to_e4m3(float x) {
  const float a = fabs(x);
  if (!(a > 0.0f)) return x;
  float q;
  if (a < 0.015625f) {
    q = rint(a * 512.0f) * 0.001953125f;
  } else {
    int ex;
    (void)frexp(a, ex);
    const float step = ldexp(1.0f, ex - 4);
    q = rint(a / step) * step;
  }
  return x < 0.0f ? -q : q;
}

inline bool is_semantic(int tok, int sem_begin, int sem_end) { return tok >= sem_begin && tok <= sem_end; }

// The samplers' settings, as the host passes them (16 + 8 bytes).
struct D3Sampling {
  float temperature;
  int top_k;
  float top_p;
  int pad;
  ulong seed;
};
