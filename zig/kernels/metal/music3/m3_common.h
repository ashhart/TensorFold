// music3 kernel helpers: bf16 as ushort bits (round to nearest even), group reductions, the Philox counter draws.
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

// Round to bf16 / fp16 and back (the reference's rounded intermediates).
inline float rb(float x) { return b2f(f2b(x)); }
inline float rh(float x) { return float(half(x)); }

inline float m3_silu(float x) { return x / (1.0f + precise::exp(-x)); }

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

// Philox4x32-10 of counter (a, b) under key `seed`, as the CUDA kernels draw it.
inline uint4 philox(ulong seed, ulong a, ulong b) {
  uint c0 = uint(a), c1 = uint(a >> 32), c2 = uint(b), c3 = uint(b >> 32);
  uint k0 = uint(seed), k1 = uint(seed >> 32);
  for (int i = 0; i < 10; ++i) {
    const uint h0 = mulhi(0xD2511F53u, c0), l0 = 0xD2511F53u * c0;
    const uint h1 = mulhi(0xCD9E8D57u, c2), l1 = 0xCD9E8D57u * c2;
    const uint n0 = h1 ^ c1 ^ k0, n2 = h0 ^ c3 ^ k1;
    c0 = n0;
    c1 = l1;
    c2 = n2;
    c3 = l0;
    k0 += 0x9E3779B9u;
    k1 += 0xBB67AE85u;
  }
  return uint4(c0, c1, c2, c3);
}

// The first word as a uniform in [0, 1) (24 bits).
inline float philox_uniform(ulong seed, ulong a, ulong b) { return float(philox(seed, a, b).x >> 8) * (1.0f / 16777216.0f); }

// Two normals (Box-Muller) of counter (a, b).
inline float2 philox_normal2(ulong seed, ulong a, ulong b) {
  const uint4 c = philox(seed, a, b);
  const float u1 = (float(c.x >> 8) + 1.0f) * (1.0f / 16777216.0f);
  const float u2 = float(c.y >> 8) * (1.0f / 16777216.0f);
  const float r = precise::sqrt(-2.0f * precise::log(u1));
  return float2(r * precise::cos(6.2831853071795864f * u2), r * precise::sin(6.2831853071795864f * u2));
}
