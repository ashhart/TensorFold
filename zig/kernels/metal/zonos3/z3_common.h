// ours: ZONOS3 Mini's Metal helpers: bf16 as ushort bits, the CUDA kernels' reduction orders, precise fp32 math.
#pragma once
// No contraction, as the build's -ffp-contract=off, so a run-time compile (the kernel contract's) takes the same steps.
#pragma METAL fp contract(off)
#include <metal_stdlib>
using namespace metal;

typedef ushort bf16;

inline float b2f(bf16 x) { return as_type<float>(uint(x) << 16); }

// xor butterfly, offsets 16 down to 1 (tf_common::warp_sum's order)
inline float warp_sum(float v) {
  for (ushort o = 16; o > 0; o >>= 1) v += simd_shuffle_xor(v, o);
  return v;
}

inline float warp_max(float v) {
  for (ushort o = 16; o > 0; o >>= 1) v = fmax(v, simd_shuffle_xor(v, o));
  return v;
}

// Each simdgroup's butterfly, then one more butterfly over the `groups` sums (zeros past them); every thread calls.
inline float block_sum_tree(float v, threadgroup float* scratch, uint groups, uint lane, uint sg) {
  v = warp_sum(v);
  if (lane == 0) scratch[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  v = warp_sum(lane < groups ? scratch[lane] : 0.0f);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return v;
}

inline float block_max_tree(float v, threadgroup float* scratch, uint groups, uint lane, uint sg) {
  v = warp_max(v);
  if (lane == 0) scratch[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  v = warp_max(lane < groups ? scratch[lane] : -INFINITY);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return v;
}

// 1 / sqrt(ss / width + eps), each step correctly rounded.
inline float rms_scale(float ss, int width, float eps) {
  return precise::divide(1.0f, precise::sqrt(precise::divide(ss, float(width)) + eps));
}

inline float z3_exp(float x) { return precise::exp(x); }
inline float z3_sigmoid(float x) { return precise::divide(1.0f, 1.0f + precise::exp(-x)); }
inline float z3_silu(float x) { return precise::divide(x, 1.0f + precise::exp(-x)); }
// log1p(u) for u >= 0 without the rounding of 1 + u: log(w) * u / (w - 1), w = 1 + u.
inline float z3_log1p(float u) {
  const float w = 1.0f + u;
  return w == 1.0f ? u : precise::log(w) * precise::divide(u, w - 1.0f);
}

inline float z3_softplus(float x) { return x > 20.0f ? x : z3_log1p(precise::exp(x)); }
