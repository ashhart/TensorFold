#include <metal_stdlib>
#pragma METAL fp contract(off)
using namespace metal;

// Cross-pass dots use fixed-order FP32 accumulation and SIMD reductions; no BF16 value accumulates.
constant float INKLING_LOOP_EPS = 1e-6f;
constant float INKLING_LOOP_SCALE = 0x1.555556p-6f; // binary32(1 / 48)
constant uint INKLING_LOOP_H = 2304;
constant uint INKLING_LOOP_RANK = 64;
constant uint INKLING_LOOP_MESSAGES = 48;

struct InkLoopArgs { uint rows; };

[[max_total_threads_per_threadgroup(256)]]
kernel void tf_inkling2_loop_messages(
    const device bfloat* bank0 [[buffer(0)]], const device bfloat* bank1 [[buffer(1)]],
    const device float* q_mlp [[buffer(2)]], const device float* kv [[buffer(3)]],
    device float* out [[buffer(4)]], constant InkLoopArgs& a [[buffer(5)]],
    uint t [[thread_position_in_threadgroup]], uint work [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint row = work / INKLING_LOOP_MESSAGES;
  const uint entry = work - row * INKLING_LOOP_MESSAGES;
  if (row >= a.rows) return;
  const uint layer = entry / 4;
  const uint role = entry - 4 * layer;
  const device float* queries = (role == 0 || role == 3) ? q_mlp : kv;
  // q/mlp and k/v occupy even/odd rows respectively.
  const device float* query = queries + size_t(2 * layer + (role >= 2 ? 1 : 0)) * INKLING_LOOP_H;
  const device bfloat* banks[2] = { bank0 + size_t(row) * INKLING_LOOP_H,
                                    bank1 + size_t(row) * INKLING_LOOP_H };
  threadgroup float sums[2 * 8], inverse[2], dots[2 * 8], weights[2];
  for (uint s = 0; s < 2; ++s) {
    float ss = 0.0f;
    for (uint d = t; d < INKLING_LOOP_H; d += 256) {
      const float x = float(banks[s][d]);
      ss += x * x;
    }
    const float reduced = simd_sum(ss);
    if (lane == 0) sums[s * 8 + t / 32] = reduced;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) for (uint s = 0; s < 2; ++s) {
    float ss = 0.0f;
    for (uint j = 0; j < 8; ++j) ss += sums[s * 8 + j];
    inverse[s] = precise::rsqrt(precise::divide(ss, 2304.0f) + INKLING_LOOP_EPS);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = 0; s < 2; ++s) {
    float dot = 0.0f;
    for (uint d = t; d < INKLING_LOOP_H; d += 256) {
      const float beta = float(banks[s][d]) * inverse[s];
      dot += query[d] * beta;
    }
    const float reduced = simd_sum(dot);
    if (lane == 0) dots[s * 8 + t / 32] = reduced;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    float logits[2];
    float peak = -INFINITY;
    for (uint s = 0; s < 2; ++s) {
      float dot = 0.0f;
      for (uint j = 0; j < 8; ++j) dot += dots[s * 8 + j];
      weights[s] = dot * INKLING_LOOP_SCALE;
      logits[s] = weights[s];
      peak = max(peak, logits[s]);
    }
    float den = 0.0f;
    for (uint s = 0; s < 2; ++s) {
      weights[s] = precise::exp(logits[s] - peak);
      den += weights[s];
    }
    for (uint s = 0; s < 2; ++s) weights[s] = precise::divide(weights[s], den);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint d = t; d < INKLING_LOOP_H; d += 256) {
    const float beta0 = float(banks[0][d]) * inverse[0];
    const float beta1 = float(banks[1][d]) * inverse[1];
    out[(size_t(row) * INKLING_LOOP_MESSAGES + entry) * INKLING_LOOP_H + d] =
        weights[0] * beta0 + weights[1] * beta1;
  }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void tf_inkling2_loop_gates(
    const device float* h [[buffer(0)]], const device float* m [[buffer(1)]],
    const device bfloat* down [[buffer(2)]], const device bfloat* forget [[buffer(3)]],
    const device bfloat* gain [[buffer(4)]], device float* out [[buffer(5)]],
    constant InkLoopArgs& a [[buffer(6)]], uint t [[thread_position_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  if (row >= a.rows) return;
  const device float* inputs[2] = { h + size_t(row) * INKLING_LOOP_H,
                                    m + size_t(row) * INKLING_LOOP_H };
  threadgroup float sums[2 * 8], inverse[2], zeta[INKLING_LOOP_RANK];
  threadgroup bfloat normalized[2 * INKLING_LOOP_H];
  for (uint s = 0; s < 2; ++s) {
    float ss = 0.0f;
    for (uint d = t; d < INKLING_LOOP_H; d += 256) {
      const float x = inputs[s][d];
      ss += x * x;
    }
    const float reduced = simd_sum(ss);
    if (lane == 0) sums[s * 8 + t / 32] = reduced;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) for (uint s = 0; s < 2; ++s) {
    float ss = 0.0f;
    for (uint j = 0; j < 8; ++j) ss += sums[s * 8 + j];
    inverse[s] = precise::rsqrt(precise::divide(ss, 2304.0f) + INKLING_LOOP_EPS);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint d = t; d < INKLING_LOOP_H; d += 256) {
    normalized[d] = bfloat(h[size_t(row) * INKLING_LOOP_H + d] * inverse[0]);
    normalized[INKLING_LOOP_H + d] = bfloat(m[size_t(row) * INKLING_LOOP_H + d] * inverse[1]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < INKLING_LOOP_RANK) {
    float dot = 0.0f;
    for (uint d = 0; d < 2 * INKLING_LOOP_H; ++d)
      dot += float(down[size_t(t) * (2 * INKLING_LOOP_H) + d]) * float(normalized[d]);
    zeta[t] = float(bfloat(dot));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint d = t; d < INKLING_LOOP_H; d += 256) {
    float fdot = 0.0f, gdot = 0.0f;
    for (uint r = 0; r < INKLING_LOOP_RANK; ++r) {
      fdot += float(forget[size_t(d) * INKLING_LOOP_RANK + r]) * zeta[r];
      gdot += float(gain[size_t(d) * INKLING_LOOP_RANK + r]) * zeta[r];
    }
    const float f = precise::tanh(fdot);
    const float g = precise::tanh(gdot);
    // Four separately rounded binary32 steps; do not fold these into one expression.
    const float keep = 1.0f - f;
    const float left = keep * h[size_t(row) * INKLING_LOOP_H + d];
    const float right = g * m[size_t(row) * INKLING_LOOP_H + d];
    out[size_t(row) * INKLING_LOOP_H + d] = left + right;
  }
}
