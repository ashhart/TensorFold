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

// Two BF16 pairs of w and x multiplied and added to sum in column order.
inline float ink_loop_pair(float sum, uint w, uint x) {
  sum += as_type<float>(w << 16u) * as_type<float>(x << 16u);
  return sum + as_type<float>(w & 0xffff0000u) * as_type<float>(x & 0xffff0000u);
}
// A BF16 pair of w times two FP32 values, added to sum in column order.
inline float ink_loop_zpair(float sum, uint w, float z0, float z1) {
  sum += as_type<float>(w << 16u) * z0;
  return sum + as_type<float>(w & 0xffff0000u) * z1;
}

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
  threadgroup float sums[2 * 8], inverse[2];
  threadgroup float4 zeta4[INKLING_LOOP_RANK / 4];
  threadgroup uint4 normalized_words[2 * INKLING_LOOP_H / 8];
  threadgroup float* zeta = (threadgroup float*)zeta4;
  threadgroup bfloat* normalized = (threadgroup bfloat*)normalized_words;
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
    // One thread a rank: the serial chain over the 4,608 normalized values, 64 BF16 loaded ahead of each step.
    const device uint4* w = (const device uint4*)(down + size_t(t) * (2 * INKLING_LOOP_H));
    float dot = 0.0f;
    for (uint j = 0; j < 2 * INKLING_LOOP_H / 8; j += 8) {
      uint4 wv[8];
      for (uint i = 0; i < 8; ++i) wv[i] = w[j + i];
      for (uint i = 0; i < 8; ++i) {
        const uint4 xv = normalized_words[j + i];
        dot = ink_loop_pair(dot, wv[i].x, xv.x);
        dot = ink_loop_pair(dot, wv[i].y, xv.y);
        dot = ink_loop_pair(dot, wv[i].z, xv.z);
        dot = ink_loop_pair(dot, wv[i].w, xv.w);
      }
    }
    zeta[t] = float(bfloat(dot));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint d = t; d < INKLING_LOOP_H; d += 256) {
    // The forget and gain dots over zeta in rank order, each row loaded whole ahead of its chain.
    const device uint4* fw = (const device uint4*)(forget + size_t(d) * INKLING_LOOP_RANK);
    const device uint4* gw = (const device uint4*)(gain + size_t(d) * INKLING_LOOP_RANK);
    uint4 fv[8], gv[8];
    for (uint j = 0; j < 8; ++j) {
      fv[j] = fw[j];
      gv[j] = gw[j];
    }
    float fdot = 0.0f, gdot = 0.0f;
    for (uint j = 0; j < 8; ++j) {
      const float4 z0 = zeta4[2 * j], z1 = zeta4[2 * j + 1];
      fdot = ink_loop_zpair(fdot, fv[j].x, z0.x, z0.y);
      fdot = ink_loop_zpair(fdot, fv[j].y, z0.z, z0.w);
      fdot = ink_loop_zpair(fdot, fv[j].z, z1.x, z1.y);
      fdot = ink_loop_zpair(fdot, fv[j].w, z1.z, z1.w);
      gdot = ink_loop_zpair(gdot, gv[j].x, z0.x, z0.y);
      gdot = ink_loop_zpair(gdot, gv[j].y, z0.z, z0.w);
      gdot = ink_loop_zpair(gdot, gv[j].z, z1.x, z1.y);
      gdot = ink_loop_zpair(gdot, gv[j].w, z1.z, z1.w);
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
