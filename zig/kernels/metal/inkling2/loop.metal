#include <metal_stdlib>
#pragma METAL fp contract(off)
using namespace metal;

// Cross-pass dots use fixed-order FP32 accumulation and SIMD reductions; no BF16 value accumulates.
constant float INKLING_LOOP_EPS = 1e-6f;
constant float INKLING_LOOP_SCALE = 0x1.555556p-6f; // binary32(1 / 48)
constant uint INKLING_LOOP_H = 2304;
constant uint INKLING_LOOP_RANK = 64;
constant uint INKLING_LOOP_MESSAGES = 48;

// stride: floats from one row's m to the next (the messages interleave 48 entries a row).
struct InkLoopArgs { uint rows, stride; };

[[max_total_threads_per_threadgroup(256)]]
kernel void tf_inkling2_loop_rms(
    const device bfloat* bank0 [[buffer(0)]], const device bfloat* bank1 [[buffer(1)]],
    device float* inv [[buffer(2)]], constant InkLoopArgs& a [[buffer(3)]],
    uint t [[thread_position_in_threadgroup]], uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  if (row >= a.rows) return;
  const device bfloat* b0 = bank0 + size_t(row) * INKLING_LOOP_H;
  const device bfloat* b1 = bank1 + size_t(row) * INKLING_LOOP_H;
  threadgroup float sums[16];
  float ss0 = 0.0f, ss1 = 0.0f;
  for (uint d = t; d < INKLING_LOOP_H; d += 256u) {
    const float x0 = float(b0[d]), x1 = float(b1[d]);
    ss0 += x0 * x0;
    ss1 += x1 * x1;
  }
  const uint sg = t / 32u;
  const float r0 = simd_sum(ss0), r1 = simd_sum(ss1);
  if (lane == 0u) {
    sums[sg] = r0;
    sums[8u + sg] = r1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < 2u) {
    float ss = 0.0f;
    for (uint j = 0u; j < 8u; ++j) ss += sums[t * 8u + j];
    inv[row * 2u + t] = precise::rsqrt(precise::divide(ss, 2304.0f) + INKLING_LOOP_EPS);
  }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void tf_inkling2_loop_messages(
    const device bfloat* bank0 [[buffer(0)]], const device bfloat* bank1 [[buffer(1)]],
    const device float* q_mlp [[buffer(2)]], const device float* kv [[buffer(3)]],
    const device float* inv [[buffer(4)]], device float* out [[buffer(5)]],
    constant InkLoopArgs& a [[buffer(6)]],
    uint t [[thread_position_in_threadgroup]], uint work [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint row = work / INKLING_LOOP_MESSAGES;
  const uint entry = work - row * INKLING_LOOP_MESSAGES;
  if (row >= a.rows) return;
  const uint layer = entry / 4u;
  const uint role = entry - 4u * layer;
  const device float* queries = (role == 0u || role == 3u) ? q_mlp : kv;
  const device float* query = queries + size_t(2u * layer + (role >= 2u ? 1u : 0u)) * INKLING_LOOP_H;
  const device bfloat* b0p = bank0 + size_t(row) * INKLING_LOOP_H;
  const device bfloat* b1p = bank1 + size_t(row) * INKLING_LOOP_H;
  const float inv0 = inv[row * 2u], inv1 = inv[row * 2u + 1u];
  threadgroup float dots[16], weights[4];
  float b0[9], b1[9], dot0 = 0.0f, dot1 = 0.0f;
  for (uint k = 0u; k < 9u; ++k) {
    const uint d = t + k * 256u;
    b0[k] = float(b0p[d]) * inv0;
    b1[k] = float(b1p[d]) * inv1;
    const float qv = query[d];
    dot0 += qv * b0[k];
    dot1 += qv * b1[k];
  }
  const uint sg = t / 32u;
  const float r0 = simd_sum(dot0), r1 = simd_sum(dot1);
  if (lane == 0u) {
    dots[sg] = r0;
    dots[8u + sg] = r1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0u) {
    float logits[2];
    float peak = -INFINITY;
    for (uint s = 0u; s < 2u; ++s) {
      float dot = 0.0f;
      for (uint j = 0u; j < 8u; ++j) dot += dots[s * 8u + j];
      weights[s] = dot * INKLING_LOOP_SCALE;
      logits[s] = weights[s];
      peak = max(peak, logits[s]);
    }
    float den = 0.0f;
    for (uint s = 0u; s < 2u; ++s) {
      weights[s] = precise::exp(logits[s] - peak);
      den += weights[s];
    }
    for (uint s = 0u; s < 2u; ++s) weights[s] = precise::divide(weights[s], den);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float w0 = weights[0], w1 = weights[1];
  device float* dst = out + (size_t(row) * INKLING_LOOP_MESSAGES + entry) * INKLING_LOOP_H;
  for (uint k = 0u; k < 9u; ++k) dst[t + k * 256u] = w0 * b0[k] + w1 * b1[k];
}

// The gates' first launch: h and m each to unit RMS and BF16, as normalized[row][h | m].
kernel void tf_inkling2_loop_gates(
    const device float* h [[buffer(0)]], const device float* m [[buffer(1)]],
    device bfloat* normalized [[buffer(2)]], constant InkLoopArgs& a [[buffer(3)]],
    uint t [[thread_position_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  if (row >= a.rows) return;
  const device float* inputs[2] = { h + size_t(row) * INKLING_LOOP_H,
                                    m + size_t(row) * a.stride };
  threadgroup float sums[2 * 8], inverse[2];
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
  device bfloat* y = normalized + size_t(row) * (2 * INKLING_LOOP_H);
  for (uint d = t; d < INKLING_LOOP_H; d += 256) {
    y[d] = bfloat(inputs[0][d] * inverse[0]);
    y[INKLING_LOOP_H + d] = bfloat(inputs[1][d] * inverse[1]);
  }
}

// The second: zeta[row][r] = bf16(down[r] . normalized[row]) in fp32, a simdgroup a rank, eight columns a lane a step.
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_inkling2_loop_zeta(
    const device bfloat* normalized [[buffer(0)]], const device bfloat* down [[buffer(1)]],
    device bfloat* zeta [[buffer(2)]], constant InkLoopArgs& a [[buffer(3)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint r = tg.x * 8 + sg, row = tg.y;
  if (r >= INKLING_LOOP_RANK || row >= a.rows) return;
  const device bfloat* w = down + size_t(r) * (2 * INKLING_LOOP_H);
  const device bfloat* x = normalized + size_t(row) * (2 * INKLING_LOOP_H);
  float acc = 0.0f;
  for (uint i = 8 * lane; i < 2 * INKLING_LOOP_H; i += 256) {
    const float4 w0 = float4(*(const device bfloat4*)(w + i)), w1 = float4(*(const device bfloat4*)(w + i + 4));
    const float4 x0 = float4(*(const device bfloat4*)(x + i)), x1 = float4(*(const device bfloat4*)(x + i + 4));
    acc += dot(w0, x0) + dot(w1, x1);
  }
  const float sum = simd_sum(acc);
  if (lane == 0) zeta[row * INKLING_LOOP_RANK + r] = bfloat(sum);
}

// The third: each output's forget and gain dots over zeta in rank order, then the four-step blend.
[[max_total_threads_per_threadgroup(32)]]
kernel void tf_inkling2_loop_blend(
    const device float* h [[buffer(0)]], const device float* m [[buffer(1)]],
    const device bfloat* zeta [[buffer(2)]], const device bfloat* forget [[buffer(3)]],
    const device bfloat* gain [[buffer(4)]], device float* out [[buffer(5)]],
    constant InkLoopArgs& a [[buffer(6)]], uint2 at [[thread_position_in_grid]]) {
  const uint d = at.x, row = at.y;
  if (d >= INKLING_LOOP_H || row >= a.rows) return;
  {
    float fdot = 0.0f, gdot = 0.0f;
    for (uint r = 0; r < INKLING_LOOP_RANK; ++r) {
      const float z = float(zeta[row * INKLING_LOOP_RANK + r]);
      fdot += float(forget[size_t(d) * INKLING_LOOP_RANK + r]) * z;
      gdot += float(gain[size_t(d) * INKLING_LOOP_RANK + r]) * z;
    }
    const float f = precise::tanh(fdot);
    const float g = precise::tanh(gdot);
    // Four separately rounded binary32 steps; do not fold these into one expression.
    const float keep = 1.0f - f;
    const float left = keep * h[size_t(row) * INKLING_LOOP_H + d];
    const float right = g * m[size_t(row) * a.stride + d];
    out[size_t(row) * INKLING_LOOP_H + d] = left + right;
  }
}
