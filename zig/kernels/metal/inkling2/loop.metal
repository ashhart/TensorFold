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
struct InkLoopArgs { uint rows, stride, roles; };
constant uint INK2_GATE_SCRATCH_STRIDE = 64u * (2u * INKLING_LOOP_H + INKLING_LOOP_RANK);

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

// The gates' first launch: h and m each to unit RMS and BF16 across up to 3 controller roles in grid.z.
kernel void tf_inkling2_loop_gates(
    const device float* h0 [[buffer(0)]], const device float* h1 [[buffer(1)]],
    const device float* h2 [[buffer(2)]], const device float* m0 [[buffer(3)]],
    const device float* m1 [[buffer(4)]], const device float* m2 [[buffer(5)]],
    device bfloat* normalized [[buffer(6)]], constant InkLoopArgs& a [[buffer(7)]],
    uint t [[thread_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint row = tg.x, role = tg.z;
  if (row >= a.rows || role >= a.roles) return;
  const device float* h = (role == 0u) ? h0 : ((role == 1u) ? h1 : h2);
  const device float* m = (role == 0u) ? m0 : ((role == 1u) ? m1 : m2);
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
  device bfloat* y = normalized + size_t(role) * INK2_GATE_SCRATCH_STRIDE + size_t(row) * (2 * INKLING_LOOP_H);
  for (uint d = t; d < INKLING_LOOP_H; d += 256) {
    y[d] = bfloat(inputs[0][d] * inverse[0]);
    y[INKLING_LOOP_H + d] = bfloat(inputs[1][d] * inverse[1]);
  }
}

// The second: zeta[row][r] = bf16(down[r] . normalized[row]) in fp32 per grid.z role, eight columns a lane a step.
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_inkling2_loop_zeta(
    const device bfloat* normalized [[buffer(0)]], const device bfloat* down0 [[buffer(1)]],
    const device bfloat* down1 [[buffer(2)]], const device bfloat* down2 [[buffer(3)]],
    device bfloat* zeta [[buffer(4)]], constant InkLoopArgs& a [[buffer(5)]],
    uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint r = tg.x * 8 + sg, row = tg.y, role = tg.z;
  if (r >= INKLING_LOOP_RANK || row >= a.rows || role >= a.roles) return;
  const device bfloat* down = (role == 0u) ? down0 : ((role == 1u) ? down1 : down2);
  const device bfloat* w = down + size_t(r) * (2 * INKLING_LOOP_H);
  const device bfloat* x = normalized + size_t(role) * INK2_GATE_SCRATCH_STRIDE + size_t(row) * (2 * INKLING_LOOP_H);
  zeta += size_t(role) * INK2_GATE_SCRATCH_STRIDE;
  float acc = 0.0f;
  for (uint i = 8 * lane; i < 2 * INKLING_LOOP_H; i += 256) {
    const float4 w0 = float4(*(const device bfloat4*)(w + i)), w1 = float4(*(const device bfloat4*)(w + i + 4));
    const float4 x0 = float4(*(const device bfloat4*)(x + i)), x1 = float4(*(const device bfloat4*)(x + i + 4));
    acc += dot(w0, x0) + dot(w1, x1);
  }
  const float sum = simd_sum(acc);
  if (lane == 0) zeta[row * INKLING_LOOP_RANK + r] = bfloat(sum);
}

// The third: each output's forget and gain dots over zeta in rank order per grid.z role, then the four-step blend.
kernel void tf_inkling2_loop_blend(
    const device float* h0 [[buffer(0)]], const device float* h1 [[buffer(1)]],
    const device float* h2 [[buffer(2)]], const device float* m0 [[buffer(3)]],
    const device float* m1 [[buffer(4)]], const device float* m2 [[buffer(5)]],
    const device bfloat* zeta [[buffer(6)]], const device bfloat* forget0 [[buffer(7)]],
    const device bfloat* forget1 [[buffer(8)]], const device bfloat* forget2 [[buffer(9)]],
    const device bfloat* gain0 [[buffer(10)]], const device bfloat* gain1 [[buffer(11)]],
    const device bfloat* gain2 [[buffer(12)]], device float* out0 [[buffer(13)]],
    device float* out1 [[buffer(14)]], device float* out2 [[buffer(15)]],
    constant InkLoopArgs& a [[buffer(16)]], uint3 at [[thread_position_in_grid]]) {
  const uint d = at.x, row = at.y, role = at.z;
  if (d >= INKLING_LOOP_H || row >= a.rows || role >= a.roles) return;
  const device float* h = (role == 0u) ? h0 : ((role == 1u) ? h1 : h2);
  const device float* m = (role == 0u) ? m0 : ((role == 1u) ? m1 : m2);
  const device bfloat* forget = (role == 0u) ? forget0 : ((role == 1u) ? forget1 : forget2);
  const device bfloat* gain = (role == 0u) ? gain0 : ((role == 1u) ? gain1 : gain2);
  device float* out = (role == 0u) ? out0 : ((role == 1u) ? out1 : out2);
  zeta += size_t(role) * INK2_GATE_SCRATCH_STRIDE;
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
