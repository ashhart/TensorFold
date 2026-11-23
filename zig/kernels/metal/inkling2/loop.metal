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

[[max_total_threads_per_threadgroup(192)]]
kernel void tf_inkling2_loop_messages(
    const device bfloat* bank0 [[buffer(0)]], const device bfloat* bank1 [[buffer(1)]],
    const device float* q_mlp [[buffer(2)]], const device float* kv [[buffer(3)]],
    device float* out [[buffer(4)]], constant InkLoopArgs& a [[buffer(5)]],
    uint t [[thread_position_in_threadgroup]], uint work [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint row = work / INKLING_LOOP_MESSAGES;
  const uint entry = work - row * INKLING_LOOP_MESSAGES;
  if (row >= a.rows) return;
  const uint sg = t / 32u;
  const uint layer = entry / 4;
  const uint role = entry - 4 * layer;
  const device float* queries = (role == 0 || role == 3) ? q_mlp : kv;
  // q/mlp and k/v occupy even/odd rows respectively.
  const device float* query = queries + size_t(2 * layer + (role >= 2 ? 1 : 0)) * INKLING_LOOP_H;
  const device bfloat* b0_ptr = bank0 + size_t(row) * INKLING_LOOP_H;
  const device bfloat* b1_ptr = bank1 + size_t(row) * INKLING_LOOP_H;
  threadgroup float sums[2 * 6], dots[2 * 6];
  float4 b0[3], b1[3], qv[3];
  float ss0 = 0.0f, ss1 = 0.0f;
  for (uint n = 0; n < 3; ++n) {
    const uint d = (n * 192u + t) * 4u;
    b0[n] = float4(*(const device bfloat4*)(b0_ptr + d));
    b1[n] = float4(*(const device bfloat4*)(b1_ptr + d));
    qv[n] = *(const device float4*)(query + d);
    ss0 += (b0[n].x * b0[n].x + b0[n].y * b0[n].y) + (b0[n].z * b0[n].z + b0[n].w * b0[n].w);
    ss1 += (b1[n].x * b1[n].x + b1[n].y * b1[n].y) + (b1[n].z * b1[n].z + b1[n].w * b1[n].w);
  }
  const float r0 = simd_sum(ss0), r1 = simd_sum(ss1);
  if (lane == 0) { sums[sg] = r0; sums[6u + sg] = r1; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float inverse[2];
  for (uint s = 0; s < 2; ++s) {
    float ss = 0.0f;
    for (uint j = 0; j < 6; ++j) ss += sums[s * 6u + j];
    inverse[s] = precise::rsqrt(precise::divide(ss, 2304.0f) + INKLING_LOOP_EPS);
  }
  float d0 = 0.0f, d1 = 0.0f;
  for (uint n = 0; n < 3; ++n) {
    const float4 p0 = qv[n] * (b0[n] * inverse[0]);
    const float4 p1 = qv[n] * (b1[n] * inverse[1]);
    d0 += (p0.x + p0.y) + (p0.z + p0.w);
    d1 += (p1.x + p1.y) + (p1.z + p1.w);
  }
  const float rd0 = simd_sum(d0), rd1 = simd_sum(d1);
  if (lane == 0) { dots[sg] = rd0; dots[6u + sg] = rd1; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float weights[2], logits[2];
  float peak = -INFINITY;
  for (uint s = 0; s < 2; ++s) {
    float dot = 0.0f;
    for (uint j = 0; j < 6; ++j) dot += dots[s * 6u + j];
    weights[s] = dot * INKLING_LOOP_SCALE;
    logits[s] = weights[s];
    peak = max(peak, logits[s]);
  }
  float den = 0.0f;
  for (uint s = 0; s < 2; ++s) { weights[s] = precise::exp(logits[s] - peak); den += weights[s]; }
  for (uint s = 0; s < 2; ++s) weights[s] = precise::divide(weights[s], den);
  device float* dst = out + (size_t(row) * INKLING_LOOP_MESSAGES + entry) * INKLING_LOOP_H;
  for (uint n = 0; n < 3; ++n) {
    const uint d = (n * 192u + t) * 4u;
    *(device float4*)(dst + d) = weights[0] * (b0[n] * inverse[0]) + weights[1] * (b1[n] * inverse[1]);
  }
}

// The gates' first launch: h and m each to unit RMS and BF16, as normalized[row][h | m].
[[max_total_threads_per_threadgroup(192)]]
kernel void tf_inkling2_loop_gates(
    const device float* h [[buffer(0)]], const device float* m [[buffer(1)]],
    device bfloat* normalized [[buffer(2)]], constant InkLoopArgs& a [[buffer(3)]],
    uint t [[thread_position_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  if (row >= a.rows) return;
  const uint sg = t / 32u;
  const device float* hp = h + size_t(row) * INKLING_LOOP_H;
  const device float* mp = m + size_t(row) * a.stride;
  threadgroup float sums[2 * 6];
  float4 vh[3], vm[3];
  float ssh = 0.0f, ssm = 0.0f;
  for (uint n = 0; n < 3; ++n) {
    const uint d = (n * 192u + t) * 4u;
    vh[n] = *(const device float4*)(hp + d);
    vm[n] = *(const device float4*)(mp + d);
    ssh += (vh[n].x * vh[n].x + vh[n].y * vh[n].y) + (vh[n].z * vh[n].z + vh[n].w * vh[n].w);
    ssm += (vm[n].x * vm[n].x + vm[n].y * vm[n].y) + (vm[n].z * vm[n].z + vm[n].w * vm[n].w);
  }
  const float rh = simd_sum(ssh), rm = simd_sum(ssm);
  if (lane == 0) { sums[sg] = rh; sums[6u + sg] = rm; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float th = 0.0f, tm = 0.0f;
  for (uint j = 0; j < 6; ++j) { th += sums[j]; tm += sums[6u + j]; }
  const float inv0 = precise::rsqrt(precise::divide(th, 2304.0f) + INKLING_LOOP_EPS);
  const float inv1 = precise::rsqrt(precise::divide(tm, 2304.0f) + INKLING_LOOP_EPS);
  device bfloat* y = normalized + size_t(row) * (2 * INKLING_LOOP_H);
  for (uint n = 0; n < 3; ++n) {
    const uint d = (n * 192u + t) * 4u;
    *(device bfloat4*)(y + d) = bfloat4(vh[n] * inv0);
    *(device bfloat4*)(y + INKLING_LOOP_H + d) = bfloat4(vm[n] * inv1);
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
