#include <metal_stdlib>
using namespace metal;

// One row's saved SwiGLU neurons, followed by their output in neuron order and one BF16 add.
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_q27_pocket_rows(const device bfloat* x [[buffer(0)]], const device float* weights [[buffer(1)]],
                              device bfloat* out [[buffer(2)]], constant uint4& p [[buffer(3)]],
                              uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float activation[64];
  const uint d = p.x, n = p.y, cap = p.z;
  const device float* gate = weights;
  const device float* up = weights + size_t(cap) * d;
  const device float* down = up + size_t(cap) * d;
  for (uint j = sg; j < n; j += 8) {
    float g = 0.0f, u = 0.0f;
    for (uint k = lane; k < d; k += 32) {
      const float value = float(x[size_t(row) * d + k]);
      g = fma(value, gate[size_t(j) * d + k], g);
      u = fma(value, up[size_t(j) * d + k], u);
    }
    g = simd_sum(g);
    u = simd_sum(u);
    if (lane == 0) activation[j] = (g / (1.0f + precise::exp(-g))) * u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint k = t; k < d; k += 256) {
    float change = 0.0f;
    for (uint j = 0; j < n; j++) change = fma(activation[j], down[size_t(j) * d + k], change);
    const size_t at = size_t(row) * d + k;
    out[at] = bfloat(float(out[at]) + change);
  }
}

// Training scales only the open fact; the ordinary serving entry and its bindings stay unchanged.
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_q27_pocket_scaled(const device bfloat* x [[buffer(0)]], const device float* weights [[buffer(1)]],
                              device bfloat* out [[buffer(2)]], constant uint4& p [[buffer(3)]], constant float& scale [[buffer(4)]],
                              uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float activation[64];
  const uint d = p.x, n = p.y, cap = p.z;
  const device float* gate = weights;
  const device float* up = weights + size_t(cap) * d;
  const device float* down = up + size_t(cap) * d;
  for (uint j = sg; j < n; j += 8) {
    float g = 0.0f, u = 0.0f;
    for (uint k = lane; k < d; k += 32) {
      const float value = float(x[size_t(row) * d + k]);
      g = fma(value, gate[size_t(j) * d + k], g);
      u = fma(value, up[size_t(j) * d + k], u);
    }
    g = simd_sum(g);
    u = simd_sum(u);
    if (lane == 0) {
      const float value = (g / (1.0f + precise::exp(-g))) * u;
      activation[j] = j >= p.w && scale != 1.0f ? value * scale : value;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint k = t; k < d; k += 256) {
    float change = 0.0f;
    for (uint j = 0; j < n; j++) change = fma(activation[j], down[size_t(j) * d + k], change);
    const size_t at = size_t(row) * d + k;
    out[at] = bfloat(float(out[at]) + change);
  }
}
