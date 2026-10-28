#include <metal_stdlib>
using namespace metal;
struct learn_shape { uint4 u; float4 f; };

inline float learn_sum(float v, threadgroup float* scratch, uint t) {
  v = simd_sum(v);
  if ((t & 31) == 0) scratch[t >> 5] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = t < 8 ? scratch[t] : 0.0f;
  if (t < 32) total = simd_sum(total);
  if (t == 0) scratch[0] = total;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float result = scratch[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return result;
}

[[max_total_threads_per_threadgroup(256)]]
kernel void tf_learn_copy(const device uchar* x [[buffer(0)]], device uchar* y [[buffer(1)]],
                          constant learn_shape& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  const uint first = i * 16;
  if (first >= p.u.x) return;
  if (p.u.y != 0 && first + 16 <= p.u.x) {
    *(device uint4*)(y + first) = *(const device uint4*)(x + first);
  } else {
    for (uint j = 0; j < 16 && first + j < p.u.x; j++) y[first + j] = x[first + j];
  }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void tf_learn_zero(device float* x [[buffer(0)]], constant learn_shape& p [[buffer(1)]],
                          uint i [[thread_position_in_grid]]) {
  if (i < p.u.x) x[i] = 0.0f;
}

[[max_total_threads_per_threadgroup(256)]]
kernel void tf_learn_nudge(device bfloat* x [[buffer(0)]], const device float* delta [[buffer(1)]],
                           constant learn_shape& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < p.u.x) x[i] = bfloat(float(x[i]) + delta[i]);
}

// Gradient of mean answer loss: the native BF16 logits in, BF16-rounded fp32 gradient and loss/probability out.
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_learn_loss(const device bfloat* logits [[buffer(0)]], const device uint* target [[buffer(1)]],
                          const device float* weight [[buffer(2)]], device float* grad [[buffer(3)]],
                          device float2* stats [[buffer(4)]], constant learn_shape& p [[buffer(5)]],
                          uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  threadgroup float scratch[8];
  const uint width = p.u.y;
  const size_t base = size_t(row) * width;
  float top = -INFINITY;
  for (uint c = t; c < width; c += 256) top = max(top, float(logits[base + c]));
  top = simd_max(top);
  if ((t & 31) == 0) scratch[t >> 5] = top;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float high = t < 8 ? scratch[t] : -INFINITY;
  if (t < 32) high = simd_max(high);
  if (t == 0) scratch[0] = high;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  top = scratch[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint c = t; c < width; c += 256) total += precise::exp(float(logits[base + c]) - top);
  total = learn_sum(total, scratch, t);
  for (uint c = t; c < width; c += 256) {
    const float probability = precise::exp(float(logits[base + c]) - top) / total;
    grad[base + c] = float(bfloat((probability - float(c == target[row])) * weight[row]));
  }
  if (t == 0) {
    const float wanted = float(logits[base + target[row]]);
    stats[row] = float2(precise::log(total) + top - wanted, precise::exp(wanted - top) / total);
  }
}

// The input gradient of RMS normalization adds to the residual stream's existing gradient.
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_learn_rms_back(const device bfloat* x [[buffer(0)]], const device bfloat* gain [[buffer(1)]],
                              const device float* dy [[buffer(2)]], device float* grad [[buffer(3)]],
                              constant learn_shape& p [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                              uint t [[thread_index_in_threadgroup]]) {
  threadgroup float scratch[8];
  const uint width = p.u.y;
  const size_t base = size_t(row) * width;
  float square = 0.0f, dot = 0.0f;
  for (uint c = t; c < width; c += 256) {
    const float v = float(x[base + c]);
    square = fma(v, v, square);
    dot = fma(v, float(gain[c]) * dy[base + c], dot);
  }
  square = learn_sum(square, scratch, t);
  dot = learn_sum(dot, scratch, t);
  const float inv = 1.0f / precise::sqrt(square / float(width) + p.f.x);
  for (uint c = t; c < width; c += 256) {
    const float g = inv * float(gain[c]) * dy[base + c];
    const float correction = inv * inv * inv * float(x[base + c]) * dot / float(width);
    grad[base + c] += g - correction;
  }
}

// The packed gate/up rows have the same member order as the serving MLP.
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_learn_swiglu_back(const device bfloat* gu [[buffer(0)]], const device float* da [[buffer(1)]],
                                 device float* dgu [[buffer(2)]], constant learn_shape& p [[buffer(3)]],
                                 uint i [[thread_position_in_grid]]) {
  const uint width = p.u.y;
  if (i >= p.u.x * width) return;
  const size_t at = size_t(i / width) * (2 * width) + i % width;
  const float gate = float(gu[at]), up = float(gu[at + width]);
  const float sigmoid = 1.0f / (1.0f + precise::exp(-gate));
  dgu[at] = da[i] * up * sigmoid * (1.0f + gate * (1.0f - sigmoid));
  dgu[at + width] = da[i] * gate * sigmoid;
}

// The pocket's contribution to its normalized input gradient, with the open neurons' per-sequence scale.
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_learn_pocket_back(const device bfloat* x [[buffer(0)]], const device float* weights [[buffer(1)]],
                                 const device float* grad [[buffer(2)]], device float* dx [[buffer(3)]],
                                 constant learn_shape& p [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                                 uint t [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float cg[64], cu[64];
  const uint width = p.u.y, n = p.u.z, cap = p.u.w;
  const device float* gate = weights;
  const device float* up = gate + size_t(cap) * width;
  const device float* down = up + size_t(cap) * width;
  const size_t base = size_t(row) * width;
  for (uint j = sg; j < n; j += 8) {
    float g = 0.0f, u = 0.0f, e = 0.0f;
    for (uint c = lane; c < width; c += 32) {
      const float value = float(x[base + c]);
      g = fma(value, gate[size_t(j) * width + c], g);
      u = fma(value, up[size_t(j) * width + c], u);
      e = fma(grad[base + c], down[size_t(j) * width + c], e);
    }
    g = simd_sum(g); u = simd_sum(u); e = simd_sum(e);
    const float sigmoid = 1.0f / (1.0f + precise::exp(-g));
    const float scale = j >= p.u.x ? p.f.x : 1.0f;
    if (lane == 0) {
      cg[j] = scale * e * u * sigmoid * (1.0f + g * (1.0f - sigmoid));
      cu[j] = scale * e * g * sigmoid;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint c = t; c < width; c += 256) {
    float change = 0.0f;
    for (uint j = 0; j < n; j++) {
      change = fma(cg[j], gate[size_t(j) * width + c], change);
      change = fma(cu[j], up[size_t(j) * width + c], change);
    }
    dx[base + c] += change;
  }
}

// Widen actual native BF16 prepared rows; the recurrence backward consumes these, not a second forward.
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_learn_widen(const device bfloat* x [[buffer(0)]], device float* y [[buffer(1)]],
                           constant learn_shape& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < p.u.x) y[i] = float(x[i]);
}


// Q/K conv activations before native RMS, with the native two BF16 SiLU nodes. V is captured directly.
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_learn_gdn_act(const device bfloat* projection [[buffer(0)]], const device bfloat* conv [[buffer(1)]],
                             device bfloat* act [[buffer(2)]], constant learn_shape& p [[buffer(3)]],
                             uint2 i [[thread_position_in_grid]]) {
  const uint c = i.x, row = i.y;
  if (c >= p.u.y || row >= p.u.x) return;
  float sum = 0.0f;
  for (int tap = 0; tap < 4; tap++) {
    const int at = int(row) - 3 + tap;
    if (at >= 0) sum += float(conv[c * 4 + tap]) * float(projection[size_t(at) * p.u.z + c]);
  }
  const float rounded = float(bfloat(sum));
  const float sigmoid = float(bfloat(1.0f / (1.0f + metal::exp(-rounded))));
  act[size_t(row) * p.u.w + c] = bfloat(rounded * sigmoid);
}
