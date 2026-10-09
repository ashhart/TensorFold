// Sliding Weights: a learned change to a projection in the forward, and the bounded step on captured rows.
#include <metal_stdlib>
using namespace metal;

// The sum over a threadgroup of `groups` simdgroups, the same in every thread.
inline float group_sum(float v, threadgroup float* scratch, uint lane, uint sg, uint groups) {
  v = simd_sum(v);
  if (lane == 0) scratch[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  v = simd_sum(lane < groups ? scratch[lane] : 0.0f);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return v;
}

inline float group_max(float v, threadgroup float* scratch, uint lane, uint sg, uint groups) {
  v = simd_max(v);
  if (lane == 0) scratch[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  v = simd_max(lane < groups ? scratch[lane] : -INFINITY);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return v;
}

// y[r] += dw k[r], dw [N, K]: a simdgroup an output, 8 rows a pass, each row summed on its own.
kernel void tf_slide_delta(const device bfloat* k [[buffer(0)]],
                           const device bfloat* dw [[buffer(1)]],
                           device bfloat* y [[buffer(2)]],
                           constant uint3& dims [[buffer(3)]],
                           uint2 pos [[thread_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]]) {
  const uint rows = dims.x, n = dims.y, width = dims.z, j = pos.x / 32, r0 = pos.y * 8;
  if (j >= n) return;
  const device bfloat4* w = (const device bfloat4*)(dw + size_t(j) * width);
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (uint i = lane; i < width / 4; i += 32) {
    const float4 wi = float4(w[i]);
    for (uint r = 0; r < 8 && r0 + r < rows; r++) acc[r] += dot(wi, float4(((const device bfloat4*)(k + size_t(r0 + r) * width))[i]));
  }
  for (uint r = 0; r < 8; r++) {
    const float s = simd_sum(acc[r]);
    if (lane == 0 && r0 + r < rows) y[size_t(r0 + r) * n + j] = bfloat(float(y[size_t(r0 + r) * n + j]) + s);
  }
}

// The 4-bit LM head [vocab, dim] dequantized into bf16 [dim, vocab] for the backward product: grid (vocab, dim).
kernel void tf_slide_head_t(const device uint* w [[buffer(0)]],
                            const device bfloat* scales [[buffer(1)]],
                            const device bfloat* biases [[buffer(2)]],
                            device bfloat* out [[buffer(3)]],
                            uint2 pos [[thread_position_in_grid]],
                            uint2 size [[threads_per_grid]]) {
  const uint v = pos.x, d = pos.y, vocab = size.x, dim = size.y;
  const size_t i = size_t(v) * dim + d;
  const uint q = (w[i / 8] >> (4 * (i % 8))) & 0xf;
  out[size_t(d) * vocab + v] = bfloat(float(scales[i / 64]) * float(q) + float(biases[i / 64]));
}

// out = a - b for captured rows: the final residual without the change the forward added (b), in f32.
kernel void tf_slide_sub(const device bfloat* a [[buffer(0)]],
                         const device bfloat* b [[buffer(1)]],
                         device float* out [[buffer(2)]],
                         uint i [[thread_position_in_grid]]) {
  out[i] = float(a[i]) - float(b[i]);
}

// A row's final RMS norm of h = base + hd: hn = bf16(h inv g), inv = rsqrt(mean(h^2) + eps); 256 threads a row.
kernel void tf_slide_norm(const device float* base [[buffer(0)]],
                          const device bfloat* hd [[buffer(1)]],
                          const device bfloat* g [[buffer(2)]],
                          device bfloat* hn [[buffer(3)]],
                          device float* inv [[buffer(4)]],
                          constant uint& dim [[buffer(5)]],
                          constant float& eps [[buffer(6)]],
                          uint r [[threadgroup_position_in_grid]],
                          uint t [[thread_position_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scratch[32];
  const device float* b = base + size_t(r) * dim;
  const device bfloat* d = hd + size_t(r) * dim;
  float sq = 0;
  for (uint j = t; j < dim; j += 256) {
    const float h = b[j] + float(d[j]);
    sq += h * h;
  }
  const float s = rsqrt(group_sum(sq, scratch, lane, sg, 8) / float(dim) + eps);
  for (uint j = t; j < dim; j += 256) hn[size_t(r) * dim + j] = bfloat((b[j] + float(d[j])) * s * float(g[j]));
  if (t == 0) inv[r] = s;
}

// A row's softmax over bf16 logits: stats = (loss, target's probability), then in place (p - onehot) weight[r].
kernel void tf_slide_softmax(device bfloat* logits [[buffer(0)]],
                             const device uint* targets [[buffer(1)]],
                             const device float* weights [[buffer(2)]],
                             device float2* stats [[buffer(3)]],
                             constant uint& vocab [[buffer(4)]],
                             uint r [[threadgroup_position_in_grid]],
                             uint t [[thread_position_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scratch[32];
  device bfloat* l = logits + size_t(r) * vocab;
  float m = -INFINITY;
  for (uint v = t; v < vocab; v += 1024) m = max(m, float(l[v]));
  m = group_max(m, scratch, lane, sg, 32);
  float s = 0;
  for (uint v = t; v < vocab; v += 1024) s += exp(float(l[v]) - m);
  s = group_sum(s, scratch, lane, sg, 32);
  const uint target = targets[r];
  const float lt = float(l[target]);
  if (t == 0) stats[r] = float2(log(s) + m - lt, exp(lt - m) / s);
  threadgroup_barrier(mem_flags::mem_device);
  const float w = weights[r];
  for (uint v = t; v < vocab; v += 1024) l[v] = bfloat((exp(float(l[v]) - m) / s - (v == target ? 1.0f : 0.0f)) * w);
}

// The final norm's backward, dhn and dh as columns of [dim, stride]: dh = inv g dhn - inv^3 h (g dhn . h) / dim.
kernel void tf_slide_norm_back(const device bfloat* dhn [[buffer(0)]],
                               const device float* base [[buffer(1)]],
                               const device bfloat* hd [[buffer(2)]],
                               const device bfloat* g [[buffer(3)]],
                               const device float* inv [[buffer(4)]],
                               device bfloat* dh [[buffer(5)]],
                               constant uint2& dims [[buffer(6)]],
                               uint r [[threadgroup_position_in_grid]],
                               uint t [[thread_position_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scratch[32];
  const uint dim = dims.x, stride = dims.y;
  const device float* b = base + size_t(r) * dim;
  const device bfloat* d = hd + size_t(r) * dim;
  float dot = 0;
  for (uint j = t; j < dim; j += 256) dot += float(g[j]) * float(dhn[size_t(j) * stride + r]) * (b[j] + float(d[j]));
  dot = group_sum(dot, scratch, lane, sg, 8);
  const float s = inv[r], k = s * s * s * dot / float(dim);
  for (uint j = t; j < dim; j += 256) {
    const size_t at = size_t(j) * stride + r;
    dh[at] = bfloat(s * float(g[j]) * float(dhn[at]) - k * (b[j] + float(d[j])));
  }
}

// Sums of squares of the gradient's n values, a threadgroup of 256 a strided slice, into part[threadgroup].
kernel void tf_slide_sumsq(const device bfloat* grad [[buffer(0)]],
                           device float* part [[buffer(1)]],
                           constant uint& n [[buffer(2)]],
                           uint tg [[threadgroup_position_in_grid]],
                           uint tgs [[threadgroups_per_grid]],
                           uint t [[thread_position_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scratch[32];
  float s = 0;
  for (uint i = tg * 256 + t; i < n; i += tgs * 256) {
    const float x = float(grad[i]);
    s += x * x;
  }
  s = group_sum(s, scratch, lane, sg, 8);
  if (t == 0) part[tg] = s;
}

// out = (rate / max(|grad|, 1), |grad|^2) from the partial sums, the step NaN when the norm is not finite.
kernel void tf_slide_scale(const device float* part [[buffer(0)]],
                           device float* out [[buffer(1)]],
                           constant uint& parts [[buffer(2)]],
                           constant float& rate [[buffer(3)]],
                           uint t [[thread_position_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scratch[32];
  float s = 0;
  for (uint i = t; i < parts; i += 256) s += part[i];
  s = group_sum(s, scratch, lane, sg, 8);
  if (t == 0) {
    out[0] = isfinite(s) ? rate / max(sqrt(s), 1.0f) : NAN;
    out[1] = s;
  }
}

// The bounded move: w = clamp(w - step grad, anchor - bound, anchor + bound) and its bf16 copy; none if step is NaN.
kernel void tf_slide_step(device float* w [[buffer(0)]],
                          device bfloat* wb [[buffer(1)]],
                          const device bfloat* anchor [[buffer(2)]],
                          const device bfloat* grad [[buffer(3)]],
                          const device float* scale [[buffer(4)]],
                          constant float& bound [[buffer(5)]],
                          uint i [[thread_position_in_grid]]) {
  const float s = scale[0];
  if (!isfinite(s)) return;
  const float a = float(anchor[i]);
  const float x = clamp(w[i] - s * float(grad[i]), a - bound, a + bound);
  w[i] = x;
  wb[i] = bfloat(x);
}
