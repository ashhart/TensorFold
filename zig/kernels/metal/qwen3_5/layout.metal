// Qwen's embedding, head normalization, text RoPE and gated attention layouts.
#include <metal_stdlib>
using namespace metal;

kernel void qwen35_embed(const device uint* ids [[buffer(0)]],
                         const device uchar* weight [[buffer(1)]],
                         const device bfloat* scales [[buffer(2)]],
                         const device bfloat* biases [[buffer(3)]],
                         device bfloat* out [[buffer(4)]],
                         uint2 pos [[thread_position_in_grid]]) {
  const uint b = pos.x, r = pos.y;
  const size_t id = ids[r];
  const uchar v = weight[id * 1024 + b];
  const size_t g = id * 32 + (2 * b) / 64;
  const float s = float(scales[g]), z = float(biases[g]);
  out[size_t(r) * 2048 + 2 * b] = bfloat(s * float(v & 15) + z);
  out[size_t(r) * 2048 + 2 * b + 1] = bfloat(s * float(v >> 4) + z);
}

kernel void qwen35_head_norm(const device bfloat* x [[buffer(0)]],
                             const device bfloat* weight [[buffer(1)]],
                             device bfloat* out [[buffer(2)]],
                             constant uint4& dims [[buffer(3)]],
                             uint row [[threadgroup_position_in_grid]],
                             uint t [[thread_position_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float sums[2], inv;
  const uint head = row % dims.y, m = row / dims.y;
  const size_t base = size_t(m) * dims.z + head * dims.w;
  float v[4], acc = 0.0f;
  for (int i = 0; i < 4; i++) {
    v[i] = float(x[base + 4 * t + i]);
    acc += v[i] * v[i];
  }
  acc = simd_sum(acc);
  if (lane == 0) sums[sg] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) inv = metal::precise::rsqrt((sums[0] + sums[1]) / 256.0f + 1e-6f);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = 0; i < 4; i++) {
    const uint d = 4 * t + i;
    out[(size_t(m) * dims.y + head) * 256 + d] = bfloat(float(weight[d]) * (v[i] * inv));
  }
}

inline bfloat qwen35_rotate(const device bfloat* x, uint d, uint position) {
  if (d >= 64) return x[d];
  const uint i = d % 32;
  const float angle = float(position) * metal::pow(10000000.0f, -float(i) / 32.0f);
  const float cs = metal::cos(angle), sn = metal::sin(angle);
  const float a = float(x[i]), b = float(x[i + 32]);
  return bfloat(d < 32 ? a * cs - b * sn : b * cs + a * sn);
}

kernel void qwen35_queries(const device bfloat* x [[buffer(0)]],
                           device bfloat* out [[buffer(1)]],
                           constant uint4& dims [[buffer(2)]],
                           uint3 p [[thread_position_in_grid]]) {
  const uint d = p.x, h = p.y, r = p.z;
  const uint m = dims.z + r;
  out[(size_t(h) * dims.y + m) * 256 + d] = qwen35_rotate(x + (size_t(m) * 8 + h) * 256, d, dims.x + r);
}

kernel void qwen35_keys(const device bfloat* k [[buffer(0)]],
                        const device bfloat* v [[buffer(1)]],
                        device bfloat* keys [[buffer(2)]],
                        device bfloat* values [[buffer(3)]],
                        constant uint4& dims [[buffer(4)]],
                        uint3 p [[thread_position_in_grid]]) {
  const uint d = p.x, h = p.y, r = p.z;
  const uint m = dims.z + r;
  const size_t dst = (size_t(h) * dims.y + dims.x + r) * 256 + d;
  keys[dst] = qwen35_rotate(k + (size_t(m) * 2 + h) * 256, d, dims.x + r);
  values[dst] = v[(size_t(m) * 2 + h) * 256 + d];
}

kernel void qwen35_attention_gate(const device bfloat* x [[buffer(0)]],
                                  const device bfloat* q [[buffer(1)]],
                                  device bfloat* out [[buffer(2)]],
                                  uint2 p [[thread_position_in_grid]]) {
  const uint d = p.x, r = p.y, head = d / 256;
  const float gate = float(q[size_t(r) * 4096 + head * 512 + 256 + d % 256]);
  const bfloat s = bfloat(1.0f / (1.0f + metal::exp(-gate)));
  out[size_t(r) * 2048 + d] = bfloat(float(x[size_t(r) * 2048 + d]) * float(s));
}
