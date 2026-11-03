#include <metal_stdlib>
using namespace metal;
struct learn_shape { uint4 u; float4 f; };

// A raw affine4 matrix's input gradient in full fp32, in output-column order; no relaxed float MMA operand.
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_learn_q4_back(const device float* grad [[buffer(0)]], const device uint* words [[buffer(1)]],
                             const device bfloat* scales [[buffer(2)]], const device bfloat* biases [[buffer(3)]],
                             device float* dx [[buffer(4)]], constant learn_shape& p [[buffer(5)]],
                             uint i [[thread_position_in_grid]]) {
  const uint rows = p.u.x, n = p.u.y, k = p.u.z;
  const uint row = (i / k) * 4, column = i % k;
  if (row >= rows) return;
  float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
  for (uint out = 0; out < n; out++) {
    const size_t at = size_t(out) * k + column;
    const uint q = (words[at / 8] >> (4 * (column & 7))) & 15;
    const size_t group = size_t(out) * (k / 64) + column / 64;
    const float weight = float(scales[group]) * float(q) + float(biases[group]);
    s0 = fma(grad[size_t(row) * n + out], weight, s0);
    if (row + 1 < rows) s1 = fma(grad[size_t(row + 1) * n + out], weight, s1);
    if (row + 2 < rows) s2 = fma(grad[size_t(row + 2) * n + out], weight, s2);
    if (row + 3 < rows) s3 = fma(grad[size_t(row + 3) * n + out], weight, s3);
  }
  const size_t at = size_t(row) * k + column;
  dx[at] = p.u.w != 0 ? dx[at] + s0 : s0;
  if (row + 1 < rows) dx[at + k] = p.u.w != 0 ? dx[at + k] + s1 : s1;
  if (row + 2 < rows) dx[at + 2 * k] = p.u.w != 0 ? dx[at + 2 * k] + s2 : s2;
  if (row + 3 < rows) dx[at + 3 * k] = p.u.w != 0 ? dx[at + 3 * k] + s3 : s3;
}

// The grouped-query key/value gradients sum query heads, then query rows, in a fixed order.
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_learn_attn_kv_back(const device bfloat* qkv [[buffer(0)]], const device float* dout [[buffer(1)]],
                                 const device float* probability [[buffer(2)]], const device float* score_grad [[buffer(3)]],
                                 device float* dk [[buffer(4)]], device float* dv [[buffer(5)]],
                                 constant learn_shape& p [[buffer(6)]], uint i [[thread_position_in_grid]]) {
  const uint rows = p.u.x, heads = p.u.y, kv_heads = p.u.z, dim = p.u.w;
  const uint kw = kv_heads * dim, qw = heads * dim, stride = qw + 2 * kw;
  if (i >= rows * kw) return;
  const uint key = i / kw, h = i % kw / dim, d = i % dim, per = heads / kv_heads;
  float key_grad = 0.0f, value_grad = 0.0f;
  for (uint head = h * per; head < (h + 1) * per; head++) {
    for (uint query = key; query < rows; query++) {
      const size_t prob = (size_t(head) * rows + query) * rows + key;
      key_grad = fma(score_grad[prob], float(qkv[size_t(query) * stride + head * dim + d]), key_grad);
      value_grad = fma(probability[prob], dout[size_t(query) * qw + head * dim + d], value_grad);
    }
  }
  dk[i] = key_grad * p.f.x;
  dv[i] = value_grad;
}

// Differentiate the output gate from P times V, without dividing a rounded gated BF16 value by its sigmoid.
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_learn_attn_gate_grad(const device float* dgated [[buffer(0)]], const device bfloat* pq [[buffer(1)]],
                                   const device bfloat* qkv [[buffer(2)]], const device float* probability [[buffer(3)]],
                                   device float* dpq [[buffer(4)]], constant learn_shape& p [[buffer(5)]],
                                   uint i [[thread_position_in_grid]]) {
  const uint rows = p.u.x, heads = p.u.y, kv_heads = p.u.z, dim = p.u.w;
  const uint qw = heads * dim, kw = kv_heads * dim, stride = qw + 2 * kw;
  if (i >= rows * qw) return;
  const uint row = i / qw, h = i % qw / dim, d = i % dim, kh = h / (heads / kv_heads);
  float context = 0.0f;
  for (uint j = 0; j <= row; j++) {
    const float prob = probability[(size_t(h) * rows + row) * rows + j];
    context = fma(prob, float(qkv[size_t(j) * stride + qw + kw + kh * dim + d]), context);
  }
  const size_t at = size_t(row) * (2 * qw) + h * (2 * dim) + dim + d;
  const float sigmoid = 1.0f / (1.0f + precise::exp(-float(pq[at])));
  dpq[at] = dgated[i] * context * sigmoid * (1.0f - sigmoid);
}
