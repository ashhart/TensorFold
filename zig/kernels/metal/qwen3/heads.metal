// Projected Q and K heads: BF16 RMS normalization before RoPE. V is untouched.
#include <metal_stdlib>
using namespace metal;
struct q3_head_args { uint rows, heads, kv_heads, stride; float eps; };
[[kernel]] void q3_head_norm(device bfloat* X [[buffer(0)]], const device bfloat* QW [[buffer(1)]],
    const device bfloat* KW [[buffer(2)]], constant q3_head_args& a [[buffer(3)]],
    uint2 g [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  if (g.x >= a.heads + a.kv_heads || g.y >= a.rows) return;
  threadgroup float part[4];
  device bfloat* x = X + (size_t)g.y * a.stride + (size_t)g.x * 128;
  const device bfloat* w = g.x < a.heads ? QW : KW;
  const float v = float(x[t]);
  const float sum = simd_sum(v * v);
  if (t % 32 == 0) part[t / 32] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint s = 0; s < 4; ++s) total += part[s];
  const float inv = precise::rsqrt(total / 128.0f + a.eps);
  x[t] = bfloat(float(bfloat(v * inv)) * float(w[t]));
}
