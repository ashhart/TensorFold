// Precise row softmax of bf16 scores, bit-identical to MLX 0.32.3's block_ and looped_softmax_precise_bfloat16.
#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;
// tf:header

constant constexpr float TF_LOW = -0x1.fffffep+127f;   // -FLT_MAX, MLX's finite_min

// Rows up to 4096: thread t holds 4t..4t+3; simd then 32-slot max and sum; out = bf16(fast::exp(x - max) / total).
inline void tf_softmax_row(const device bfloat* in, device bfloat* out, int n, uint row, uint t, uint sg, uint lane,
                           threadgroup float* tops, threadgroup float* sums) {
  const device bfloat* x = in + long(row) * n + 4 * long(t);
  float e[4];
  const bool whole = int(4 * t) + 4 <= n;
  for (int i = 0; i < 4; ++i) e[i] = whole || int(4 * t) + i < n ? float(x[i]) : -INFINITY;
  if (sg == 0) {
    tops[lane] = -INFINITY;
    sums[lane] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float top = TF_LOW;
  for (int i = 0; i < 4; ++i) top = (top < e[i]) ? e[i] : top;
  top = simd_max(top);
  if (lane == 0) tops[sg] = top;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    top = simd_max(tops[lane]);
    if (lane == 0) tops[0] = top;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  top = tops[0];
  float total = 0;
  for (int i = 0; i < 4; ++i) {
    const float ex = fast::exp(e[i] - top);
    e[i] = ex;
    total += ex;
  }
  total = simd_sum(total);
  if (lane == 0) sums[sg] = total;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    total = simd_sum(sums[lane]);
    if (lane == 0) sums[0] = total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = 1 / sums[0];
  device bfloat* y = out + long(row) * n + 4 * long(t);
  for (int i = 0; i < 4; ++i)
    if (whole || int(4 * t) + i < n) y[i] = bfloat(e[i] * inv);
}

// Longer rows (1024 threads): runs of 4 with an online max and rescaled sum, merged as MLX's looped kernel merges.
inline void tf_softmax_looped(const device bfloat* in, device bfloat* out, int n, uint row, uint t, uint threads,
                              uint lane, uint sg, threadgroup float* tops, threadgroup float* sums) {
  const device bfloat* x = in + long(row) * n;
  float top = TF_LOW, total = 0;
  const int steps = (n + 4 * int(threads) - 1) / (4 * int(threads));
  for (int r = 0; r < steps; ++r) {
    const int at = r * int(threads) * 4 + int(t) * 4;
    float v[4];
    if (at + 4 <= n) {
      for (int i = 0; i < 4; ++i) v[i] = x[at + i];
    } else {
      for (int i = 0; i < 4; ++i) v[i] = at + i < n ? float(x[at + i]) : -INFINITY;
    }
    const float before = top;
    for (int i = 0; i < 4; ++i) top = (top < v[i]) ? v[i] : top;
    total *= fast::exp(before - top);
    for (int i = 0; i < 4; ++i) total += fast::exp(v[i] - top);
  }
  float before = top;
  top = simd_max(top);
  total *= fast::exp(before - top);
  total = simd_sum(total);
  before = top;
  if (lane == 0) tops[sg] = top;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  top = simd_max(tops[lane]);
  total *= fast::exp(before - top);
  if (lane == 0) sums[sg] = total;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  total = simd_sum(sums[lane]);
  const float inv = 1 / total;
  device bfloat* y = out + long(row) * n;
  for (int r = 0; r < steps; ++r) {
    const int at = r * int(threads) * 4 + int(t) * 4;
    if (at + 4 <= n) {
      for (int i = 0; i < 4; ++i) y[at + i] = bfloat(fast::exp(float(x[at + i]) - top) * inv);
    } else {
      for (int i = 0; i < 4; ++i)
        if (at + i < n) y[at + i] = bfloat(fast::exp(float(x[at + i]) - top) * inv);
    }
  }
}

// tf:kernel tf_softmax_bf16 inputs=X,P outputs=OUT
[[kernel]] void tf_softmax_bf16(
  const device bfloat16_t* X [[buffer(0)]],
  const constant int32_t* P [[buffer(1)]],
  device bfloat16_t* OUT [[buffer(2)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  threadgroup float tops[32];
  threadgroup float sums[32];
  tf_softmax_row(X, OUT, P[0], threadgroup_position_in_grid.x, thread_position_in_threadgroup.x,
                 simdgroup_index_in_threadgroup, thread_index_in_simdgroup, tops, sums);
}

// tf:kernel tf_softmax_looped_bf16 inputs=X,P outputs=OUT
[[kernel]] void tf_softmax_looped_bf16(
  const device bfloat16_t* X [[buffer(0)]],
  const constant int32_t* P [[buffer(1)]],
  device bfloat16_t* OUT [[buffer(2)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
  uint3 threads_per_threadgroup [[threads_per_threadgroup]]) {
  threadgroup float tops[32];
  threadgroup float sums[32];
  tf_softmax_looped(X, OUT, P[0], threadgroup_position_in_grid.x, thread_position_in_threadgroup.x,
                    threads_per_threadgroup.x, thread_index_in_simdgroup, simdgroup_index_in_threadgroup, tops, sums);
}
