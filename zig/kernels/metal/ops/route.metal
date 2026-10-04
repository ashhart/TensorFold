// The MTP head's expert choice (mlx-lm's compiled group_expert_select, n_group 1) in one launch, MLX 0.32.3's bits.
#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;
// tf:header

// MLX's Sigmoid fusion on fp32: y = 1 / (1 + precise exp(|x|)), then y or 1 - y by the sign.
inline float tf_sigmoid(float x) {
  const float y = 1 / (1 + metal::precise::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}

// MLX's sort order for floats: NaN after every number, otherwise <.
inline bool tf_before(float a, float b) {
  const bool an = isnan(a), bn = isnan(b);
  if (an | bn) return (!an) & bn;
  return a < b;
}

// Element i's place in a stable ascending sort of n keys (carg_block_sort is a stable merge sort).
inline int tf_rank(const threadgroup float* keys, int n, int i) {
  const float v = keys[i];
  int rank = 0;
  for (int j = 0; j < n; ++j) rank += (tf_before(keys[j], v) || (j < i && !tf_before(v, keys[j]))) ? 1 : 0;
  return rank;
}

// tf:kernel tf_route_topk inputs=G,BIAS,C,P outputs=IDS,WT
[[kernel]] void tf_route_topk(
  const device bfloat16_t* G [[buffer(0)]],
  const device float* BIAS [[buffer(1)]],
  const constant float* C [[buffer(2)]],
  const constant int32_t* P [[buffer(3)]],
  device uint32_t* IDS [[buffer(4)]],
  device float* WT [[buffer(5)]],
  uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
  uint3 threads_per_threadgroup [[threads_per_threadgroup]]) {
  threadgroup float prob[1024];
  threadgroup float key[1024];
  threadgroup uint pick[64];
  const int n = P[0], k = P[1], t = int(thread_position_in_threadgroup.x), tn = int(threads_per_threadgroup.x);
  const long row = threadgroup_position_in_grid.y;
  for (int i = t; i < n; i += tn) {
    const float p = tf_sigmoid(static_cast<float>(G[row * n + i]));
    const float s = p + BIAS[i];
    prob[i] = p;
    key[i] = -s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = t; i < n; i += tn) {
    const int rank = tf_rank(key, n, i);
    if (rank < k) pick[rank] = uint(i);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t != 0) return;
  float total = 0;
  for (int j = 0; j < k; ++j) total = prob[pick[j]] + total;
  const float den = total + C[0];
  for (int j = 0; j < k; ++j) {
    const float q = prob[pick[j]] / den;
    IDS[row * k + j] = pick[j];
    WT[row * k + j] = q * C[1];
  }
}
