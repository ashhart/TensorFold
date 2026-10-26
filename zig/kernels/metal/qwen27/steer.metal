// Control vectors steer the residual stream: the direction's component leaves it, or it is added as a bias.
#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;

// The hidden width this kernel serves; a host that serves another width defines Q27_HIDDEN ahead of it.
#ifndef Q27_HIDDEN
#define Q27_HIDDEN 5120
#endif

// One threadgroup a row, 256 threads a group, each thread striding the hidden width; MODE 0 projects, else adds.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void custom_kernel_qwen27_steer_bfloat16_t_float_float_float(
  device bfloat16_t* X [[buffer(0)]],
  const device float* DIR [[buffer(1)]],
  constant float& ALPHA [[buffer(2)]],
  constant uint& MODE [[buffer(3)]],
  uint3 at [[thread_position_in_grid]],
  uint3 tpg [[thread_position_in_threadgroup]],
  uint group [[simdgroup_index_in_threadgroup]],
  uint lane [[thread_index_in_simdgroup]],
  uint sgs [[simdgroups_per_threadgroup]]) {
  constexpr uint K = Q27_HIDDEN;
  constexpr uint TPG = 256;
  constexpr uint GROUPS = 32;
  const uint tid = tpg.x;
  // The row: a group dispatch may lay the rows out along x or y, a thread dispatch leaves the group at zero.
  const uint row = at.y != 0 ? at.y : at.x / TPG;
  const uint base = row * K;
  threadgroup float red[GROUPS];
  if (MODE == 0) {
    float acc = 0.0f;
    for (uint i = tid; i < K; i += TPG) acc += float(X[base + i]) * DIR[i];
    // The dot over the group: a simdgroup tree, then the groups' partials.
    const float part = simd_sum(acc);
    if (lane == 0) red[group] = part;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float dot = 0.0f;
    for (uint i = 0; i < sgs; i++) dot += red[i];
    // Every thread waited for the dot, so the row still holds its original values here.
    const float k = ALPHA * dot;
    for (uint i = tid; i < K; i += TPG) X[base + i] = bfloat16_t(float(X[base + i]) - k * DIR[i]);
  } else {
    for (uint i = tid; i < K; i += TPG) X[base + i] = bfloat16_t(float(X[base + i]) + ALPHA * DIR[i]);
  }
}
