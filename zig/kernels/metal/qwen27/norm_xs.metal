// Runtime constants preserve the Python operator's projection-shaped arithmetic.
#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;
typedef half float16_t;
[[kernel]] void custom_kernel_q27_norm_xs_bfloat16_t_bfloat16_t_bfloat16_t_floatc_int32_t_bfloat16_t_bfloat16_t_float(
  const device bfloat16_t* H [[buffer(0)]],
  const device bfloat16_t* R [[buffer(1)]],
  const device bfloat16_t* Wt [[buffer(2)]],
  const constant float* eps [[buffer(3)]],
  const device int32_t* dims [[buffer(4)]],
  device bfloat16_t* HO [[buffer(5)]],
  device bfloat16_t* XO [[buffer(6)]],
  device float* XS [[buffer(7)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
Q27_SPECIALIZE

  const uint t = thread_position_in_threadgroup.x;
  const uint m = threadgroup_position_in_grid.y;
  const int M = dims[0], MP = dims[1];
  constexpr int E = 16;
  constexpr int TPG = K / E;
  threadgroup float red[TPG / 32];
  if (int(m) >= M) {
    if ((t & 3) == 0) XS[(t >> 2) * MP + m] = 0.0f;
    return;
  }
  const int base = int(m) * K + int(t) * E;
  float hv[E];
  float ss = 0.0f;
  for (int i = 0; i < E; i++) {
    bfloat h = H[base + i];
    h = bfloat(float(h) + float(R[base + i]));
HO[base + i] = h;
    hv[i] = float(h);
    ss = fma(hv[i], hv[i], ss);
  }
  ss = simd_sum(ss);
  if (thread_index_in_simdgroup == 0) red[simdgroup_index_in_threadgroup] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (int i = 0; i < TPG / 32; i++) total += red[i];
  const float inv = metal::rsqrt(total / float(K) + eps[0]);
  float xv[E];
  for (int i = 0; i < E; i++) {
    const bfloat x = bfloat(float(Wt[int(t) * E + i]) * (hv[i] * inv));
    XO[base + i] = x;
    xv[i] = float(x);
  }
  // a group's 64 values summed in order from zero, as the lane projection's sums are: its four threads take turns
  float gs = 0.0f;
  for (uint turn = 0; turn < 4; turn++) {
    if ((t & 3) == turn) for (int i = 0; i < E; i++) gs += xv[i];
    gs = simd_shuffle(gs, ushort((thread_index_in_simdgroup & ~3u) | turn));
  }
  if ((t & 3) == 0) XS[(t >> 2) * MP + m] = gs;

}
