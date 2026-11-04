// The residual norm with the control vector's projection inside it: one dispatch, one pass over the row.
#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;
typedef half float16_t;

// The norm of a steered residual: the row is summed, the direction's component leaves it, then it is normalised.
[[kernel]] void custom_kernel_q27_norm_steer_bfloat16_t_bfloat16_t_bfloat16_t_floatc_int32_t_bfloat16_t_bfloat16_t_float_float_float_uint(
  const device bfloat16_t* H [[buffer(0)]],
  const device bfloat16_t* R [[buffer(1)]],
  const device bfloat16_t* Wt [[buffer(2)]],
  const constant float* eps [[buffer(3)]],
  const device int32_t* dims [[buffer(4)]],
  device bfloat16_t* HO [[buffer(5)]],
  device bfloat16_t* XO [[buffer(6)]],
  device float* XS [[buffer(7)]],
  const device float* DIR [[buffer(8)]],
  constant float& ALPHA [[buffer(9)]],
  constant uint& MODE [[buffer(10)]],
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
  threadgroup float dot_parts[TPG / 32];
  threadgroup float sq_parts[TPG / 32];
  if (int(m) >= M) {
    if ((t & 3) == 0) XS[(t >> 2) * MP + m] = 0.0f;
    return;
  }
  const int base = int(m) * K + int(t) * E;
  const int dbase = int(t) * E;
  float hv[E];
  for (int i = 0; i < E; i++) {
    // The residual as the block before it left it, in the precision the stream carries.
    hv[i] = float(bfloat(float(H[base + i]) + float(R[base + i])));
  }
  // The direction's component in this row: the dot over the row, a simdgroup tree and then the groups' partials.
  float sd = 0.0f;
  for (int i = 0; i < E; i++) sd = fma(hv[i], DIR[dbase + i], sd);
  sd = simd_sum(sd);
  if (thread_index_in_simdgroup == 0) dot_parts[simdgroup_index_in_threadgroup] = sd;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float dot = 0.0f;
  for (int i = 0; i < TPG / 32; i++) dot += dot_parts[i];
  const float k = ALPHA * dot;

  // What the stream holds after the direction acts on it, and the squares the normalisation reads.
  float pv[E];
  float ss = 0.0f;
  for (int i = 0; i < E; i++) {
    const float d = DIR[dbase + i];
    const float v = MODE == 0 ? hv[i] - k * d : hv[i] + ALPHA * d;
    const bfloat rounded = bfloat(v);
    HO[base + i] = rounded;
    pv[i] = float(rounded);
    ss = fma(pv[i], pv[i], ss);
  }
  ss = simd_sum(ss);
  if (thread_index_in_simdgroup == 0) sq_parts[simdgroup_index_in_threadgroup] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (int i = 0; i < TPG / 32; i++) total += sq_parts[i];
  const float inv = metal::rsqrt(total / float(K) + eps[0]);
  float xv[E];
  for (int i = 0; i < E; i++) {
    const bfloat x = bfloat(float(Wt[dbase + i]) * (pv[i] * inv));
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
