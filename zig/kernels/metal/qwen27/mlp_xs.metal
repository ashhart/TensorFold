// Runtime constants preserve the Python operator's projection-shaped arithmetic.
#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;
typedef half float16_t;
[[kernel]] void custom_kernel_q27_mlp_xs_bfloat16_t_bfloat16_t_int32_t_bfloat16_t_float(
  const device bfloat16_t* GATE [[buffer(0)]],
  const device bfloat16_t* UP [[buffer(1)]],
  const device int32_t* dims [[buffer(2)]],
  device bfloat16_t* HOUT [[buffer(3)]],
  device float* XS [[buffer(4)]],
  uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
Q27_SPECIALIZE

  const uint t = thread_position_in_threadgroup.x;
  const uint g = threadgroup_position_in_grid.x;
  const uint m = threadgroup_position_in_grid.y;
  const int M = dims[0], MP = dims[1];
  threadgroup bfloat hb[64];
  if (int(m) >= M) {
    if (t == 0) XS[g * MP + m] = 0.0f;
    return;
  }
  const int e = int(m) * N + int(g) * 64 + int(t);
  const int fused = int(m) * 2 * N + int(g) * 64 + int(t);  // GATE and UP point into the fused gate_up rows
  const float gf = float(GATE[fused]);
  const bfloat h = bfloat(gf / (1.0f + metal::exp(-gf)) * float(UP[fused]));
  HOUT[e] = h;
  hb[t] = h;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    float acc = 0.0f;
    for (int i = 0; i < 64; i++) acc += float(hb[i]);
    XS[g * MP + m] = acc;
  }

}
