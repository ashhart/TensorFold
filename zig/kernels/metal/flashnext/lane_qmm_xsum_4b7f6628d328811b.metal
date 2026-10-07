#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;
typedef half float16_t;

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
[[kernel]] void custom_kernel_lane_qmm_xsum_4b7f6628d328811b_bfloat16_t_int32_t_float(
  const device bfloat16_t* X [[buffer(0)]],
  const device int32_t* mdims [[buffer(1)]],
  device float* XS [[buffer(2)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
  constexpr int K = 6144;
  constexpr int GS = 32;

  const int M = mdims[0], MP = mdims[1];
  const uint m = thread_position_in_grid.y;
  const uint g = thread_position_in_grid.x;
  if (g >= K / GS || int(m) >= MP) return;
  float acc = 0.0f;
  if (int(m) < M) for (int i = 0; i < GS; i++) acc += float(X[m * K + g * GS + i]);
  XS[g * MP + m] = acc;

}
