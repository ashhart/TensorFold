// One thread per column: a layer's residual rows summed into that layer's fp32 accumulator.
#include <metal_stdlib>
using namespace metal;

struct Sum { uint layer, width; };

kernel void q27_sumrows(device const bfloat* rows [[buffer(0)]], device const int* dims [[buffer(1)]],
                        device float* acc [[buffer(2)]], constant Sum& p [[buffer(3)]],
                        uint gid [[thread_position_in_grid]]) {
  if (gid >= p.width) return;
  const int count = dims[0];
  float sum = 0.0f;
  for (int r = 0; r < count; ++r) sum += float(rows[r * p.width + gid]);
  acc[p.layer * p.width + gid] += sum;
}
