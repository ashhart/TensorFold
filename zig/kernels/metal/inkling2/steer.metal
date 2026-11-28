// A steered decoder visit's output: the residual add rounded to BF16, then the strength times the direction added.
#include <metal_stdlib>
// alpha * d is rounded on its own before the sum, as the separately ordered reference rounds it: never one FMA.
#pragma clang fp contract(off)
using namespace metal;
struct InkSteerArgs { uint count; float alpha; };
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_mlp_partial_steer(const device float* t [[buffer(0)]], const device bfloat* x [[buffer(1)]], device bfloat* p [[buffer(2)]], const device float* dir [[buffer(3)]], constant InkSteerArgs& a [[buffer(4)]], uint i [[thread_position_in_grid]]) {
  if (i >= a.count) return;
  const float hv = float(bfloat(t[i] + float(x[i])));
  const float step = a.alpha * dir[i % 2304];
  // a zero step leaves the rounded residual alone, so a negative zero stays negative
  p[i] = step == 0.0f ? bfloat(hv) : bfloat(hv + step);
}
// The reference's second launch: main's mlp_partial wrote p, and this adds the direction to it in place.
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_steer_add(device bfloat* p [[buffer(0)]], const device float* dir [[buffer(1)]], constant InkSteerArgs& a [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i >= a.count) return;
  const float step = a.alpha * dir[i % 2304];
  if (step == 0.0f) return;
  p[i] = bfloat(float(p[i]) + step);
}
