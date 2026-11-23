#include <metal_stdlib>
using namespace metal;
#pragma METAL fp contract(off)
struct InkRuntimeArgs { uint count, boundary; };
kernel void tf_ink2_copy_bf16(const device bfloat* x [[buffer(0)]], device bfloat* y [[buffer(1)]], constant InkRuntimeArgs& a [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < a.count) y[i] = x[i];
}
kernel void tf_ink2_copy_f32(const device float* x [[buffer(0)]], device float* y [[buffer(1)]], constant InkRuntimeArgs& a [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < a.count) y[i] = x[i];
}
kernel void tf_ink2_widen(const device bfloat* x [[buffer(0)]], device float* y [[buffer(1)]], constant InkRuntimeArgs& a [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < a.count) y[i] = float(x[i]);
}
kernel void tf_ink2_attention_partial(const device bfloat* p [[buffer(0)]], const device bfloat* x [[buffer(1)]], device bfloat* blocks [[buffer(2)]], device float* t [[buffer(3)]], constant InkRuntimeArgs& a [[buffer(4)]], uint i [[thread_position_in_grid]]) {
  if (i >= a.count) return;
  const uint row = i / 2304, d = i % 2304;
  t[i] = a.boundary ? float(x[i]) : float(p[i]) + float(x[i]);
  if (a.boundary) blocks[row * 4 * 2304 + (a.boundary - 1) * 2304 + d] = p[i];
}
kernel void tf_ink2_mlp_partial(const device float* t [[buffer(0)]], const device bfloat* x [[buffer(1)]], device bfloat* p [[buffer(2)]], constant InkRuntimeArgs& a [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  if (i < a.count) p[i] = bfloat(t[i] + float(x[i]));
}

kernel void tf_ink2_capture_tap(const device bfloat* x [[buffer(0)]], device bfloat* y [[buffer(1)]], constant InkRuntimeArgs& a [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i >= a.count) return;
  const uint row = i / 2304, d = i % 2304;
  y[row * 4 * 2304 + a.boundary * 2304 + d] = x[i];
}

[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_bank_and_reset(
    const device bfloat* blocks [[buffer(0)]], device bfloat* p [[buffer(1)]],
    const device bfloat* prelude [[buffer(2)]], device bfloat* bank0 [[buffer(3)]],
    device bfloat* bank1 [[buffer(4)]], constant InkRuntimeArgs& a [[buffer(5)]],
    uint2 at [[thread_position_in_grid]]) {
  const uint q = at.x, row = at.y;
  if (q >= 576u || row >= a.count) return;
  const size_t idx = size_t(row) * 576u + q;
  const size_t blk = (size_t(row) * 4u + a.boundary) * 576u + q;
  const bfloat4 prev = ((const device bfloat4*)p)[idx];
  ((device bfloat4*)bank0)[idx] = ((const device bfloat4*)blocks)[blk];
  ((device bfloat4*)bank1)[idx] = prev;
  ((device bfloat4*)p)[idx] = ((const device bfloat4*)prelude)[idx];
}
