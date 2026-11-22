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
// Row i of a convolution's taps: a stream's three old rows, then the launch's own.
inline float ink2_tap(const device bfloat* old, const device bfloat* fresh, uint i, uint d) {
  return i < 3u ? float(old[i * 2304u + d]) : float(fresh[size_t(i - 3u) * 2304 + d]);
}
// Glue's static convolution of row `row` at d, as the BF16 it stores.
inline bfloat ink2_conv(const device bfloat* old, const device bfloat* fresh, const device bfloat* w, uint row, uint d) {
  float sum = ink2_tap(old, fresh, row + 3u, d);
  for (uint tap = 0; tap < 4; ++tap) sum += float(w[d * 4 + tap]) * ink2_tap(old, fresh, row + tap, d);
  return bfloat(precise::divide(sum, precise::sqrt(5.0f)));
}
// The attention output convolved, then the running sum: t = p + conv, or conv alone where a boundary banks p.
kernel void tf_ink2_attention_partial(const device bfloat* p [[buffer(0)]], const device bfloat* old [[buffer(1)]], const device bfloat* fresh [[buffer(2)]], const device bfloat* w [[buffer(3)]], device bfloat* blocks [[buffer(4)]], device float* t [[buffer(5)]], constant InkRuntimeArgs& a [[buffer(6)]], uint i [[thread_position_in_grid]]) {
  if (i >= a.count) return;
  const uint row = i / 2304, d = i % 2304;
  const float x = float(ink2_conv(old, fresh, w, row, d));
  t[i] = a.boundary ? x : float(p[i]) + x;
  if (a.boundary) blocks[row * 4 * 2304 + (a.boundary - 1) * 2304 + d] = p[i];
}
// The expert output convolved into the running p, and p widened into `wide`, the next visit's partial source.
kernel void tf_ink2_mlp_partial(const device float* t [[buffer(0)]], const device bfloat* old [[buffer(1)]], const device bfloat* fresh [[buffer(2)]], const device bfloat* w [[buffer(3)]], device bfloat* p [[buffer(4)]], device float* wide [[buffer(5)]], constant InkRuntimeArgs& a [[buffer(6)]], uint i [[thread_position_in_grid]]) {
  if (i >= a.count) return;
  const bfloat v = bfloat(t[i] + float(ink2_conv(old, fresh, w, i / 2304, i % 2304)));
  p[i] = v;
  wide[i] = float(v);
}

kernel void tf_ink2_capture_tap(const device bfloat* x [[buffer(0)]], device bfloat* y [[buffer(1)]], constant InkRuntimeArgs& a [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i >= a.count) return;
  const uint row = i / 2304, d = i % 2304;
  y[row * 4 * 2304 + a.boundary * 2304 + d] = x[i];
}
