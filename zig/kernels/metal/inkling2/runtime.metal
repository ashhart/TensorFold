#include <metal_stdlib>
using namespace metal;
#pragma METAL fp contract(off)
struct InkRuntimeArgs { uint count, boundary; };
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_copy_bf16(const device bfloat* x [[buffer(0)]], device bfloat* y [[buffer(1)]], constant InkRuntimeArgs& a [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < a.count) y[i] = x[i];
}
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_copy_f32(const device float* x [[buffer(0)]], device float* y [[buffer(1)]], constant InkRuntimeArgs& a [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < a.count) y[i] = x[i];
}
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_widen(const device bfloat* x [[buffer(0)]], device float* y [[buffer(1)]], constant InkRuntimeArgs& a [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < a.count) y[i] = float(x[i]);
}
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_attention_partial(const device bfloat* p [[buffer(0)]], const device bfloat* x [[buffer(1)]], device bfloat* blocks [[buffer(2)]], device float* t [[buffer(3)]], constant InkRuntimeArgs& a [[buffer(4)]], uint i [[thread_position_in_grid]]) {
  if (i >= a.count) return;
  const uint row = i / 2304, d = i % 2304;
  t[i] = a.boundary ? float(x[i]) : float(p[i]) + float(x[i]);
  if (a.boundary) blocks[row * 4 * 2304 + (a.boundary - 1) * 2304 + d] = p[i];
}
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_mlp_partial(const device float* t [[buffer(0)]], const device bfloat* x [[buffer(1)]], device bfloat* p [[buffer(2)]], constant InkRuntimeArgs& a [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  if (i < a.count) p[i] = bfloat(t[i] + float(x[i]));
}

[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_capture_tap(const device bfloat* x [[buffer(0)]], device bfloat* y [[buffer(1)]], constant InkRuntimeArgs& a [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i >= a.count) return;
  const uint row = i / 2304, d = i % 2304;
  y[row * 4 * 2304 + a.boundary * 2304 + d] = x[i];
}

struct InkCommitArgs { uint segments, taps_stride; uint accepted[8]; };
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_commit_histories(
    const device bfloat* src_conv0 [[buffer(0)]], const device bfloat* src_conv1 [[buffer(1)]],
    const device bfloat* src_conv2 [[buffer(2)]], const device bfloat* src_conv3 [[buffer(3)]],
    const device bfloat* src_conv4 [[buffer(4)]], const device bfloat* src_conv5 [[buffer(5)]],
    const device bfloat* src_conv6 [[buffer(6)]], const device bfloat* src_conv7 [[buffer(7)]],
    const device bfloat* taps0 [[buffer(8)]], const device bfloat* taps1 [[buffer(9)]],
    const device bfloat* taps2 [[buffer(10)]], const device bfloat* taps3 [[buffer(11)]],
    const device bfloat* taps4 [[buffer(12)]], const device bfloat* taps5 [[buffer(13)]],
    const device bfloat* taps6 [[buffer(14)]], const device bfloat* taps7 [[buffer(15)]],
    device bfloat* dst_conv0 [[buffer(16)]], device bfloat* dst_conv1 [[buffer(17)]],
    device bfloat* dst_conv2 [[buffer(18)]], device bfloat* dst_conv3 [[buffer(19)]],
    device bfloat* dst_conv4 [[buffer(20)]], device bfloat* dst_conv5 [[buffer(21)]],
    device bfloat* dst_conv6 [[buffer(22)]], device bfloat* dst_conv7 [[buffer(23)]],
    constant InkCommitArgs& a [[buffer(24)]],
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint seg = tg.y;
  if (seg >= a.segments) return;
  const uint slot = tg.x / 26u, tile = tg.x % 26u;
  if (slot >= 48u) return;
  const uint cs = tile < 4u ? 0u : (tile < 8u ? 1u : (tile < 17u ? 2u : 3u));
  const uint widths[4] = { 1024u, 1024u, 2304u, 2304u };
  const uint conv_bases[4] = { 0u, 3072u, 6144u, 13056u };
  const uint taps_bases[4] = { 0u, 1024u, 2048u, 4352u };
  const uint tile_off[4] = { 0u, 4u, 8u, 17u };
  const uint width = widths[cs];
  const uint c = (tile - tile_off[cs]) * 256u + tid;
  if (c >= width) return;
  const device bfloat* sc = seg == 0u ? src_conv0 : seg == 1u ? src_conv1 : seg == 2u ? src_conv2 : seg == 3u ? src_conv3 : seg == 4u ? src_conv4 : seg == 5u ? src_conv5 : seg == 6u ? src_conv6 : src_conv7;
  const device bfloat* tp = seg == 0u ? taps0 : seg == 1u ? taps1 : seg == 2u ? taps2 : seg == 3u ? taps3 : seg == 4u ? taps4 : seg == 5u ? taps5 : seg == 6u ? taps6 : taps7;
  device bfloat* dc = seg == 0u ? dst_conv0 : seg == 1u ? dst_conv1 : seg == 2u ? dst_conv2 : seg == 3u ? dst_conv3 : seg == 4u ? dst_conv4 : seg == 5u ? dst_conv5 : seg == 6u ? dst_conv6 : dst_conv7;
  const size_t hist_off = size_t(slot) * 19968u + conv_bases[cs] + c;
  const size_t taps_off = size_t(slot) * size_t(a.taps_stride) * 6656u + size_t(a.taps_stride) * taps_bases[cs] + c;
  const uint acc = a.accepted[seg];
  bfloat v[3];
  for (uint t = 0; t < 3u; ++t) {
    const uint idx = acc + t;
    v[t] = idx < 3u ? sc[hist_off + size_t(idx) * width] : tp[taps_off + size_t(idx - 3u) * width];
  }
  for (uint t = 0; t < 3u; ++t) dc[hist_off + size_t(t) * width] = v[t];
}
