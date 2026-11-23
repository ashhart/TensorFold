#include <metal_stdlib>
using namespace metal;
#pragma METAL fp contract(off)
struct InkRuntimeArgs { uint count, boundary; };
struct InkStageConvArgs { uint rows, width, commit, pairs; };
struct InkCommitHistoriesArgs { uint offset, slots, tap_rows; };
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
kernel void tf_ink2_stage_conv(
    device bfloat* hist0 [[buffer(0)]], const device bfloat* raw0 [[buffer(1)]], device bfloat* staged0 [[buffer(2)]],
    device bfloat* hist1 [[buffer(3)]], const device bfloat* raw1 [[buffer(4)]], device bfloat* staged1 [[buffer(5)]],
    constant InkStageConvArgs& a [[buffer(6)]], uint tid [[thread_position_in_grid]]) {
  const uint stride = a.pairs * a.width;
  uint block = 0u, rem = tid;
  if (a.rows > 16u) {
    const uint blocks = (a.rows + 15u) / 16u;
    if (tid >= stride * blocks) return;
    block = tid / stride;
    rem = tid - block * stride;
  } else if (tid >= stride) {
    return;
  }
  const uint p = (a.pairs == 1u) ? 0u : (rem >= a.width ? 1u : 0u);
  const uint c = (p == 0u) ? rem : (rem - a.width);
  device bfloat* hist = (p == 0) ? hist0 : hist1;
  const device bfloat* raw = (p == 0) ? raw0 : raw1;
  device bfloat* staged = (p == 0) ? staged0 : staged1;
  const size_t w = a.width;
  const uint r0 = block * 16u;
  const uint r1 = min(r0 + 16u, a.rows);
  for (uint r = r0; r < r1; ++r) staged[(size_t)(3 + r) * w + c] = raw[(size_t)r * w + c];
  if (block != 0u) return;
  const bfloat h0 = hist[c];
  const bfloat h1 = hist[w + c];
  const bfloat h2 = hist[2 * w + c];
  staged[c] = h0;
  staged[w + c] = h1;
  staged[2 * w + c] = h2;
  if (a.commit != 0) {
    if (a.rows == 1u) {
      hist[c] = h1;
      hist[w + c] = h2;
      hist[2 * w + c] = raw[c];
    } else if (a.rows == 2u) {
      hist[c] = h2;
      hist[w + c] = raw[c];
      hist[2 * w + c] = raw[w + c];
    } else {
      const size_t base = (size_t)(a.rows - 3u) * w + c;
      hist[c] = raw[base];
      hist[w + c] = raw[base + w];
      hist[2 * w + c] = raw[base + 2 * w];
    }
  }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_commit_histories(
    const device bfloat* taps [[buffer(0)]], device bfloat* conv [[buffer(1)]],
    constant InkCommitHistoriesArgs& a [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  const uint slot_elems = 3u * 6656u;
  if (i >= a.slots * slot_elems) return;
  const uint slot = i / slot_elems;
  const uint rem = i % slot_elems;
  uint w;
  uint prefix;
  uint local;
  if (rem < 3072u) {
    w = 1024u; prefix = 0u; local = rem;
  } else if (rem < 6144u) {
    w = 1024u; prefix = 1024u; local = rem - 3072u;
  } else if (rem < 13056u) {
    w = 2304u; prefix = 2048u; local = rem - 6144u;
  } else {
    w = 2304u; prefix = 4352u; local = rem - 13056u;
  }
  const uint row = local / w;
  const uint c = local % w;
  const size_t src = (size_t)slot * a.tap_rows * 6656u + (size_t)a.tap_rows * prefix + (size_t)(a.offset + row) * w + c;
  conv[i] = taps[src];
}
