//! Inkling 2 Mini tree-verify convolutions and arbitrary surviving-path KV and conv compaction.
#include <metal_stdlib>
using namespace metal;
#pragma METAL fp contract(off)

struct InkConvTreeArgs { uint rows, channels; };

struct InkKeepKvArgs {
  ulong key_at[48];
  ulong value_at[48];
  uint rings[48];
  uint start;
  uint path_len;
  uint path[16];
};

struct InkKeepConvArgs { uint src[3]; uint tap_rows; };

[[max_total_threads_per_threadgroup(32)]]
kernel void tf_ink2_conv_static_tree(
    const device bfloat* history [[buffer(0)]], const device bfloat* raw [[buffer(1)]],
    const device bfloat* weight [[buffer(2)]], const device uint* src [[buffer(3)]],
    device bfloat* out [[buffer(4)]], constant InkConvTreeArgs& a [[buffer(5)]],
    uint2 at [[thread_position_in_grid]]) {
  const uint c = at.x, r = at.y;
  if (c >= a.channels || r >= a.rows) return;
  const uint idx3 = src[r * 4 + 3];
  float sum = float(idx3 < 3 ? history[size_t(idx3) * a.channels + c] : raw[size_t(idx3 - 3) * a.channels + c]);
  for (uint tap = 0; tap < 4; ++tap) {
    const uint idx = src[r * 4 + tap];
    const bfloat x = idx < 3 ? history[size_t(idx) * a.channels + c] : raw[size_t(idx - 3) * a.channels + c];
    sum += float(weight[c * 4 + tap]) * float(x);
  }
  out[size_t(r) * a.channels + c] = bfloat(precise::divide(sum, precise::sqrt(5.0f)));
}

[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_conv_dynamic_tree(
    const device bfloat* history [[buffer(0)]], const device bfloat* raw [[buffer(1)]],
    const device bfloat* input [[buffer(2)]], const device bfloat* static_w [[buffer(3)]],
    const device bfloat* projection [[buffer(4)]], const device uint* src [[buffer(5)]],
    device bfloat* out [[buffer(6)]], constant InkConvTreeArgs& a [[buffer(7)]],
    uint t [[thread_position_in_threadgroup]], uint r [[threadgroup_position_in_grid]]) {
  if (r >= a.rows) return;
  threadgroup float projected[32];
  if (t < 32) {
    float dot = 0.0f;
    for (uint d = 0; d < 2304; ++d)
      dot += float(projection[size_t(t) * 2304 + d]) * float(input[size_t(r) * 2304 + d]);
    projected[t] = float(bfloat(dot));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint c = t; c < a.channels; c += 256) {
    const uint idx3 = src[r * 4 + 3];
    float sum = float(idx3 < 3 ? history[size_t(idx3) * a.channels + c] : raw[size_t(idx3 - 3) * a.channels + c]);
    for (uint lag = 0; lag < 4; ++lag) {
      const float w = float(static_w[lag * a.channels + c]) + projected[4 * (c / 128) + lag];
      const uint idx = src[r * 4 + 3 - lag];
      const bfloat x = idx < 3 ? history[size_t(idx) * a.channels + c] : raw[size_t(idx - 3) * a.channels + c];
      sum += w * float(x);
    }
    out[size_t(r) * a.channels + c] = bfloat(precise::divide(sum, 3.0f));
  }
}

[[max_total_threads_per_threadgroup(128)]]
kernel void tf_ink2_keep_kv(
    device bfloat* kv [[buffer(0)]], constant InkKeepKvArgs& a [[buffer(1)]],
    uint2 tg [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]]) {
  const uint slot = tg.x, kvh = tg.y;
  if (slot >= 48 || kvh >= 8 || d >= 128 || a.path_len == 0 || a.path_len > 16) return;
  const uint ring = a.rings[slot];
  const ulong head_off = ulong(kvh) * ulong(ring) * 128;
  device bfloat* k_base = (device bfloat*)((device uchar*)kv + a.key_at[slot]) + head_off;
  device bfloat* v_base = (device bfloat*)((device uchar*)kv + a.value_at[slot]) + head_off;
  bfloat ks[16], vs[16];
  for (uint j = 0; j < a.path_len; ++j) {
    const ulong src = ulong((a.start + a.path[j]) % ring) * 128 + d;
    ks[j] = k_base[src];
    vs[j] = v_base[src];
  }
  for (uint j = 0; j < a.path_len; ++j) {
    const ulong dst = ulong((a.start + j) % ring) * 128 + d;
    k_base[dst] = ks[j];
    v_base[dst] = vs[j];
  }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_keep_conv(
    const device bfloat* src_conv [[buffer(0)]], const device bfloat* taps [[buffer(1)]],
    device bfloat* conv [[buffer(2)]], constant InkKeepConvArgs& a [[buffer(3)]],
    uint2 at [[thread_position_in_grid]]) {
  const uint c = at.x, row = at.y;
  if (row >= 48 * 4) return;
  const uint slot = row / 4, cs = row % 4;
  const uint widths[4] = {1024, 1024, 2304, 2304};
  const uint width = widths[cs];
  if (c >= width) return;
  uint conv_base = 0, tap_base = 0;
  for (uint s = 0; s < cs; ++s) {
    conv_base += 3 * widths[s];
    tap_base += a.tap_rows * widths[s];
  }
  const ulong hist_base = ulong(slot) * 3 * 6656 + conv_base + c;
  const ulong tap_off = ulong(slot) * ulong(a.tap_rows) * 6656 + tap_base + c;
  bfloat v[3];
  for (uint out_r = 0; out_r < 3; ++out_r) {
    const uint idx = a.src[out_r];
    v[out_r] = idx < 3 ? src_conv[hist_base + ulong(idx) * width] : taps[tap_off + ulong(idx - 3) * width];
  }
  for (uint out_r = 0; out_r < 3; ++out_r) conv[hist_base + ulong(out_r) * width] = v[out_r];
}
