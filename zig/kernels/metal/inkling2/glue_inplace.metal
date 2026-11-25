// Direct-history static and dynamic short convolutions.

[[max_total_threads_per_threadgroup(32)]]
kernel void tf_inkling2_conv_static_inplace(
    const device bfloat* history [[buffer(0)]], const device bfloat* raw [[buffer(1)]],
    const device bfloat* weight [[buffer(2)]], device bfloat* out [[buffer(3)]],
    constant InkConvArgs& a [[buffer(4)]], uint2 at [[thread_position_in_grid]]) {
  const uint c = at.x, r = at.y;
  if (c >= a.channels || r >= a.rows) return;
  float sum = float(raw[size_t(r) * a.channels + c]);
  for (uint tap = 0; tap < 4; ++tap) {
    const uint idx = r + tap;
    const bfloat x = idx < 3 ? history[size_t(idx) * a.channels + c] : raw[size_t(idx - 3) * a.channels + c];
    sum += float(weight[c * 4 + tap]) * float(x);
  }
  out[size_t(r) * a.channels + c] = bfloat(precise::divide(sum, precise::sqrt(5.0f)));
}

[[max_total_threads_per_threadgroup(256)]]
kernel void tf_inkling2_conv_dynamic_inplace(
    const device bfloat* history [[buffer(0)]], const device bfloat* raw [[buffer(1)]],
    const device bfloat* input [[buffer(2)]], const device bfloat* static_w [[buffer(3)]],
    const device bfloat* projection [[buffer(4)]], device bfloat* out [[buffer(5)]],
    constant InkConvArgs& a [[buffer(6)]], uint t [[thread_position_in_threadgroup]],
    uint r [[threadgroup_position_in_grid]]) {
  if (r >= a.rows) return;
  threadgroup float projected[32];
  if (t < 32) {
    projected[t] = float(bfloat(ink_chain(projection + size_t(t) * 2304, input + size_t(r) * 2304, 2304)));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint c = t; c < a.channels; c += 256) {
    float sum = float(raw[size_t(r) * a.channels + c]);
    for (uint lag = 0; lag < 4; ++lag) {
      const float w = float(static_w[lag * a.channels + c]) + projected[4 * (c / 128) + lag];
      const uint idx = r + 3 - lag;
      const bfloat x = idx < 3 ? history[size_t(idx) * a.channels + c] : raw[size_t(idx - 3) * a.channels + c];
      sum += w * float(x);
    }
    out[size_t(r) * a.channels + c] = bfloat(precise::divide(sum, 3.0f));
  }
}
