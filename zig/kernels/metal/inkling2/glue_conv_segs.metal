struct InkConvDynSeg { ulong taps, input, out; uint row0, rows; };
struct InkConvDynSegsArgs { uint n_segs, total_rows, channels; };

inline void ink2_conv_dynamic_t(
    const device bfloat* taps, const device bfloat* input,
    const device bfloat* static_w, const device bfloat* projection,
    device bfloat* out, uint r, uint channels, uint t, threadgroup float* projected) {
  if (t < 32) {
    float dot = 0.0f;
    for (uint d = 0; d < 2304; ++d)
      dot += float(projection[size_t(t) * 2304 + d]) * float(input[size_t(r) * 2304 + d]);
    projected[t] = float(bfloat(dot));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint c = t; c < channels; c += 256) {
    float sum = float(taps[size_t(r + 3) * channels + c]);
    for (uint lag = 0; lag < 4; ++lag) {
      const float w = float(static_w[lag * channels + c]) + projected[4 * (c / 128) + lag];
      sum += w * float(taps[size_t(r + 3 - lag) * channels + c]);
    }
    out[size_t(r) * channels + c] = bfloat(precise::divide(sum, 3.0f));
  }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void tf_inkling2_conv_dynamic_segs(
    const device InkConvDynSeg* segs [[buffer(0)]], const device bfloat* static_w [[buffer(1)]],
    const device bfloat* projection [[buffer(2)]], constant InkConvDynSegsArgs& a [[buffer(3)]],
    uint t [[thread_position_in_threadgroup]], uint global_r [[threadgroup_position_in_grid]]) {
  if (global_r >= a.total_rows) return;
  uint si = 0;
  while (si + 1u < a.n_segs && global_r >= segs[si].row0 + segs[si].rows) ++si;
  const device InkConvDynSeg& seg = segs[si];
  const uint r = global_r - seg.row0;
  threadgroup float projected[32];
  ink2_conv_dynamic_t(
      (const device bfloat*)seg.taps, (const device bfloat*)seg.input,
      static_w, projection, (device bfloat*)seg.out, r, a.channels, t, projected);
}
