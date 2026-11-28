// A steered visit's output fused with the next mix and its norm, as tf_inkling2_mlp_partial_ar_norm writes them.
struct InkSteerNormArgs { uint rows, sources; float alpha; };
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_mlp_partial_ar_norm_steer(
    const device float* partial_in [[buffer(0)]], const device bfloat* mlp_conv [[buffer(1)]],
    const device bfloat* blocks [[buffer(2)]], const device float* block_inv_rms [[buffer(3)]],
    const device float* query [[buffer(4)]], const device bfloat* norm_w [[buffer(5)]],
    const device float* dir [[buffer(6)]], device bfloat* p_out [[buffer(7)]],
    device float* partial_out [[buffer(8)]], device bfloat* norm_out [[buffer(9)]],
    constant InkSteerNormArgs& a [[buffer(10)]], uint t [[thread_position_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  if (row >= a.rows || a.sources == 0 || a.sources > 5) return;
  const device float* pin = partial_in + size_t(row) * 2304;
  const device bfloat* conv = mlp_conv + size_t(row) * 2304;
  const device bfloat* blocks_row = blocks + size_t(row) * 4 * 2304;
  const device float* inv_row = block_inv_rms + size_t(row) * 4;
  device bfloat* p_row = p_out + size_t(row) * 2304;
  device float* pout_row = partial_out + size_t(row) * 2304;
  device bfloat* out_row = norm_out + size_t(row) * 2304;
  float partial_regs[9];
  float part_ss = 0.0f;
  for (uint k = 0; k < 9; ++k) {
    const uint d = t + k * 256;
    const float hv = float(bfloat(pin[d] + float(conv[d])));
    const float step = a.alpha * dir[d];
    // a zero step leaves the rounded residual alone, so a negative zero stays negative
    const bfloat next_p_n = step == 0.0f ? bfloat(hv) : bfloat(hv + step);
    const float widened = float(next_p_n);
    p_row[d] = next_p_n;
    pout_row[d] = widened;
    partial_regs[k] = widened;
    part_ss += widened * widened;
  }
  threadgroup float red[8], dots[5 * 8], means[5], weights[5];
  if (t + 1 < a.sources) means[t] = inv_row[t];
  const float inv = ink_partial_inv_rms(part_ss, t, lane, red);
  if (t == 0) means[a.sources - 1] = inv;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const bfloat dummy[1] = { bfloat(0.0f) };
  ink_ar_finish_norm_regs(blocks_row, partial_regs, dummy, 5, query, norm_w, out_row, a.sources, t, lane, dots, means, weights, red);
}
