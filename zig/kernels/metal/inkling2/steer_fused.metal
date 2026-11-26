// tf_inkling2_mlp_partial_ar with p steered exactly as tf_ink2_mlp_partial_steer rounds it; the mix is main's.
struct InkSteerArArgs { uint rows, sources, queries; float alpha; };
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_mlp_partial_ar_steer(
    const device float* partial_in [[buffer(0)]], const device bfloat* mlp_conv [[buffer(1)]],
    const device bfloat* blocks [[buffer(2)]], const device float* block_inv_rms [[buffer(3)]],
    const device float* queries [[buffer(4)]], device bfloat* p_out [[buffer(5)]],
    device float* partial_out [[buffer(6)]], device float* ar_out [[buffer(7)]],
    const device float* dir [[buffer(8)]], constant InkSteerArArgs& a [[buffer(9)]],
    uint t [[thread_position_in_threadgroup]], uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  if (row >= a.rows || a.sources == 0 || a.sources > 5 || a.queries == 0 || a.queries > 3) return;
  const device float* pin_row = partial_in + size_t(row) * 2304;
  const device bfloat* conv_row = mlp_conv + size_t(row) * 2304;
  const device bfloat* blocks_row = blocks + size_t(row) * 4 * 2304;
  const device float* inv_row = block_inv_rms + size_t(row) * 4;
  device bfloat* p_row = p_out + size_t(row) * 2304;
  device float* pout_row = partial_out + size_t(row) * 2304;
  float partial_regs[9];
  float part_ss = 0.0f;
  for (uint k = 0; k < 9; ++k) {
    const uint d = t + k * 256;
    const float hv_ar = float(bfloat(pin_row[d] + float(conv_row[d])));
    const float step_ar = a.alpha * dir[d];
    const bfloat next_p = bfloat(hv_ar + step_ar);
    const float widened = float(next_p);
    p_row[d] = next_p;
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
  for (uint q = 0; q < a.queries; ++q) {
    const device float* query = queries + size_t(q) * 2304;
    device float* out_row = ar_out + (size_t(q) * a.rows + row) * 2304;
    ink_ar_finish_regs(blocks_row, partial_regs, dummy, 5, query, out_row, a.sources, t, lane, dots, means, weights);
    if (q + 1 < a.queries) threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
