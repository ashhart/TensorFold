// Exact sp_norm RMSNorm from BF16 and fused ungated AR -> RMSNorm, after glue.metal.

// Matches sp_norm's exact reduction order: per-thread fma(v, v, ss), simd_sum, 8-simdgroup serial sum, precise rsqrt.
inline float ink_norm_inv(float ss, uint t, uint lane, threadgroup float* part) {
  const float reduced = simd_sum(ss);
  if (lane == 0) part[t / 32] = reduced;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint s = 0; s < 8; ++s) total += part[s];
  return precise::rsqrt(precise::divide(total, 2304.0f) + INKLING_EPS);
}

inline void ink_ar_finish_norm_regs(
    const device bfloat* blocks, const thread float* partial_regs,
    const thread bfloat* override_bf, uint override_slot,
    const device float* query, const device bfloat* norm_w, device bfloat* out,
    uint sources, uint t, uint lane,
    threadgroup float* dots, threadgroup float* means, threadgroup float* weights, threadgroup float* norm_red) {
  for (uint s = 0; s < sources; ++s) {
    float dot = 0.0f;
    for (uint k = 0; k < 9; ++k) {
      const uint d = t + k * 256;
      const float z = ink_reg_value(blocks, partial_regs, override_bf, override_slot, s, sources, k, d) * means[s];
      dot += query[d] * z;
    }
    const float reduced = simd_sum(dot);
    if (lane == 0) dots[s * 8 + t / 32] = reduced;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  ink_ar_softmax(sources, t, dots, weights);
  float vals[9];
  float ss = 0.0f;
  for (uint k = 0; k < 9; ++k) {
    const uint d = t + k * 256;
    float acc = 0.0f;
    for (uint s = 0; s < sources; ++s)
      acc += weights[s] * (ink_reg_value(blocks, partial_regs, override_bf, override_slot, s, sources, k, d) * means[s]);
    vals[k] = acc;
    ss = fma(acc, acc, ss);
  }
  const float inv = ink_norm_inv(ss, t, lane, norm_red);
  for (uint k = 0; k < 9; ++k) {
    const uint d = t + k * 256;
    out[d] = bfloat(float(norm_w[d]) * (vals[k] * inv));
  }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void tf_inkling2_norm_bf16(
    const device bfloat* x [[buffer(0)]], const device bfloat* norm_w [[buffer(1)]],
    device bfloat* out [[buffer(2)]], constant InkEmbedArgs& a [[buffer(3)]],
    uint t [[thread_position_in_threadgroup]], uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  if (row >= a.rows) return;
  const device bfloat* x_row = x + size_t(row) * 2304;
  device bfloat* out_row = out + size_t(row) * 2304;
  float norm_vals[9];
  float ss = 0.0f;
  for (uint k = 0; k < 9; ++k) {
    const float v = float(x_row[t + k * 256]);
    norm_vals[k] = v;
    ss = fma(v, v, ss);
  }
  threadgroup float red[8];
  const float norm_inv = ink_norm_inv(ss, t, lane, red);
  for (uint k = 0; k < 9; ++k) {
    const uint d = t + k * 256;
    out_row[d] = bfloat(float(norm_w[d]) * (norm_vals[k] * norm_inv));
  }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void tf_inkling2_attn_partial_ar_norm(
    const device bfloat* p [[buffer(0)]], const device bfloat* attn_conv [[buffer(1)]],
    device bfloat* blocks [[buffer(2)]], device float* block_inv_rms [[buffer(3)]],
    device float* partial [[buffer(4)]], const device float* query [[buffer(5)]],
    const device bfloat* norm_w [[buffer(6)]], device bfloat* out [[buffer(7)]],
    constant InkAttnPartialArArgs& a [[buffer(8)]],
    uint t [[thread_position_in_threadgroup]], uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  if (row >= a.rows || a.sources == 0 || a.sources > 5 || a.boundary > 4) return;
  const device bfloat* p_row = p + size_t(row) * 2304;
  const device bfloat* x_row = attn_conv + size_t(row) * 2304;
  device bfloat* blocks_row = blocks + size_t(row) * 4 * 2304;
  device float* inv_row = block_inv_rms + size_t(row) * 4;
  device float* partial_row = partial + size_t(row) * 2304;
  device bfloat* out_row = out + size_t(row) * 2304;
  float partial_regs[9];
  bfloat p_bf[9];
  float part_ss = 0.0f;
  float block_ss = 0.0f;
  for (uint k = 0; k < 9; ++k) {
    const uint d = t + k * 256;
    const float x = float(x_row[d]);
    float val;
    if (a.boundary) {
      const bfloat pb = p_row[d];
      p_bf[k] = pb;
      blocks_row[size_t(a.boundary - 1) * 2304 + d] = pb;
      const float pv = float(pb);
      block_ss += pv * pv;
      val = x;
    } else {
      val = float(p_row[d]) + x;
    }
    partial_row[d] = val;
    partial_regs[k] = val;
    part_ss += val * val;
  }
  threadgroup float red[8], block_red[8], dots[5 * 8], means[5], weights[5];
  if (t + 1 < a.sources && !(a.boundary != 0 && t == a.boundary - 1)) means[t] = inv_row[t];
  const float attn_part_inv = ink_partial_inv_rms(part_ss, t, lane, red);
  float block_inv = 0.0f;
  if (a.boundary) block_inv = ink_partial_inv_rms(block_ss, t, lane, block_red);
  if (t == 0) {
    if (a.boundary) {
      inv_row[a.boundary - 1] = block_inv;
      if (a.boundary < a.sources) means[a.boundary - 1] = block_inv;
    }
    means[a.sources - 1] = attn_part_inv;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint override_slot = a.boundary ? (a.boundary - 1) : 5;
  ink_ar_finish_norm_regs(blocks_row, partial_regs, p_bf, override_slot, query, norm_w, out_row, a.sources, t, lane, dots, means, weights, red);
}

struct InkMlpPartialArNormArgs { uint rows, sources; };
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_inkling2_mlp_partial_ar_norm(
    const device float* partial_in [[buffer(0)]], const device bfloat* mlp_conv [[buffer(1)]],
    const device bfloat* blocks [[buffer(2)]], const device float* block_inv_rms [[buffer(3)]],
    const device float* query [[buffer(4)]], const device bfloat* norm_w [[buffer(5)]],
    device bfloat* p_out [[buffer(6)]], device float* partial_out [[buffer(7)]],
    device bfloat* norm_out [[buffer(8)]], constant InkMlpPartialArNormArgs& a [[buffer(9)]],
    uint t [[thread_position_in_threadgroup]], uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
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
    const bfloat next_p_n = bfloat(pin[d] + float(conv[d]));
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
