// Sorted MoE residuals, after lfm2.metal: the original pick order and both BF16 boundaries.
inline bfloat4 lfm2_sorted_state(const device bfloat* y, const device bfloat* wts, device bfloat* h,
                                 const device uint* pos, uint row, uint t) {
  const size_t base = size_t(row) * D + 4 * t;
  bfloat4 saved;
  for (int j = 0; j < 4; ++j) {
    float acc = 0.0f;
    for (uint k = 0; k < TOPK; ++k) {
      const size_t slot = size_t(row) * TOPK + k;
      acc += float(static_cast<bfloat>(float(y[size_t(pos[slot]) * D + 4 * t + j]) * float(wts[slot])));
    }
    const bfloat hv = h[base + j] + static_cast<bfloat>(acc);
    h[base + j] = hv;
    saved[j] = hv;
  }
  return saved;
}

// lfm2_combine_sorted followed by lfm2_norm, without writing and rereading r.
[[max_total_threads_per_threadgroup(512)]]
[[kernel]] void lfm2_combine_sorted_norm(const device bfloat* y [[buffer(0)]], const device bfloat* wts [[buffer(1)]],
                                         device bfloat* h [[buffer(2)]], const device bfloat* w [[buffer(3)]],
                                         constant float& eps [[buffer(4)]], device bfloat* x [[buffer(5)]],
                                         const device uint* pos [[buffer(8)]], uint row [[threadgroup_position_in_grid]],
                                         uint t [[thread_position_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                                         uint lane [[thread_index_in_simdgroup]]) {
  const bfloat4 saved = lfm2_sorted_state(y, wts, h, pos, row, t);
  float v[4];
  for (int j = 0; j < 4; ++j) v[j] = saved[j];
  threadgroup float sums[32], inv[1];
  lfm2_rms(v, w, eps, x, size_t(row) * D + 4 * t, t, sg, lane, sums, inv);
}
