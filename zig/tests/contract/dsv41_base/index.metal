// The base's index per-slot kernels, renamed tf_base_*: a test appends them to today's source to compare bytes.

// Decode's few rows: eight simdgroups share a row's decoded query and score a key each in turn, the same bits as above.
[[kernel, max_total_threads_per_threadgroup(256)]] void tf_base_dsv41_ix_scores_dec(
    const device uchar* Q [[buffer(0)]], const device uchar* QS [[buffer(1)]], const device bfloat* W [[buffer(2)]],
    const device uchar* IK [[buffer(3)]], device float* S [[buffer(4)]], const device uchar* CM [[buffer(5)]],
    constant TfDsv41IxArgs& a [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
    uint t [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint sgs [[simdgroups_per_threadgroup]]) {
  const uint row = tg.y;
  if (row >= a.rows) return;
  threadgroup float qf[IH * TF_IX_QPAD]; // 16 KiB: this row's query, decoded once for the whole chunk
  threadgroup float wf[IH];
  const size_t at = size_t(row) * QROW;
  for (uint i = t; i < IH * ID; i += sgs * 32u) qf[(i / ID) * TF_IX_QPAD + i % ID] = tf_ix_q(Q, QS, at, i / ID, i % ID);
  // the row's weights times the softmax scale once, in fp32 (model.py:556), before any head meets a score
  if (t < IH) wf[t] = float(W[size_t(row) * a.w_stride + t]) * a.w_scale;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint first = tg.x * a.chunk;
  const uint last = min(first + a.chunk, a.keys);
  // keys a query reaches: (start + row) / ratio (model.py:565 prefill with start 1, 568 decode with end_pos)
  const uint visible = (a.start + row) / a.ratio;
  const threadgroup float* qh = qf + size_t(lane) * TF_IX_QPAD;
  const float wh = wf[lane];
  for (uint key = first + sg; key < last; key += sgs) {
#if TF_CTL == 4 // gate 2's control: one float past the LAST output, into the guard bytes (lane 0, before any mask)
    if (lane == 0u && row == a.rows - 1u && key == a.keys - 1u) S[size_t(a.rows) * a.out_stride] = 1.0f;
#endif
    const bool live = key < visible && (a.mask_stride == 0u || CM[size_t(row) * a.mask_stride + key / CBS] != 0u);
    if (!live) {
      if (lane == 0u) S[size_t(row) * a.out_stride + key] = -INFINITY; // model.py:566, 576 mask unreachable keys
      continue;
    }
    const device uchar* ik = IK + size_t(key) * a.ik_stride;
    const device uchar* ks = ik + ID / 2u; // a key's row: 64 FP4 bytes then its 4 e8m0 scales
    float p[4];
    for (uint q = 0; q < 4u; ++q) {
#if TF_CTL == 3 // gate 4's control: the group sums accumulated in half
      half h = 0.0h;
      for (uint i = 32u * q; i < 32u * q + 32u; ++i) h += half(qh[i]) * half(tf_ix_e2m1((ik[i >> 1] >> (4u * (i & 1u))) & 0xFu));
      p[q] = float(h);
#else
      float acc = 0.0f;
      for (uint i = 32u * q; i < 32u * q + 32u; ++i) acc += qh[i] * tf_ix_e2m1((ik[i >> 1] >> (4u * (i & 1u))) & 0xFu);
      p[q] = acc;
#endif
    }
    const float prod = ((p[0] * tf_ix_e8m0(ks[0])) + (p[1] * tf_ix_e8m0(ks[1]))) + ((p[2] * tf_ix_e8m0(ks[2])) + (p[3] * tf_ix_e8m0(ks[3])));
    const float term = tf_ix_relu(prod) * wh;
    float acc = 0.0f;
#if TF_CTL == 2 // gate 3's control: the head fold in reverse, and head 0 (the last one folded) dropped
    for (uint h = IH - 1u; h >= 1u; --h) acc += simd_shuffle(term, ushort(h));
#else
    for (uint h = 0u; h < IH; ++h) acc += simd_shuffle(term, ushort(h));
#endif
    if (lane == 0u) S[size_t(row) * a.out_stride + key] = acc;
  }
}
