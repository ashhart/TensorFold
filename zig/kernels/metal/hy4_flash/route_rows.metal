
// Appended to the core router source (its TF_E, TF_TOPK, PER, RouteArgs and tf_sigmoid_precise). A prompt chunk's
// picks and weights: tf_route_select's per-row top-k (sigmoid + bias, ties to the lower id, weights normalized and
// scaled) step for step, one simdgroup a row and a threadgroup per row, so every row runs at once.
[[kernel]] void tf_hy4_route_select_rows(const device float* LOGITS [[buffer(0)]], const device float* BIAS [[buffer(1)]],
                                         constant RouteArgs& a [[buffer(2)]], device int* PICK [[buffer(3)]],
                                         device float* WTS [[buffer(4)]], uint r_ [[threadgroup_position_in_grid]],
                                         uint lane [[thread_index_in_simdgroup]]) {
  const int r = int(r_);
  if (r >= a.rows) return;
  float c[PER], sc[PER];
  for (int j = 0; j < PER; j++) {
    const int id = j * 32 + int(lane);
    if (id < TF_E) {
      sc[j] = tf_sigmoid_precise(LOGITS[r * TF_E + id]);
      c[j] = sc[j] + BIAS[id];
    } else {
      sc[j] = 0.0f; c[j] = -INFINITY;
    }
  }
  float w[TF_TOPK];
  for (int k = 0; k < TF_TOPK; k++) {
    float best = -INFINITY, bsc = 0.0f;
    int bid = TF_E;
    for (int j = 0; j < PER; j++) {
      const int id = j * 32 + int(lane);
      if (id < TF_E && (c[j] > best || (c[j] == best && id < bid))) { best = c[j]; bid = id; bsc = sc[j]; }
    }
    for (int off = 16; off > 0; off /= 2) {
      const float ob = simd_shuffle_xor(best, off);
      const int oi = simd_shuffle_xor(bid, off);
      const float os = simd_shuffle_xor(bsc, off);
      if (ob > best || (ob == best && oi < bid)) { best = ob; bid = oi; bsc = os; }
    }
    w[k] = bsc;
    if (int(lane) == bid % 32) c[bid / 32] = -INFINITY;
    if (lane == 0) PICK[r * TF_TOPK + k] = bid;
  }
  if (lane == 0) {
    float total = w[0];
    for (int k = 1; k < TF_TOPK; k++) total = total + w[k];
    for (int k = 0; k < TF_TOPK; k++) WTS[r * TF_TOPK + k] = (w[k] / total) * a.scale;
  }
}

// tf_hy4_route_select_rows as ranks: one thread an expert (TF_E <= 512, threadgroup 512), each expert's rank the
// count of experts ahead of it (larger sigmoid + bias, ties to the lower id), so the top-k write themselves in pick
// order; then the weights normalized in that order as the loop does. The same picks, order and weights (same
// sigmoid, same sums), without the k sequential arg-max rounds.
[[kernel]] void tf_hy4_route_select_rank(const device float* LOGITS [[buffer(0)]], const device float* BIAS [[buffer(1)]],
                                         constant RouteArgs& a [[buffer(2)]], device int* PICK [[buffer(3)]],
                                         device float* WTS [[buffer(4)]], uint r_ [[threadgroup_position_in_grid]],
                                         uint t [[thread_index_in_threadgroup]]) {
  threadgroup float cs[512];
  threadgroup float w[TF_TOPK];
  const int r = int(r_), id = int(t);
  if (r >= a.rows) return;
  float sc = 0.0f, c = -INFINITY;
  if (id < TF_E) {
    sc = tf_sigmoid_precise(LOGITS[r * TF_E + id]);
    c = sc + BIAS[id];
  }
  cs[t] = c;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (id < TF_E) {
    int rank = 0;
    for (int j = 0; j < TF_E; ++j) {
      const float o = cs[j];
      rank += (o > c || (o == c && j < id)) ? 1 : 0;
    }
    if (rank < TF_TOPK) {
      PICK[r * TF_TOPK + rank] = id;
      w[rank] = sc;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    float total = w[0];
    for (int k = 1; k < TF_TOPK; k++) total = total + w[k];
    for (int k = 0; k < TF_TOPK; k++) WTS[r * TF_TOPK + k] = (w[k] / total) * a.scale;
  }
}
