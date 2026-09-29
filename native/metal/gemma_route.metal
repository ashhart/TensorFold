
  const uint lane = thread_index_in_simdgroup;
  const uint r = threadgroup_position_in_grid.x;
  float sc[NE / 32];
  for (int j = 0; j < NE / 32; j++) sc[j] = float(G[int(r) * NE + int(lane) + 32 * j]);
  float picked[K];
  int ids[K];
  for (int k = 0; k < K; k++) {
    float best = -INFINITY;
    int best_e = 1 << 20;
    for (int j = 0; j < NE / 32; j++) {
      if (sc[j] > best) { best = sc[j]; best_e = int(lane) + 32 * j; }
    }
    const float top = simd_max(best);
    const int winner = simd_min(best == top ? best_e : (1 << 20));
    for (int j = 0; j < NE / 32; j++) {
      if (int(lane) + 32 * j == winner) sc[j] = -INFINITY;
    }
    picked[k] = top;
    ids[k] = winner;
  }
  if (lane == 0) {
    float total = 0.0f;
    for (int k = 0; k < K; k++) total += metal::exp(picked[k] - picked[0]);
    for (int k = 0; k < K; k++) {
      const float p = float(bfloat(metal::exp(picked[k] - picked[0]) / total));
      IDX[int(r) * K + k] = uint(ids[k]);
      WT[int(r) * K + k] = bfloat(p * float(PES[ids[k]]));
    }
  }
