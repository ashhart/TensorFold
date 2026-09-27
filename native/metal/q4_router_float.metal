
  // one simdgroup per expert: lane l reads 8 consecutive inputs at a time, 256 apart; rows in order
  const uint lane = thread_index_in_simdgroup;
  const int e = int(threadgroup_position_in_grid.x) * (T / 32) + int(simdgroup_index_in_threadgroup);
  const int R = rows[0];
  if (e >= NE) return;
  const device bfloat* w = GW + size_t(e) * D;
  float acc[MAXR];
  for (int r = 0; r < MAXR; r++) acc[r] = 0.0f;
  for (int c = 8 * int(lane); c < D; c += 256) {
    float wv[8];
    for (int j = 0; j < 8; j++) wv[j] = float(w[c + j]);
    for (int r = 0; r < MAXR; r++) {
      if (r >= R) break;
      const device bfloat* xr = X + r * D + c;
      float a = acc[r];
      for (int j = 0; j < 8; j++) a = fma(float(xr[j]), wv[j], a);
      acc[r] = a;
    }
  }
  for (int r = 0; r < MAXR; r++) {
    if (r >= R) break;
    const float total = simd_sum(acc[r]);
    if (lane == 0) OUT[r * NE + e] = float(total);
  }
