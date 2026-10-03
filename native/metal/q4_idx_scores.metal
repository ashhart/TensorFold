
  // Simdgroup s of threadgroup (x, y) scores blocks (8 x + s) BB .. + BB - 1 for rows RB y .. RB y + RB - 1, each
  // block's keys read once for those rows: block b's score for row r is the sum over the HI indexer heads (in order)
  // of relu(q . pooled b) (fp32: a lane's DI / 32 dims in order, then simd_sum), over sqrt(DI). Only rows past TOP
  // complete blocks, and only their complete blocks, are scored.
  const uint lane = thread_index_in_simdgroup;
  const int b0 = (int(threadgroup_position_in_grid.x) * 8 + int(simdgroup_index_in_threadgroup)) * BB;
  const int r0 = int(threadgroup_position_in_grid.y) * RB;
  const int nb = int(POOLED_shape[0]), r1 = metal::min(int(Q_shape[0]), r0 + RB);
  constexpr int PER = DI / 32;
  for (int b = b0; b < b0 + BB && b < nb; b++) {
    const device bfloat* pb = POOLED + size_t(b) * DI + lane * PER;
    float p[PER];
    for (int i = 0; i < PER; i++) p[i] = float(pb[i]);
    for (int r = r0; r < r1; r++) {
      const int complete = COMPLETE[r];
      if (complete <= TOP || b >= complete) continue;
      float s = 0.0f;
      for (int h = 0; h < HI; h++) {
        const device bfloat* qh = Q + (r * HI + h) * DI + lane * PER;
        float dot = 0.0f;
        for (int i = 0; i < PER; i++) dot = fma(float(qh[i]), p[i], dot);
        s += metal::max(simd_sum(dot), 0.0f);
      }
      if (lane == 0) SC[size_t(r) * nb + b] = s / metal::precise::sqrt(float(DI));
    }
  }
