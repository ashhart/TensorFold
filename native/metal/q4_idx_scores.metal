
  // A simdgroup a block, rows in grid y: block b's score for row r is the sum over the HI indexer heads (in order)
  // of relu(q . pooled b) (fp32: a lane's DI / 32 dims in order, then simd_sum), over sqrt(DI). Only rows past TOP
  // complete blocks, and only their complete blocks, are scored (nothing else is read).
  const uint lane = thread_index_in_simdgroup;
  const int b = int(threadgroup_position_in_grid.x) * 8 + int(simdgroup_index_in_threadgroup);
  const int r = int(threadgroup_position_in_grid.y);
  const int complete = COMPLETE[r];
  if (complete <= TOP || b >= complete) return;
  constexpr int PER = DI / 32;
  const device bfloat* pb = POOLED + size_t(b) * DI + lane * PER;
  float p[PER];
  for (int i = 0; i < PER; i++) p[i] = float(pb[i]);
  float s = 0.0f;
  for (int h = 0; h < HI; h++) {
    const device bfloat* qh = Q + (r * HI + h) * DI + lane * PER;
    float dot = 0.0f;
    for (int i = 0; i < PER; i++) dot = fma(float(qh[i]), p[i], dot);
    s += metal::max(simd_sum(dot), 0.0f);
  }
  if (lane == 0) SC[size_t(r) * POOLED_shape[0] + b] = s / metal::precise::sqrt(float(DI));
