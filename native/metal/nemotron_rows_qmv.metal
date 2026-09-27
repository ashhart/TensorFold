
  // Threadgroup: one simdgroup per input row (x) for BLK blocks of RPS output rows (y). The row count is a launch
  // dimension only: every simdgroup runs the same instructions over its own row.
  const uint lane = thread_index_in_simdgroup;
  const int r = int(thread_position_in_threadgroup.x) / 32;
  const int row0 = int(thread_position_in_grid.y) * RPS;
  float acc[RPS];
  tf_rowdot<K, GS, RPS>((const device uint8_t*)W + size_t(row0) * (K / 2), S + size_t(row0) * (K / GS),
                        B + size_t(row0) * (K / GS), X + size_t(r) * K, lane, acc);
  if (lane == 0)
    for (int j = 0; j < RPS; j++) OUT[size_t(r) * N + row0 + j] = bfloat(acc[j]);
