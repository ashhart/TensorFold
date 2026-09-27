
  // fc2 over the pair's activation (bf16 out)
  const uint lane = thread_index_in_simdgroup;
  const int p = int(threadgroup_position_in_grid.z);
  const size_t e = size_t(IDS[p]);
  const int row0 = (int(threadgroup_position_in_grid.y) * SG + int(simdgroup_index_in_threadgroup)) * RPS;
  const size_t at = e * N + size_t(row0);
  float acc[RPS];
  tf_rowdot<K, GS, RPS>((const device uint8_t*)W + at * (K / 2), S + at * (K / GS), B + at * (K / GS),
                        X + size_t(p) * K, lane, acc);
  if (lane == 0)
    for (int j = 0; j < RPS; j++) Y[size_t(p) * N + row0 + j] = bfloat(acc[j]);
