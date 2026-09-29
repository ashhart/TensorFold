
  const uint lane = thread_index_in_simdgroup;
  const uint n = threadgroup_position_in_grid.x * 8 + simdgroup_index_in_threadgroup;
  const uint r = threadgroup_position_in_grid.y;
  if (n >= uint(N)) return;
  const size_t xb = size_t(r) * K, wb = size_t(n) * K;
  float acc = 0.0f;
  for (int k = int(lane); k < K; k += 32) acc = fma(float(X[xb + k]), WT[wb + k], acc);
  acc = simd_sum(acc);
  if (lane == 0) OUT[size_t(r) * N + n] = bfloat(acc);
