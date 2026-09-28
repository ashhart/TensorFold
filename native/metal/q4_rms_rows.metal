
  // One threadgroup of 1024 threads per row (per group of G features when G < W): bf16((x * rinv) * scale), the
  // sum of squares in fp32 (each thread's features in order, then the simdgroups in order).
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const int r = int(threadgroup_position_in_grid.y);
  const int grp = int(threadgroup_position_in_grid.x);
  const size_t base = size_t(r) * W + size_t(grp) * G;
  threadgroup float part[32];
  float ss = 0.0f;
  for (int i = int(t); i < G; i += 1024) { const float v = float(X[base + i]); ss = fma(v, v, ss); }
  ss = simd_sum(ss);
  if (lane == 0) part[sg] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (int k = 0; k < 32; k++) total += part[k];
  const float rinv = metal::precise::rsqrt(total / float(G) + eps[0]);
  for (int i = int(t); i < G; i += 1024)
    OUT[base + i] = bfloat((float(X[base + i]) * rinv) * SCALE[(grp * G + i) % SW]);
