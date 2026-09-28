
  // residual add + RMSNorm of row m: one threadgroup of K / 16 threads, thread t holds [16 t, 16 t + 16); the
  // row's sum of squares is each thread's sequential fma over its 16, then simd_sum, then the simdgroups in order
  const uint t = thread_position_in_threadgroup.x;
  const uint m = threadgroup_position_in_grid.y;
  constexpr int E = 16;
  constexpr int TPG = K / E;
  threadgroup float red[TPG / 32];
  const int base = int(m) * K + int(t) * E;
  float hv[E];
  float ss = 0.0f;
  for (int i = 0; i < E; i++) {
    bfloat h = H[base + i];
    h = bfloat(float(h) + float(R[base + i]));
    HO[base + i] = h;
    hv[i] = float(h);
    ss = fma(hv[i], hv[i], ss);
  }
  ss = simd_sum(ss);
  if (thread_index_in_simdgroup == 0) red[simdgroup_index_in_threadgroup] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (int i = 0; i < TPG / 32; i++) total += red[i];
  const float inv = metal::precise::rsqrt(total / float(K) + eps[0]);
  for (int i = 0; i < E; i++) XO[base + i] = bfloat(float(Wt[int(t) * E + i]) * (hv[i] * inv));
