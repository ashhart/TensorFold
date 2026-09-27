
  // one threadgroup of K / 16 threads per row: thread t holds elements [16 t, 16 t + 16) in registers.
  // The row's sum of squares is each thread's sequential fma over its 16, then simd_sum, then the
  // simdgroups' sums in order; a 64-group's input sum (for the next lane matmul) is ((g0 + g1) + (g2 + g3))
  // over its 4 threads' sequential sums. Arithmetic since 2026-09-24 (it was 256 strided threads, two
  // barriers and 64-long sums read back from threadgroup memory): 11.5 -> 7 us a call at 16 rows.
  const uint t = thread_position_in_threadgroup.x;
  const uint m = threadgroup_position_in_grid.y;
  const int M = dims[0], MP = dims[1];
  constexpr int E = 16;
  constexpr int TPG = K / E;
  threadgroup float red[TPG / 32];
  if (int(m) >= M) {
    if ((t & 3) == 0) XS[(t >> 2) * MP + m] = 0.0f;
    return;
  }
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
  const float inv = metal::rsqrt(total / float(K) + eps[0]);
  float gs = 0.0f;
  for (int i = 0; i < E; i++) {
    const bfloat x = bfloat(float(Wt[int(t) * E + i]) * (hv[i] * inv));
    XO[base + i] = x;
    gs += float(x);
  }
  gs += simd_shuffle_xor(gs, 1);
  gs += simd_shuffle_xor(gs, 2);
  if ((t & 3) == 0) XS[(t >> 2) * MP + m] = gs;
