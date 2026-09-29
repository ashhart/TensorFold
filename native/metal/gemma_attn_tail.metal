
  const uint t = thread_position_in_threadgroup.x;
  const uint r = threadgroup_position_in_grid.x;
  constexpr int PER = D / T;
  threadgroup float p1[T / 32], p2[T / 32];
  float ov[PER], hin[PER], wa[PER], w1[PER], w2[PER], w3[PER];
  for (int i = 0; i < PER; i++) {
    const int c = int(t) + i * T;
    ov[i] = float(O[int(r) * D + c]); hin[i] = float(H[int(r) * D + c]);
    wa[i] = float(WA[c]); w1[i] = float(W1[c]); w2[i] = float(W2[c]); w3[i] = float(W3[c]);
  }
  float ss = 0.0f;
  for (int i = 0; i < PER; i++) ss = fma(ov[i], ov[i], ss);
  float total1;
  
  {
    float s = simd_sum(ss);
    if (thread_index_in_simdgroup == 0) p1[simdgroup_index_in_threadgroup] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total1 = 0.0f;
    for (int q = 0; q < T / 32; q++) total1 += p1[q];
  }

  const float inv1 = metal::precise::rsqrt(total1 / float(D) + eps[0]);
  float hv[PER];
  float ss2 = 0.0f;
  for (int i = 0; i < PER; i++) {
    const int c = int(t) + i * T;
    const float a = float(bfloat(wa[i] * float(bfloat(ov[i] * inv1))));
    const bfloat hn = bfloat(hin[i] + a);
    HN[int(r) * D + c] = hn;
    hv[i] = float(hn);
    ss2 = fma(hv[i], hv[i], ss2);
  }
  float total2;
  
  {
    float s = simd_sum(ss2);
    if (thread_index_in_simdgroup == 0) p2[simdgroup_index_in_threadgroup] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total2 = 0.0f;
    for (int q = 0; q < T / 32; q++) total2 += p2[q];
  }

  const float inv2 = metal::precise::rsqrt(total2 / float(D) + eps[0]);
  for (int i = 0; i < PER; i++) {
    const int c = int(t) + i * T;
    const float n = float(bfloat(hv[i] * inv2));
    N1[int(r) * D + c] = bfloat(w1[i] * n);
    N2[int(r) * D + c] = bfloat(w2[i] * n);
    N3[int(r) * D + c] = bfloat(w3[i] * n);
  }
