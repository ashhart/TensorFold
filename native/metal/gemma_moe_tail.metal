
  const uint t = thread_position_in_threadgroup.x;
  const uint r = threadgroup_position_in_grid.x;
  constexpr int PER = D / T;
  threadgroup float p1[T / 32], p2[T / 32], p3[T / 32], p4[T / 32];
  // every load before the first reduction, as in attn_tail
  float y1v[PER], h2v[PER], hin[PER], w1[PER], w2[PER], wp[PER], wn[PER];
  for (int i = 0; i < PER; i++) {
    const int c = int(t) + i * T;
    y1v[i] = float(Y1[int(r) * D + c]); h2v[i] = float(Y2[int(r) * D + c]); hin[i] = float(H[int(r) * D + c]);
    w1[i] = float(W1[c]); w2[i] = float(W2[c]); wp[i] = float(WP[c]); wn[i] = float(WN[c]);
  }
  const float sc = float(SC[0]);
  float ss1 = 0.0f, ss2 = 0.0f;
  for (int i = 0; i < PER; i++) {
    ss1 = fma(y1v[i], y1v[i], ss1);
    ss2 = fma(h2v[i], h2v[i], ss2);
  }
  float total1, total2;
  
  {
    float s = simd_sum(ss1);
    if (thread_index_in_simdgroup == 0) p1[simdgroup_index_in_threadgroup] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total1 = 0.0f;
    for (int q = 0; q < T / 32; q++) total1 += p1[q];
  }

  
  {
    float s = simd_sum(ss2);
    if (thread_index_in_simdgroup == 0) p2[simdgroup_index_in_threadgroup] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total2 = 0.0f;
    for (int q = 0; q < T / 32; q++) total2 += p2[q];
  }

  const float inv1 = metal::precise::rsqrt(total1 / float(D) + eps[0]);
  const float inv2 = metal::precise::rsqrt(total2 / float(D) + eps[0]);
  float sv[PER];
  float ss3 = 0.0f;
  for (int i = 0; i < PER; i++) {
    const float a1 = float(bfloat(w1[i] * float(bfloat(y1v[i] * inv1))));
    const float a2 = float(bfloat(w2[i] * float(bfloat(h2v[i] * inv2))));
    sv[i] = float(bfloat(a1 + a2));
    ss3 = fma(sv[i], sv[i], ss3);
  }
  float total3;
  
  {
    float s = simd_sum(ss3);
    if (thread_index_in_simdgroup == 0) p3[simdgroup_index_in_threadgroup] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total3 = 0.0f;
    for (int q = 0; q < T / 32; q++) total3 += p3[q];
  }

  const float inv3 = metal::precise::rsqrt(total3 / float(D) + eps[0]);
  float hv[PER];
  float ss4 = 0.0f;
  for (int i = 0; i < PER; i++) {
    const int c = int(t) + i * T;
    const float b = float(bfloat(wp[i] * float(bfloat(sv[i] * inv3))));
    const bfloat hs = bfloat(hin[i] + b);
    const bfloat hn = bfloat(float(hs) * sc);
    HN[int(r) * D + c] = hn;
    hv[i] = float(hn);
    ss4 = fma(hv[i], hv[i], ss4);
  }
  float total4;
  
  {
    float s = simd_sum(ss4);
    if (thread_index_in_simdgroup == 0) p4[simdgroup_index_in_threadgroup] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total4 = 0.0f;
    for (int q = 0; q < T / 32; q++) total4 += p4[q];
  }

  const float inv4 = metal::precise::rsqrt(total4 / float(D) + eps[0]);
  for (int i = 0; i < PER; i++) {
    const int c = int(t) + i * T;
    NEXT[int(r) * D + c] = bfloat(wn[i] * float(bfloat(hv[i] * inv4)));
  }
