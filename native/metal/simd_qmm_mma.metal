  #define LOAD8(r, j) ((((const device uint4*)X)[size_t(r) * (K / 8) + (j)]))

  // R rows: threadgroup (x, y) takes rows 8 RT y .. 8 RT y + 8 RT - 1 in RT tiles of 8 (rows >= R read row R - 1;
  // their results are dropped) and S simdgroups, simdgroup c running chunk c for NT tiles of 8 outputs.
  const uint lane = thread_index_in_simdgroup;
  const int c = int(simdgroup_index_in_threadgroup);
  const int qid = int(lane) / 4;
  const int fm = (qid & 4) + ((int(lane) / 2) % 4);
  const int fn = (qid & 2) * 2 + (int(lane) % 2) * 2;
  const int R = X_shape[0];
  constexpr int G = K / 64;
  const float one = ONE[0];
  const int nb = int(threadgroup_position_in_grid.x) * (8 * NT);
  const int rb = int(threadgroup_position_in_grid.y) * (8 * RT);
  threadgroup float red[S > 1 ? S * RT * NT * 64 : 1];
  const device uint2* W2 = (const device uint2*)W;
  int wrow[NT];
  for (int t = 0; t < NT; t++) wrow[t] = min(nb + 8 * t + fm, N - 1);
  int xr0[RT], xr1[RT];
  for (int rt = 0; rt < RT; rt++) { xr0[rt] = min(rb + 8 * rt + fn, R - 1); xr1[rt] = min(rb + 8 * rt + fn + 1, R - 1); }
  float acc[RT][NT][2];
  for (int rt = 0; rt < RT; rt++)
    for (int t = 0; t < NT; t++) { acc[rt][t][0] = 0.0f; acc[rt][t][1] = 0.0f; }
  for (int g = c; g < G; g += S) {
    uint2 wv[NT];
    PRAGMA_UNROLL
    for (int t = 0; t < NT; t++) wv[t] = W2[size_t(wrow[t]) * (K / 16) + 4 * g + fn / 2];
    uint4 xa[RT], xb[RT];
    float xs0[RT], xs1[RT];
    PRAGMA_UNROLL
    for (int rt = 0; rt < RT; rt++) {
      xa[rt] = LOAD8(xr0[rt], 8 * g + fm);
      xb[rt] = LOAD8(xr1[rt], 8 * g + fm);
      float v = sum8(xa[rt], one), u = sum8(xb[rt], one);
      v = fma(simd_shuffle_xor(v, ushort(2)), one, v); u = fma(simd_shuffle_xor(u, ushort(2)), one, u);
      v = fma(simd_shuffle_xor(v, ushort(4)), one, v); u = fma(simd_shuffle_xor(u, ushort(4)), one, u);
      v = fma(simd_shuffle_xor(v, ushort(16)), one, v); u = fma(simd_shuffle_xor(u, ushort(16)), one, u);
      xs0[rt] = v; xs1[rt] = u;
    }
    simdgroup_matrix<float, 8, 8> P[RT][NT];
    PRAGMA_UNROLL
    for (int rt = 0; rt < RT; rt++)
      for (int t = 0; t < NT; t++) P[rt][t] = simdgroup_matrix<float, 8, 8>(0.0f);
    PRAGMA_UNROLL
    for (int s = 0; s < 8; s++) {
      const float ps = pre(s);
      const uint mask = 0xFu << (4 * s);
      simdgroup_matrix<float, 8, 8> bm[RT];
      PRAGMA_UNROLL
      for (int rt = 0; rt < RT; rt++) {
        bm[rt].thread_elements()[0] = bf8(xa[rt], s) * ps;
        bm[rt].thread_elements()[1] = bf8(xb[rt], s) * ps;
      }
      PRAGMA_UNROLL
      for (int t = 0; t < NT; t++) {
        simdgroup_matrix<float, 8, 8> am;
        am.thread_elements()[0] = float(wv[t].x & mask);
        am.thread_elements()[1] = float(wv[t].y & mask);
        PRAGMA_UNROLL
        for (int rt = 0; rt < RT; rt++) simdgroup_multiply_accumulate(P[rt][t], am, bm[rt], P[rt][t]);
      }
    }
    PRAGMA_UNROLL
    for (int t = 0; t < NT; t++) {
      const float sc = float(SC[size_t(wrow[t]) * G + g]);
      const float bi = float(BI[size_t(wrow[t]) * G + g]);
      PRAGMA_UNROLL
      for (int rt = 0; rt < RT; rt++) {
        acc[rt][t][0] = fma(bi, xs0[rt], fma(sc, P[rt][t].thread_elements()[0], acc[rt][t][0]));
        acc[rt][t][1] = fma(bi, xs1[rt], fma(sc, P[rt][t].thread_elements()[1], acc[rt][t][1]));
      }
    }
  }
  if (S == 1) {
    for (int rt = 0; rt < RT; rt++)
      for (int t = 0; t < NT; t++)
        for (int e = 0; e < 2; e++) {
          const int row = rb + 8 * rt + fn + e, n = nb + 8 * t + fm;
          if (row < R && n < N) OUT[size_t(row) * N + n] = bfloat(acc[rt][t][e]);
        }
    return;
  }
  for (int rt = 0; rt < RT; rt++)
    for (int t = 0; t < NT; t++)
      for (int e = 0; e < 2; e++) red[((c * RT + rt) * NT + t) * 64 + int(lane) * 2 + e] = acc[rt][t][e];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int idx = c * 32 + int(lane); idx < RT * NT * 64; idx += S * 32) {
    float v[S];
    for (int k = 0; k < S; k++) v[k] = red[k * (RT * NT * 64) + idx];
    for (int w = 1; w < S; w *= 2)
      for (int k = 0; k + w < S; k += 2 * w) v[k] = fma(v[k + w], one, v[k]);
    const int rt = idx / (NT * 64), t = (idx / 64) % NT, l = (idx % 64) / 2, e = idx % 2;
    const int lq = l / 4;
    const int row = rb + 8 * rt + (lq & 2) * 2 + (l % 2) * 2 + e, n = nb + 8 * t + (lq & 4) + ((l / 2) % 4);
    if (row < R && n < N) OUT[size_t(row) * N + n] = bfloat(v[0]);
  }
  #undef LOAD8
