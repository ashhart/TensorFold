  #define LOAD8(r, j) ((((const device uint4*)X)[size_t(r) * (K / 8) + (j)]))

  // R rows: threadgroup (x, y) takes rows 8 RT y .. in RT tiles of 8 (rows >= R read row R - 1, dropped); SGS
  // simdgroups compute the S chunks in order. A lane's A elements are codes 8 fn .. 8 fn + 15 of its group.
  const uint lane = thread_index_in_simdgroup;
  const int sg = int(simdgroup_index_in_threadgroup);
  const int qid = int(lane) / 4;
  const int fm = (qid & 4) + ((int(lane) / 2) % 4);
  const int fn = (qid & 2) * 2 + (int(lane) % 2) * 2;
  const int R = X_shape[0];
  constexpr int G = K / 64, SG = K / GS, WPR = K * B / 32, LW = B == 8 ? 4 : 3;
  const float one = ONE[0];
  const int nb = int(threadgroup_position_in_grid.x) * (8 * NT);
  const int rb = int(threadgroup_position_in_grid.y) * (8 * RT);
  threadgroup float red[S > 1 ? S * RT * NT * 64 : 1];
  int wrow[NT];
  for (int t = 0; t < NT; t++) wrow[t] = min(nb + 8 * t + fm, N - 1);
  int xr0[RT], xr1[RT];
  for (int rt = 0; rt < RT; rt++) { xr0[rt] = min(rb + 8 * rt + fn, R - 1); xr1[rt] = min(rb + 8 * rt + fn + 1, R - 1); }
  for (int c = sg; c < S; c += SGS) {
    float acc[RT][NT][2];
    for (int rt = 0; rt < RT; rt++)
      for (int t = 0; t < NT; t++) { acc[rt][t][0] = 0.0f; acc[rt][t][1] = 0.0f; }
    for (int g = c; g < G; g += S) {
      const int bit0 = 64 * B * g + 8 * B * fn;
      const bool half_word = (bit0 & 31) != 0;               // 5-bit lanes fn = 2, 6 start 16 bits into a word
      uint v[NT][4];
      PRAGMA_UNROLL
      for (int t = 0; t < NT; t++) {
        const device uint* wp = W + size_t(wrow[t]) * WPR + (bit0 >> 5);
        uint w[4];
        PRAGMA_UNROLL
        for (int i = 0; i < 4; i++) w[i] = i < LW ? wp[i] : 0u;
        v[t][0] = half_word ? (w[0] >> 16) | (w[1] << 16) : w[0];
        v[t][1] = half_word ? (w[1] >> 16) | (w[2] << 16) : w[1];
        v[t][2] = half_word ? (w[2] >> 16) : w[2];
        v[t][3] = w[3];
      }
      uint4 xa[RT], xb[RT];
      float xs0[RT], xs1[RT];
      PRAGMA_UNROLL
      for (int rt = 0; rt < RT; rt++) {
        xa[rt] = LOAD8(xr0[rt], 8 * g + fm);
        xb[rt] = LOAD8(xr1[rt], 8 * g + fm);
        float a = sum8(xa[rt], one), u = sum8(xb[rt], one);
        a = fma(simd_shuffle_xor(a, ushort(2)), one, a); u = fma(simd_shuffle_xor(u, ushort(2)), one, u);
        a = fma(simd_shuffle_xor(a, ushort(4)), one, a); u = fma(simd_shuffle_xor(u, ushort(4)), one, u);
        a = fma(simd_shuffle_xor(a, ushort(16)), one, a); u = fma(simd_shuffle_xor(u, ushort(16)), one, u);
        xs0[rt] = a; xs1[rt] = u;
      }
      simdgroup_matrix<float, 8, 8> P[RT][NT];
      PRAGMA_UNROLL
      for (int rt = 0; rt < RT; rt++)
        for (int t = 0; t < NT; t++) P[rt][t] = simdgroup_matrix<float, 8, 8>(0.0f);
      PRAGMA_UNROLL
      for (int s = 0; s < 8; s++) {
        simdgroup_matrix<float, 8, 8> bm[RT];
        PRAGMA_UNROLL
        for (int rt = 0; rt < RT; rt++) {
          bm[rt].thread_elements()[0] = bf8(xa[rt], s);
          bm[rt].thread_elements()[1] = bf8(xb[rt], s);
        }
        PRAGMA_UNROLL
        for (int t = 0; t < NT; t++) {
          simdgroup_matrix<float, 8, 8> am;
          am.thread_elements()[0] = cf(code_at<B>(v[t], s));
          am.thread_elements()[1] = cf(code_at<B>(v[t], 8 + s));
          PRAGMA_UNROLL
          for (int rt = 0; rt < RT; rt++) simdgroup_multiply_accumulate(P[rt][t], am, bm[rt], P[rt][t]);
        }
      }
      PRAGMA_UNROLL
      for (int t = 0; t < NT; t++) {
        const int si = GS == 128 ? (g >> 1) : g;          // one scale covers two groups of 64
        const float sc = float(SC[size_t(wrow[t]) * SG + si]);
        const float bi = float(BI[size_t(wrow[t]) * SG + si]);
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
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int idx = sg * 32 + int(lane); idx < RT * NT * 64; idx += SGS * 32) {
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
