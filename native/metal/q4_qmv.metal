
  // MLX's qmv_fast inner loop: threadgroup b has SG simdgroups of RPS output rows each; lane l reads 16 inputs of
  // each 512-input step. The R input rows are taken in order inside, each with the same sums at any R (and MLX's
  // one-row sums). (Converting a word's nibbles once for all rows kept the bits but was slower at 4 rows on the
  // M3 Ultra, 124 -> 165 us, register pressure.)
  const uint lane = thread_index_in_simdgroup;
  const uint g = simdgroup_index_in_threadgroup;
  const int row0 = int(threadgroup_position_in_grid.y) * (SG * RPS) + int(g) * RPS;
  constexpr int KB = K / 2;
  constexpr int KG = K / 32;
  const device uint8_t* w = (const device uint8_t*)W + size_t(row0) * KB + lane * 8;
  const device bfloat* sc = S + size_t(row0) * KG + lane / 2;
  const device bfloat* bi = B + size_t(row0) * KG + lane / 2;
  float acc[R][RPS];
  for (int r = 0; r < R; r++) for (int j = 0; j < RPS; j++) acc[r][j] = 0.0f;
  for (int k0 = 0; k0 < K; k0 += 512) {
    float xt[R][16], sum[R];
    for (int r = 0; r < R; r++) sum[r] = load16(X + r * K + k0 + lane * 16, xt[r]);
    for (int j = 0; j < RPS; j++) {
      uint16_t ws[4];
      const device uint16_t* wp = (const device uint16_t*)(w + j * KB);
      for (int i = 0; i < 4; i++) ws[i] = wp[i];
      const float s = float(sc[j * KG]), b = float(bi[j * KG]);
      for (int r = 0; r < R; r++) acc[r][j] += qdot16w(ws, xt[r], s, b, sum[r]);
    }
    w += 256; sc += 16; bi += 16;
  }
  for (int r = 0; r < R; r++)
    for (int j = 0; j < RPS; j++) {
      const float v = simd_sum(acc[r][j]);
      if (lane == 0) OUT[r * N + row0 + j] = bfloat(v);
    }
