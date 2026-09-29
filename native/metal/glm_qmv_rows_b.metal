
  // _QMV_ROWS at BITS (5, 6, 8): simdgroup r takes input row r through MLX's one-row qmv_fast loop at that width
  const uint lane = thread_index_in_simdgroup;
  const int r = int(simdgroup_index_in_threadgroup);
  const int row0 = int(threadgroup_position_in_grid.y) * RPS;
  constexpr int BLK = 32 * V;
  constexpr int KB = K * BITS / 8;
  constexpr int KG = K / 64;
  constexpr int SDIV = 64 / V;
  constexpr int SSTEP = BLK / 64;
  constexpr int WSTEP = 32 * LB;
  const device uint8_t* w = (const device uint8_t*)W + size_t(row0) * KB + lane * LB;
  const device bfloat* sc = S + size_t(row0) * KG + lane / SDIV;
  const device bfloat* bi = B + size_t(row0) * KG + lane / SDIV;
  const device bfloat* x = X + r * K + lane * V;
  float acc[RPS];
  for (int j = 0; j < RPS; j++) acc[j] = 0.0f;
  for (int k0 = 0; k0 < K; k0 += BLK) {
    float xt[V];
    const float sum = loadv<BITS, V>(x, xt);
    for (int j = 0; j < RPS; j++)
      acc[j] += qdotv<BITS, V>(w + j * KB, xt, float(sc[j * KG]), float(bi[j * KG]), sum);
    w += WSTEP; sc += SSTEP; bi += SSTEP; x += BLK;
  }
  for (int j = 0; j < RPS; j++) {
    const float v = simd_sum(acc[j]);
    if (lane == 0) OUT[r * N + row0 + j] = bfloat(v);
  }
