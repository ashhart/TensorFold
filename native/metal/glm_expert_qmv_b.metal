
  // _EXPERT_QMV at BITS (5, 6, 8): pick m of expert u through the one-row loop affine_gather_qmv_fast runs
  const uint lane = thread_index_in_simdgroup;
  const int m = int(simdgroup_index_in_threadgroup);
  const int u = int(threadgroup_position_in_grid.z);
  if (u >= UCOUNT[0]) return;
  const int pick = UMEM[u * MAXR + m];
  if (pick < 0) return;
  const size_t e = size_t(UIDS[u]);
  const int row0 = int(threadgroup_position_in_grid.y) * RPS;
  constexpr int BLK = 32 * V;
  constexpr int KB = K * BITS / 8;
  constexpr int KG = K / 64;
  constexpr int SDIV = 64 / V;
  constexpr int SSTEP = BLK / 64;
  constexpr int WSTEP = 32 * LB;
  const device uint8_t* w = (const device uint8_t*)W + (e * N + row0) * KB + lane * LB;
  const device bfloat* sc = S + (e * N + row0) * KG + lane / SDIV;
  const device bfloat* bi = B + (e * N + row0) * KG + lane / SDIV;
  const device bfloat* x = X + size_t(PER_PICK ? pick : pick / TOPK) * K + lane * V;
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
    if (lane == 0) OUT[size_t(pick) * N + row0 + j] = bfloat(v);
  }
