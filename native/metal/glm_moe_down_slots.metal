
  // Simdgroup m: down rows for pick m of expert u (MAXU: the shared expert) by the one-row qmv_fast loop
  const uint lane = thread_index_in_simdgroup;
  const int m = int(simdgroup_index_in_threadgroup);
  if (PART == 1 && SB != 4) {
    // the shared expert's down projection alone at SB bits: row r = m, MLX's one-row qmv_fast loop
    constexpr int SBLK = 32 * SV;
    constexpr int SKB = K * SB / 8;
    constexpr int KG1 = K / 64;
    constexpr int SDIV = 64 / SV;
    constexpr int SSTEP = SBLK / 64;
    constexpr int WSTEP = 32 * SLB;
    const int R1 = int(ACT_shape[0]);
    if (m >= R1) return;
    const int r = m;
    const int row0 = int(threadgroup_position_in_grid.y) * RPS;
    const device uint8_t* w = (const device uint8_t*)SDW + size_t(row0) * SKB + lane * SLB;
    const device bfloat* sc = SDS + size_t(row0) * KG1 + lane / SDIV;
    const device bfloat* bi = SDB + size_t(row0) * KG1 + lane / SDIV;
    const device bfloat* x = ACT + size_t(r) * K + lane * SV;
    float acc[RPS];
    for (int j = 0; j < RPS; j++) acc[j] = 0.0f;
    for (int k0 = 0; k0 < K; k0 += SBLK) {
      float xt[SV];
      const float sum = loadv<SB, SV>(x, xt);
      for (int j = 0; j < RPS; j++) acc[j] += qdotv<SB, SV>(w + j * SKB, xt, float(sc[j * KG1]), float(bi[j * KG1]), sum);
      w += WSTEP; sc += SSTEP; bi += SSTEP; x += SBLK;
    }
    for (int j = 0; j < RPS; j++) {
      const float v = simd_sum(acc[j]);
      if (lane == 0) Y[size_t(r) * N + row0 + j] = bfloat(v);
    }
    return;
  }
  const int u = PART == 1 ? MAXU : int(threadgroup_position_in_grid.z);
  const int R = int(ACT_shape[0]);
  constexpr int SLOTS = PART == 0 ? TOPK + 1 : (PART == 1 ? 1 : TOPK);
  const bool shared = u == MAXU;
  if (!shared && u >= UCOUNT[0]) return;
  const int pick = shared ? (m < R ? m * TOPK : -1) : UMEM[u * MAXR + m];
  if (pick < 0) return;
  const int r = pick / TOPK, slot = shared ? (PART == 1 ? 0 : TOPK) : pick % TOPK;
  const int row0 = int(threadgroup_position_in_grid.y) * RPS;
  constexpr int KB = K / 2;
  constexpr int KG = K / 64;
  const size_t at = shared ? size_t(row0) : size_t(SLOTOF[LAYER[0] * NE + UIDS[u]]) * N + row0;
  const device uint8_t* w = (const device uint8_t*)(shared ? SDW : DW) + at * KB + lane * 8;
  const device bfloat* sc = (shared ? SDS : DS) + at * KG + lane / 4;
  const device bfloat* bi = (shared ? SDB : DB) + at * KG + lane / 4;
  const device bfloat* x = ACT + (size_t(r) * SLOTS + slot) * K + lane * 16;
  float acc[RPS];
  for (int j = 0; j < RPS; j++) acc[j] = 0.0f;
  for (int k0 = 0; k0 < K; k0 += 512) {
    float xt[16];
    const float sum = load16(x, xt);
    for (int j = 0; j < RPS; j++) acc[j] += qdot16(w + j * KB, xt, float(sc[j * KG]), float(bi[j * KG]), sum);
    w += 256; sc += 8; bi += 8; x += 512;
  }
  for (int j = 0; j < RPS; j++) {
    const float v = simd_sum(acc[j]);
    if (lane == 0) Y[(size_t(r) * SLOTS + slot) * N + row0 + j] = bfloat(v);
  }
