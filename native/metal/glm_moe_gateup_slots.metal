
  // Simdgroup m: gate and up for pick m of expert u (MAXU: the shared expert) by the one-row loop, then SwiGLU
  const uint lane = thread_index_in_simdgroup;
  const int m = int(simdgroup_index_in_threadgroup);
  if (PART == 1 && SB != 4) {
    // the shared expert alone at SB bits: row r = m by MLX's one-row qmv_fast loop at that width, then SwiGLU
    constexpr int SBLK = 32 * SV;
    constexpr int SKB = K * SB / 8;
    constexpr int KG1 = K / 64;
    constexpr int SDIV = 64 / SV;
    constexpr int SSTEP = SBLK / 64;
    constexpr int WSTEP = 32 * SLB;
    const int R = int(X_shape[0]);
    if (m >= R) return;
    const int r = m;
    const int row0 = int(threadgroup_position_in_grid.y) * RPS;
    const device uint8_t* gw = (const device uint8_t*)SGU + size_t(row0) * SKB + lane * SLB;
    const device uint8_t* uw = (const device uint8_t*)SGU + size_t(N + row0) * SKB + lane * SLB;
    const device bfloat* gs = SGUS + size_t(row0) * KG1 + lane / SDIV;
    const device bfloat* gb = SGUB + size_t(row0) * KG1 + lane / SDIV;
    const device bfloat* us = SGUS + size_t(N + row0) * KG1 + lane / SDIV;
    const device bfloat* ub = SGUB + size_t(N + row0) * KG1 + lane / SDIV;
    const device bfloat* x = X + size_t(r) * K + lane * SV;
    float ag[RPS], au[RPS];
    for (int j = 0; j < RPS; j++) { ag[j] = 0.0f; au[j] = 0.0f; }
    for (int k0 = 0; k0 < K; k0 += SBLK) {
      float xt[SV];
      const float sum = loadv<SB, SV>(x, xt);
      for (int j = 0; j < RPS; j++) {
        ag[j] += qdotv<SB, SV>(gw + j * SKB, xt, float(gs[j * KG1]), float(gb[j * KG1]), sum);
        au[j] += qdotv<SB, SV>(uw + j * SKB, xt, float(us[j * KG1]), float(ub[j * KG1]), sum);
      }
      gw += WSTEP; uw += WSTEP; gs += SSTEP; gb += SSTEP; us += SSTEP; ub += SSTEP; x += SBLK;
    }
    for (int j = 0; j < RPS; j++) {
      const float gv = simd_sum(ag[j]), uv = simd_sum(au[j]);
      if (lane == 0) {
        const float lim = float(bfloat(LIM[0]));
        const bfloat gt = bfloat(metal::min(float(bfloat(gv)), lim));
        const bfloat up = bfloat(metal::min(metal::max(float(bfloat(uv)), -lim), lim));
        const bfloat sl = gt * sigmoid_fast(gt);
        ACT[size_t(r) * N + row0 + j] = sl * up;
      }
    }
    return;
  }
  // PART 0: routed and shared; 1: the shared expert alone; 2: the routed experts alone (the same arithmetic)
  const int u = PART == 1 ? MAXU : int(threadgroup_position_in_grid.z);
  const int R = int(X_shape[0]);
  constexpr int SLOTS = PART == 0 ? TOPK + 1 : (PART == 1 ? 1 : TOPK);
  const bool shared = u == MAXU;
  if (!shared && u >= UCOUNT[0]) return;
  const int pick = shared ? (m < R ? m * TOPK : -1) : UMEM[u * MAXR + m];
  if (pick < 0) return;
  const int r = pick / TOPK, slot = shared ? (PART == 1 ? 0 : TOPK) : pick % TOPK;
  const int row0 = int(threadgroup_position_in_grid.y) * RPS;
  constexpr int KB = K / 2;
  constexpr int KG = K / 64;
  const size_t e = shared ? 0 : size_t(SLOTOF[LAYER[0] * NE + UIDS[u]]);
  // shared: one stacked matrix [gate (N) ; up (N)] rows
  const size_t grow = shared ? size_t(row0) : e * N + row0;
  const size_t urow = shared ? size_t(N + row0) : e * N + row0;
  const device uint8_t* gw = (const device uint8_t*)(shared ? SGU : GW) + grow * KB + lane * 8;
  const device uint8_t* uw = (const device uint8_t*)(shared ? SGU : UW) + urow * KB + lane * 8;
  const device bfloat* gs = (shared ? SGUS : GS) + grow * KG + lane / 4;
  const device bfloat* gb = (shared ? SGUB : GB) + grow * KG + lane / 4;
  const device bfloat* us = (shared ? SGUS : US) + urow * KG + lane / 4;
  const device bfloat* ub = (shared ? SGUB : UB) + urow * KG + lane / 4;
  const device bfloat* x = X + size_t(r) * K + lane * 16;
  float ag[RPS], au[RPS];
  for (int j = 0; j < RPS; j++) { ag[j] = 0.0f; au[j] = 0.0f; }
  for (int k0 = 0; k0 < K; k0 += 512) {
    float xt[16];
    const float sum = load16(x, xt);
    for (int j = 0; j < RPS; j++) {
      ag[j] += qdot16(gw + j * KB, xt, float(gs[j * KG]), float(gb[j * KG]), sum);
      au[j] += qdot16(uw + j * KB, xt, float(us[j * KG]), float(ub[j * KG]), sum);
    }
    gw += 256; uw += 256; gs += 8; gb += 8; us += 8; ub += 8; x += 512;
  }
  for (int j = 0; j < RPS; j++) {
    const float gv = simd_sum(ag[j]), uv = simd_sum(au[j]);
    if (lane == 0) {
      // minimum / clip on bf16 return one of their inputs: exact in float
      const float lim = float(bfloat(LIM[0]));
      const bfloat gt = bfloat(metal::min(float(bfloat(gv)), lim));
      const bfloat up = bfloat(metal::min(metal::max(float(bfloat(uv)), -lim), lim));
      const bfloat sl = gt * sigmoid_fast(gt);
      ACT[(size_t(r) * SLOTS + slot) * N + row0 + j] = sl * up;
    }
  }
