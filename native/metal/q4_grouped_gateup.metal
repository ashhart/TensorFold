
  // Threadgroup (b, u): MAXM simdgroups, simdgroup m takes the m-th row that picked distinct expert u (u = MAXU:
  // the shared expert, every row, slot TOPK) and computes its gate and up rows RPS b .. RPS b + RPS - 1 with
  // expert_gateup's loop (its bits). The simdgroups read the same weight rows: memory serves them once.
  const uint lane = thread_index_in_simdgroup;
  const int m = int(simdgroup_index_in_threadgroup);
  const int u = int(threadgroup_position_in_grid.z);
  const int R = rows[0];
  constexpr int SLOTS = TOPK + 1;
  const bool shared = u == MAXU;
  if (!shared && u >= UCOUNT[0]) return;
  const int member = shared ? (m < R ? m * 32 + TOPK : -1) : UMEM[u * MAXR + m];
  if (member < 0) return;
  const size_t e = shared ? 0 : size_t(UIDS[u]);
  const int r = member / 32, slot = member % 32;
  const int row0 = int(threadgroup_position_in_grid.y) * RPS;
  constexpr int KB = K / 2;
  constexpr int KG = K / 32;
  const device uint32_t* GWp = shared ? SGW : GW;
  const device uint32_t* UWp = shared ? SUW : UW;
  const device bfloat* GSp = shared ? SGS : GS;
  const device bfloat* GBp = shared ? SGB : GB;
  const device bfloat* USp = shared ? SUS : US;
  const device bfloat* UBp = shared ? SUB : UB;
  const device uint8_t* gw = (const device uint8_t*)GWp + (e * N + row0) * KB + lane * 8;
  const device uint8_t* uw = (const device uint8_t*)UWp + (e * N + row0) * KB + lane * 8;
  const device bfloat* gs = GSp + (e * N + row0) * KG + lane / 2;
  const device bfloat* gb = GBp + (e * N + row0) * KG + lane / 2;
  const device bfloat* us = USp + (e * N + row0) * KG + lane / 2;
  const device bfloat* ub = UBp + (e * N + row0) * KG + lane / 2;
  const device bfloat* x = X + r * K + lane * 16;
  float xt[16];
  float ag[RPS], au[RPS];
  for (int row = 0; row < RPS; row++) { ag[row] = 0.0f; au[row] = 0.0f; }
  for (int k0 = 0; k0 < K; k0 += 512) {
    const float sum = load16(x, xt);
    for (int row = 0; row < RPS; row++) {
      ag[row] += qdot16(gw + row * KB, xt, float(gs[row * KG]), float(gb[row * KG]), sum);
      au[row] += qdot16(uw + row * KB, xt, float(us[row * KG]), float(ub[row * KG]), sum);
    }
    gw += 256; uw += 256; gs += 16; gb += 16; us += 16; ub += 16; x += 512;
  }
  for (int row = 0; row < RPS; row++) {
    const float gv = simd_sum(ag[row]), uv = simd_sum(au[row]);
    if (lane == 0) ACT[(r * SLOTS + slot) * N + row0 + row] = bfloat(bsilu(float(bfloat(gv))) * float(bfloat(uv)));
  }
