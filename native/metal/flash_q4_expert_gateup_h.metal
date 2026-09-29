
  // Threadgroup (b, p): 2 simdgroups, rows 8 b + 4 g .. + 3 of slot p (p = r * SLOTS + k; slot k < TOPK is the
  // row's k-th expert by router logit, each simdgroup finding it itself), gate and up,
  // over K in steps of 512 (MLX's qmv_fast loop); then bf16(SiLU(bf16(gate)) * bf16(up)).
  // A slot past TOPK (SHARED = 1) is the shared expert, from its own matrices. Threadgroup (0, p)'s first
  // simdgroup also writes the slot's expert to PICK and, for the last routed slot (whose selection rounds give the
  // top-k logits), the weights exp(l_k - l_0) / their sum (fp32, bf16-rounded) to WTS: expert_down reads them.
  const uint lane = thread_index_in_simdgroup;
  const uint g = simdgroup_index_in_threadgroup;
  const int p = int(threadgroup_position_in_grid.z);
  constexpr int SLOTS = TOPK + SHARED;
  const int r = p / SLOTS, slot = p % SLOTS;
  const bool shared = slot == TOPK;
  float picked[TOPK];
  const size_t e = shared ? 0 : size_t(simd_topk<NE>(LOGITS + r * NL, slot, lane, picked));
  if (!shared && threadgroup_position_in_grid.y == 0 && g == 0 && lane == 0) {
    PICK[r * TOPK + slot] = uint32_t(e);
    if (slot == TOPK - 1) {
      float total = 0.0f;
      float ex[TOPK];
      for (int kk = 0; kk < TOPK; kk++) { ex[kk] = metal::exp(picked[kk] - picked[0]); total += ex[kk]; }
      for (int kk = 0; kk < TOPK; kk++) WTS[r * TOPK + kk] = float(bfloat(ex[kk] / total));
    }
  }
  const int row0 = int(threadgroup_position_in_grid.y) * (SG * RPS) + int(g) * RPS;
  constexpr int KB = K / 2;                         // bytes a row
  constexpr int KG = K / 32;                        // groups a row
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
    const float sum = load16h(x, xt);
    for (int row = 0; row < RPS; row++) {
      ag[row] += qdot16h(gw + row * KB, xt, float(gs[row * KG]), float(gb[row * KG]), sum);
      au[row] += qdot16h(uw + row * KB, xt, float(us[row * KG]), float(ub[row * KG]), sum);
    }
    gw += 256; uw += 256; gs += 16; gb += 16; us += 16; ub += 16; x += 512;
  }
  for (int row = 0; row < RPS; row++) {
    const float gv = simd_sum(ag[row]), uv = simd_sum(au[row]);
    if (lane == 0) ACT[p * N + row0 + row] = bfloat(bsilu(float(bfloat(gv))) * float(bfloat(uv)));
  }
