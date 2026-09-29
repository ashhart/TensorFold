
  const uint lane = thread_index_in_simdgroup;
  const int m = int(simdgroup_index_in_threadgroup);
  const int u = int(threadgroup_position_in_grid.z);
  if (u >= UCOUNT[0]) return;
  const int pick = UMEM[u * MAXR + m];
  if (pick < 0) return;
  const size_t e = size_t(UIDS[u]);
  const int row0 = int(threadgroup_position_in_grid.y) * RPS;
  constexpr int KB = K / 2;
  constexpr int KG = K / 32;
  const device uint8_t* wg = (const device uint8_t*)GW + (e * N + row0) * KB + lane * 8;
  const device uint8_t* wu = (const device uint8_t*)UW + (e * N + row0) * KB + lane * 8;
  const device uint8_t* sg = GS + (e * N + row0) * KG + lane / 2;
  const device uint8_t* su = US + (e * N + row0) * KG + lane / 2;
  const device bfloat* x = X + size_t(pick / TOPK) * K + lane * 16;
  float ag[RPS], au[RPS];
  for (int j = 0; j < RPS; j++) { ag[j] = 0.0f; au[j] = 0.0f; }
  for (int k0 = 0; k0 < K; k0 += 512) {
    float xt[16];
    for (int i = 0; i < 16; i++) xt[i] = float(x[i]);
    for (int j = 0; j < RPS; j++) {
      ag[j] += fp4dot16(wg + j * KB, xt, e8m0(sg[j * KG]));
      au[j] += fp4dot16(wu + j * KB, xt, e8m0(su[j * KG]));
    }
    wg += 256; wu += 256; sg += 16; su += 16; x += 512;
  }
  const float wt = WTS[pick];
  const float lim = LIM[0];
  for (int j = 0; j < RPS; j++) {
    const float gs = simd_sum(ag[j]);
    const float us = simd_sum(au[j]);
    if (lane == 0) {
      float gb = float(bfloat(gs)), ub = float(bfloat(us));
      if (lim > 0.0f) { ub = metal::clamp(ub, -lim, lim); gb = metal::min(gb, lim); }
      ACT[size_t(pick) * N + row0 + j] = bfloat(wt * (gb / (1.0f + metal::precise::exp(-gb))) * ub);
    }
  }
