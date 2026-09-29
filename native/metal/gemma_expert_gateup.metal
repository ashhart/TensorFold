
  const uint lane = thread_index_in_simdgroup;
  const uint g = simdgroup_index_in_threadgroup;
  const int p = int(threadgroup_position_in_grid.z);
  const int r = p / TOPK;
  const size_t e = size_t(IDX[p]);
  const int row0 = int(threadgroup_position_in_grid.y) * (SG * RPS) + int(g) * RPS;
  constexpr int KB = K / 2;
  constexpr int KG = K / GS;
  const device uint8_t* gw = (const device uint8_t*)GW + (e * N + row0) * KB + lane * 8;
  const device uint8_t* uw = (const device uint8_t*)UW + (e * N + row0) * KB + lane * 8;
  const device bfloat* gs = GSC + (e * N + row0) * KG + lane / (GS / 16);
  const device bfloat* gb = GBI + (e * N + row0) * KG + lane / (GS / 16);
  const device bfloat* us = USC + (e * N + row0) * KG + lane / (GS / 16);
  const device bfloat* ub = UBI + (e * N + row0) * KG + lane / (GS / 16);
  const device bfloat* x = X + r * K + lane * 16;
  float xt[16];
  float ag[RPS], au[RPS];
  for (int row = 0; row < RPS; row++) { ag[row] = 0.0f; au[row] = 0.0f; }
  for (int k0 = 0; k0 < K; k0 += 512) {
    if (k0 + int(lane) * 16 < K) {
      const float sum = load16(x, xt);
      for (int row = 0; row < RPS; row++) {
        ag[row] += qdot16(gw + row * KB, xt, float(gs[row * KG]), float(gb[row * KG]), sum);
        au[row] += qdot16(uw + row * KB, xt, float(us[row * KG]), float(ub[row * KG]), sum);
      }
    }
    gw += 256; uw += 256; gs += 512 / GS; gb += 512 / GS; us += 512 / GS; ub += 512 / GS; x += 512;
  }
  for (int row = 0; row < RPS; row++) {
    const float gv = simd_sum(ag[row]), uv = simd_sum(au[row]);
    if (lane == 0) ACT[p * N + row0 + row] = bfloat(gelu_tanh(float(bfloat(gv))) * float(bfloat(uv)));
  }
