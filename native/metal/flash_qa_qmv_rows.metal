
  // qmv_rows for any width: simdgroup r runs the qmv_fast loop (VPT codes a lane a step) for input row r
  const uint lane = thread_index_in_simdgroup;
  const int r = int(simdgroup_index_in_threadgroup);
  const int row0 = int(threadgroup_position_in_grid.y) * RPS;
  constexpr int VPT = lane_values(BITS), RB = K * BITS / 8, KG = K / GS;
  const device uint8_t* w = (const device uint8_t*)W + size_t(row0) * RB;
  const device bfloat* x = X + r * K;
  float acc[RPS];
  for (int j = 0; j < RPS; j++) acc[j] = 0.0f;
  for (int v0 = int(lane) * VPT; v0 < K; v0 += 32 * VPT) {
    float xv[VPT], sum = 0.0f;
    for (int i = 0; i < VPT; i++) { xv[i] = float(x[v0 + i]); sum += xv[i]; }
    for (int j = 0; j < RPS; j++) {
      float q[VPT];
      lane_codes<BITS, VPT>(w + j * RB, v0, q);
      float d = 0.0f;
      for (int i = 0; i < VPT; i++) d = fma(q[i], xv[i], d);
      const size_t at = size_t(row0 + j) * KG + v0 / GS;
      acc[j] += fma(float(S[at]), d, float(B[at]) * sum);
    }
  }
  for (int j = 0; j < RPS; j++) {
    const float v = simd_sum(acc[j]);
    if (lane == 0) OUT[r * N + row0 + j] = bfloat(v);
  }
