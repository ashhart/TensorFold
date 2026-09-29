
  const uint lane = thread_index_in_simdgroup;
  const int k = int(simdgroup_index_in_threadgroup);
  const int r = int(threadgroup_position_in_grid.z);
  const int d0 = int(threadgroup_position_in_grid.y) * 8;
  constexpr int KB = NI / 2;
  constexpr int KG = NI / GS;
  constexpr int NC = NI / 16;
  threadgroup float ys[TOPK][8];
  const size_t e = size_t(IDX[r * TOPK + k]);
  const device bfloat* x = ACT + (r * TOPK + k) * NI;
  float xa[16], xb[16];
  const float sa = load16(x + lane * 16, xa);
  const bool second = int(lane) < NC - 32;
  const float sb = second ? load16(x + (32 + lane) * 16, xb) : 0.0f;
  // the 8 rows' loads before the first simd_sum, so their reads are in flight together
  float acc[8];
  #pragma unroll
  for (int row = 0; row < 8; row++) {
    const size_t at = e * D + d0 + row;
    const device uint8_t* w = (const device uint8_t*)DW + at * KB;
    acc[row] = qdot16(w + lane * 8, xa, float(DSC[at * KG + lane / (GS / 16)]),
                      float(DBI[at * KG + lane / (GS / 16)]), sa);
  }
  if (second) {
    #pragma unroll
    for (int row = 0; row < 8; row++) {
      const size_t at = e * D + d0 + row;
      const device uint8_t* w = (const device uint8_t*)DW + at * KB;
      acc[row] += qdot16(w + (32 + lane) * 8, xb, float(DSC[at * KG + (32 + lane) / (GS / 16)]),
                         float(DBI[at * KG + (32 + lane) / (GS / 16)]), sb);
    }
  }
  #pragma unroll
  for (int row = 0; row < 8; row++) {
    const float s = simd_sum(acc[row]);
    if (lane == 0) ys[k][row] = float(bfloat(s));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (k == 0 && lane < 8) {
    float routed = 0.0f;
    for (int kk = 0; kk < TOPK; kk++) routed += float(bfloat(ys[kk][lane] * float(WT[r * TOPK + kk])));
    OUT[r * D + d0 + int(lane)] = bfloat(routed);
  }
