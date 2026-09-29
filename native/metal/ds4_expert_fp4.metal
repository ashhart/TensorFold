
  // Simdgroup m runs pick m of expert u with the one-row fp_qmv_fast loop; an expert's picks share reads
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
  const device uint8_t* w = (const device uint8_t*)W + (e * N + row0) * KB + lane * 8;
  const device uint8_t* sc = S + (e * N + row0) * KG + lane / 2;
  const device bfloat* x = X + size_t(PER_PICK ? pick : pick / TOPK) * K + lane * 16;
  float acc[RPS];
  for (int j = 0; j < RPS; j++) acc[j] = 0.0f;
  for (int k0 = 0; k0 < K; k0 += 512) {
    float xt[16];
    for (int i = 0; i < 16; i++) xt[i] = float(x[i]);
    for (int j = 0; j < RPS; j++)
      acc[j] += fp4dot16(w + j * KB, xt, e8m0(sc[j * KG]));
    w += 256; sc += 16; x += 512;
  }
  for (int j = 0; j < RPS; j++) {
    const float v = simd_sum(acc[j]);
    if (lane == 0) OUT[size_t(pick) * N + row0 + j] = bfloat(v);
  }
