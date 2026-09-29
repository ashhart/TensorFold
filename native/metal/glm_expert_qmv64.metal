
  // Simdgroup m runs pick m of expert u with the one-row gather qmv_fast loop; an expert's picks share reads
  const uint lane = thread_index_in_simdgroup;
  const int m = int(simdgroup_index_in_threadgroup);
  const int u = int(threadgroup_position_in_grid.z);
  if (u >= UCOUNT[0]) return;
  const int pick = UMEM[u * MAXR + m];
  if (pick < 0) return;
  const size_t e = size_t(UIDS[u]);
  const int row0 = int(threadgroup_position_in_grid.y) * RPS;
  constexpr int KB = K / 2;
  constexpr int KG = K / 64;
  const device uint8_t* w = (const device uint8_t*)W + (e * N + row0) * KB + lane * 8;
  const device bfloat* sc = S + (e * N + row0) * KG + lane / 4;
  const device bfloat* bi = B + (e * N + row0) * KG + lane / 4;
  const device bfloat* x = X + size_t(PER_PICK ? pick : pick / TOPK) * K + lane * 16;
  float acc[RPS];
  for (int j = 0; j < RPS; j++) acc[j] = 0.0f;
  for (int k0 = 0; k0 < K; k0 += 512) {
    float xt[16];
    const float sum = load16(x, xt);
    for (int j = 0; j < RPS; j++)
      acc[j] += qdot16(w + j * KB, xt, float(sc[j * KG]), float(bi[j * KG]), sum);
    w += 256; sc += 8; bi += 8; x += 512;
  }
  for (int j = 0; j < RPS; j++) {
    const float v = simd_sum(acc[j]);
    if (lane == 0) OUT[size_t(pick) * N + row0 + j] = bfloat(v);
  }
