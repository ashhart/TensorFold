
  const uint lane = thread_index_in_simdgroup;
  const int r = int(threadgroup_position_in_grid.y);
  const int row0 = (int(threadgroup_position_in_grid.x) * SG + int(simdgroup_index_in_threadgroup)) * RPS;
  constexpr int KG = K / GS;
  const device uint8_t* w = (const device uint8_t*)W + size_t(row0) * K + lane * 8;
  const device bfloat* sc = S + size_t(row0) * KG + (lane * 8) / GS;
  const device bfloat* bi = B + size_t(row0) * KG + (lane * 8) / GS;
  const device bfloat* x = X + size_t(r) * K + lane * 8;
  float acc[RPS];
  for (int j = 0; j < RPS; j++) acc[j] = 0.0f;
  for (int k0 = 0; k0 < K; k0 += 256) {
    float xt[8];
    float sum = 0.0f;
    for (int i = 0; i < 8; i++) { xt[i] = float(x[i]); sum += xt[i]; }
    for (int j = 0; j < RPS; j++) {
      float a = 0.0f;
      for (int i = 0; i < 8; i++) a = fma(xt[i], float(w[j * K + i]), a);
      acc[j] += float(sc[j * KG]) * a + float(bi[j * KG]) * sum;
    }
    w += 256; sc += 256 / GS; bi += 256 / GS; x += 256;
  }
  for (int j = 0; j < RPS; j++) {
    const float total = simd_sum(acc[j]);
    if (lane == 0) OUT[size_t(r) * N + row0 + j] = bfloat(total);
  }
