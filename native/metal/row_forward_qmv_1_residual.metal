
  // threadgroup b has SG simdgroups of RPS output rows each; a group of GS inputs spans GS / 16 lanes. Each
  // 512-input step loads the RPS weight rows' words once, then takes the R input rows in turn (one row's 16
  // inputs live at a time: holding all R rows' inputs spilled registers from 3 rows on an M3 Ultra; converting the
  // nibbles once for all rows, kept as halves, gave the same bits but ran slower).
  const uint lane = thread_index_in_simdgroup;
  const uint g = simdgroup_index_in_threadgroup;
  const int row0 = int(threadgroup_position_in_grid.y) * (SG * RPS) + int(g) * RPS;
  constexpr int KB = K / 2;
  constexpr int KG = K / GS;
  const device uint8_t* w = (const device uint8_t*)W + size_t(row0) * KB + lane * 8;
  const device bfloat* sc = S + size_t(row0) * KG + lane / (GS / 16);
  const device bfloat* bi = B + size_t(row0) * KG + lane / (GS / 16);
  float acc[R][RPS];
  for (int r = 0; r < R; r++) for (int j = 0; j < RPS; j++) acc[r][j] = 0.0f;
  // each input row's RMSNorm scale from the T partial sums of squares its producer wrote, added in a fixed order
  float inv[R];
  for (int r = 0; r < R; r++) {
    float part_sum = 0.0f;
    for (int t = int(lane); t < T; t += 32) part_sum += PART[r * T + t];
    inv[r] = metal::rsqrt(simd_sum(part_sum) / float(K) + eps[0]);
  }
  if (R == 1) {
    for (int k0 = 0; k0 < K; k0 += 512) {
      float xt[16];
      const float sum = load16n(X + k0 + lane * 16, NW + k0 + lane * 16, inv[0], xt);
      for (int j = 0; j < RPS; j++) {
        const device uint16_t* wp = (const device uint16_t*)(w + j * KB);
        uint16_t ws1[4];
        for (int i = 0; i < 4; i++) ws1[i] = wp[i];
        acc[0][j] += qdot16w(ws1, xt, float(sc[j * KG]), float(bi[j * KG]), sum);
      }
      w += 256; sc += 512 / GS; bi += 512 / GS;
    }
  } else {
  for (int k0 = 0; k0 < K; k0 += 512) {
    uint16_t ws[RPS][4];
    float s[RPS], b[RPS];
    for (int j = 0; j < RPS; j++) {
      const device uint16_t* wp = (const device uint16_t*)(w + j * KB);
      for (int i = 0; i < 4; i++) ws[j][i] = wp[i];
      s[j] = float(sc[j * KG]);
      b[j] = float(bi[j * KG]);
    }
    for (int r = 0; r < R; r++) {
      float xt[16];
      const float sum = load16n(X + r * K + k0 + lane * 16, NW + k0 + lane * 16, inv[r], xt);
      for (int j = 0; j < RPS; j++) acc[r][j] += qdot16w(ws[j], xt, s[j], b[j], sum);
    }
    w += 256; sc += 512 / GS; bi += 512 / GS;
  }
  }
  // h = res + y (bf16, as mlx_lm's residual add) and the partial sum of squares of this threadgroup's outputs
  threadgroup float part[SG][R];
  for (int r = 0; r < R; r++) {
    float ss = 0.0f;
    for (int j = 0; j < RPS; j++) {
      const float v = simd_sum(acc[r][j]);
      const bfloat h = bfloat(float(RES[r * N + row0 + j]) + float(bfloat(v)));
      if (lane == 0) OUT[r * N + row0 + j] = h;
      ss = fma(float(h), float(h), ss);
    }
    if (lane == 0) part[g][r] = ss;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (g == 0 && lane == 0)
    for (int r = 0; r < R; r++) {
      float total = 0.0f;
      for (int s2 = 0; s2 < SG; s2++) total += part[s2][r];
      PO[r * (N / (SG * RPS)) + int(threadgroup_position_in_grid.y)] = total;
    }
