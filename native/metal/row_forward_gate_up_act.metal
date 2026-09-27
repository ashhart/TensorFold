
  // threadgroup b has SG simdgroups of RPS output rows each; a group of GS inputs spans GS / 16 lanes. Each
  // 512-input step loads the RPS weight rows' words once, then takes the R input rows in turn (one row's 16
  // inputs live at a time: holding all R rows' inputs spilled registers from 3 rows on an M3 Ultra; converting the
  // nibbles once for all rows, kept as halves, gave the same bits but ran slower).
  const uint lane = thread_index_in_simdgroup;
  const uint g = simdgroup_index_in_threadgroup;
  const int row0 = int(g) * NH + int(threadgroup_position_in_grid.y) * RPS;
  constexpr int KB = K / 2;
  constexpr int KG = K / GS;
  const device uint8_t* w = (const device uint8_t*)W + size_t(row0) * KB + lane * 8;
  const device bfloat* sc = S + size_t(row0) * KG + lane / (GS / 16);
  const device bfloat* bi = B + size_t(row0) * KG + lane / (GS / 16);
  float acc[R][RPS];
  for (int r = 0; r < R; r++) for (int j = 0; j < RPS; j++) acc[r][j] = 0.0f;
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
      const float sum = load16(X + r * K + k0 + lane * 16, xt);
      for (int j = 0; j < RPS; j++) acc[r][j] += qdot16w(ws[j], xt, s[j], b[j], sum);
    }
    w += 256; sc += 512 / GS; bi += 512 / GS;
  }
  threadgroup float ups[R][RPS];
  for (int r = 0; r < R; r++)
    for (int j = 0; j < RPS; j++) {
      const float v = simd_sum(acc[r][j]);
      if (lane == 0 && g == 1) ups[r][j] = float(bfloat(v));
      acc[r][j] = v;
    }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (g == 0 && lane == 0)
    for (int r = 0; r < R; r++)
      for (int j = 0; j < RPS; j++) {
        const float gf = float(bfloat(acc[r][j]));
        OUT[r * NH + row0 + j] = bfloat(gf / (1.0f + metal::exp(-gf)) * ups[r][j]);
      }
