
  // A simdgroup: OPS outputs of up to RT rows. Lane l takes 32-code blocks l, l + 32, ... of each output: it reads the
  // block's BITS words once, dequantizes each code once (fma(scale, q, bias)) and folds it into every row's own fma
  // chain, then one simd_sum. A row's bits never depend on OPS, RT or the rows beside it.
  constexpr int BLOCKS = K / 32, GROUPS = K / GS, BPG = GS / 32, WPR = K * BITS / 32, STEPS = (BLOCKS + 31) / 32;
  const int lane = int(thread_index_in_simdgroup);
  const int n0 = (int(threadgroup_position_in_grid.x) * SG + int(simdgroup_index_in_threadgroup)) * OPS;
  const int r0 = int(threadgroup_position_in_grid.y) * RT;
  const int rows = min(RT, int(X_shape[0]) - r0);
  const device uint* xw = (const device uint*)X;
  int nn[OPS];
  float acc[OPS][RT];
  PRAGMA_UNROLL
  for (int u = 0; u < OPS; u++) {
    nn[u] = min(n0 + u, N - 1);
    PRAGMA_UNROLL
    for (int r = 0; r < RT; r++) acc[u][r] = 0.0f;
  }
  for (int s = 0; s < STEPS; s++) {
    const int b = s * 32 + lane;
    if (b >= BLOCKS) break;
    uint wd[OPS][BITS];
    float sc[OPS], bi[OPS];
    PRAGMA_UNROLL
    for (int u = 0; u < OPS; u++) {
      const device uint* p = W + size_t(nn[u]) * WPR + size_t(b) * BITS;
      PRAGMA_UNROLL
      for (int j = 0; j < BITS; j++) wd[u][j] = p[j];
      const size_t g = size_t(nn[u]) * GROUPS + b / BPG;
      sc[u] = float(SC[g]);
      bi[u] = float(BI[g]);
    }
    PRAGMA_UNROLL
    for (int c = 0; c < 4; c++) {
      float wv[OPS][8];
      PRAGMA_UNROLL
      for (int u = 0; u < OPS; u++)
        PRAGMA_UNROLL
        for (int i = 0; i < 8; i++) wv[u][i] = fma(sc[u], float(code_at<BITS>(wd[u], 8 * c + i)), bi[u]);
      PRAGMA_UNROLL
      for (int r = 0; r < RT; r++) {
        if (r < rows) {
          const size_t at = (size_t(r0 + r) * K + size_t(b) * 32 + 8 * c) / 2;
          float xv[8];
          PRAGMA_UNROLL
          for (int h = 0; h < 4; h++) {
            const uint v = xw[at + h];
            xv[2 * h] = bf_half(v, 0);
            xv[2 * h + 1] = bf_half(v, 1);
          }
          PRAGMA_UNROLL
          for (int u = 0; u < OPS; u++)
            PRAGMA_UNROLL
            for (int i = 0; i < 8; i++) acc[u][r] = fma(xv[i], wv[u][i], acc[u][r]);
        }
      }
    }
  }
  PRAGMA_UNROLL
  for (int r = 0; r < RT; r++) {
    if (r < rows) {
      PRAGMA_UNROLL
      for (int u = 0; u < OPS; u++) {
        const float total = simd_sum(acc[u][r]);
        if (lane == 0 && n0 + u < N) OUT[size_t(r0 + r) * N + n0 + u] = bfloat(total);
      }
    }
  }
