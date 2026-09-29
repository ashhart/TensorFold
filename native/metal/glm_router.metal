
  // MLX's one-row gemv_t (BM 1, BN 2, SM 8, SN 4, TM 4, TN 4) over the repacked bf16 matrix, rows sharing reads
  const uint lane = thread_index_in_simdgroup;
  const int thrM = int(lane) / 4, thrN = int(lane) % 4;
  const int q = int(threadgroup_position_in_grid.x) * 4 + thrN;       // column quad: columns 4 q .. 4 q + 3
  constexpr int ITERS = K / 32;
  float acc[RR][4];
  for (int r = 0; r < RR; r++) for (int tn = 0; tn < 4; tn++) acc[r][tn] = 0.0f;
  const device uint4* w = (const device uint4*)(RP + (size_t(q) * 8 + thrM) * ITERS * 16);
  for (int i0 = 0; i0 < ITERS; i0 += U) {
    uint4 raw[U][2];                                                   // U iterations x 16 bf16
    for (int u = 0; u < U; u++) { raw[u][0] = w[(i0 + u) * 2]; raw[u][1] = w[(i0 + u) * 2 + 1]; }
    for (int u = 0; u < U; u++) {
      float inter[4][4];
      for (int h = 0; h < 2; h++) {
        const uint4 v = raw[u][h];
        const uint words[4] = {v.x, v.y, v.z, v.w};
        for (int j = 0; j < 4; j++) {
          const int e = h * 8 + j * 2;                                 // bf16 pairs: low half first
          inter[e / 4][e % 4] = as_type<float>(words[j] << 16);
          inter[(e + 1) / 4][(e + 1) % 4] = as_type<float>(words[j] & 0xffff0000u);
        }
      }
      const int bm = 4 * thrM + 32 * (i0 + u);
      for (int r = 0; r < RR; r++) {
        float vc[4];
        for (int tm = 0; tm < 4; tm++) vc[tm] = X[size_t(r) * K + bm + tm];
        for (int tm = 0; tm < 4; tm++)
          for (int tn = 0; tn < 4; tn++) acc[r][tn] += vc[tm] * inter[tm][tn];
      }
    }
  }
  for (int r = 0; r < RR; r++)
    for (int tn = 0; tn < 4; tn++) {
      float v = acc[r][tn];
      for (ushort sm = 4; sm >= 1; sm >>= 1) v += simd_shuffle_down(v, 4 * sm);
      if (thrM == 0) OUT[size_t(r) * NE + 4 * q + tn] = v;
    }
