
  constexpr int S = 4;
  constexpr int F = S * D;                          // flattened streams
  constexpr int MIX = (2 + S) * S;                  // 24 mixes

  // _HC_MIX's arithmetic on the bf16 matrix repacked per thread, loads issued U iterations ahead
  const int og = int(threadgroup_position_in_grid.x);
  const int r = int(threadgroup_position_in_grid.y);
  const uint lane = thread_index_in_simdgroup, sgn = simdgroup_index_in_threadgroup;
  constexpr int ITERS = F / 1024;
  threadgroup float part[8][4];
  const float inv = INV[r];
  device const bfloat* xs = X + size_t(r) * F;
  const device uint4* w = (const device uint4*)(FNP + ((size_t(og) * 8 + sgn) * 32 + lane) * ITERS * 16);
  const int bn0 = (32 * int(sgn) + int(lane)) * 4;
  float res[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int i0 = 0; i0 < ITERS; i0 += U) {
    uint4 raw[U][2];
    float xv[U][4];
    for (int u = 0; u < U; u++) {
      raw[u][0] = w[(i0 + u) * 2]; raw[u][1] = w[(i0 + u) * 2 + 1];
      for (int tn = 0; tn < 4; tn++) xv[u][tn] = float(xs[bn0 + 1024 * (i0 + u) + tn]);
    }
    for (int u = 0; u < U; u++) {
      float vc[4];
      for (int tn = 0; tn < 4; tn++) vc[tn] = xv[u][tn] * inv;
      float inter[4][4];
      for (int h = 0; h < 2; h++) {
        const uint4 v = raw[u][h];
        const uint words[4] = {v.x, v.y, v.z, v.w};
        for (int j = 0; j < 4; j++) {
          const int e = h * 8 + j * 2;
          inter[e / 4][e % 4] = as_type<float>(words[j] << 16);
          inter[(e + 1) / 4][(e + 1) % 4] = as_type<float>(words[j] & 0xffff0000u);
        }
      }
      for (int tm = 0; tm < 4; tm++)
        for (int tn = 0; tn < 4; tn++) res[tm] += inter[tm][tn] * vc[tn];
    }
  }
  for (int tm = 0; tm < 4; tm++)
    for (ushort sn = 16; sn >= 1; sn >>= 1) res[tm] += simd_shuffle_down(res[tm], sn);
  if (lane == 0) for (int tm = 0; tm < 4; tm++) part[sgn][tm] = res[tm];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgn == 0 && lane < 4) {
    float a = part[0][lane];
    for (int k = 1; k < 8; k++) a += part[k][lane];
    MIXES[r * MIX + og * 4 + lane] = a;
  }
