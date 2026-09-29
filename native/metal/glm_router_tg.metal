
  // _ROUTER's arithmetic in simdgroup 0, the others double-buffering its weights in threadgroup memory
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
  const int thrM = int(lane) / 4, thrN = int(lane) % 4;
  const int g = int(threadgroup_position_in_grid.x);
  constexpr int ITERS = K / 32;
  constexpr int NCH = ITERS / C;
  constexpr int UNITS = 32 * C * 2;                                    // uint4 units a chunk
  constexpr int XS = 32 * C;                                           // x values a row a chunk
  threadgroup uint4 buf[2][UNITS];
  threadgroup float xb[2][RR][XS];
  const device uint4* rp = (const device uint4*)RP;
  auto fetch = [&](int ch, int b) {
    for (int u = int(t) - 32; u < UNITS + RR * XS; u += int(NT) - 32) {
      if (u < 0) continue;
      if (u < UNITS) {
        const int l = u / (C * 2), it = (u % (C * 2)) / 2, hh = u % 2;
        const int q = g * 4 + (l % 4), m = l / 4;
        buf[b][u] = rp[((size_t(q) * 8 + m) * ITERS + ch * C + it) * 2 + hh];
      } else {
        const int v = u - UNITS, r = v / XS, k = v % XS;
        xb[b][r][k] = X[size_t(r) * K + ch * XS + k];
      }
    }
  };
  float acc[RR][4];
  for (int r = 0; r < RR; r++) for (int tn = 0; tn < 4; tn++) acc[r][tn] = 0.0f;
  fetch(0, 0);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int ch = 0; ch < NCH; ch++) {
    const int b = ch & 1;
    if (sg != 0) {
      if (ch + 1 < NCH) fetch(ch + 1, b ^ 1);
    } else {
      for (int it = 0; it < C; it++) {
        float inter[4][4];
        for (int h = 0; h < 2; h++) {
          const uint4 v = buf[b][(int(lane) * C + it) * 2 + h];
          const uint words[4] = {v.x, v.y, v.z, v.w};
          for (int j = 0; j < 4; j++) {
            const int e = h * 8 + j * 2;
            inter[e / 4][e % 4] = as_type<float>(words[j] << 16);
            inter[(e + 1) / 4][(e + 1) % 4] = as_type<float>(words[j] & 0xffff0000u);
          }
        }
        const int bm = 4 * thrM + 32 * it;                             // within the chunk
        for (int r = 0; r < RR; r++) {
          float vc[4];
          for (int tm = 0; tm < 4; tm++) vc[tm] = xb[b][r][bm + tm];
          for (int tm = 0; tm < 4; tm++)
            for (int tn = 0; tn < 4; tn++) acc[r][tn] += vc[tm] * inter[tm][tn];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (sg != 0) return;
  const int q = g * 4 + thrN;
  for (int r = 0; r < RR; r++)
    for (int tn = 0; tn < 4; tn++) {
      float v = acc[r][tn];
      for (ushort sm = 4; sm >= 1; sm >>= 1) v += simd_shuffle_down(v, 4 * sm);
      if (thrM == 0) OUT[size_t(r) * NE + 4 * q + tn] = v;
    }
