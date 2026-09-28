  #define LOAD8(r, j) ((((const device uint4*)X)[size_t(r) * (K / 8) + (j)]))

  // RS rows (1 to 4). Lane (chunk c = lane % S, slot j = lane / S) runs chunk c of NR outputs n0 + j + (32 / S) u,
  // a whole group (WPG words) of each in registers, once a row; the threadgroup stages XB groups of each row's
  // inputs pre-scaled in chain order. A row's chain is the same at any RS.
  constexpr int WPG = GS / 8, NS = 8 / WPG;     // words a group; nibble stride of an MMA step
  constexpr int XP = GS == 64 ? 76 : 44;        // floats a staged group: GS inputs, WPG sums, pad (bank spread)
  threadgroup float xs[RS * XB * XP];
  const uint lane = thread_index_in_simdgroup;
  const int tid = int(simdgroup_index_in_threadgroup) * 32 + int(lane);
  const int c = int(lane) % S;
  constexpr int SLOTS = 32 / S;
  const int n0 = (int(threadgroup_position_in_grid.x) * SGS + int(simdgroup_index_in_threadgroup)) * (SLOTS * NR)
                 + int(lane) / S;
  constexpr int G = K / GS;
  const float one = ONE[0];
  const device uint4* wr[NR];
  const device bfloat* sr[NR];
  const device bfloat* br[NR];
  float acc[NR][RS];
  PRAGMA_UNROLL
  for (int u = 0; u < NR; u++) {
    const int nn = min(n0 + SLOTS * u, N - 1);
    wr[u] = (const device uint4*)(W + size_t(nn) * (K / 8));
    sr[u] = SC + size_t(nn) * G;
    br[u] = BI + size_t(nn) * G;
    PRAGMA_UNROLL
    for (int r = 0; r < RS; r++) acc[u][r] = 0.0f;
  }
  uint4 nw[NR][WPG / 4];
  PRAGMA_UNROLL
  for (int u = 0; u < NR; u++) for (int h = 0; h < WPG / 4; h++) nw[u][h] = c < G ? wr[u][(WPG / 4) * c + h] : uint4(0);
  for (int b0 = 0; b0 < G; b0 += XB) {
    const int nbk = min(XB, G - b0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int idx = tid; idx < RS * nbk * WPG; idx += SGS * 32) {
      const int r = RS == 1 ? 0 : idx / (nbk * WPG);
      const int gl = (RS == 1 ? idx : idx - r * (nbk * WPG)) / WPG, j = idx % WPG;
      const uint4 v = LOAD8(r, WPG * (b0 + gl) + j);
      threadgroup float* xr = xs + r * (XB * XP) + gl * XP;
      PRAGMA_UNROLL
      for (int e = 0; e < 8; e++) xr[8 * (e / NS) + NS * j + e % NS] = bf8(v, e) * pre(e);
      xr[GS + j] = sum8(v, one);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int g = b0 + c; g < b0 + nbk; g += S) {
      uint4 wv[NR][WPG / 4];
      PRAGMA_UNROLL
      for (int u = 0; u < NR; u++) for (int h = 0; h < WPG / 4; h++) wv[u][h] = nw[u][h];
      if (g + S < G) {
        PRAGMA_UNROLL
        for (int u = 0; u < NR; u++) for (int h = 0; h < WPG / 4; h++) nw[u][h] = wr[u][(WPG / 4) * (g + S) + h];
      }
      float xsum[RS];
      float P[NR][RS];
      PRAGMA_UNROLL
      for (int r = 0; r < RS; r++) {
        const threadgroup float* xg = xs + r * (XB * XP) + (g - b0) * XP;
        const float4 p0 = *(const threadgroup float4*)(xg + GS);
        xsum[r] = fma(fma(p0.w, one, p0.z), one, fma(p0.y, one, p0.x));
        if (GS == 64) {
          const float4 p1 = *(const threadgroup float4*)(xg + GS + 4);
          xsum[r] = fma(fma(fma(p1.w, one, p1.z), one, fma(p1.y, one, p1.x)), one, xsum[r]);
        }
        PRAGMA_UNROLL
        for (int u = 0; u < NR; u++) P[u][r] = 0.0f;
      }
      PRAGMA_UNROLL
      for (int s = 0; s < WPG; s++) {
        float xq[RS][8];
        PRAGMA_UNROLL
        for (int r = 0; r < RS; r++) {
          const threadgroup float* xg = xs + r * (XB * XP) + (g - b0) * XP + 8 * s;
          const float4 lo = *(const threadgroup float4*)(xg), hi = *(const threadgroup float4*)(xg + 4);
          xq[r][0] = lo.x; xq[r][1] = lo.y; xq[r][2] = lo.z; xq[r][3] = lo.w;
          xq[r][4] = hi.x; xq[r][5] = hi.y; xq[r][6] = hi.z; xq[r][7] = hi.w;
        }
        PRAGMA_UNROLL
        for (int u = 0; u < NR; u++)
          PRAGMA_UNROLL
          for (int i = 0; i < 8; i++) {
            const float q = float(wv[u][i / NS / 4][(i / NS) % 4] & (0xFu << (4 * (NS * s + i % NS))));
            PRAGMA_UNROLL
            for (int r = 0; r < RS; r++) P[u][r] = fma(xq[r][i], q, P[u][r]);
          }
      }
      PRAGMA_UNROLL
      for (int u = 0; u < NR; u++) {
        const float sc = float(sr[u][g]), bi = float(br[u][g]);
        PRAGMA_UNROLL
        for (int r = 0; r < RS; r++) {
          acc[u][r] = fma(sc, P[u][r], acc[u][r]);
          acc[u][r] = fma(bi, xsum[r], acc[u][r]);
        }
      }
    }
  }
  PRAGMA_UNROLL
  for (int u = 0; u < NR; u++)
    PRAGMA_UNROLL
    for (int r = 0; r < RS; r++) {
      float v = acc[u][r];
      PRAGMA_UNROLL
      for (int m = 1; m < S; m <<= 1) v = fma(simd_shuffle_xor(v, ushort(m)), one, v);
      const int n = n0 + SLOTS * u;
      if (n < N && c == 0) OUT[size_t(r) * N + n] = bfloat(v);
    }
  #undef LOAD8
