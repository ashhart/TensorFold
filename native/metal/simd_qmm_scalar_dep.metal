  #define LOAD8(r, j) ((((const device uint4*)X)[size_t(r) * (K / 8) + (j)]))

  // one row. Threadgroup of SGS simdgroups; lane (chunk c = lane % S, slot j = lane / S) runs chunk c of NR outputs
  // n0 + j + (32 / S) u, a whole group (8 words) of each in registers, the next group's loaded before this one is
  // used. The threadgroup stages XB groups of inputs at a time in threadgroup memory, pre-scaled and in chain order
  // (x'[g][8 s + i] = x[64 g + 8 i + s] * 2^-4s), with each group's 8 left-to-right sums. (Loading each block of a
  // row contiguously and swapping halves between lane pairs gave the same bits and ran 1.4% slower, 2026-09-26.)
  constexpr int XP = 76;                        // floats a staged group: 64 inputs, 8 sums, 4 pad
  threadgroup float xs[XB * XP];
  const uint lane = thread_index_in_simdgroup;
  const int tid = int(simdgroup_index_in_threadgroup) * 32 + int(lane);
  const int c = int(lane) % S;
  constexpr int SLOTS = 32 / S;
  const int n0 = (int(threadgroup_position_in_grid.x) * SGS + int(simdgroup_index_in_threadgroup)) * (SLOTS * NR)
                 + int(lane) / S;
  constexpr int G = K / 64;
  const float one = ONE[0];
  const device uint4* wr[NR];
  const device bfloat* sr[NR];
  const device bfloat* br[NR];
  float acc[NR];
  PRAGMA_UNROLL
  for (int u = 0; u < NR; u++) {
    const int nn = min(n0 + SLOTS * u, N - 1);
    wr[u] = (const device uint4*)(W + size_t(nn) * (K / 8));
    sr[u] = SC + size_t(nn) * G;
    br[u] = BI + size_t(nn) * G;
    acc[u] = 0.0f;
  }
  uint4 na[NR], nb[NR];
  PRAGMA_UNROLL
  for (int u = 0; u < NR; u++) { na[u] = uint4(0); nb[u] = uint4(0); }
  if (c < G) {
    PRAGMA_UNROLL
    for (int u = 0; u < NR; u++) { na[u] = wr[u][2 * c]; nb[u] = wr[u][2 * c + 1]; }
  }
  for (int b0 = 0; b0 < G; b0 += XB) {
    const int nbk = min(XB, G - b0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int idx = tid; idx < nbk * 8; idx += SGS * 32) {
      const int gl = idx / 8, i = idx % 8;
      const uint4 v = LOAD8(0, 8 * (b0 + gl) + i);
      PRAGMA_UNROLL
      for (int s = 0; s < 8; s++) xs[gl * XP + 8 * s + i] = bf8(v, s) * pre(s);
      xs[gl * XP + 64 + i] = sum8(v, one);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int g = b0 + c; g < b0 + nbk; g += S) {
      uint4 wa[NR], wb[NR];
      PRAGMA_UNROLL
      for (int u = 0; u < NR; u++) { wa[u] = na[u]; wb[u] = nb[u]; }
      if (g + S < G) {
        PRAGMA_UNROLL
        for (int u = 0; u < NR; u++) { na[u] = wr[u][2 * (g + S)]; nb[u] = wr[u][2 * (g + S) + 1]; }
      }
      const threadgroup float* xg = xs + (g - b0) * XP;
      const float4 p0 = *(const threadgroup float4*)(xg + 64), p1 = *(const threadgroup float4*)(xg + 68);
      const float xsum = fma(fma(fma(p1.w, one, p1.z), one, fma(p1.y, one, p1.x)), one,
                             fma(fma(p0.w, one, p0.z), one, fma(p0.y, one, p0.x)));
      float P[NR];
      PRAGMA_UNROLL
      for (int u = 0; u < NR; u++) P[u] = 0.0f;
      PRAGMA_UNROLL
      for (int s = 0; s < 8; s++) {
        const uint mask = 0xFu << (4 * s);
        const float4 lo = *(const threadgroup float4*)(xg + 8 * s), hi = *(const threadgroup float4*)(xg + 8 * s + 4);
        const float xq[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
        PRAGMA_UNROLL
        for (int u = 0; u < NR; u++) {
          const uint wd[8] = {wa[u].x, wa[u].y, wa[u].z, wa[u].w, wb[u].x, wb[u].y, wb[u].z, wb[u].w};
          PRAGMA_UNROLL
          for (int i = 0; i < 8; i++) P[u] = fma(xq[i], float(wd[i] & mask), P[u]);
        }
      }
      PRAGMA_UNROLL
      for (int u = 0; u < NR; u++) {
        acc[u] = fma(float(sr[u][g]), P[u], acc[u]);
        acc[u] = fma(float(br[u][g]), xsum, acc[u]);
      }
    }
  }
  PRAGMA_UNROLL
  for (int u = 0; u < NR; u++) {
    float v = acc[u];
    PRAGMA_UNROLL
    for (int m = 1; m < S; m <<= 1) v = fma(simd_shuffle_xor(v, ushort(m)), one, v);
    const int n = n0 + SLOTS * u;
    if (n < N && c == 0) OUT[n] = bfloat(v);
  }
  #undef LOAD8
