// TF_FN_GX_GATEUP_MB2: gx_gateup's rows for two pairs of one expert, its weights decoded once; appended to fn_gx.metal
#ifdef GX_MB2_INDEXED
inline float2 mb2_sb(const device bfloat* S, const device bfloat* B, size_t at) {
  const uint i = uint(((const device ushort*)S)[at]) * 2u;
  return float2(float(B[i]), float(B[i + 1]));
}
#else
inline float2 mb2_sb(const device bfloat* S, const device bfloat* B, size_t at) {
  return float2(float(S[at]), float(B[at]));
}
#endif

// gateup_rows for inputs x[0] and x[1]: each input's sums, products, FMAs and order are the one-input path's
inline void gateup_rows2(const device uint32_t* GWp, const device uint32_t* UWp, const device bfloat* GSp,
                         const device bfloat* GBp, const device bfloat* USp, const device bfloat* UBp, size_t first,
                         const device bfloat* xa, const device bfloat* xb, uint lane, thread float (*ag)[4],
                         thread float (*au)[4]) {
  constexpr int RB = K / 2, RPS = 4;
  const device uint8_t* gw = (const device uint8_t*)GWp + first * RB;
  const device uint8_t* uw = (const device uint8_t*)UWp + first * RB;
  for (int v0 = int(lane) * 16; v0 < K; v0 += 512) {
    float va[16], vb[16], suma = 0.0f, sumb = 0.0f;
    for (int i = 0; i < 16; i++) { va[i] = float(xa[v0 + i]); suma += va[i]; }
    for (int i = 0; i < 16; i++) { vb[i] = float(xb[v0 + i]); sumb += vb[i]; }
    for (int row = 0; row < RPS; row++) {
      float qg[16], qu[16];
      lane_codes(gw + row * RB, v0, qg);
      lane_codes(uw + row * RB, v0, qu);
      float dga = 0.0f, dua = 0.0f, dgb = 0.0f, dub = 0.0f;
      for (int i = 0; i < 16; i++) { dga = fma(qg[i], va[i], dga); dua = fma(qu[i], va[i], dua); }
      for (int i = 0; i < 16; i++) { dgb = fma(qg[i], vb[i], dgb); dub = fma(qu[i], vb[i], dub); }
      const size_t at = (first + row) * KG + v0 / 32;
      const float2 g = mb2_sb(GSp, GBp, at), u = mb2_sb(USp, UBp, at);
      ag[0][row] += fma(g.x, dga, g.y * suma);
      au[0][row] += fma(u.x, dua, u.y * suma);
      ag[1][row] += fma(g.x, dgb, g.y * sumb);
      au[1][row] += fma(u.x, dub, u.y * sumb);
    }
  }
}

// A list slot's expert: the routed expert, or -1 for a shared row
inline int mb2_key(const device int* LIST, int n, int y) { return y >= n ? -1 : LIST[L_MEM + y] >> 16; }

// Slots pair (even, next) when both hold one expert; any other slot runs alone, so each slot runs exactly once
[[max_total_threads_per_threadgroup(64)]]
[[kernel]] void gx_gateup_mb2(const device bfloat* X [[buffer(0)]], const device int* LIST [[buffer(1)]],
    const device uint* GW [[buffer(2)]], const device bfloat* GS [[buffer(3)]], const device bfloat* GB [[buffer(4)]],
    const device uint* UW [[buffer(5)]], const device bfloat* US [[buffer(6)]], const device bfloat* UB [[buffer(7)]],
    const device uint* SGW [[buffer(8)]], const device bfloat* SGS [[buffer(9)]], const device bfloat* SGB [[buffer(10)]],
    const device uint* SUW [[buffer(11)]], const device bfloat* SUS [[buffer(12)]], const device bfloat* SUB [[buffer(13)]],
    device bfloat* ACT [[buffer(14)]], uint3 tg [[threadgroup_position_in_grid]],
    uint sgi [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const int n = LIST[L_N], y = int(tg.y), slots = LIST[L_GU + 1], key = mb2_key(LIST, n, y);
  if ((y & 1) != 0 && mb2_key(LIST, n, y - 1) == key) return;
  const bool two = (y & 1) == 0 && y + 1 < slots && mb2_key(LIST, n, y + 1) == key;
  const bool shared = y >= n;
  const int pa = shared ? (y - n) * SLOTS + TOPK : LIST[L_MEM + y] & 0xFFFF;
  const int row0 = int(tg.x) * 8 + int(sgi) * 4;
  const size_t first = shared ? size_t(row0) : size_t(key) * N + row0;
  const device uint* gw = shared ? SGW : GW;
  const device uint* uw = shared ? SUW : UW;
  const device bfloat* gs = shared ? SGS : GS;
  const device bfloat* gb = shared ? SGB : GB;
  const device bfloat* us = shared ? SUS : US;
  const device bfloat* ub = shared ? SUB : UB;
  if (two) {
    const int pb = y + 1 >= n ? (y + 1 - n) * SLOTS + TOPK : LIST[L_MEM + y + 1] & 0xFFFF;
    float ag[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
    float au[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
    gateup_rows2(gw, uw, gs, gb, us, ub, first, X + (pa / SLOTS) * K, X + (pb / SLOTS) * K, lane, ag, au);
    for (int m = 0; m < 2; m++) for (int row = 0; row < 4; row++) {
      const float gv = simd_sum(ag[m][row]), uv = simd_sum(au[m][row]);
      if (lane == 0) ACT[(m == 0 ? pa : pb) * N + row0 + row] = bfloat(bsilu(float(bfloat(gv))) * float(bfloat(uv)));
    }
    return;
  }
  float ag[4] = {0.0f, 0.0f, 0.0f, 0.0f}, au[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  gateup_rows(gw, uw, gs, gb, us, ub, first, X + (pa / SLOTS) * K, lane, ag, au);
  for (int row = 0; row < 4; row++) {
    const float gv = simd_sum(ag[row]), uv = simd_sum(au[row]);
    if (lane == 0) ACT[pa * N + row0 + row] = bfloat(bsilu(float(bfloat(gv))) * float(bfloat(uv)));
  }
}
