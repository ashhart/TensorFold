// MiniMax-M3's indexer block scores on the tensor units: S[row, h, b] = max over b's keys of IQ[row, h] . IK[key].
#include "../nax.h"
using namespace tfp;

// Defines: DH (128, the index dim), IH (index heads).

constant constexpr int LD = DH;

// torch's max: a NaN wins (Metal's max is fmax and drops it).
inline float nmax(float x, float y) { return isnan(x) || isnan(y) ? NAN : max(x, y); }

// Row max over a fragment's 32 columns into r[0], r[1] (rows home.y, home.y + 8).
inline void fold_max(thread const frag<float>& f, thread float* r) {
  TF_UNROLL
  for (short h = 0; h < 2; h++) {
    const short b = 4 * h;
    float t = nmax(nmax(f[b], f[b + 1]), nmax(f[b + 2], f[b + 3]));
    t = nmax(t, simd_shuffle_xor(t, ushort(1)));
    t = nmax(t, simd_shuffle_xor(t, ushort(8)));
    r[h] = nmax(r[h], t);
  }
}

constant constexpr int RT = 32;                    // query rows a tile
constant constexpr int NS = DH / 64;               // simdgroups a tile, each its own 64 output columns
constant constexpr int KS = 32;                    // keys a simdgroup a block
constant constexpr int BKT = KS * NS;              // keys a block

template <bool TR>
inline frag<bfloat> blk(const device bfloat* p, int ld, int r, int c, short2 home) {
#ifndef TF_SIMD_FRAGS
  const device bfloat* q = p + (r + home.y) * ld + c + home.x;
  return frag<bfloat>(*(const device bfloat4*)q, *(const device bfloat4*)(q + 8 * ld));
#else
  frag<bfloat> f;
  if (TR) {
    frag_get_t(f, p, ld, r, c, home);
  } else {
    frag_get(f, p, ld, r, c, home);
  }
  return f;
#endif
}

// IQ [R, IH, DH] at P.y..; IK [slot, DH]; S [R, IH, P.z]. A tile is RT rows of an index head, 64 keys a step.
[[max_total_threads_per_threadgroup(256)]]
[[kernel]] void mm_index_scores_nax(const device bfloat* IQ [[buffer(0)]], const device bfloat* IK [[buffer(1)]],
                                    device float* S [[buffer(2)]], constant int3& P [[buffer(3)]],
                                    uint sg [[simdgroup_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  const int R = P.x, p0 = P.y, blocks = P.z;
  const int h = int(tg.y);
  const int r0 = int(tg.x) * RT;
  const int q_rows = min(RT, R - r0);
  const int qs = IH * DH;
  const device bfloat* q = IQ + r0 * qs + h * DH;
  const short2 home = frag_home(ushort(lane));
  threadgroup float rowmax[NS][RT];
  float best[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
  const int first_pos = p0 + r0, last_pos = p0 + r0 + q_rows - 1;
  const int steps = last_pos / BKT + 1;
  for (int kb = 0; kb < steps; kb++) {
    const int kpos = kb * BKT + int(sg) * KS;
    const device bfloat* kblk = IK + long(kpos) * LD;
    frag<float> s[4];
    TF_UNROLL
    for (short i = 0; i < 4; i++) s[i] = frag<float>(0);
    for (int d = 0; d < DH / 16; d++) {
      const frag<bfloat> k0 = blk<true>(kblk, LD, 0, 16 * d, home);
      const frag<bfloat> k1 = blk<true>(kblk, LD, 16, 16 * d, home);
      TF_UNROLL
      for (short hh = 0; hh < 2; hh++) {
        const frag<bfloat> qf = blk<false>(q, qs, 16 * hh, 16 * d, home);
        mma_16x32<false, true>(s[2 * hh], s[2 * hh + 1], qf, k0, k1);
      }
    }
    TF_UNROLL
    for (short i = 0; i < 4; i++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        const int local = 16 * (i >> 1) + home.y + (e >> 2) * 8, row = first_pos + local;
        const int col = kpos + 16 * (i & 1) + home.x + TF_COL(e);
        if (col > row) s[i][e] = -INFINITY; // a key past the row (or never written): no part of the block's max
      }
    }
    float m[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    TF_UNROLL
    for (short i = 0; i < 4; i++) fold_max(s[i], m + 2 * (i >> 1));
    threadgroup_barrier(mem_flags::mem_threadgroup); // the step before has read rowmax
    if (home.x == 0) {
      TF_UNROLL
      for (short j = 0; j < 4; j++) rowmax[sg][home.y + 8 * j] = m[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    TF_UNROLL
    for (short j = 0; j < 4; j++)
      for (int u = 0; u < NS; u++) best[j] = nmax(best[j], rowmax[u][home.y + 8 * j]);
    if (kb % 2 == 1 || kb + 1 == steps) { // a block's two steps done (or its first, when it is the tile's last)
      const int b = kb / 2;
      if (sg == 0 && home.x == 0 && b < blocks) {
        TF_UNROLL
        for (short j = 0; j < 4; j++) {
          const int local = home.y + 8 * j;
          if (local < q_rows) S[(long(r0 + local) * IH + h) * blocks + b] = best[j];
        }
      }
      TF_UNROLL
      for (short j = 0; j < 4; j++) best[j] = -INFINITY;
    }
  }
}
