// Gemma 4 prompt attention on the tensor units: 32 query rows of a head a threadgroup, 32 keys a simdgroup a block.
#include "../nax.h"
using namespace tfp;

// Compiled with DH (head dim), NQ, G (query heads a key head), SL (= DH / 128), WINDOW (0: global), RING (0: linear).

constant constexpr float kMasked = -3.402823466e+38f;
constant constexpr float kLog2e = 1.44269504089f;
constant constexpr int BK = 32;

// Row max (MAX) or sum over a fragment's 32 columns into r[0], r[1] (rows home.y, home.y + 8), as attention_nax folds.
template <bool MAX>
inline void fold(thread const frag<float>& f, thread float* r) {
  TF_UNROLL
  for (short h = 0; h < 2; h++) {
    const short b = 4 * h;
    float t = MAX ? max(max(f[b], f[b + 1]), max(f[b + 2], f[b + 3])) : (f[b] + f[b + 1]) + (f[b + 2] + f[b + 3]);
    const float u = simd_shuffle_xor(t, ushort(1));
    t = MAX ? max(t, u) : t + u;
    const float w = simd_shuffle_xor(t, ushort(8));
    t = MAX ? max(t, w) : t + w;
    r[h] = MAX ? max(r[h], t) : r[h] + t;
  }
}

// A tile is RT query rows of one head: NS simdgroups each score KS keys of a block over the whole head (Q hot in L1).
constant constexpr int RT = 32;          // query rows a tile
// Each simdgroup accumulates its own 64 output columns; once a block the tile's row maxes cross threadgroup memory.
constant constexpr int NS = DH / 64;     // simdgroups a tile
constant constexpr int KS = 32;          // keys a simdgroup a block
// The fragment-step loops stay rolled: fully unrolled, the code outgrew the instruction cache and ran 5x slower.
constant constexpr int BKT = KS * NS;    // keys a block
// The probabilities cross threadgroup memory PH keys at a time, as fp32 (bf16 moved an 8k prompt's logits off mlx-lm).
constant constexpr int PH = 128;         // keys whose probabilities cross at once
constant constexpr int PP = PH + 8;      // the probabilities' row pitch
using PT = float;

// A bf16 matrix's 16x16 block at (r, c): a left or (TR) transposed right operand (on M5 a lane's row in one load).
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

// The probabilities' 16 x 16 block at (r, c) of the tile, and a scored fragment's store into it.
inline frag<PT> prob_frag(const threadgroup PT* pt, int r, int c, short2 home) {
#ifndef TF_SIMD_FRAGS
  const threadgroup PT* q = pt + (r + home.y) * PP + c + home.x;
  return frag<PT>(*(const threadgroup vec<PT, 4>*)q, *(const threadgroup vec<PT, 4>*)(q + 8 * PP));
#else
  frag<PT> f;
  frag_get(f, pt, PP, r, c, home);
  return f;
#endif
}

inline void prob_put(threadgroup PT* pt, int r, int c, short2 home, thread const frag<float>& s) {
#ifndef TF_SIMD_FRAGS
  threadgroup PT* q = pt + (r + home.y) * PP + c + home.x;
  *(threadgroup vec<PT, 4>*)q = vec<PT, 4>(s[0], s[1], s[2], s[3]);
  *(threadgroup vec<PT, 4>*)(q + 8 * PP) = vec<PT, 4>(s[4], s[5], s[6], s[7]);
#else
  TF_UNROLL
  for (short e = 0; e < 8; e++) pt[(r + home.y + (e >> 2) * 8) * PP + c + home.x + TF_COL(e)] = PT(s[e]);
#endif
}

// Q [R, NQ, DH] at positions p0..; K, V the cache [HK, cap, DH]; row j of top and total is the tile's home.y + 8j.
[[max_total_threads_per_threadgroup(256)]]
// Fragment i of s: tile rows 16 (i >> 1).., keys 16 (i & 1)..; of acc: rows 16 (i >> 2).., own columns 16 (i & 3)...
[[kernel]] void gemma_prompt_attention(const device bfloat* Q [[buffer(0)]], const device bfloat* K [[buffer(1)]],
                                       const device bfloat* V [[buffer(2)]], device bfloat* O [[buffer(3)]],
                                       constant int3& P [[buffer(4)]],
                                       const device int* HI [[buffer(5)]], // sliding layers: a row's last key (image blocks)
                                       uint sg [[simdgroup_index_in_threadgroup]],
                                       uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  const int R = P.x, p0 = P.y, cap = P.z;
  const int qh = int(tg.y), kv = qh / G;
  const int r0 = int(tg.x) * RT;
  const int q_rows = min(RT, R - r0);
  const int qs = NQ * DH;
  const device bfloat* q = Q + r0 * qs + qh * DH;
  const device bfloat* kc = K + long(kv) * cap * DH;
  const device bfloat* vc = V + long(kv) * cap * DH + int(sg) * 64;
  const short2 home = frag_home(ushort(lane));
  threadgroup PT pt[RT * PP];
  threadgroup float rowmax[NS][RT];
  frag<float> acc[8];
  TF_UNROLL
  for (short i = 0; i < 8; i++) acc[i] = frag<float>(0);
  float top[4] = {kMasked, kMasked, kMasked, kMasked}, total[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  const int first_pos = p0 + r0, last_pos = WINDOW ? HI[r0 + q_rows - 1] : p0 + r0 + q_rows - 1; // the tile's last key
  const int lo = WINDOW ? max(0, first_pos - WINDOW + 1) : 0;
  for (int kb = lo / BKT; kb <= last_pos / BKT; kb++) {
    const int kpos = kb * BKT + int(sg) * KS;
    const int slot = RING ? kpos % RING : kpos;
    const device bfloat* kblk = kc + long(slot) * DH;
    frag<float> s[4];
    TF_UNROLL
    for (short i = 0; i < 4; i++) s[i] = frag<float>(0);
    for (int d = 0; d < DH / 16; d++) {
      const frag<bfloat> k0 = blk<true>(kblk, DH, 0, 16 * d, home);
      const frag<bfloat> k1 = blk<true>(kblk, DH, 16, 16 * d, home);
      TF_UNROLL
      for (short h = 0; h < 2; h++) { // rows past q_rows sit inside the 512-row buffer; their outputs stay unwritten
        const frag<bfloat> qf = blk<false>(q, qs, 16 * h, 16 * d, home);
        mma_16x32<false, true>(s[2 * h], s[2 * h + 1], qf, k0, k1);
      }
    }
    TF_UNROLL
    for (short i = 0; i < 4; i++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        const int local = 16 * (i >> 1) + home.y + (e >> 2) * 8, row = first_pos + local;
        const int col = kpos + 16 * (i & 1) + home.x + TF_COL(e);
        const int hi = WINDOW && local < q_rows ? HI[r0 + local] : row;
        const bool out = col > hi || (WINDOW && col < row - WINDOW + 1);
        s[i][e] = out ? kMasked : s[i][e] * kLog2e;
      }
    }
    float top_new[4] = {top[0], top[1], top[2], top[3]};
    TF_UNROLL
    for (short i = 0; i < 4; i++) fold<true>(s[i], top_new + 2 * (i >> 1));
    if (home.x == 0) {
      TF_UNROLL
      for (short j = 0; j < 4; j++) rowmax[sg][home.y + 8 * j] = top_new[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    TF_UNROLL
    for (short j = 0; j < 4; j++) {
      for (int t = 0; t < NS; t++) top_new[j] = max(top_new[j], rowmax[t][home.y + 8 * j]);
    }
    TF_UNROLL
    for (short i = 0; i < 4; i++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        s[i][e] = s[i][e] == kMasked ? 0.0f : fast::exp2(s[i][e] - top_new[2 * (i >> 1) + (e >> 2)]);
      }
    }
    float keep[4];
    TF_UNROLL
    for (short j = 0; j < 4; j++) {
      keep[j] = top[j] == kMasked ? 0.0f : fast::exp2(top[j] - top_new[j]);
      top[j] = top_new[j];
      total[j] = total[j] * keep[j];
    }
    TF_UNROLL
    for (short i = 0; i < 4; i++) fold<false>(s[i], total + 2 * (i >> 1));
    TF_UNROLL
    for (short i = 0; i < 8; i++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) acc[i][e] = acc[i][e] * keep[2 * (i >> 2) + (e >> 2)];
    }
    for (int ph = 0; ph < BKT / PH; ph++) {
      if (ph > 0) threadgroup_barrier(mem_flags::mem_threadgroup); // the phase before has read the tile
      if (int(sg) / (PH / KS) == ph) {
        TF_UNROLL
        for (short i = 0; i < 4; i++) prob_put(pt, 16 * (i >> 1), int(sg) * KS - ph * PH + 16 * (i & 1), home, s[i]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int kk = 0; kk < PH / 16; kk++) {
        const int kp = kb * BKT + ph * PH + 16 * kk;
        if (kp > last_pos) break; // keys past the tile's rows: masked, and their cache rows may hold anything
        const int valid = min(16, last_pos + 1 - kp); // a value row never written would poison the tile as 0 x NaN
        const device bfloat* vblk = vc + long(RING ? kp % RING : kp) * DH;
        frag<bfloat> v0, v1;
        TF_UNROLL
        for (short d = 0; d < 4; d += 2) {
          if (valid == 16) {
            v0 = blk<false>(vblk, DH, 0, 16 * d, home);
            v1 = blk<false>(vblk, DH, 0, 16 * d + 16, home);
          } else {
            frag_get_in(v0, vblk, DH, 0, 16 * d, home, valid, DH);
            frag_get_in(v1, vblk, DH, 0, 16 * d + 16, home, valid, DH);
          }
          TF_UNROLL
          for (short h = 0; h < 2; h++) {
            const frag<PT> pf = prob_frag(pt, 16 * h, 16 * kk, home);
            mma_16x32<false, false>(acc[4 * h + d], acc[4 * h + d + 1], pf, v0, v1);
          }
        }
      }
    }
  }
  // the tile's totals: every simdgroup's partial sums, scaled alike by the shared maxes
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (home.x == 0) {
    TF_UNROLL
    for (short j = 0; j < 4; j++) rowmax[sg][home.y + 8 * j] = total[j];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float inv[4];
  TF_UNROLL
  for (short j = 0; j < 4; j++) {
    float t = 0.0f;
    for (int u = 0; u < NS; u++) t += rowmax[u][home.y + 8 * j];
    inv[j] = t > 0.0f ? 1.0f / t : 0.0f;
  }
  device bfloat* o = O + r0 * qs + qh * DH + int(sg) * 64;
  TF_UNROLL
  for (short i = 0; i < 8; i++) {
    TF_UNROLL
    for (short e = 0; e < 8; e++) acc[i][e] = acc[i][e] * inv[2 * (i >> 2) + (e >> 2)];
    frag_put_in(acc[i], o, qs, 16 * (i >> 2), 16 * (i & 3), home, q_rows, 64);
  }
}


// Decode rows, a key head's G query heads as a tile's rows; partials by CD-key chunk. P: rows, p0, cap, chunks.
constant constexpr int CD = 256;

[[max_total_threads_per_threadgroup(128)]]
[[kernel]] void gemma_decode_attention(const device bfloat* Q [[buffer(0)]], const device bfloat* K [[buffer(1)]],
                                       const device bfloat* V [[buffer(2)]], constant int4& P [[buffer(3)]],
                                       device float* PM [[buffer(4)]], device float* PL [[buffer(5)]],
                                       device float* PO [[buffer(6)]], uint sg [[simdgroup_index_in_threadgroup]],
                                       uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr int RPT = 16 / G; // decode rows a tile
  const int R = P.x, p0 = P.y, cap = P.z, NCH = P.w;
  const int c = int(tg.x), kv = int(tg.y), r0 = int(tg.z) * RPT;
  const int qs = NQ * DH;
  const short2 home = frag_home(ushort(lane));
  // tile row t: decode row r0 + t / G, query head kv * G + t % G; the fragment's two row halves are two decode rows
  const int ra = r0 + home.y / G, rb = r0 + (home.y + 8) / G;
  const bool la = ra < R, lb = rb < R;
  const device bfloat* qa = Q + ra * qs + (kv * G + home.y % G) * DH + int(sg) * 128;
  const device bfloat* qb = Q + rb * qs + (kv * G + (home.y + 8) % G) * DH + int(sg) * 128;
  frag<bfloat> qf[8];
  TF_UNROLL
  for (short d = 0; d < 8; d++) frag_get_rows(qf[d], qa, qb, 16 * d, home, la, lb);
  const device bfloat* kc = K + long(kv) * cap * DH + int(sg) * 128;
  const device bfloat* vc = V + long(kv) * cap * DH + int(sg) * 128;
  frag<float> acc[8];
  TF_UNROLL
  for (short i = 0; i < 8; i++) acc[i] = frag<float>(0);
  float top[2] = {kMasked, kMasked}, total[2] = {0.0f, 0.0f};
  const int pos_a = p0 + ra, pos_b = p0 + rb;
  const int last = min(p0 + R - 1, (c + 1) * CD - 1);
  threadgroup float part[SL][32][16];
  for (int kpos = c * CD; kpos <= last; kpos += BK) {
    const device bfloat* kblk = kc + long(kpos) * DH;
    const device bfloat* vblk = vc + long(kpos) * DH;
    frag<float> s[2] = {frag<float>(0), frag<float>(0)};
    TF_UNROLL
    for (short d = 0; d < 8; d++) {
      frag<bfloat> k0, k1;
      frag_get_t(k0, kblk, DH, 0, 16 * d, home);
      frag_get_t(k1, kblk, DH, 16, 16 * d, home);
      mma_16x32<false, true>(s[0], s[1], qf[d], k0, k1);
    }
    if (SL > 1) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) { part[sg][lane][e] = s[0][e]; part[sg][lane][8 + e] = s[1][e]; }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        float a = part[0][lane][e], b = part[0][lane][8 + e];
        for (int t = 1; t < SL; t++) { a += part[t][lane][e]; b += part[t][lane][8 + e]; }
        s[0][e] = a;
        s[1][e] = b;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    TF_UNROLL
    for (short f = 0; f < 2; f++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        const int col = kpos + 16 * f + home.x + TF_COL(e);
        s[f][e] = col > ((e >> 2) ? pos_b : pos_a) ? kMasked : s[f][e] * 1.44269504089f;
      }
    }
    float top_new[2] = {top[0], top[1]};
    fold<true>(s[0], top_new);
    fold<true>(s[1], top_new);
    TF_UNROLL
    for (short f = 0; f < 2; f++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) s[f][e] = s[f][e] == kMasked ? 0.0f : fast::exp2(s[f][e] - top_new[e >> 2]);
    }
    float keep[2];
    TF_UNROLL
    for (short h = 0; h < 2; h++) {
      keep[h] = top[h] == kMasked ? 0.0f : fast::exp2(top[h] - top_new[h]);
      top[h] = top_new[h];
      total[h] = total[h] * keep[h];
    }
    fold<false>(s[0], total);
    fold<false>(s[1], total);
    TF_UNROLL
    for (short i = 0; i < 8; i++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) acc[i][e] = acc[i][e] * keep[e >> 2];
    }
    TF_UNROLL
    for (short d = 0; d < 8; d += 2) {
      TF_UNROLL
      for (short k = 0; k < 2; k++) {
        frag<bfloat> v0, v1;
        frag_get(v0, vblk, DH, 16 * k, 16 * d, home);
        frag_get(v1, vblk, DH, 16 * k, 16 * d + 16, home);
        mma_16x32<false, false>(acc[d], acc[d + 1], s[k], v0, v1);
      }
    }
  }
  // partials in gemma_attention_merge's layout: max and sum in log2 units back to natural (merge uses exp)
  TF_UNROLL
  for (short h = 0; h < 2; h++) {
    const int r = h ? rb : ra;
    if (!(h ? lb : la)) continue;
    const int qh = kv * G + (home.y + 8 * h) % G;
    const size_t slot = (size_t(qh) * R + r) * NCH + c;
    if (home.x == 0 && sg == 0) { PM[slot] = top[h] == kMasked ? -INFINITY : top[h] * 0.69314718056f; PL[slot] = total[h]; }
    TF_UNROLL
    for (short i = 0; i < 8; i++) {
      TF_UNROLL
      for (short e = 0; e < 4; e++) PO[slot * DH + int(sg) * 128 + 16 * i + home.x + TF_COL(e)] = acc[i][4 * h + e];
    }
  }
}
