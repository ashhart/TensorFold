// Gemma 4 prompt attention on the tensor units: 16 query rows of a head a threadgroup, the head in 128-wide slices.
#include "../nax.h"
using namespace tfp;

// Compiled with DH (head dim), NQ, G (query heads a key head), SL (= DH / 128), WINDOW (0: global), RING (0: linear).

constant constexpr float kMasked = -3.402823466e+38f;
constant constexpr int BK = 32;

// Row max (MAX) or sum over a fragment's 32 columns into r[2] (rows home.y and home.y + 8), as attention_nax folds.
template <bool MAX>
inline void fold(thread const frag<float>& f, thread float (&r)[2]) {
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

// Q [R, NQ, DH] at positions p0..; K, V the cache [HK, cap, DH]: slice scores add in order, each keeps its outputs.
[[max_total_threads_per_threadgroup(128)]]
[[kernel]] void gemma_prompt_attention(const device bfloat* Q [[buffer(0)]], const device bfloat* K [[buffer(1)]],
                                       const device bfloat* V [[buffer(2)]], device bfloat* O [[buffer(3)]],
                                       constant int3& P [[buffer(4)]], uint sg [[simdgroup_index_in_threadgroup]],
                                       uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  const int R = P.x, p0 = P.y, cap = P.z;
  const int qh = int(tg.y), kv = qh / G;
  const int r0 = int(tg.x) * 16;
  const int q_rows = min(16, R - r0);
  const int qs = NQ * DH;
  const device bfloat* q = Q + r0 * qs + qh * DH + int(sg) * 128;
  const device bfloat* kc = K + long(kv) * cap * DH + int(sg) * 128;
  const device bfloat* vc = V + long(kv) * cap * DH + int(sg) * 128;
  const short2 home = frag_home(ushort(lane));
  frag<bfloat> qf[8];
  TF_UNROLL
  for (short d = 0; d < 8; d++) frag_get_in(qf[d], q, qs, 0, 16 * d, home, q_rows, 128);
  frag<float> acc[8];
  TF_UNROLL
  for (short i = 0; i < 8; i++) acc[i] = frag<float>(0);
  float top[2] = {kMasked, kMasked}, total[2] = {0.0f, 0.0f};
  const int first_pos = p0 + r0, last_pos = p0 + r0 + q_rows - 1;
  const int lo = WINDOW ? max(0, first_pos - WINDOW + 1) : 0;
  threadgroup float part[SL][32][16];
  for (int kb = lo / BK; kb <= last_pos / BK; kb++) {
    const int kpos = kb * BK;
    const int slot = RING ? kpos % RING : kpos;
    const device bfloat* kblk = kc + long(slot) * DH;
    const device bfloat* vblk = vc + long(slot) * DH;
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
        const int row = first_pos + home.y + (e >> 2) * 8, col = kpos + 16 * f + home.x + TF_COL(e);
        const bool out = col > row || (WINDOW && col < row - WINDOW + 1);
        s[f][e] = out ? kMasked : s[f][e] * 1.44269504089f;
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
  float inv[2];
  TF_UNROLL
  for (short h = 0; h < 2; h++) inv[h] = total[h] > 0.0f ? 1.0f / total[h] : 0.0f;
  device bfloat* o = O + r0 * qs + qh * DH + int(sg) * 128;
  TF_UNROLL
  for (short i = 0; i < 8; i++) {
    TF_UNROLL
    for (short e = 0; e < 8; e++) acc[i][e] = acc[i][e] * inv[e >> 2];
    frag_put_in(acc[i], o, qs, 0, 16 * i, home, q_rows, 128);
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
