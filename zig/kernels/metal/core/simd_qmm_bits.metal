// Row-exact 2- to 8-bit projections before the M5 on simd_qmm tiles, rows at a stride; 4-bit gives simd_qmm sums.
#include <metal_stdlib>
using namespace metal;

// K inputs, N outputs, R rows, OUT's row pitch LD; one is 1.0 read at run time so the compiler can't fold the sums.
struct TfSqwArgs { int K, N, R, LD; float one; };

// The bf16 at index e (0..7) of 8 packed bf16, as fp32.
inline float tf_sqw_bf8(uint4 v, int e) {
  const uint w = v[e / 2];
  return as_type<float>((e % 2) ? (w & 0xFFFF0000u) : (w << 16));
}

// A row's 8 inputs summed left to right.
inline float tf_sqw_sum8(uint4 v, float one) {
  float t = tf_sqw_bf8(v, 0);
  for (int e = 1; e < 8; e++) t = fma(tf_sqw_bf8(v, e), one, t);
  return t;
}

// 2^-4s: 4-bit inputs pre-scaled so a nibble masked in place (code * 2^4s) multiplies to the exact product.
inline float tf_sqw_pre(int s) { return as_type<float>(uint(127 - 4 * s) << 23); }

// 2^-(2s + 16h): 2-bit inputs pre-scaled for codes masked in place in a word's low (h = 0) or high (h = 1) half.
inline float tf_sqw_pre2(int s, int h) { return as_type<float>(uint(127 - 2 * s - 16 * h) << 23); }

// A lane's 16 codes of a group row (2 * BITS bytes, little-endian bits) as 32-bit words; 4-bit loads them as simd_qmm.
template <int BITS>
inline void tf_sqw_load(const device uchar* p, thread uint* u) {
  if (BITS == 2) {
    u[0] = *(const device uint*)p;
  } else if (BITS == 4) {
    const uint2 v = *(const device uint2*)p;
    u[0] = v.x; u[1] = v.y;
  } else if (BITS == 6) {
    const device uint* q = (const device uint*)p;
    u[0] = q[0]; u[1] = q[1]; u[2] = q[2];
  } else if (BITS == 8) {
    const uint4 v = *(const device uint4*)p;
    u[0] = v.x; u[1] = v.y; u[2] = v.z; u[3] = v.w;
  } else {
    const device ushort* q = (const device ushort*)p;
    for (int j = 0; j < BITS / 2; j++) u[j] = uint(q[2 * j]) | (uint(q[2 * j + 1]) << 16);
    u[BITS / 2] = uint(q[BITS - 1]);
  }
}

// Code i (0..15) of the words, exact as a float.
template <int BITS>
inline float tf_sqw_code(const thread uint* u, int i) {
  const int at = BITS * i, w = at / 32, off = at % 32;
  uint v = u[w] >> off;
  if (off + BITS > 32) v |= u[w + 1] << (32 - off);
  return float(v & ((1u << BITS) - 1u));
}

// R rows in 8-row tiles, NT 8-column tiles; simdgroups take K chunks c, c + SGS, ..., summed by a fixed tree.
template <int BITS, int S, int NT, int RT>
[[kernel]] void tf_sqw_mma(const device uint4* X [[buffer(0)]],
                           const device uchar* W [[buffer(1)]],
                           const device bfloat* SC [[buffer(2)]],
                           const device bfloat* BI [[buffer(3)]],
                           constant TfSqwArgs& a [[buffer(4)]],
                           device bfloat* OUT [[buffer(5)]],
                           uint sgi [[simdgroup_index_in_threadgroup]],
                           uint sgs [[simdgroups_per_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint3 tg [[threadgroup_position_in_grid]]) {
  const int K = a.K, N = a.N, R = a.R, LD = a.LD, G = K / 64, SGS = int(sgs);
  const float one = a.one;
  const int sg = int(sgi);
  const int qid = int(lane) / 4;
  const int fm = (qid & 4) + ((int(lane) / 2) % 4);
  const int fn = (qid & 2) * 2 + (int(lane) % 2) * 2;
  const int nb = int(tg.x) * (8 * NT);
  const int rb = int(tg.y) * (8 * RT);
  threadgroup float red[S > 1 ? S * RT * NT * 64 : 1];
  const size_t RB = size_t(K) * BITS / 8; // a weight row's bytes
  int wrow[NT];
  for (int t = 0; t < NT; t++) wrow[t] = min(nb + 8 * t + fm, N - 1);
  int xr0[RT], xr1[RT];
  for (int rt = 0; rt < RT; rt++) {
    xr0[rt] = min(rb + 8 * rt + fn, R - 1);
    xr1[rt] = min(rb + 8 * rt + fn + 1, R - 1);
  }
  for (int c = sg; c < S; c += SGS) {
    float acc[RT][NT][2];
    for (int rt = 0; rt < RT; rt++)
      for (int t = 0; t < NT; t++) { acc[rt][t][0] = 0.0f; acc[rt][t][1] = 0.0f; }
    for (int g = c; g < G; g += S) {
      uint wv[NT][(BITS + 1) / 2 + 1];
      for (int t = 0; t < NT; t++) tf_sqw_load<BITS>(W + size_t(wrow[t]) * RB + g * (8 * BITS) + (fn / 2) * (2 * BITS), wv[t]);
      uint4 xa[RT], xb[RT];
      float xs0[RT], xs1[RT];
      for (int rt = 0; rt < RT; rt++) {
        xa[rt] = X[size_t(xr0[rt]) * (K / 8) + 8 * g + fm];
        xb[rt] = X[size_t(xr1[rt]) * (K / 8) + 8 * g + fm];
        float v = tf_sqw_sum8(xa[rt], one), u = tf_sqw_sum8(xb[rt], one);
        v = fma(simd_shuffle_xor(v, ushort(2)), one, v); u = fma(simd_shuffle_xor(u, ushort(2)), one, u);
        v = fma(simd_shuffle_xor(v, ushort(4)), one, v); u = fma(simd_shuffle_xor(u, ushort(4)), one, u);
        v = fma(simd_shuffle_xor(v, ushort(16)), one, v); u = fma(simd_shuffle_xor(u, ushort(16)), one, u);
        xs0[rt] = v; xs1[rt] = u;
      }
      simdgroup_matrix<float, 8, 8> P[RT][NT];
      for (int rt = 0; rt < RT; rt++)
        for (int t = 0; t < NT; t++) P[rt][t] = simdgroup_matrix<float, 8, 8>(0.0f);
      #pragma clang loop unroll(full)
      for (int s = 0; s < 8; s++) {
        const float ps = BITS == 4 ? tf_sqw_pre(s) : BITS == 2 ? tf_sqw_pre2(s, fm & 1) : 1.0f;
        simdgroup_matrix<float, 8, 8> bm[RT];
        for (int rt = 0; rt < RT; rt++) {
          bm[rt].thread_elements()[0] = BITS <= 4 ? tf_sqw_bf8(xa[rt], s) * ps : tf_sqw_bf8(xa[rt], s);
          bm[rt].thread_elements()[1] = BITS <= 4 ? tf_sqw_bf8(xb[rt], s) * ps : tf_sqw_bf8(xb[rt], s);
        }
        for (int t = 0; t < NT; t++) {
          simdgroup_matrix<float, 8, 8> am;
          if (BITS == 4) {
            const uint mask = 0xFu << (4 * s);
            am.thread_elements()[0] = float(wv[t][0] & mask);
            am.thread_elements()[1] = float(wv[t][1] & mask);
          } else if (BITS == 2) { // codes 8fn + s and 8fn + 8 + s in place: scaled 2^2s and 2^(16 + 2s)
            am.thread_elements()[0] = float(wv[t][0] & (3u << (2 * s)));
            am.thread_elements()[1] = float(wv[t][0] & (3u << (16 + 2 * s)));
          } else {
            am.thread_elements()[0] = tf_sqw_code<BITS>(wv[t], s);
            am.thread_elements()[1] = tf_sqw_code<BITS>(wv[t], 8 + s);
          }
          for (int rt = 0; rt < RT; rt++) simdgroup_multiply_accumulate(P[rt][t], am, bm[rt], P[rt][t]);
        }
      }
      for (int t = 0; t < NT; t++) {
        const float sc = float(SC[size_t(wrow[t]) * G + g]);
        const float bi = float(BI[size_t(wrow[t]) * G + g]);
        for (int rt = 0; rt < RT; rt++) {
          acc[rt][t][0] = fma(bi, xs0[rt], fma(sc, P[rt][t].thread_elements()[0], acc[rt][t][0]));
          acc[rt][t][1] = fma(bi, xs1[rt], fma(sc, P[rt][t].thread_elements()[1], acc[rt][t][1]));
        }
      }
    }
    if (S == 1) {
      for (int rt = 0; rt < RT; rt++)
        for (int t = 0; t < NT; t++)
          for (int e = 0; e < 2; e++) {
            const int row = rb + 8 * rt + fn + e, n = nb + 8 * t + fm;
            if (row < R && n < N) OUT[size_t(row) * LD + n] = bfloat(acc[rt][t][e]);
          }
      return;
    }
    for (int rt = 0; rt < RT; rt++)
      for (int t = 0; t < NT; t++)
        for (int e = 0; e < 2; e++) red[((c * RT + rt) * NT + t) * 64 + int(lane) * 2 + e] = acc[rt][t][e];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int idx = sg * 32 + int(lane); idx < RT * NT * 64; idx += SGS * 32) {
    float v[S];
    for (int k = 0; k < S; k++) v[k] = red[k * (RT * NT * 64) + idx];
    for (int w = 1; w < S; w *= 2)
      for (int k = 0; k + w < S; k += 2 * w) v[k] = fma(v[k + w], one, v[k]);
    const int rt = idx / (NT * 64), t = (idx / 64) % NT, l = (idx % 64) / 2, e = idx % 2;
    const int lq = l / 4;
    const int row = rb + 8 * rt + (lq & 2) * 2 + (l % 2) * 2 + e, n = nb + 8 * t + (lq & 4) + ((l / 2) % 4);
    if (row < R && n < N) OUT[size_t(row) * LD + n] = bfloat(v[0]);
  }
}

// A row group's 64 codes (8 * BITS bytes) as 2 * BITS words, loaded 8 bytes at a time.
template <int BITS>
inline void tf_sqw_group(const device uchar* p, thread uint* u) {
  const device uint2* q = (const device uint2*)p;
  #pragma clang loop unroll(full)
  for (int j = 0; j < BITS; j++) {
    const uint2 v = q[j];
    u[2 * j] = v.x; u[2 * j + 1] = v.y;
  }
}

// 1-2 rows over the same weight loads: each lane runs one chunk of NR outputs in the MMA kernel's order.
template <int BITS, int S, int NR, int XB, int RS>
[[kernel]] void tf_sqw_scalar(const device uint4* X [[buffer(0)]],
                              const device uchar* W [[buffer(1)]],
                              const device bfloat* SC [[buffer(2)]],
                              const device bfloat* BI [[buffer(3)]],
                              constant TfSqwArgs& a [[buffer(4)]],
                              device bfloat* OUT [[buffer(5)]],
                              uint sgi [[simdgroup_index_in_threadgroup]],
                              uint sgs [[simdgroups_per_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr int XP = 76;  // floats a staged group: 64 inputs, 8 word sums, pad (bank spread)
  constexpr int SLOTS = 32 / S;
  constexpr int WW = 2 * BITS; // words a row group
  threadgroup float xs[RS * XB * XP];
  const int K = a.K, N = a.N, LD = a.LD, G = K / 64, SGS = int(sgs);
  const float one = a.one;
  const int tid = int(sgi) * 32 + int(lane);
  const int c = int(lane) % S;
  const int n0 = (int(tg.x) * SGS + int(sgi)) * (SLOTS * NR) + int(lane) / S;
  const size_t RB = size_t(K) * BITS / 8;
  const device uchar* wr[NR];
  const device bfloat* sr[NR];
  const device bfloat* br[NR];
  float acc[NR][RS];
  for (int u = 0; u < NR; u++) {
    const int nn = min(n0 + SLOTS * u, N - 1);
    wr[u] = W + size_t(nn) * RB;
    sr[u] = SC + size_t(nn) * G;
    br[u] = BI + size_t(nn) * G;
    for (int r = 0; r < RS; r++) acc[u][r] = 0.0f;
  }
  uint nw[NR][WW];
  for (int u = 0; u < NR; u++) {
    if (c < G) tf_sqw_group<BITS>(wr[u] + c * (8 * BITS), nw[u]);
    else for (int j = 0; j < WW; j++) nw[u][j] = 0;
  }
  for (int b0 = 0; b0 < G; b0 += XB) {
    const int nbk = min(XB, G - b0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int idx = tid; idx < RS * nbk * 8; idx += SGS * 32) {
      const int r = RS == 1 ? 0 : idx / (nbk * 8);
      const int gl = (RS == 1 ? idx : idx - r * (nbk * 8)) / 8, j = idx % 8;
      const uint4 v = X[size_t(r) * (K / 8) + 8 * (b0 + gl) + j];
      threadgroup float* xr = xs + r * (XB * XP) + gl * XP;
      for (int e = 0; e < 8; e++) xr[8 * e + j] = BITS == 4 ? tf_sqw_bf8(v, e) * tf_sqw_pre(e) : BITS == 2 ? tf_sqw_bf8(v, e) * tf_sqw_pre2(e, j & 1) : tf_sqw_bf8(v, e);
      xr[64 + j] = tf_sqw_sum8(v, one);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int g = b0 + c; g < b0 + nbk; g += S) {
      uint wv[NR][WW];
      for (int u = 0; u < NR; u++)
        for (int j = 0; j < WW; j++) wv[u][j] = nw[u][j];
      if (g + S < G)
        for (int u = 0; u < NR; u++) tf_sqw_group<BITS>(wr[u] + (g + S) * (8 * BITS), nw[u]);
      float xsum[RS];
      float P[NR][RS];
      for (int r = 0; r < RS; r++) {
        const threadgroup float* xg = xs + r * (XB * XP) + (g - b0) * XP;
        const float4 p0 = *(const threadgroup float4*)(xg + 64);
        const float4 p1 = *(const threadgroup float4*)(xg + 68);
        xsum[r] = fma(fma(p0.w, one, p0.z), one, fma(p0.y, one, p0.x));
        xsum[r] = fma(fma(fma(p1.w, one, p1.z), one, fma(p1.y, one, p1.x)), one, xsum[r]);
        for (int u = 0; u < NR; u++) P[u][r] = 0.0f;
      }
      #pragma clang loop unroll(full)
      for (int s = 0; s < 8; s++) {
        float xq[RS][8];
        for (int r = 0; r < RS; r++) {
          const threadgroup float* xg = xs + r * (XB * XP) + (g - b0) * XP + 8 * s;
          const float4 lo = *(const threadgroup float4*)(xg), hi = *(const threadgroup float4*)(xg + 4);
          xq[r][0] = lo.x; xq[r][1] = lo.y; xq[r][2] = lo.z; xq[r][3] = lo.w;
          xq[r][4] = hi.x; xq[r][5] = hi.y; xq[r][6] = hi.z; xq[r][7] = hi.w;
        }
        for (int u = 0; u < NR; u++)
          #pragma clang loop unroll(full)
          for (int i = 0; i < 8; i++) {
            const float q = BITS == 4 ? float(wv[u][i] & (0xFu << (4 * s))) : BITS == 2 ? float(wv[u][i / 2] & (3u << (16 * (i & 1) + 2 * s))) : tf_sqw_code<BITS>(wv[u], 8 * i + s);
            for (int r = 0; r < RS; r++) P[u][r] = fma(xq[r][i], q, P[u][r]);
          }
      }
      for (int u = 0; u < NR; u++) {
        const float sc = float(sr[u][g]), bi = float(br[u][g]);
        for (int r = 0; r < RS; r++) {
          acc[u][r] = fma(sc, P[u][r], acc[u][r]);
          acc[u][r] = fma(bi, xsum[r], acc[u][r]);
        }
      }
    }
  }
  for (int u = 0; u < NR; u++)
    for (int r = 0; r < RS; r++) {
      float v = acc[u][r];
      for (int m = 1; m < S; m <<= 1) v = fma(simd_shuffle_xor(v, ushort(m)), one, v);
      const int n = n0 + SLOTS * u;
      if (n < N && c == 0) OUT[size_t(r) * LD + n] = bfloat(v);
    }
}

#define TF_SQW(B, S, NT, RT) template [[host_name("tf_sqw_mma_q" #B "_" #S "_" #NT "_" #RT)]] [[kernel]] decltype(tf_sqw_mma<B, S, NT, RT>) tf_sqw_mma<B, S, NT, RT>;
#define TF_SQW_BITS(B) \
  TF_SQW(B, 8, 1, 1) TF_SQW(B, 8, 2, 1) TF_SQW(B, 8, 4, 1) TF_SQW(B, 8, 1, 2) TF_SQW(B, 8, 2, 2) TF_SQW(B, 8, 4, 2) \
  TF_SQW(B, 16, 1, 1) TF_SQW(B, 16, 2, 1) TF_SQW(B, 16, 4, 1) TF_SQW(B, 16, 1, 2) TF_SQW(B, 16, 2, 2) \
  TF_SQW(B, 32, 1, 1) TF_SQW(B, 32, 2, 1) TF_SQW(B, 32, 1, 2)
TF_SQW_BITS(2) TF_SQW_BITS(3) TF_SQW_BITS(4) TF_SQW_BITS(5) TF_SQW_BITS(6) TF_SQW_BITS(8)
#define TF_SQWS(B, S, NR, XB, RS) template [[host_name("tf_sqw_scalar_q" #B "_" #S "_" #NR "_" #RS)]] [[kernel]] decltype(tf_sqw_scalar<B, S, NR, XB, RS>) tf_sqw_scalar<B, S, NR, XB, RS>;
#define TF_SQWS_BITS(B) \
  TF_SQWS(B, 8, 1, 32, 1) TF_SQWS(B, 8, 2, 32, 1) TF_SQWS(B, 8, 1, 16, 2) TF_SQWS(B, 8, 2, 16, 2) \
  TF_SQWS(B, 16, 1, 32, 1) TF_SQWS(B, 16, 2, 32, 1) TF_SQWS(B, 16, 1, 16, 2) TF_SQWS(B, 16, 2, 16, 2) \
  TF_SQWS(B, 32, 1, 32, 1) TF_SQWS(B, 32, 2, 32, 1) TF_SQWS(B, 32, 1, 32, 2) TF_SQWS(B, 32, 2, 32, 2)
TF_SQWS_BITS(2) TF_SQWS_BITS(3) TF_SQWS_BITS(4) TF_SQWS_BITS(5) TF_SQWS_BITS(6) TF_SQWS_BITS(8)
