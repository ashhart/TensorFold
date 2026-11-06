// Row-exact 4-bit projections before the M5: 8-row simdgroup-matrix tiles, and a scalar twin for 1-2 rows.
#include <metal_stdlib>
using namespace metal;

// K inputs, N outputs, R rows; one is 1.0 read at run time so the compiler can't fold or reorder the fma sums.
struct TfSqArgs { int K, N, R; float one; };

// The bf16 at index e (0..7) of 8 packed bf16, as fp32.
inline float tf_sq_bf8(uint4 v, int e) {
  const uint w = v[e / 2];
  return as_type<float>((e % 2) ? (w & 0xFFFF0000u) : (w << 16));
}

// A row's 8 inputs summed left to right.
inline float tf_sq_sum8(uint4 v, float one) {
  float t = tf_sq_bf8(v, 0);
  for (int e = 1; e < 8; e++) t = fma(tf_sq_bf8(v, e), one, t);
  return t;
}

// 2^-4s: inputs pre-scaled so a nibble masked in place (code * 2^4s) multiplies to the exact product.
inline float tf_sq_pre(int s) { return as_type<float>(uint(127 - 4 * s) << 23); }

// R rows in 8-row tiles, NT 8-column tiles; K chunks c, c + SGS, ... summed by a fixed tree; loops unrolled (+15-20%).
template <int S, int NT, int RT>
[[kernel]] void tf_sq_mma(const device uint4* X [[buffer(0)]],
                          const device uint* W [[buffer(1)]],
                          const device bfloat* SC [[buffer(2)]],
                          const device bfloat* BI [[buffer(3)]],
                          constant TfSqArgs& a [[buffer(4)]],
                          device bfloat* OUT [[buffer(5)]],
                          uint sgi [[simdgroup_index_in_threadgroup]],
                          uint sgs [[simdgroups_per_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint3 tg [[threadgroup_position_in_grid]]) {
  const int K = a.K, N = a.N, R = a.R, G = K / 64, SGS = int(sgs);
  const float one = a.one;
  const int sg = int(sgi);
  const int qid = int(lane) / 4;
  const int fm = (qid & 4) + ((int(lane) / 2) % 4);
  const int fn = (qid & 2) * 2 + (int(lane) % 2) * 2;
  const int nb = int(tg.x) * (8 * NT);
  const int rb = int(tg.y) * (8 * RT);
  threadgroup float red[S > 1 ? S * RT * NT * 64 : 1];
  const device uint2* W2 = (const device uint2*)W;
  int wrow[NT];
  #pragma unroll
  for (int t = 0; t < NT; t++) wrow[t] = min(nb + 8 * t + fm, N - 1);
  int xr0[RT], xr1[RT];
  #pragma unroll
  for (int rt = 0; rt < RT; rt++) {
    xr0[rt] = min(rb + 8 * rt + fn, R - 1);
    xr1[rt] = min(rb + 8 * rt + fn + 1, R - 1);
  }
  for (int c = sg; c < S; c += SGS) {
    float acc[RT][NT][2];
    #pragma unroll
    for (int rt = 0; rt < RT; rt++)
      #pragma unroll
      for (int t = 0; t < NT; t++) { acc[rt][t][0] = 0.0f; acc[rt][t][1] = 0.0f; }
    for (int g = c; g < G; g += S) {
      uint2 wv[NT];
      #pragma unroll
      for (int t = 0; t < NT; t++) wv[t] = W2[size_t(wrow[t]) * (K / 16) + 4 * g + fn / 2];
      uint4 xa[RT], xb[RT];
      float xs0[RT], xs1[RT];
      #pragma unroll
      for (int rt = 0; rt < RT; rt++) {
        xa[rt] = X[size_t(xr0[rt]) * (K / 8) + 8 * g + fm];
        xb[rt] = X[size_t(xr1[rt]) * (K / 8) + 8 * g + fm];
        float v = tf_sq_sum8(xa[rt], one), u = tf_sq_sum8(xb[rt], one);
        v = fma(simd_shuffle_xor(v, ushort(2)), one, v); u = fma(simd_shuffle_xor(u, ushort(2)), one, u);
        v = fma(simd_shuffle_xor(v, ushort(4)), one, v); u = fma(simd_shuffle_xor(u, ushort(4)), one, u);
        v = fma(simd_shuffle_xor(v, ushort(16)), one, v); u = fma(simd_shuffle_xor(u, ushort(16)), one, u);
        xs0[rt] = v; xs1[rt] = u;
      }
      simdgroup_matrix<float, 8, 8> P[RT][NT];
      #pragma unroll
      for (int rt = 0; rt < RT; rt++)
        #pragma unroll
        for (int t = 0; t < NT; t++) P[rt][t] = simdgroup_matrix<float, 8, 8>(0.0f);
      #pragma unroll
      for (int s = 0; s < 8; s++) {
        const float ps = tf_sq_pre(s);
        const uint mask = 0xFu << (4 * s);
        simdgroup_matrix<float, 8, 8> bm[RT];
        #pragma unroll
        for (int rt = 0; rt < RT; rt++) {
          bm[rt].thread_elements()[0] = tf_sq_bf8(xa[rt], s) * ps;
          bm[rt].thread_elements()[1] = tf_sq_bf8(xb[rt], s) * ps;
        }
        #pragma unroll
        for (int t = 0; t < NT; t++) {
          simdgroup_matrix<float, 8, 8> am;
          am.thread_elements()[0] = float(wv[t].x & mask);
          am.thread_elements()[1] = float(wv[t].y & mask);
          #pragma unroll
          for (int rt = 0; rt < RT; rt++) simdgroup_multiply_accumulate(P[rt][t], am, bm[rt], P[rt][t]);
        }
      }
      #pragma unroll
      for (int t = 0; t < NT; t++) {
        const float sc = float(SC[size_t(wrow[t]) * G + g]);
        const float bi = float(BI[size_t(wrow[t]) * G + g]);
        #pragma unroll
        for (int rt = 0; rt < RT; rt++) {
          acc[rt][t][0] = fma(bi, xs0[rt], fma(sc, P[rt][t].thread_elements()[0], acc[rt][t][0]));
          acc[rt][t][1] = fma(bi, xs1[rt], fma(sc, P[rt][t].thread_elements()[1], acc[rt][t][1]));
        }
      }
    }
    if (S == 1) {
      #pragma unroll
      for (int rt = 0; rt < RT; rt++)
        #pragma unroll
        for (int t = 0; t < NT; t++)
          #pragma unroll
          for (int e = 0; e < 2; e++) {
            const int row = rb + 8 * rt + fn + e, n = nb + 8 * t + fm;
            if (row < R && n < N) OUT[size_t(row) * N + n] = bfloat(acc[rt][t][e]);
          }
      return;
    }
    #pragma unroll
    for (int rt = 0; rt < RT; rt++)
      #pragma unroll
      for (int t = 0; t < NT; t++)
        #pragma unroll
        for (int e = 0; e < 2; e++) red[((c * RT + rt) * NT + t) * 64 + int(lane) * 2 + e] = acc[rt][t][e];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int idx = sg * 32 + int(lane); idx < RT * NT * 64; idx += SGS * 32) {
    float v[S];
    #pragma unroll
    for (int k = 0; k < S; k++) v[k] = red[k * (RT * NT * 64) + idx];
    #pragma unroll
    for (int w = 1; w < S; w *= 2)
      #pragma unroll
      for (int k = 0; k + w < S; k += 2 * w) v[k] = fma(v[k + w], one, v[k]);
    const int rt = idx / (NT * 64), t = (idx / 64) % NT, l = (idx % 64) / 2, e = idx % 2;
    const int lq = l / 4;
    const int row = rb + 8 * rt + (lq & 2) * 2 + (l % 2) * 2 + e, n = nb + 8 * t + (lq & 4) + ((l / 2) % 4);
    if (row < R && n < N) OUT[size_t(row) * N + n] = bfloat(v[0]);
  }
}

#define TF_SQ_MMA(S, NT, RT) template [[host_name("tf_sq_mma_" #S "_" #NT "_" #RT)]] [[kernel]] decltype(tf_sq_mma<S, NT, RT>) tf_sq_mma<S, NT, RT>;
TF_SQ_MMA(8, 1, 1) TF_SQ_MMA(8, 2, 1) TF_SQ_MMA(8, 4, 1) TF_SQ_MMA(8, 1, 2) TF_SQ_MMA(8, 2, 2) TF_SQ_MMA(8, 4, 2)
TF_SQ_MMA(16, 1, 1) TF_SQ_MMA(16, 2, 1) TF_SQ_MMA(16, 4, 1) TF_SQ_MMA(16, 1, 2) TF_SQ_MMA(16, 2, 2)
TF_SQ_MMA(32, 1, 1) TF_SQ_MMA(32, 2, 1) TF_SQ_MMA(32, 1, 2)
