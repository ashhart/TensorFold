// DeepSeek V4.1 small matvecs: an output is 32 lane chains of FMAs over K in steps of 32, then one butterfly.
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

#ifndef TF_CTL
#define TF_CTL 0
#endif

struct TfDsv41MvArgs {
  uint k, n, rows;
};

inline float tf_dsv41_mv_x(bfloat v) {
  const float f = float(v);
  return (TF_CTL == 1 && !isfinite(f)) ? 0.0f : f;
}

inline float tf_dsv41_mv_step(float acc, float x, float w) {
  const float r = fma(x, w, acc);
  return TF_CTL == 3 ? float(half(r)) : r;
}

inline float tf_dsv41_mv_tree(float v) {
  for (ushort m = 1; m < 32; m *= 2) v += simd_shuffle_xor(v, m);
  return v;
}

// A simdgroup: NB columns by RT rows, one weight load shared by the rows; the loads of U steps are issued together.
template <typename W, typename T, uint RT, uint NB, uint U>
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_dsv41_mv(
    const device bfloat* X [[buffer(0)]], const device W* Wt [[buffer(1)]], device T* Y [[buffer(2)]],
    constant TfDsv41MvArgs& a [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint n0 = tg.x * NB, r0 = tg.y * RT;
  const uint cols = min(NB, a.n - n0), live = min(RT, a.rows - r0);
  float acc[NB][RT];
  for (uint nb = 0; nb < NB; ++nb)
    for (uint r = 0; r < RT; ++r) acc[nb][r] = 0.0f;
  uint i = lane;
  if (TF_CTL != 2) {
    for (; i + 32u * (U - 1) < a.k; i += 32u * U) {
      float w[U][NB];
      for (uint u = 0; u < U; ++u)
        for (uint nb = 0; nb < NB; ++nb) w[u][nb] = nb < cols ? float(Wt[ulong(n0 + nb) * a.k + i + 32u * u]) : 0.0f;
      for (uint r = 0; r < RT; ++r) {
        if (r >= live) break;
        float x[U];
        for (uint u = 0; u < U; ++u) x[u] = tf_dsv41_mv_x(X[ulong(r0 + r) * a.k + i + 32u * u]);
        for (uint u = 0; u < U; ++u)
          for (uint nb = 0; nb < NB; ++nb) acc[nb][r] = tf_dsv41_mv_step(acc[nb][r], x[u], w[u][nb]);
      }
    }
  }
  for (; i < a.k; i += 32u) {
    if (TF_CTL == 2 && i - lane + 32u > a.k) break;
    float w[NB];
    for (uint nb = 0; nb < NB; ++nb) w[nb] = nb < cols ? float(Wt[ulong(n0 + nb) * a.k + i]) : 0.0f;
    for (uint r = 0; r < RT; ++r) {
      if (r >= live) break;
      const float x = tf_dsv41_mv_x(X[ulong(r0 + r) * a.k + i]);
      for (uint nb = 0; nb < NB; ++nb) acc[nb][r] = tf_dsv41_mv_step(acc[nb][r], x, w[nb]);
    }
  }
  for (uint r = 0; r < RT; ++r) {
    if (r >= live) break;
    for (uint nb = 0; nb < NB; ++nb) {
      const float v = tf_dsv41_mv_tree(acc[nb][r]);
      if (lane == 0 && nb < cols) Y[ulong(r0 + r) * a.n + n0 + nb] = T(v);
    }
  }
  if (TF_CTL == 4 && lane == 0 && tg.x == 0 && tg.y == 0) Y[ulong(a.rows) * a.n] = T(1.0f);
}

#define TF_MV(W, WN, T, TN, RT, NB, U) \
  template [[host_name("tf_dsv41_mv_" WN "_" TN "_r" #RT "c" #NB)]] [[kernel]] decltype(tf_dsv41_mv<W, T, RT, NB, U>) \
      tf_dsv41_mv<W, T, RT, NB, U>;
// The launch geometries (rows by columns a simdgroup); each output's lane chains and butterfly are the same in all.
#define TF_MV_TIERS(W, WN, T, TN) \
  TF_MV(W, WN, T, TN, 1, 1, 32)   \
  TF_MV(W, WN, T, TN, 2, 1, 16)   \
  TF_MV(W, WN, T, TN, 4, 1, 16)   \
  TF_MV(W, WN, T, TN, 4, 2, 16)   \
  TF_MV(W, WN, T, TN, 8, 2, 16)   \
  TF_MV(W, WN, T, TN, 16, 2, 16)
TF_MV_TIERS(bfloat, "bf16w", float, "f32o")
TF_MV_TIERS(bfloat, "bf16w", bfloat, "bf16o")
TF_MV_TIERS(float, "f32w", float, "f32o")
TF_MV_TIERS(float, "f32w", bfloat, "bf16o")
