// Decode glue's bf16 linear: Y = bf16(X W^T) with fp32 sums, kernels P_matvec_ROWS for 1 to 32 rows of X.
#include <metal_stdlib>
using namespace metal;

struct tf_bf16_mv_args { uint rows, n, k, ldy; };

// Row-chunk helper for bf16 linear: FULL means active >= ROWS so r never exceeds X row bounds.
template <uint ROWS, bool FULL>
inline void tf_bf16_matvec_chunk(const device bfloat* X, const device bfloat* w, constant tf_bf16_mv_args& a,
    device bfloat* Y, uint j, uint lane, uint base, uint active) {
  float acc[ROWS];
  _Pragma("clang loop unroll(full)")
  for (uint r = 0; r < ROWS; ++r) acc[r] = 0.0f;
  for (uint i = 8 * lane; i < a.k; i += 256) {
    const float4 w0 = float4(*(const device bfloat4*)(w + i)), w1 = float4(*(const device bfloat4*)(w + i + 4));
    _Pragma("clang loop unroll(full)")
    for (uint r = 0; r < ROWS; ++r) {
      const uint rr = FULL ? r : min(r, active - 1u);
      const device bfloat* x = X + (size_t)(base + rr) * a.k + i;
      acc[r] += dot(w0, float4(*(const device bfloat4*)x)) + dot(w1, float4(*(const device bfloat4*)(x + 4)));
    }
  }
  _Pragma("clang loop unroll(full)")
  for (uint r = 0; r < ROWS; ++r) {
    const float s = simd_sum(acc[r]);
    if (lane == 0 && r < active) Y[(size_t)(base + r) * a.ldy + j] = bfloat(s);
  }
}

// Y[r, j] = bf16(sum_i X[r, i] * W[j, i]) over bf16 W [n, k] in fp32 (torch's bf16 linear): a simdgroup a row j.
template <uint ROWS>
inline void tf_bf16_matvec_t(const device bfloat* X, const device bfloat* W, constant tf_bf16_mv_args& a,
    device bfloat* Y, uint3 tg, uint sg, uint lane) {
  constexpr uint SG = 8;
  const uint j = tg.x * SG + sg;
  if (a.rows == 0 || j >= a.n) return;
  const device bfloat* w = W + (size_t)j * a.k;
  if (ROWS <= 16) {
    if (a.rows >= ROWS) tf_bf16_matvec_chunk<ROWS, true>(X, w, a, Y, j, lane, 0, ROWS);
    else tf_bf16_matvec_chunk<ROWS, false>(X, w, a, Y, j, lane, 0, a.rows);
    return;
  }
  // Beyond 16 rows, acc[32] spills on Apple GPUs; tile in <=16-row chunks with W kept hot.
  const uint rows = min(a.rows, ROWS);
  for (uint base = 0; base < rows; base += 16) {
    const uint left = rows - base;
    if (left >= 16) tf_bf16_matvec_chunk<16, true>(X, w, a, Y, j, lane, base, 16);
    else if (left <= 1) tf_bf16_matvec_chunk<1, true>(X, w, a, Y, j, lane, base, 1);
    else if (left <= 2) tf_bf16_matvec_chunk<2, false>(X, w, a, Y, j, lane, base, left);
    else if (left <= 4) tf_bf16_matvec_chunk<4, false>(X, w, a, Y, j, lane, base, left);
    else if (left <= 8) tf_bf16_matvec_chunk<8, false>(X, w, a, Y, j, lane, base, left);
    else tf_bf16_matvec_chunk<16, false>(X, w, a, Y, j, lane, base, left);
  }
}

#define TF_BF16_MATVEC(NAME, ROWS)                                                                                 \
  [[kernel]] void NAME(const device bfloat* X [[buffer(0)]], const device bfloat* W [[buffer(1)]],                 \
      constant tf_bf16_mv_args& a [[buffer(2)]], device bfloat* Y [[buffer(3)]],                                   \
      uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],                      \
      uint lane [[thread_index_in_simdgroup]]) {                                                                   \
    tf_bf16_matvec_t<ROWS>(X, W, a, Y, tg, sg, lane);                                                              \
  }
#define TF_BF16_MATVECS(P)                                                                                         \
  TF_BF16_MATVEC(P##_matvec_1, 1)                                                                                  \
  TF_BF16_MATVEC(P##_matvec_2, 2)                                                                                  \
  TF_BF16_MATVEC(P##_matvec_4, 4)                                                                                  \
  TF_BF16_MATVEC(P##_matvec_8, 8)                                                                                  \
  TF_BF16_MATVEC(P##_matvec_16, 16)                                                                                \
  TF_BF16_MATVEC(P##_matvec_32, 32)
