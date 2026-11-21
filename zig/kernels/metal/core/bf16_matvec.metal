// Decode glue's bf16 linear: Y = bf16(X W^T) with fp32 sums, kernels P_matvec_ROWS for 1 to 32 rows of X.
#include <metal_stdlib>
using namespace metal;

struct tf_bf16_mv_args { uint rows, n, k, ldy; };

// Y[r, j] = bf16(sum_i X[r, i] * W[j, i]) over bf16 W [n, k] in fp32 (torch's bf16 linear): a simdgroup a row j.
template <uint ROWS>
inline void tf_bf16_matvec_t(const device bfloat* X, const device bfloat* W, constant tf_bf16_mv_args& a,
    device bfloat* Y, uint3 tg, uint sg, uint lane) {
  constexpr uint SG = 8;
  const uint j = tg.x * SG + sg;
  if (j >= a.n) return;
  const device bfloat* w = W + (size_t)j * a.k;
  float acc[ROWS];
  _Pragma("clang loop unroll(full)")
  for (uint r = 0; r < ROWS; ++r) acc[r] = 0.0f;
  for (uint i = 8 * lane; i < a.k; i += 256) {
    const float4 w0 = float4(*(const device bfloat4*)(w + i)), w1 = float4(*(const device bfloat4*)(w + i + 4));
    _Pragma("clang loop unroll(full)")
    for (uint r = 0; r < ROWS; ++r) {
      const device bfloat* x = X + (size_t)r * a.k + i;
      acc[r] += dot(w0, float4(*(const device bfloat4*)x)) + dot(w1, float4(*(const device bfloat4*)(x + 4)));
    }
  }
  _Pragma("clang loop unroll(full)")
  for (uint r = 0; r < ROWS; ++r) {
    const float s = simd_sum(acc[r]);
    if (lane == 0 && r < a.rows) Y[(size_t)r * a.ldy + j] = bfloat(s);
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
