// Scalar GLM epilogues shared by their standalone rows and their TP2 RoCE fusion.
#pragma once
#include "glm_triton.cuh"

namespace glm_post {
using namespace glm_triton;

__device__ __forceinline__ float add(float rank0, float rank1) { return rank0 + rank1; }
__device__ __forceinline__ float residual(float x, float sum) {
  return x + rbf(sum);
}
__device__ __forceinline__ float hc(float x0, float x1, float x2, float x3,
                                     float c0, float c1, float c2, float c3, float p, float branch) {
  float m = __fmul_rn(x1, c1);
  m = __fmaf_rn(x0, c0, m);
  m = __fmaf_rn(x2, c2, m);
  m = __fmaf_rn(x3, c3, m);
  return __fmaf_rn(rbf(branch), p, m);
}
__device__ __forceinline__ void residual4(const uint4 rank0, const uint4 rank1,
                                          const __nv_bfloat16* x, __nv_bfloat16* out) {
  const uint32_t a[4] = {rank0.x, rank0.y, rank0.z, rank0.w};
  const uint32_t b[4] = {rank1.x, rank1.y, rank1.z, rank1.w};
  for (int j = 0; j < 4; ++j)
    out[j] = __float2bfloat16_rn(residual(bf(x[j]), add(__uint_as_float(a[j]), __uint_as_float(b[j]))));
}
__device__ __forceinline__ void hc4(const uint4 rank0, const uint4 rank1, const __nv_bfloat16* x,
                                    __nv_bfloat16* out, int64_t d, int s, const float* post, const float* comb) {
  const uint32_t a[4] = {rank0.x, rank0.y, rank0.z, rank0.w};
  const uint32_t b[4] = {rank1.x, rank1.y, rank1.z, rank1.w};
  const float c0 = comb[s], c1 = comb[4 + s], c2 = comb[8 + s], c3 = comb[12 + s];
  const float p = post[s];
  for (int j = 0; j < 4; ++j)
    out[j] = __float2bfloat16_rn(hc(bf(x[j]), bf(x[d + j]), bf(x[2 * d + j]), bf(x[3 * d + j]),
                                    c0, c1, c2, c3, p, add(__uint_as_float(a[j]), __uint_as_float(b[j]))));
}

}  // namespace glm_post
