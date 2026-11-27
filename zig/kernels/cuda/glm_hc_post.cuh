// GLM hc_post for one four-column RoCE pack: the served kernel's operand and FMA order.
#pragma once

#include "glm_triton.cuh"

namespace glm_hc {

__device__ __forceinline__ void ld4(const __nv_bfloat16* p, float v[4]) {
  const uint2 u = *reinterpret_cast<const uint2*>(p);
  const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&u);
  for (int j = 0; j < 4; ++j) v[j] = glm_triton::bf(h[j]);
}

__device__ __forceinline__ void st4(__nv_bfloat16* p, const float v[4]) {
  uint2 u;
  __nv_bfloat16* h = reinterpret_cast<__nv_bfloat16*>(&u);
  for (int j = 0; j < 4; ++j) h[j] = __float2bfloat16_rn(v[j]);
  *reinterpret_cast<uint2*>(p) = u;
}

// i is a four-column offset in one rank's rows-by-d shard; branch is summed rank 0 first.
__device__ __forceinline__ void post4(const __nv_bfloat16* x, __nv_bfloat16* xout, const float* post,
                                      const float* comb, int64_t d, int d_shift, int64_t i, const float branch[4]) {
  const int64_t r = i >> d_shift, c = i & (d - 1);
  float xv[4][4];
  for (int t = 0; t < 4; ++t) ld4(x + (r * 4 + t) * d + c, xv[t]);
  for (int s = 0; s < 4; ++s) {
    const float* cm = comb + r * 16 + s;
    const float c0 = cm[0], c1 = cm[4], c2 = cm[8], c3 = cm[12], p = post[r * 4 + s];
    float o[4];
    for (int j = 0; j < 4; ++j) {
      float m = __fmul_rn(xv[1][j], c1);
      m = __fmaf_rn(xv[0][j], c0, m);
      m = __fmaf_rn(xv[2][j], c2, m);
      m = __fmaf_rn(xv[3][j], c3, m);
      o[j] = __fmaf_rn(glm_triton::rbf(branch[j]), p, m);
    }
    st4(xout + (r * 4 + s) * d + c, o);
  }
}

}  // namespace glm_hc
