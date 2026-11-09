// glue._swiglu's rows by hand: silu with the clamp before it (GLM) or after it (Ling), and the 64-value group sums.
#pragma once

#include "glm_triton.cuh"

using namespace glm_triton;

namespace glm_moe {

// A thread's S consecutive values summed from the second, then the butterfly over the 64 / S lanes of their group.
template <int S>
__device__ __forceinline__ float group64(const float* q) {
  float g = q[0] + q[1];
  for (int j = 2; j < S; ++j) g = q[j] + g;
  for (int o = 32 / S; o > 0; o >>= 1) g = g + __shfl_xor_sync(0xffffffffu, g, o);
  return g;
}

// AFTER false: GLM's silu(min(g, L)); true: Ling's min(silu(g), L), the gate clipped after silu. Both clip u.
template <bool AFTER>
__device__ __forceinline__ float swiglu1(float g, float u, float limit) {
  if (!AFTER) g = fminf(g, limit);
  if (AFTER) u = u > limit ? limit : (u < 0.0f - limit ? 0.0f - limit : u);  // a NaN u stays NaN
  else u = fminf(fmaxf(u, 0.0f - limit), limit);
  float s = rbf(div_full(g, exp_approx(0.0f - g) + 1.0f));
  if (AFTER) s = s > limit ? limit : s;
  return rbf(__fmul_rn(u, s));
}

}  // namespace glm_moe

// glue._swiglu with BLOCK 512 (128 threads, 4 values each, 64-group sums over 16 lanes), column block blockIdx.y.
#define TF_GLM_SWIGLU_512(NAME, AFTER)                                                                               \
  extern "C" __global__ void __launch_bounds__(128)                                                                  \
      NAME(const __nv_bfloat16* gu, __nv_bfloat16* out, float* xs, float limit, int64_t w) {                         \
    const int64_t r = blockIdx.x, d0 = blockIdx.y * 512 + threadIdx.x * 4;                                           \
    const uint2 gv = *reinterpret_cast<const uint2*>(gu + r * 2 * w + d0);                                           \
    const uint2 uv = *reinterpret_cast<const uint2*>(gu + r * 2 * w + w + d0);                                       \
    const __nv_bfloat16* gh = reinterpret_cast<const __nv_bfloat16*>(&gv);                                           \
    const __nv_bfloat16* uh = reinterpret_cast<const __nv_bfloat16*>(&uv);                                           \
    float a[4];                                                                                                      \
    uint2 ov;                                                                                                        \
    __nv_bfloat16* oh = reinterpret_cast<__nv_bfloat16*>(&ov);                                                       \
    for (int j = 0; j < 4; ++j) {                                                                                    \
      a[j] = glm_moe::swiglu1<AFTER>(bf(gh[j]), bf(uh[j]), limit);                                                   \
      oh[j] = __float2bfloat16_rn(a[j]);                                                                             \
    }                                                                                                                \
    *reinterpret_cast<uint2*>(out + r * w + d0) = ov;                                                                \
    const float g = glm_moe::group64<4>(a);                                                                          \
    if ((threadIdx.x & 15) == 0) xs[r * (w / 64) + d0 / 64] = g;                                                     \
  }

// glue._swiglu with BLOCK 64 (the TP=3 shared expert's 704): one value a thread, the warp's butterfly, then 2 warps.
#define TF_GLM_SWIGLU_64(NAME, AFTER)                                                                                \
  extern "C" __global__ void __launch_bounds__(64)                                                                   \
      NAME(const __nv_bfloat16* gu, __nv_bfloat16* out, float* xs, float limit, int64_t w) {                         \
    __shared__ float smem[2];                                                                                        \
    const int64_t r = blockIdx.x, d = blockIdx.y * 64 + threadIdx.x;                                                 \
    const float a = glm_moe::swiglu1<AFTER>(bf(gu[r * 2 * w + d]), bf(gu[r * 2 * w + w + d]), limit);                \
    out[r * w + d] = __float2bfloat16_rn(a);                                                                         \
    const float g = block_sum<2>(a, smem);                                                                           \
    if (threadIdx.x == 0) xs[r * (w / 64) + blockIdx.y] = g;                                                         \
  }
