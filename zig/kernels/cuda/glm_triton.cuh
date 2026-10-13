// What GLM's Triton kernels compile to, for our hand-written copies (glm_glue.cu): approximate PTX ops and tl.sum.
#pragma once

#include <cuda_bf16.h>
#include <stdint.h>

namespace glm_triton {

__device__ __forceinline__ float bf(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float rbf(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

__device__ __forceinline__ float sqrt_approx(float x) {
  float y;
  asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float div_full(float a, float b) {
  float y;
  asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}

// Triton's add.f32 and mul.f32 carry no rounding mode, so ptxas may contract them into an fma as it did Triton's.
__device__ __forceinline__ float fadd(float a, float b) {
  float y;
  asm("add.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}
__device__ __forceinline__ float fmul(float a, float b) {
  float y;
  asm("mul.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}

// tl.exp: ex2.approx of x times log2(e) rounded to fp32.
__device__ __forceinline__ float exp_approx(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(__fmul_rn(x, 0x1.715476p+0f)));
  return y;
}

// tl.sum over a block: a butterfly over each warp, then over the warps' partials in the first NW lanes of warp 0.
template <int NW>
__device__ __forceinline__ float block_sum(float acc, float* smem) {
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
  if (NW == 1) return acc;
  if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = acc;
  __syncthreads();
  float v = threadIdx.x < NW ? smem[threadIdx.x] : 0.0f;
  for (int o = NW / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  __syncthreads();
  if (threadIdx.x == 0) smem[0] = v;
  __syncthreads();
  const float r = smem[0];
  __syncthreads();
  return r;
}

}  // namespace glm_triton
