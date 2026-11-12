// MiniMax-M3.1-Flash's CUDA helpers: bf16 rounding, sigmoid, warp and block reductions.
#pragma once

#include <cuda_bf16.h>
#include <math_constants.h>
#include <stdint.h>
#include "common/reduce.cuh"
#include "common/act.cuh"

namespace tf_mm3 {

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float ld(const bf16* p) { return __bfloat162float(*p); }
__device__ __forceinline__ void st(bf16* p, float v) { *p = __float2bfloat16_rn(v); }

// torch's sigmoid of an fp32 value, 1 / (1 + e^-x) on either side.
__device__ __forceinline__ float sigmoid(float x) { return tf_common::sigmoidf_(x); }

using tf_common::warp_sum;

using tf_common::warp_max;

// A block of 1024 threads sums `v` over its warps; every thread gets the total (two syncs).
__device__ __forceinline__ float block_sum(float v, float* part) {
  v = warp_sum(v);
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, warps = blockDim.x >> 5;
  __syncthreads();
  if (lane == 0) part[w] = v;
  __syncthreads();
  float t = 0.0f;
  for (int i = 0; i < warps; ++i) t += part[i];
  return t;
}

__device__ __forceinline__ void mma16816(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&v);
}

}  // namespace tf_mm3
