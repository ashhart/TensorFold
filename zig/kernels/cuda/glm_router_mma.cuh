// glue's mma.m16n8k16 (bf16 in, fp32 accumulate) and its register pairs: GLM's router and the MLA chunk tiles.
#pragma once

#include "glm_triton.cuh"
#include "common/qmm_frag.cuh"

namespace glm_router {

// Two consecutive bf16 as one mma register (the lower K in the low half), zero outside the rows or columns.
__device__ __forceinline__ uint32_t pair(const __nv_bfloat16* p, bool ok) {
  if (!ok) return 0u;
  return (uint32_t)__bfloat16_as_ushort(p[0]) | ((uint32_t)__bfloat16_as_ushort(p[1]) << 16);
}


__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, const uint32_t* b) {
  qmm_frag::mma_v(c, a, b);
}

}  // namespace glm_router
