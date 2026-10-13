// glue._router_part by hand: tl.dot's chain of mma.m16n8k16 (bf16 in, fp32 accumulate) from zero, K ascending.
#pragma once

#include "glm_triton.cuh"

namespace glm_router {

// Two consecutive bf16 as one mma register (the lower K in the low half), zero outside the rows or columns.
__device__ __forceinline__ uint32_t pair(const __nv_bfloat16* p, bool ok) {
  if (!ok) return 0u;
  return (uint32_t)__bfloat16_as_ushort(p[0]) | ((uint32_t)__bfloat16_as_ushort(p[1]) << 16);
}


__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

}  // namespace glm_router

// glue._router_part: part[s][m][e] = x[m] . w[e] over K slice s of ks (d / ks inputs), each warp one 16 x 8 tile.
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_router_part(const __nv_bfloat16* x, const __nv_bfloat16* w, float* part, int64_t rows, int64_t x_stride,
                       int64_t d, int64_t ne, int ks) {
  const int lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
  const int64_t m0 = (int64_t)blockIdx.x * 16, n = (int64_t)blockIdx.y * 32 + (threadIdx.x >> 5) * 8;
  const int64_t s = blockIdx.z, span = d / ks, lo = s * span;
  const int64_t ra = m0 + g, rb = ra + 8, e = n + g;
  const bool oka = ra < rows, okb = rb < rows, oke = e < ne;
  float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int64_t k = lo; k < lo + span; k += 16) {
    const int64_t kq = k + q * 2;
    const uint32_t a[4] = {glm_router::pair(x + ra * x_stride + kq, oka), glm_router::pair(x + rb * x_stride + kq, okb),
                           glm_router::pair(x + ra * x_stride + kq + 8, oka),
                           glm_router::pair(x + rb * x_stride + kq + 8, okb)};
    const uint32_t b[2] = {glm_router::pair(w + e * d + kq, oke), glm_router::pair(w + e * d + kq + 8, oke)};
    glm_router::mma16816(c, a, b);
  }
  for (int h = 0; h < 2; ++h) {
    const int64_t r = h == 0 ? ra : rb;
    for (int j = 0; j < 2; ++j) {
      const int64_t col = n + q * 2 + j;
      if (r < rows && col < ne) part[(s * rows + r) * ne + col] = c[h * 2 + j];
    }
  }
}
