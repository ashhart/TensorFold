// Gate 3 controls for the shared routers: tiles that drop a tile edge or a K slice.
#pragma once

#include "common/router_tiles.cuh"

extern "C" __global__ void __launch_bounds__(256) tf_ctl_moe_router_tiles_row(const __nv_bfloat16* __restrict__ X, int x_stride,
                                                                                const __nv_bfloat16* __restrict__ W,
                                                                                float* __restrict__ OUT, int R, int D, int E,
                                                                                const int* __restrict__ P) {
    tf_router_tiles::tiles(X, W, OUT, R - 1, D, E, 1);
}

#include "common/router_mma.cuh"

// Gate 3's control for tf_glm_router_tile: the same sliced mma tiles that leave the last K slice unwritten.
extern "C" __global__ void __launch_bounds__(128)
    tf_ctl_glm_router_tile_slice(const __nv_bfloat16* x, const __nv_bfloat16* w, float* part, int64_t rows, int64_t x_stride,
                                 int64_t d, int64_t ne, int ks) {
    if (static_cast<int>(blockIdx.z) == ks - 1) return;
    tf_router_mma::logits<4, 2, 64, 4, false, true>(x, static_cast<int>(x_stride), w, part, static_cast<int>(rows),
                                                     static_cast<int>(d), static_cast<int>(ne), ks);
}

#include "glm_glue_router.cuh"

// Gate 3's baseline for tf_glm_router_tile: the removed tf_glm_router_part, kept here byte for byte (16 x 32 a block).
extern "C" __global__ void __launch_bounds__(128)
    tf_ref_glm_router_part(const __nv_bfloat16* x, const __nv_bfloat16* w, float* part, int64_t rows, int64_t x_stride,
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
