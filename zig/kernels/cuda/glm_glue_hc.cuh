// glue._hc_partial and glue._hc_finish (hyper-connections' pre side): common/mhc.cuh's bodies with glm's mix RMS.
#pragma once

#include "glm_pdl.cuh"

#include "common/mhc.cuh"
#include "glm_triton.cuh"

using namespace glm_triton;

// glue._hc_partial, block (4 rows, K block b) of 128 threads: a row's 24 dots and squares in Triton's fma order.
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_hc_partial(
            const __nv_bfloat16* x, const __nv_bfloat16* fn, float* part, int64_t wide, int nb, int64_t rows) {
  glm_pdl::enter();
  TF_MHC_PARTIAL_BODY
}

// The same for decode windows (16 blocks a row): 8 steps of 128 at once, a 128-thread group each; same ops, same order.
extern "C" __global__ void __launch_bounds__(1024)
    tf_glm_hc_partial8(const __nv_bfloat16* x, const __nv_bfloat16* fn, float* part, int64_t wide, int nb) {
  glm_pdl::enter();
  tf_mhc::partial8(x, fn, part, wide, nb);
}

namespace glm_hc {

// glm's mix RMS: one over the approximate root of eps plus the mean square, divided in full (Triton's ops).
struct MixRms {
  __device__ __forceinline__ static float inv(float ss, float n, float eps) {
    return div_full(1.0f, sqrt_approx(eps + div_full(ss, n)));
  }
};

}  // namespace glm_hc

extern "C" __global__ void __launch_bounds__(256)
    tf_glm_hc_finish(const __nv_bfloat16* x, const float* part, const float* base, const float* scale,
                     const __nv_bfloat16* nw, __nv_bfloat16* out, float* xs, float* post, float* comb, float eps,
                     float hc_eps, int nb, int iters) {
  glm_pdl::enter();
  tf_mhc::finish<glm_hc::MixRms>(x, part, base, scale, nw, out, xs, post, comb, eps, hc_eps, nb, iters);
}

// FP4-Local's float32 post-attention norm weight: hc_finish's arithmetic, the weight read as float32.
extern "C" __global__ void __launch_bounds__(256)
    tf_glm_hc_finish_f32w(const __nv_bfloat16* x, const float* part, const float* base, const float* scale,
                          const float* nw, __nv_bfloat16* out, float* xs, float* post, float* comb, float eps,
                          float hc_eps, int nb, int iters) {
  glm_pdl::enter();
  tf_mhc::finish<glm_hc::MixRms>(x, part, base, scale, nw, out, xs, post, comb, eps, hc_eps, nb, iters);
}

// Four decode rows share fn loads; each retains partial8's step groups, shuffle tree and final step order.
extern "C" __global__ void __launch_bounds__(1024)
    tf_glm_hc_partial8_shared(
            const __nv_bfloat16* x, const __nv_bfloat16* fn, float* part, int64_t wide, int nb, int64_t rows) {
  glm_pdl::enter();
  constexpr int R = 4;
  __shared__ float dots[R][8][24];
  __shared__ float smem[R][4];
  const int64_t r0 = (int64_t)blockIdx.x * R, b = blockIdx.y, kb = wide / nb;
  const int grp = threadIdx.x >> 7, t = threadIdx.x & 127, c = t & 15, m0 = (t >> 4) & 7;
  const int steps = static_cast<int>(kb / 128);
  if (grp < steps) {
    const int64_t base = b * kb + grp * 128;
    float xv[R][8];
    for (int q = 0; q < R; ++q) {
      uint4 u = make_uint4(0, 0, 0, 0);
      if (r0 + q < rows) u = *reinterpret_cast<const uint4*>(x + (r0 + q) * wide + base + c * 8);
      const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&u);
      for (int j = 0; j < 8; ++j) xv[q][j] = bf(h[j]);
    }
    for (int i = 0; i < 3; ++i) {
      const uint4 u = __ldg(reinterpret_cast<const uint4*>(fn + (int64_t)(m0 + 8 * i) * wide + base + c * 8));
      const __nv_bfloat16* w = reinterpret_cast<const __nv_bfloat16*>(&u);
      for (int q = 0; q < R; ++q) {
        float d = __fmul_rn(xv[q][1], bf(w[1]));
        d = __fmaf_rn(xv[q][0], bf(w[0]), d);
        for (int j = 2; j < 8; ++j) d = __fmaf_rn(xv[q][j], bf(w[j]), d);
        for (int o = 8; o > 0; o >>= 1) d = d + __shfl_xor_sync(0xffffffffu, d, o);
        if (c == 0) dots[q][grp][m0 + 8 * i] = d;
      }
    }
  }
  if (grp == 0) {
    for (int q = 0; q < R; ++q) {
      float ss = 0.0f;
      for (int s = 0; s < steps; ++s) {
        const float xt = r0 + q < rows ? bf(x[(r0 + q) * wide + b * kb + s * 128 + t]) : 0.0f;
        ss = __fmaf_rn(xt, xt, ss);
      }
      for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
      if ((t & 31) == 0) smem[q][t >> 5] = ss;
    }
  }
  __syncthreads();
  for (int q = 0; q < R; ++q) {
    if (r0 + q >= rows) break;
    float* out = part + ((r0 + q) * nb + b) * 32;
    if (threadIdx.x < 24) {
      float acc = 0.0f;
      for (int s = 0; s < steps; ++s) acc = acc + dots[q][s][threadIdx.x];
      out[threadIdx.x] = acc;
    } else if (threadIdx.x == 32) {
      out[24] = (smem[q][0] + smem[q][2]) + (smem[q][1] + smem[q][3]);
    }
  }
}
