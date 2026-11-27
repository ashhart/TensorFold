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
