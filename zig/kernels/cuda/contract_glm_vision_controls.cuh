// Planted control for glm_vision's in-place q and k: the rotation of a head before its bf16 rounding.

#include "glm_vis_qk.cuh"

extern "C" __global__ void tf_ctl_glm_vis_qk_unrounded(__nv_bfloat16* qkv, const __nv_bfloat16* __restrict__ qn,
                                                       const __nv_bfloat16* __restrict__ kn,
                                                       const int* __restrict__ pos, const float* __restrict__ inv_freq,
                                                       int N, int H, float eps) {
    glm_vis_rows::qk<false>(qkv, qn, kn, pos, inv_freq, N, H, eps);
}
