// Flash Next's BF16 read-out prompt tiles: the down at three row tiles and the up+mix (flashnext_hcn.cuh's bodies).
#include "flashnext_hcn.cuh"

#define HCN_DOWN(NAME, MT)                                                                                                 \
    extern "C" __global__ void __launch_bounds__(hcn::DN_THREADS) NAME(                                                   \
        const __nv_bfloat16* __restrict__ H, const float* __restrict__ PSS, const float* __restrict__ SCALE,              \
        const uint2* __restrict__ W, float* __restrict__ PART, int M, int N, int K, int SK, int D, int NC, int SS,        \
        float eps) {                                                                                                       \
        hcn::down_body<MT, 3>(H, PSS, SCALE, W, PART, M, N, K, SK, D, NC, SS, eps);                                       \
    }

HCN_DOWN(fn_hcn_down2, 2)
HCN_DOWN(fn_hcn_down3, 3)
HCN_DOWN(fn_hcn_down4, 4)

extern "C" __global__ void __launch_bounds__(256) fn_hcn_up3(const __nv_bfloat16* __restrict__ ACT, const uint2* __restrict__ W,
                                                             const __nv_bfloat16* __restrict__ H, const float* __restrict__ PSS,
                                                             const float* __restrict__ SCALE, __nv_bfloat16* __restrict__ MIXED,
                                                             float* __restrict__ XSM, int M, int D, int NC, float eps) {
    hcn::upmix_body<3, 5>(ACT, W, H, PSS, SCALE, MIXED, XSM, M, D, NC, eps);
}
