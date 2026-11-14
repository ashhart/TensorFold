// The fused read-out's planted controls: fn_hcn_down3 rounding toward zero, fn_hcn_up3 gating unrounded up rows.
#pragma once

#include "flashnext_hcn.cuh"

extern "C" __global__ void __launch_bounds__(hcn::DN_THREADS) tf_ctl_hcn_down3_rz(
    const __nv_bfloat16* __restrict__ H, const float* __restrict__ PSS, const float* __restrict__ SCALE,
    const uint2* __restrict__ W, float* __restrict__ PART, int M, int N, int K, int SK, int D, int NC, int SS, float eps) {
    hcn::down_body<3, 3, true>(H, PSS, SCALE, W, PART, M, N, K, SK, D, NC, SS, eps);
}

extern "C" __global__ void __launch_bounds__(256) tf_ctl_hcn_up3_raw(
    const __nv_bfloat16* __restrict__ ACT, const uint2* __restrict__ W, const __nv_bfloat16* __restrict__ H,
    const float* __restrict__ PSS, const float* __restrict__ SCALE, __nv_bfloat16* __restrict__ MIXED,
    float* __restrict__ XSM, int M, int D, int NC, float eps) {
    hcn::upmix_body<3, 5, true>(ACT, W, H, PSS, SCALE, MIXED, XSM, M, D, NC, eps);
}
