// Liquid's 3-tap short conv and commit, one planted fault each: conv bx unrounded, commits to stream 0's slot.
#pragma once
#include "liquid_conv.cuh"

extern "C" __global__ void __launch_bounds__(256) tf_ctl_liquid_conv_3(const __nv_bfloat16* P, const __nv_bfloat16* Wc, const __nv_bfloat16* STATE,
                                                                        const int* SLOT, const int* OFF, __nv_bfloat16* Y, __nv_bfloat16* SNAP,
                                                                        int SNAP0, int D, long long SS) {
    tf_liquid::liquid_conv<3, 1>(P, Wc, STATE, SLOT, OFF, Y, SNAP, SNAP0, D, SS);
}

extern "C" __global__ void __launch_bounds__(256) tf_ctl_liquid_commit_3(__nv_bfloat16* STATE, const __nv_bfloat16* SNAP, const int* SLOT,
                                                                          const int* ROW, int D, long long state_stride, long long snap_stride,
                                                                          long long SS) {
    tf_liquid::liquid_commit<3, 1>(STATE, SNAP, SLOT, ROW, D, state_stride, snap_stride, SS);
}

// LFM2.5-24B-A2B's router, one planted fault: ties go to the higher expert id.
#include "lfm2_moe.cuh"

extern "C" __global__ void __launch_bounds__(256) tf_ctl_lfm2m_route(const __nv_bfloat16* X, const float* W, const float* BIAS, int* PICK,
                                                                      float* WTS, float* LOGITS, int rows, int D, float scale) {
    tf_lfm2m::lfm2m_route<1>(X, W, BIAS, PICK, WTS, LOGITS, rows, D, scale);
}
