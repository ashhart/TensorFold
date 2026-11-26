// Planted contract controls for Hy4-flash hyper-connection kernels (built only into fatbin_contract_controls).
#pragma once

#include "hy4_flash/hc.cuh"

#define HY4_HC_CTL_MIX(NAME, MIXES, CONTROL)                                                                            \
    extern "C" __global__ void __launch_bounds__(hy4_hc::kThreads)                                                      \
        NAME(const unsigned short* x, const float* fn, const float* base, const float* scale, float eps, float hc_eps, \
             float magnitude, unsigned short* out, float* post, int hidden, int repeat) {                               \
        hy4_hc::hc_mix<MIXES, CONTROL>(x, fn, base, scale, eps, hc_eps, magnitude, out, post, hidden, repeat);         \
    }
HY4_HC_CTL_MIX(hy4_hc_pre_bad_nan, 8, 1)
HY4_HC_CTL_MIX(hy4_hc_pre_bad_repeat, 8, 2)
HY4_HC_CTL_MIX(hy4_hc_pre_bad_tail, 8, 3)
HY4_HC_CTL_MIX(hy4_hc_pre_bad_sums, 8, 4)
HY4_HC_CTL_MIX(hy4_hc_head_bad_nan, 4, 1)
HY4_HC_CTL_MIX(hy4_hc_head_bad_repeat, 4, 2)
HY4_HC_CTL_MIX(hy4_hc_head_bad_tail, 4, 3)
HY4_HC_CTL_MIX(hy4_hc_head_bad_sums, 4, 4)
#undef HY4_HC_CTL_MIX

#define HY4_HC_CTL_MERGE(NAME, CONTROL)                                                                                \
    extern "C" __global__ void __launch_bounds__(hy4_hc::kThreads)                                                     \
        NAME(const unsigned short* residual, const unsigned short* f, const float* post, unsigned short* out,          \
             int rows, int hidden, int repeat) {                                                                       \
        hy4_hc::hc_merge<CONTROL>(residual, f, post, out, rows, hidden, repeat);                                       \
    }
HY4_HC_CTL_MERGE(hy4_hc_merge_bad_nan, 1)
HY4_HC_CTL_MERGE(hy4_hc_merge_bad_repeat, 2)
HY4_HC_CTL_MERGE(hy4_hc_merge_bad_tail, 3)
#undef HY4_HC_CTL_MERGE
