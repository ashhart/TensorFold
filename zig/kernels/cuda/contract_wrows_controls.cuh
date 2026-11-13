// window_rows' equality baselines (contract_window_rows.zig's `equal` registrations), after contract_controls.cu.
#pragma once

#include "window_rows.cuh"

// The equality tests' baseline: the merge over every chunk, empty or not (Fault ALL), at head dims 256 and 512.
extern "C" __global__ void __launch_bounds__(tf_wrows::NT) tf_ctl_wrows_merge_all_256(const float* po, const float* pm, const float* pl,
                                                                                     __nv_bfloat16* out, int h, int hk, int chunks,
                                                                                     const __nv_bfloat16* q, const __nv_bfloat16* kc,
                                                                                     const __nv_bfloat16* vc, const int* pos, const int* slot,
                                                                                     int win, long long ring, float scale) {
    tf_wrows::merge<256, tf_wrows::ALL>(po, pm, pl, out, h, hk, chunks, q, kc, vc, pos, slot, win, ring, scale);
}

extern "C" __global__ void __launch_bounds__(tf_wrows::NT) tf_ctl_wrows_merge_all_512(const float* po, const float* pm, const float* pl,
                                                                                     __nv_bfloat16* out, int h, int hk, int chunks,
                                                                                     const __nv_bfloat16* q, const __nv_bfloat16* kc,
                                                                                     const __nv_bfloat16* vc, const int* pos, const int* slot,
                                                                                     int win, long long ring, float scale) {
    tf_wrows::merge<512, tf_wrows::ALL>(po, pm, pl, out, h, hk, chunks, q, kc, vc, pos, slot, win, ring, scale);
}
