// Liquid's gated short conv over stream state and its commit, by tap count (LFM2.5 3, LFM3 7); CTL 1: control.

#pragma once

#include <cuda_bf16.h>
#include <stdint.h>

#include "common/glue_math.cuh"

namespace tf_liquid {

constexpr int NT = 256;  // threads a block

// Block (row, channel block): y = bf16(C * bf16(sum of w * bf16(B * x), oldest tap first)); SNAP rows keep state.
template <int TAPS, int CTL>  // CTL 1: bx = B * x not rounded to bf16 before its tap.
__device__ __forceinline__ void liquid_conv(const __nv_bfloat16* __restrict__ P, const __nv_bfloat16* __restrict__ Wc,
                                            const __nv_bfloat16* __restrict__ STATE, const int* __restrict__ SLOT,
                                            const int* __restrict__ OFF, __nv_bfloat16* __restrict__ Y,
                                            __nv_bfloat16* __restrict__ SNAP, int SNAP0, int D, long long SS) {
    constexpr int S = TAPS - 1;  // state rows: the earlier taps come from the round (OFF before the row), else the slot
    const int64_t r = blockIdx.x, D3 = 3 * static_cast<int64_t>(D);
    const int c = blockIdx.y * NT + threadIdx.x;
    if (c >= D) return;
    const int o = OFF[r];
    const __nv_bfloat16* st = STATE + SLOT[r] * SS;
    __nv_bfloat16 bx[TAPS];
    float wide[TAPS];
#pragma unroll
    for (int j = 0; j < TAPS; ++j) wide[j] = 0.0f;
#pragma unroll
    for (int j = 0; j < TAPS; ++j) {
        const int back = S - j;  // rows before this one
        if (o >= back) {
            const int64_t src = r - back;
            const float v = __fmul_rn(tf_glue::bf(P[src * D3 + c]), tf_glue::bf(P[src * D3 + 2 * D + c]));
            bx[j] = __float2bfloat16_rn(v);
            if (CTL == 1) wide[j] = v;
        } else {
            bx[j] = st[static_cast<int64_t>(S - back + o) * D + c];
            if (CTL == 1) wide[j] = tf_glue::bf(bx[j]);
        }
    }
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < TAPS; ++j) acc = __fadd_rn(acc, __fmul_rn(tf_glue::bf(Wc[c * TAPS + j]), CTL == 1 ? wide[j] : tf_glue::bf(bx[j])));
    Y[r * D + c] = __float2bfloat16_rn(__fmul_rn(tf_glue::bf(P[r * D3 + D + c]), tf_glue::rbf(acc)));
    if (SNAP == nullptr || r < SNAP0) return;
#pragma unroll
    for (int j = 1; j < TAPS; ++j) SNAP[((r - SNAP0) * S + (j - 1)) * D + c] = bx[j];
}

// Block (layer, stream): stream k's [TAPS - 1][D] conv state set to its snapshot row ROW[k]; ROW < 0 keeps the state.
template <int TAPS, int CTL>  // CTL 1: every stream's state written to the first stream's slot.
__device__ __forceinline__ void liquid_commit(__nv_bfloat16* __restrict__ STATE, const __nv_bfloat16* __restrict__ SNAP,
                                              const int* __restrict__ SLOT, const int* __restrict__ ROW, int D,
                                              long long state_stride, long long snap_stride, long long SS) {
    constexpr int S = TAPS - 1;  // state rows
    const int layer = blockIdx.x, k = blockIdx.y, row = ROW[k];
    if (row < 0) return;
    __nv_bfloat16* dst = STATE + layer * state_stride + SLOT[CTL == 1 ? 0 : k] * SS;
    const __nv_bfloat16* src = SNAP + layer * snap_stride + static_cast<int64_t>(row) * S * D;
    for (int i = threadIdx.x; i < S * D; i += NT) dst[i] = src[i];
}

}  // namespace tf_liquid
