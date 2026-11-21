#pragma once
// dots.mocr's vision glue (HF modeling_dots_vision): the bodies the kernels and their planted controls share.

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

namespace tf_dots_vision {
constexpr int HD = 128, HALF = HD / 2, QUARTER = HALF / 2, HEADS = 12, ROW = 3 * HEADS * HD, NT = 256;

__device__ __forceinline__ float bfr(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

// Block (row, head of q then k), thread j: dims j and j + 64 of QKV [n][q 12 | k 12 | v 12][128], in place.
template <bool HF>  // true: fp32 cos and sin as HF's freqs.cos(); false: each rounded to bf16 (the text rotary's)
__device__ __forceinline__ void rope(__nv_bfloat16* __restrict__ QKV, const int* __restrict__ HW,
                                     const float* __restrict__ INV) {
    const int r = blockIdx.x, j = threadIdx.x;
    __nv_bfloat16* t = QKV + static_cast<int64_t>(r) * ROW + blockIdx.y * HD;
    const float a = __bfloat162float(t[j]), b = __bfloat162float(t[j + HALF]);
    // HF: freqs = cat(h x inv_freq, w x inv_freq), each an fp32 product; cos and sin in fp32 (no fast math)
    const float angle = __fmul_rn(static_cast<float>(HW[2 * r + (j < QUARTER ? 0 : 1)]), INV[j % QUARTER]);
    float c = cosf(angle), s = sinf(angle);
    if (!HF) {
        c = bfr(c);
        s = bfr(s);
    }
    // t x cos + rotate_half(t) x sin: each fp32 product rounded, then the fp32 sum, then bf16
    t[j] = __float2bfloat16_rn(__fsub_rn(__fmul_rn(a, c), __fmul_rn(b, s)));
    t[j + HALF] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(b, c), __fmul_rn(a, s)));
}

__device__ __forceinline__ float wide(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float wide(float v) { return v; }

// Block (row, column block): O = bf16(Y + B[col]); a bf16 Y is HIP's second rounding, an fp32 Y rounds once.
template <typename T>
__device__ __forceinline__ void patch_bias(const T* Y, const __nv_bfloat16* __restrict__ B, __nv_bfloat16* O, int N) {
    const int c = blockIdx.y * NT + threadIdx.x;
    if (c >= N) return;
    const int64_t at = static_cast<int64_t>(blockIdx.x) * N + c;
    O[at] = __float2bfloat16_rn(__fadd_rn(wide(Y[at]), __bfloat162float(B[c])));
}
}  // namespace tf_dots_vision
