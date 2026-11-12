#pragma once
// dots.mocr's q|k|v bias and RoPE (HF Qwen2 in BF16): the body the kernel and its planted control share.

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

namespace tf_dots {
constexpr int HD = 128, HALF = HD / 2;

__device__ __forceinline__ float bfr(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

// Block (row, head), thread j: dims j and j + 64 of rows [q | k | v heads of 128]; Q [rows][H], KV [rows][k HK | v HK].
template <bool TORCH>  // true: each product and the sum round to bf16 (HF's ops); false: the fp32 sum rounds once
__device__ __forceinline__ void qkv_rope(const float* __restrict__ ACC, const __nv_bfloat16* __restrict__ BIAS,
                                         const float* __restrict__ INV, const int* __restrict__ POS,
                                         __nv_bfloat16* __restrict__ Q, __nv_bfloat16* __restrict__ KV, int H, int HK) {
    const int r = blockIdx.x, h = blockIdx.y, j = threadIdx.x;
    const int64_t at = static_cast<int64_t>(r) * (H + 2 * HK) * HD + h * HD;
    // t = bf16(acc + bias): the projection's fp32 sum and its bias, rounded once
    const float a = bfr(__fadd_rn(ACC[at + j], __bfloat162float(BIAS[h * HD + j])));
    const float b = bfr(__fadd_rn(ACC[at + j + HALF], __bfloat162float(BIAS[h * HD + j + HALF])));
    const int64_t q = (static_cast<int64_t>(r) * H + h) * HD, kv = (static_cast<int64_t>(r) * 2 * HK + h - H) * HD;
    __nv_bfloat16* out = h < H ? Q + q : KV + kv;
    if (h >= H + HK) {
        out[j] = __float2bfloat16_rn(a);
        out[j + HALF] = __float2bfloat16_rn(b);
        return;
    }
    // HF: the fp32 angle position x inv_freq, fp32 cos and sin (no fast math), each cast to bf16
    const float angle = __fmul_rn(static_cast<float>(POS[r]), INV[j]);
    const float c = bfr(cosf(angle)), s = bfr(sinf(angle));
    if (TORCH) {
        out[j] = __float2bfloat16_rn(__fadd_rn(bfr(__fmul_rn(a, c)), bfr(__fmul_rn(-b, s))));
        out[j + HALF] = __float2bfloat16_rn(__fadd_rn(bfr(__fmul_rn(b, c)), bfr(__fmul_rn(a, s))));
    } else {
        out[j] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(a, c), __fmul_rn(-b, s)));
        out[j + HALF] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(b, c), __fmul_rn(a, s)));
    }
}
}  // namespace tf_dots
