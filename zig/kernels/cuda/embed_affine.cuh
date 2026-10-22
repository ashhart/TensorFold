// MLX affine token rows with a bf16 scale (embed_rows.cu's tf_embed_rows_affine); `keep`: its rounding point.
#pragma once

#include <cuda_bf16.h>
#include <stdint.h>

#include "affine_codes.cuh"

namespace tf_embed_affine {

// keep's bit: the dequantized value rounded to bf16 before the scale (MLX's dequantize, then its multiply).
constexpr int ROUND = 1;
// A control's only: the scale folded into the group's scale and bias, q (s scale) + b scale, rounded once.
constexpr int FOLD = 2;

// Row ids[blockIdx.x], 8 values a thread: bf16(bf16(fma(q, s, b)) * scale), q `bits` wide in MLX's words.
__device__ __forceinline__ void rows(const int* __restrict__ ids, const uint32_t* __restrict__ w,
                                     const __nv_bfloat16* __restrict__ s, const __nv_bfloat16* __restrict__ b,
                                     __nv_bfloat16* __restrict__ out, int d, int bits, int group, float scale, int keep) {
    const int64_t r = blockIdx.x;
    const int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (8 * c >= d) return;
    const int64_t tok = ids[r];
    const uint32_t* row = w + tok * (d / 32) * bits;
    const int64_t g = tok * (d / group) + 8 * c / group;
    float sc = __bfloat162float(s[g]), bi = __bfloat162float(b[g]);
    if (keep & FOLD) sc = __fmul_rn(sc, scale), bi = __fmul_rn(bi, scale);
    uint4 o;
    __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float v[2];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const float e = __fmaf_rn(static_cast<float>(tf_affine8::code_at(row, 8 * c + 2 * i + j, bits)), sc, bi);
            if (keep & FOLD) v[j] = e;
            else v[j] = __fmul_rn((keep & ROUND) ? __bfloat162float(__float2bfloat16_rn(e)) : e, scale);
        }
        h[i] = __floats2bfloat162_rn(v[0], v[1]);
    }
    *reinterpret_cast<uint4*>(out + r * d + 8 * c) = o;
}

}  // namespace tf_embed_affine
