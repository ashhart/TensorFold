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

// A control's only: codes read in MLX's order within a tile word, as if the tiles were MLX's words.
constexpr int MLX_ORDER = 4;

// Values 8c to 8c + 7 of a row from code(i), i < 8, and the group's s and b: bf16(bf16(fma(q, s, b)) * scale).
template <class Code>
__device__ __forceinline__ uint4 eight(Code code, float sc, float bi, float scale, int keep) {
    if (keep & FOLD) sc = __fmul_rn(sc, scale), bi = __fmul_rn(bi, scale);
    uint4 o;
    __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float v[2];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const float e = __fmaf_rn(static_cast<float>(code(2 * i + j)), sc, bi);
            if (keep & FOLD) v[j] = e;
            else v[j] = __fmul_rn((keep & ROUND) ? __bfloat162float(__float2bfloat16_rn(e)) : e, scale);
        }
        h[i] = __floats2bfloat162_rn(v[0], v[1]);
    }
    return o;
}

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
    const auto code = [&](int i) { return tf_affine8::code_at(row, 8 * c + i, bits); };
    *reinterpret_cast<uint4*>(out + r * d + 8 * c) = eight(code, __bfloat162float(s[g]), __bfloat162float(b[g]), scale, keep);
}

// rows' values from affine4_pack.cu's tiles: group value 32v + 8p + 2l + e is lane 4 (row % 8) + l, nibble p + 4e.
__device__ __forceinline__ void tile4_rows(const int* __restrict__ ids, const uint32_t* __restrict__ w,
                                           const __nv_bfloat16* __restrict__ s, const __nv_bfloat16* __restrict__ b,
                                           __nv_bfloat16* __restrict__ out, int d, int npad, float scale, int keep) {
    const int64_t r = blockIdx.x;
    const int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (8 * c >= d) return;
    const int64_t tok = ids[r];
    const int g = 8 * c / 64, v = (8 * c / 32) & 1, p = (c & 3);
    const int64_t lane0 = (((tok / 64) * (d / 64) + g) * 8 + (tok & 63) / 8) * 32 + 4 * (tok & 7);
    uint32_t q[8];
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint32_t word = w[(lane0 + l) * 2 + v];
#pragma unroll
        for (int e = 0; e < 2; ++e)
            q[2 * l + e] = (keep & MLX_ORDER) ? (word >> (4 * (2 * l + e))) & 0xFu : (word >> (4 * (p + 4 * e))) & 0xFu;
    }
    const int64_t at = static_cast<int64_t>(g) * npad + tok;
    const auto code = [&](int i) { return q[i]; };
    *reinterpret_cast<uint4*>(out + r * d + 8 * c) = eight(code, __bfloat162float(s[at]), __bfloat162float(b[at]), scale, keep);
}

}  // namespace tf_embed_affine
