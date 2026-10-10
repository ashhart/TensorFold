#pragma once

// MLX affine codes read out of a 32-code piece of a column's row.

#include <hip/hip_fp16.h>

#include <cstdint>

#include "quant/mlx.hpp"

namespace tf {
namespace rocm {

// 32 codes are exactly BITS words: 16-byte loads when aligned and BITS % 4 == 0, 8-byte for even BITS.
template <int BITS>
__device__ inline void load_piece(const uint32_t* src, uint32_t (&w)[BITS]) {
    const uintptr_t at = reinterpret_cast<uintptr_t>(src);
    if constexpr (BITS % 4 == 0) {
        if ((at & 15u) == 0) {
#pragma unroll
            for (int i = 0; i < BITS / 4; ++i) {
                const uint4 v = reinterpret_cast<const uint4*>(src)[i];
                w[4 * i] = v.x;
                w[4 * i + 1] = v.y;
                w[4 * i + 2] = v.z;
                w[4 * i + 3] = v.w;
            }
            return;
        }
    }
    if constexpr (BITS % 2 == 0) {
        if ((at & 7u) == 0) {
#pragma unroll
            for (int i = 0; i < BITS / 2; ++i) {
                const uint2 v = reinterpret_cast<const uint2*>(src)[i];
                w[2 * i] = v.x;
                w[2 * i + 1] = v.y;
            }
            return;
        }
    }
#pragma unroll
    for (int i = 0; i < BITS; ++i) w[i] = src[i];
}

// Code t of a 32-code piece as FP16. Codes are below 256, exact in BF16 and FP16, so this is code_f16(code).
template <int BITS>
__device__ inline uint32_t piece_bits(const uint32_t (&w)[BITS], int t) {
    const int bit = t * BITS;
    const int word = bit >> 5;
    const int shift = bit & 31;
    uint32_t value = w[word] >> shift;
    if (shift + BITS > 32) value |= w[word + 1] << (32 - shift);
    return value & ((1u << BITS) - 1u);
}

template <int BITS>
__device__ inline __half piece_code(const uint32_t (&w)[BITS], int t) {
    return __ushort2half_rn(static_cast<unsigned short>(piece_bits<BITS>(w, t)));
}

}  // namespace rocm
}  // namespace tf
