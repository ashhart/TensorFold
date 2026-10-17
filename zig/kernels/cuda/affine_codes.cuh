// Ours: MLX affine codes read for the lane matmul's MMAs, shared by affine8.cu and affine_prefill.cu.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace tf_affine8 {

// Code i of an MLX row packed `bits` wide, low code first, in little-endian words (a code may cross a word).
__device__ __forceinline__ uint32_t code_at(const uint32_t* __restrict__ row, long long i, int bits) {
    const long long bit = i * bits;
    const int sh = static_cast<int>(bit & 31);
    const long long wd = bit >> 5;
    uint32_t v = row[wd] >> sh;
    if (sh + bits > 32) v |= row[wd + 1] << (32 - sh);
    return v & ((1u << bits) - 1u);
}

// Bytes at s and s + 8 of w as a bf16 pair (the first low): exact, as every byte is a bf16 integer.
__device__ __forceinline__ uint32_t byte_pair(uint32_t w, int s) {
    const float a = __uint_as_float(0x4B000000u | ((w >> s) & 0xFFu)) - 8388608.0f;
    const float b = __uint_as_float(0x4B000000u | ((w >> (s + 8)) & 0xFFu)) - 8388608.0f;
    const __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&h);
}

// Codes c0 (low half) and c1 (high) of up to 6 bits as a bf16 pair: bf16 0x4300 | c is 128 + c, exactly.
__device__ __forceinline__ uint32_t code_pair(uint32_t r) {
    r |= 0x43004300u;
    const __nv_bfloat162 h = __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&r), __floats2bfloat162_rn(128.0f, 128.0f));
    return *reinterpret_cast<const uint32_t*>(&h);
}

// Pair i of a lane's BITS words: first codes stream through the low halves, second through the high halves.
template <int BITS>
__device__ __forceinline__ uint32_t pair_at(const uint32_t (&w)[BITS], int i) {
    constexpr uint32_t M = (1u << BITS) - 1u;
    const int o = i * BITS, h = o >> 4, sh = o & 15;
    if (sh + BITS <= 16) return code_pair((w[h] >> sh) & (M | M << 16));
    const uint32_t lo = __byte_perm(w[h], w[h + 1], 0x5410), hi = __byte_perm(w[h], w[h + 1], 0x7632);
    return code_pair(((lo >> sh) & M) | ((hi >> sh) & M) << 16);
}

}  // namespace tf_affine8
