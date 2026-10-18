// 3-bit MLX affine experts (groups of 64) beside experts.cuh's 4-bit blocks: bit planes that rebuild the nibble words.
#pragma once

#include <cuda_bf16.h>
#include <stdint.h>

#include "experts.cuh"

namespace {

// A block of BITS-wide codes: lanes' weight words (4-bit as Geo<64>; 3-bit a 2-bit then a 1-bit plane), scales, biases.
template <int BITS>
struct Lane {
  static_assert(BITS == 3 || BITS == 4, "3- and 4-bit blocks only");
  static constexpr int WORDS = 2 * BITS;               // weight uint32 a lane a group
  static constexpr int WU = 8 * WORDS;                 // weight uint4 a block
  static constexpr int BLOCK = WU + 4 * Geo<64>::SBV;  // uint4 a block
};

// A lane's raw words of one block (b: the block's first uint4); L4 loads a uint4, L2 a uint2.
template <int BITS, class L4, class L2>
__device__ __forceinline__ void lane_words(uint32_t (&r)[Lane<BITS>::WORDS], const uint4* b, int lane, L4 ld4, L2 ld2) {
  if constexpr (BITS == 3) {
    const uint4 v = ld4(b + lane);
    const uint2 u = ld2(reinterpret_cast<const uint2*>(b + 32) + lane);
    r[0] = v.x; r[1] = v.y; r[2] = v.z; r[3] = v.w; r[4] = u.x; r[5] = u.y;
  } else {
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const uint4 v = ld4(b + c * 32 + lane);
      r[4 * c] = v.x; r[4 * c + 1] = v.y; r[4 * c + 2] = v.z; r[4 * c + 3] = v.w;
    }
  }
}

// A lane's nibble word wi: low bits from the 2-bit plane's words wi / 2, bit 2 from the 1-bit plane's word 4 + wi / 4.
template <int BITS, int N>
__device__ __forceinline__ uint32_t nibword(const uint32_t (&r)[N], int wi) {
  if constexpr (BITS == 4) {
    return r[wi];
  } else {
    const uint32_t lo = (r[wi >> 1] >> (2 * (wi & 1))) & 0x33333333u;
    return lo | (((r[4 + (wi >> 2)] >> (wi & 3)) & 0x11111111u) << 2);
  }
}

}  // namespace
