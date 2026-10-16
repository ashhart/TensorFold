#pragma once

// The matrix-core Dot of gfx11: v_wmma_f32_16x16x16_bf16 with its wave32 layouts (gfx12's differs), over BF16 operands.

#include "common/dot2.hpp"
#include "common/vec.hpp"

namespace tf {
namespace rocm {

typedef __bf16 bf16x16 __attribute__((ext_vector_type(16)));
typedef unsigned u32x8 __attribute__((ext_vector_type(8)));
typedef float f32x8 __attribute__((ext_vector_type(8)));

// The 16 BF16 of a row's 16 k as one fragment: two 16-byte words of the staged tile.
__device__ inline bf16x16 frag16(const uint32_t* p) {
    const u32x4 lo = *reinterpret_cast<const u32x4*>(p);
    const u32x4 hi = *reinterpret_cast<const u32x4*>(p + 4);
    return __builtin_bit_cast(bf16x16, __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7));
}

// The BF16 elements and sums of DotBF16, and the 16 x 16 x 16 product of two fragments added to an accumulator.
struct WmmaBF16 : DotBF16 {
#if TF_DEVICE_WMMA_GFX11
    __device__ static f32x8 mma(bf16x16 a, bf16x16 b, f32x8 c) { return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c); }
#endif
};

}  // namespace rocm
}  // namespace tf
