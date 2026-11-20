#pragma once

// ModelOpt NVFP4's words: e2m1 codes eight to a word, one E4M3 group scale a 16 inputs, one global scale.

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

#include "quant/mlx.hpp"
#include "tiles/plan.hpp"

namespace tf {
namespace rocm {


// e2m1's sixteen levels as BF16 bits, so a nibble's level is one load and no conversion.
__device__ constexpr unsigned short kE2m1Bf16[16] = {
    0x0000, 0x3F00, 0x3F80, 0x3FC0, 0x4000, 0x4040, 0x4080, 0x40C0,
    0x8000, 0xBF00, 0xBF80, 0xBFC0, 0xC000, 0xC040, 0xC080, 0xC0C0,
};
struct Nvfp4 {
    const void* x;          // (m, k) row major, bf16
    const uint32_t* words;  // (n, k / 8) row major: 4-bit codes, 8 to a word
    GroupTable scale;       // (n, k / 16), E4M3's values as BF16
    float* out;             // (m, n)
    int m, n, k, group;
    float global = 1.0f;  // the checkpoint's second scale, applied to the widened group scale
};


}  // namespace rocm
}  // namespace tf
