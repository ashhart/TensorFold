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
