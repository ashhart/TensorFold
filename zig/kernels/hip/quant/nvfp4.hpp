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

// e2m1's sixteen levels, in the order its sign and three magnitude bits name them.
constexpr double kE2m1[16] = {
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
};

// The same levels as BF16 and FP16 bits, so a tile makes a pair with two loads and no conversion.
__device__ constexpr unsigned short kE2m1Bf16[16] = {
    0x0000, 0x3F00, 0x3F80, 0x3FC0, 0x4000, 0x4040, 0x4080, 0x40C0,
    0x8000, 0xBF00, 0xBF80, 0xBFC0, 0xC000, 0xC040, 0xC080, 0xC0C0,
};
__device__ constexpr unsigned short kE2m1F16[16] = {
    0x0000, 0x3800, 0x3C00, 0x3E00, 0x4000, 0x4200, 0x4400, 0x4600,
    0x8000, 0xB800, 0xBC00, 0xBE00, 0xC000, 0xC200, 0xC400, 0xC600,
};

struct Nvfp4 {
    const void* x;          // (m, k) row major, bf16 or fp16
    const uint32_t* words;  // (n, k / 8) row major: 4-bit codes, 8 to a word
    GroupTable scale;       // (n, k / 16), E4M3's values as BF16
    float* out;             // (m, n)
    int m, n, k, group;
    int fp16;  // 0 is bf16, 1 is fp16
    Routing route = {};
    __half* out16 = nullptr;  // set: the tile rounds each output to the activation type here instead of writing out
    float global = 1.0f;      // the checkpoint's second scale, applied to the widened group scale
};

// A routed block takes item ``z``: its expert's weights and its rows. False when the item has no rows.
__device__ inline bool take_item(Nvfp4& a, int z) {
    if (!a.route.items) return true;
    const int* item = a.route.items + 3 * z;
    const int count = item[2];
    if (count <= 0) return false;
    const long long expert = item[0];
    const long long groups = a.k / a.group;
    a.words += expert * a.n * (static_cast<long long>(a.k) / 8);
    a.scale.p = static_cast<const char*>(a.scale.p) + expert * a.n * groups * 2;
    a.route.first = item[1];
    a.m = count;
    return true;
}

}  // namespace rocm
}  // namespace tf
