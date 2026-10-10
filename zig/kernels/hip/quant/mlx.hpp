#pragma once

// MLX affine words: y[m, n] = sum_g (scale[n, g] * sum_k x[m, k] * code[n, k] + bias[n, g] * sum_k x[m, k]).

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

#include "tiles/plan.hpp"

namespace tf {
namespace rocm {

enum ScaleKind : int { kScaleF32 = 0, kScaleBF16 = 1, kScaleF16 = 2 };

// One (n, k / group) table of scales or biases, row major, in its stored type.
struct GroupTable {
    const void* p;
    int kind;  // ScaleKind
    __device__ float operator[](long long i) const {
        if (kind == kScaleBF16) return static_cast<float>(static_cast<const hip_bfloat16*>(p)[i]);
        if (kind == kScaleF16) return __half2float(static_cast<const __half*>(p)[i]);
        return static_cast<const float*>(p)[i];
    }
};

struct Affine {
    const void* x;           // (m, k) row major, bf16 or fp16
    const uint32_t* words;   // (n, k * bits / 32) row major
    GroupTable scale;
    GroupTable bias;
    float* out;  // (m, n)
    int m, n, k, bits, group;
    int fp16;  // 0 is bf16, 1 is fp16
    Routing route = {};
    __half* out16 = nullptr;  // set: the decode tile rounds each output to fp16 here instead of writing out
};

// A routed block takes item ``z``: its expert's weights and its rows. False when the item has no rows.
__device__ inline bool take_item(Affine& a, int z) {
    if (!a.route.items) return true;
    const int* item = a.route.items + 3 * z;
    const int count = item[2];
    if (count <= 0) return false;
    const long long expert = item[0];
    const long long groups = a.k / a.group;
    const int table = a.scale.kind == kScaleF32 ? 4 : 2;
    a.words += expert * a.n * (static_cast<long long>(a.k) * a.bits / 32);
    a.scale.p = static_cast<const char*>(a.scale.p) + expert * a.n * groups * table;
    a.bias.p = static_cast<const char*>(a.bias.p) + expert * a.n * groups * table;
    a.route.first = item[1];
    a.m = count;
    return true;
}

__host__ __device__ inline uint32_t affine_code(const uint32_t* row, int k, int bits, int words) {
    int bit = k * bits;
    int word = bit >> 5;
    int shift = bit & 31;
    uint32_t low = row[word];
    uint32_t high = (shift + bits > 32 && word + 1 < words) ? row[word + 1] : 0u;
    uint32_t value = (low >> shift);
    if (shift + bits > 32) value |= high << ((32 - shift) & 31);
    return value & ((1u << bits) - 1u);
}

// Integer code as the BF16 value the product sees. 129 is not exact in BF16.
__host__ __device__ inline float code_bf16(uint32_t code) {
    return static_cast<float>(hip_bfloat16(static_cast<float>(code)));
}

}  // namespace rocm
}  // namespace tf
