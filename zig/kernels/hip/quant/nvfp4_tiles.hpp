#pragma once

// ModelOpt NVFP4's product: a group's 16 inputs folded in f32, its scale applied last, a row's bits its own.

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

#include "quant/nvfp4.hpp"

namespace tf {
namespace rocm {

// The group table's entry at `i` as the float it holds: BF16's bits shifted into f32, FP16 converted, or F32 read.
__device__ __forceinline__ float nvfp4_scale(const GroupTable& t, long long i) {
    const unsigned short* h = static_cast<const unsigned short*>(t.p);
    if (t.kind == kScaleF32) return static_cast<const float*>(t.p)[i];
    if (t.kind == kScaleF16) return __half2float(h[i]);
    return __uint_as_float(static_cast<unsigned>(h[i]) << 16);
}

// A nibble's e2m1 level as a float: the BF16 bit table shifted into f32, which is the same value.
__device__ __forceinline__ float nvfp4_level(uint32_t code) {
    return __uint_as_float(static_cast<unsigned>(kE2m1Bf16[code & 0xfu]) << 16);
}

template <typename E>
__device__ __forceinline__ void nvfp4_product(Nvfp4 a) {
    const int col_words = a.k / 8;             // a column's code words: eight nibbles each
    const int groups = a.k / a.group;
    const int per_group_words = a.group / 8;   // a group's words: 16 inputs are two
    const long long total = static_cast<long long>(a.m) * a.n;
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += stride) {
        const int row = static_cast<int>(i / a.n);
        const int col = static_cast<int>(i % a.n);
        const uint32_t* w = a.words + static_cast<long long>(col) * col_words;
        const E* x = static_cast<const E*>(a.x) + static_cast<long long>(row) * a.k;
        float acc = 0.f;
        for (int g = 0; g < groups; ++g) {
            float q = 0.f;
            for (int wd = 0; wd < per_group_words; ++wd) {
                const uint32_t word = w[g * per_group_words + wd];
#pragma unroll
                for (int c = 0; c < 8; ++c) {
                    q = fmaf(nvfp4_level(word >> (4 * c)), static_cast<float>(x[g * a.group + wd * 8 + c]), q);
                }
            }
            // each E4M3 scale is exact in f32: fold the group with its own scale, the global once at the end
            acc = fmaf(q, nvfp4_scale(a.scale, static_cast<long long>(col) * groups + g), acc);
        }
        acc *= a.global;
        if (a.out16) a.out16[i] = __float2half_rn(acc);
        else a.out[i] = acc;
    }
}

}  // namespace rocm
}  // namespace tf

// This path serves bf16 activations only: the engine refuses an fp16 build for a ModelOpt checkpoint.
extern "C" __global__ void __launch_bounds__(256) tf_nvfp4_gemm(tf::rocm::Nvfp4 a) {
    tf::rocm::nvfp4_product<hip_bfloat16>(a);
}
