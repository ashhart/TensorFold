#pragma once

// ModelOpt NVFP4: e2m1 codes eight to a word, one E4M3 group scale a 16 inputs, and the projection's global scale.

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

// A warp's sum of one lane's value: a tree, so the order is the same for every row count.
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

// A warp an output element: its lanes read one column's consecutive words, a group's two words a lane pair.
__device__ __forceinline__ void nvfp4_product(Nvfp4 a) {
    const int lane = threadIdx.x & 31;
    const int warps = blockDim.x >> 5;
    const long long total = static_cast<long long>(a.m) * a.n;
    const int words_row = a.k / 8;             // eight codes a word
    const int groups = a.k / a.group;
    const long long stride = static_cast<long long>(gridDim.x) * warps;
    for (long long item = static_cast<long long>(blockIdx.x) * warps + (threadIdx.x >> 5); item < total;
         item += stride) {
        const int row = static_cast<int>(item / a.n);
        const int col = static_cast<int>(item % a.n);
        const uint32_t* w = a.words + static_cast<long long>(col) * words_row;
        const hip_bfloat16* x = static_cast<const hip_bfloat16*>(a.x) + static_cast<long long>(row) * a.k;
        float sum = 0.f;
        for (int base = 0; base < words_row; base += 32) {
            const int wd = base + lane;
            if (wd < words_row) {
                const uint32_t word = w[wd];
                float q = 0.f;
#pragma unroll
                for (int c = 0; c < 8; ++c) {
                    q = fmaf(nvfp4_level(word >> (4 * c)), static_cast<float>(x[wd * 8 + c]), q);
                }
                // a group is 16 inputs: two words, so the lane pair folds one and its even lane scales it
                q += __shfl_xor_sync(0xffffffffu, q, 1);
                if ((lane & 1) == 0) {
                    const float scale = nvfp4_scale(a.scale, static_cast<long long>(col) * groups + (wd >> 1));
                    sum = fmaf(q, scale, sum);
                }
            }
        }
        sum = warp_sum(sum);
        if (lane == 0) a.out[item] = sum * a.global;
    }
}

}  // namespace rocm
}  // namespace tf

// This path serves bf16 activations only: the engine refuses an fp16 build for a ModelOpt checkpoint.
extern "C" __global__ void __launch_bounds__(256) tf_nvfp4_gemm(tf::rocm::Nvfp4 a) {
    tf::rocm::nvfp4_product(a);
}
