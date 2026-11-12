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

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

// A warp R rows of one output column: its lanes read the column's consecutive words, so a column's weights load once
// and serve the R rows; a group's two words are folded by a lane pair, its even lane applies the group's scale.
template <int R>
__device__ __forceinline__ void nvfp4_rows(Nvfp4 a) {
    const int lane = threadIdx.x & 31;
    const int warps = blockDim.x >> 5;
    const int row_blocks = (a.m + R - 1) / R;
    const long long total = static_cast<long long>(row_blocks) * a.n;
    const int words_row = a.k / 8;
    const int groups = a.k / a.group;
    const long long stride = static_cast<long long>(gridDim.x) * warps;
    for (long long item = static_cast<long long>(blockIdx.x) * warps + (threadIdx.x >> 5); item < total;
         item += stride) {
        const int rb = static_cast<int>(item / a.n);
        const int col = static_cast<int>(item % a.n);
        const int row0 = rb * R;
        const int rows = a.m - row0 < R ? a.m - row0 : R;
        const uint32_t* w = a.words + static_cast<long long>(col) * words_row;
        const hip_bfloat16* x = static_cast<const hip_bfloat16*>(a.x) + static_cast<long long>(row0) * a.k;
        float acc[R];
#pragma unroll
        for (int r = 0; r < R; ++r) acc[r] = 0.f;
        for (int base = 0; base < words_row; base += 32) {
            const int wd = base + lane;
            if (wd < words_row) {
                const uint32_t word = w[wd];
                float lvl[8];
#pragma unroll
                for (int c = 0; c < 8; ++c) lvl[c] = nvfp4_level(word >> (4 * c));
                const float scale = nvfp4_scale(a.scale, static_cast<long long>(col) * groups + (wd >> 1));
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    float q = 0.f;
                    if (r < rows) {
#pragma unroll
                        for (int c = 0; c < 8; ++c) {
                            q = fmaf(lvl[c], static_cast<float>(x[static_cast<long long>(r) * a.k + wd * 8 + c]), q);
                        }
                    }
                    q += __shfl_xor_sync(0xffffffffu, q, 1);
                    if ((lane & 1) == 0) acc[r] = fmaf(q, scale, acc[r]);
                }
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float sum = warp_sum(acc[r]);
            if (lane == 0 and r < rows) a.out[static_cast<long long>(row0 + r) * a.n + col] = sum * a.global;
        }
    }
}

}  // namespace rocm
}  // namespace tf

// This path serves bf16 activations only: the engine refuses an fp16 build for a ModelOpt checkpoint.
extern "C" __global__ void __launch_bounds__(256) tf_nvfp4_gemm(tf::rocm::Nvfp4 a) {
    if (a.m >= 8) tf::rocm::nvfp4_rows<8>(a);
    else if (a.m >= 4) tf::rocm::nvfp4_rows<4>(a);
    else tf::rocm::nvfp4_rows<1>(a);
}
