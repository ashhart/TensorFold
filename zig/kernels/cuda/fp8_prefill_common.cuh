// Native SM120 FP8 instructions and fragment helpers for prompt-only projections.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "common/qmm_frag.cuh"

namespace tf_fp8_fast {

__device__ __forceinline__ int input_byte(int logical) {
    const int r = logical & 15;
    return (logical & ~15) | (r & 1) | ((r & 6) << 1) | ((r & 8) >> 2);
}

__device__ __forceinline__ int chunk(int row, int logical) {
    return logical ^ ((row >> 1) & 3);
}

__device__ __forceinline__ void mma(float (&sum)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
    constexpr uint32_t one = 0x7f;
    constexpr uint16_t selector = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X."
        "m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
        "{%10}, {%11,%12}, {%13}, {%14,%15};"
        : "+f"(sum[0]), "+f"(sum[1]), "+f"(sum[2]), "+f"(sum[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
          "r"(one), "h"(selector), "h"(selector), "r"(one), "h"(selector), "h"(selector));
#else
    __trap();
#endif
}

__device__ __forceinline__ float decode(uint8_t code) {
    return __half2float(__half(__nv_cvt_fp8_to_halfraw(code, __NV_E4M3)));
}

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t selectors) {
    uint32_t out;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(out) : "r"(a), "r"(b), "r"(selectors));
    return out;
}

// A quartet of integer bytes in 0..15 becomes the exact E4M3 encodings of the same integers.
__device__ __forceinline__ uint32_t integer_fp8(uint32_t bytes) {
    const uint32_t select = (bytes & 7u) | ((bytes >> 4) & 0x70u) |
                            ((bytes >> 8) & 0x700u) | ((bytes >> 12) & 0x7000u);
    const uint32_t low = prmt(0x44403800u, 0x4e4c4a48u, select);
    const uint32_t high = prmt(0x53525150u, 0x57565554u, select);
    const uint32_t mask = ((bytes >> 3) & 0x01010101u) * 255u;
    return (low & ~mask) | (high & mask);
}

// TensorFold's affine word places K positions 0,8,16,24,1,9,17,25 in its eight nibbles.
__device__ __forceinline__ uint32_t affine_fragment(uint32_t word, int half) {
    const int shift = half * 8;
    const uint32_t a = (word >> shift) & 0x000f000fu;
    const uint32_t b = (word >> (shift + 4)) & 0x000f000fu;
    return integer_fp8(prmt(a, b, 0x6420u));
}

template<bool F32>
__device__ __forceinline__ void store(void* output, size_t at, float value) {
    if constexpr (F32) static_cast<float*>(output)[at] = value;
    else static_cast<__nv_bfloat16*>(output)[at] = __float2bfloat16_rn(value);
}

template<int BM, int WN, int MT, int NT, bool F32, bool RowScaled = false>
__device__ __forceinline__ void store_tile(float (&sums)[MT][NT][4], void* output,
    int m0, int n0, int m, int n, float alpha, const float* row_scales = nullptr,
    const float* weight_scales = nullptr, int k_groups = 0) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + (warp / WN) * (MT * 16) + i * 16 + (lane >> 2) + h * 8;
                const int col = n0 + (warp % WN) * (NT * 8) + j * 8 + (lane & 3) * 2;
                if (row < m) for (int pair = 0; pair < 2; ++pair) if (col + pair < n) {
                    float scale = alpha;
                    if constexpr (RowScaled) {
                        const int column = col + pair;
                        scale = __fmul_rn(row_scales[row],
                            weight_scales[static_cast<size_t>(column / 64) * k_groups * 64 + column % 64]);
                    }
                    store<F32>(output, static_cast<size_t>(row) * n + col + pair,
                               __fmul_rn(sums[i][j][h * 2 + pair], scale));
                }
            }
}
}
