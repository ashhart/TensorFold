// The checkpoint math's tensor-core steps, shared by the lane matmul and the prompt GEMM: the block-scaled FP4 mma
// (NVFP4 x NVFP4, sm_120a / sm_121a), the e4m3 mma, and the swizzle their ldmatrix rows use.
#pragma once

#include <stdint.h>

namespace mma4 {

enum Mode : int { A4 = 0, A8 = 1 };

// 16-byte chunk c of row r: rows of 32 bytes swap chunks every fourth row, rows of 64 rotate by row pairs, so the
// eight rows of an ldmatrix hit eight bank groups.
template <int MODE>
__device__ __forceinline__ int chunk(int r, int c) {
    return MODE == A4 ? (c ^ ((r >> 2) & 1)) : (c ^ ((r >> 1) & 3));
}

__device__ __forceinline__ void mma_fp4(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sa,
                                        uint32_t sb) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
    const uint16_t z = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, %10, {%11, %12}, %13, {%14, %15};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "h"(z), "h"(z), "r"(sb), "h"(z),
          "h"(z));
#else
    __trap();
#endif
}

__device__ __forceinline__ void mma_fp8(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

}  // namespace mma4
