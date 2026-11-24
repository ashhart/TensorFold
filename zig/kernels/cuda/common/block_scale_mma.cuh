// Block-scaled MMA (sm_120a / sm_121a): m16n8k32 E4M3 x {E4M3, E2M1} with UE8M0 scales, one scale a row / column.
#pragma once

#include <stdint.h>

namespace tf_common {

// One 32-wide K group: a ue8m0 scale a row of A (lane 4r, row r + 8 lane 4r + 1) and a column of B (lane 4c).
__device__ __forceinline__ void mma_mx_e4m3(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2], uint32_t sa,
                                            uint32_t sb) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
    const uint16_t z = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, %10, {%11, %12}, %13, {%14, %15};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa), "h"(z), "h"(z), "r"(sb), "h"(z),
          "h"(z));
#else
    __trap();
#endif
}

// The same step with B in e2m1, each code in bits [5:2] of its byte.
__device__ __forceinline__ void mma_mx_e2m1(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2], uint32_t sa,
                                            uint32_t sb) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
    const uint16_t z = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, %10, {%11, %12}, %13, {%14, %15};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa), "h"(z), "h"(z), "r"(sb), "h"(z),
          "h"(z));
#else
    __trap();
#endif
}

}  // namespace tf_common
