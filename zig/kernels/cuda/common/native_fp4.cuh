// TensorFold native FP4 register operations, using NVIDIA PTX-defined fragments.
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

namespace tf_nvfp4 {
__device__ __forceinline__ uint32_t contiguous8(uint32_t packed, int lane) {
    const int nibble = lane & 3;
    uint32_t result = 0;
#pragma unroll
    for (int source = 0; source < 4; ++source) {
        const uint32_t word = __shfl_sync(0xffffffffu, packed, (lane & ~3) + source);
        result |= ((word >> (nibble * 4)) & 15u) << (source * 8);
        result |= ((word >> ((nibble + 4) * 4)) & 15u) << (source * 8 + 4);
    }
    return result;
}

__device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0,
                                   uint32_t b1, uint32_t sa, uint32_t sb) {
    constexpr uint16_t zero = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
        "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
        "{%10}, {%11,%12}, {%13}, {%14,%15};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
          "r"(sa), "h"(zero), "h"(zero), "r"(sb), "h"(zero), "h"(zero));
}

}
