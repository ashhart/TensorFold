// Native SM120/SM121 FP8 block-scaled register instruction with unit scaling.
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

namespace tf_native_fp8 {
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

}
