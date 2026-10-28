// FP4 (E2M1) and FP8 (E4M3) codes widened to bf16 pairs on the bf16x2 FMA, and one E4M3 scale to float.
#pragma once

#include <cuda_fp8.h>
#include <stdint.h>

namespace tf_common {

// Two E2M1 codes, bits s..s+3 of each 16-bit half of w, as a bf16x2 pair.
__device__ __forceinline__ uint32_t fp4pair(uint32_t w, int s) {
    const uint32_t v = w >> s;
    const uint32_t t = ((v & 0x00070007u) << 6) | ((v & 0x00080008u) << 12);
    uint32_t r;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(t), "r"(0x7E807E80u), "r"(0x80008000u));
    return r;
}

// Two E4M3 codes, the low two bytes of v, as a bf16x2 pair.
__device__ __forceinline__ uint32_t fp8pair(uint32_t v) {
    const uint32_t t = ((v & 0x7Fu) << 4) | ((v & 0x80u) << 8) | ((v & 0x7F00u) << 12) | ((v & 0x8000u) << 16);
    uint32_t r;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(t), "r"(0x7B807B80u), "r"(0x80008000u));
    return r;
}

__device__ __forceinline__ float e4m3f(uint32_t b) {
    return __half2float(__half(__nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(b), __NV_E4M3)));
}

}  // namespace tf_common
