// The checkpoint's NVFP4 activation format, shared by the row quantizer (act.cu) and the fused SwiGLU epilogue
// (gemm_ws.cu): per 16 inputs an e4m3 scale of amax / 6 under the static global scale, e2m1 codes to nearest.
#pragma once

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

namespace nvfp4q {

// |v| to the nearest e2m1 magnitude code (ties to the even code), saturating at 6, with v's sign.
__device__ __forceinline__ uint32_t e2m1(float v) {
    const float a = fabsf(v);
    uint32_t c;
    if (a <= 0.25f) c = 0;            // 0
    else if (a < 0.75f) c = 1;        // 0.5
    else if (a <= 1.25f) c = 2;       // 1
    else if (a < 1.75f) c = 3;        // 1.5
    else if (a <= 2.5f) c = 4;        // 2
    else if (a < 3.5f) c = 5;         // 3
    else if (a <= 5.0f) c = 6;        // 4
    else c = 7;                       // 6
    return (v < 0.0f && c != 0) ? (c | 8u) : c;
}

struct Scale {
    uint32_t sf8;                     // the block's e4m3 scale byte
    float mul;                        // what each input is multiplied by before e2m1
};

// A 16-input block's scale from its amax under the global scale ``g`` (1 / input_scale).
__device__ __forceinline__ Scale block_scale(float amax, float g) {
    const __nv_fp8_storage_t sf8 = __nv_cvt_float_to_fp8(g * (amax * (1.0f / 6.0f)), __NV_SATFINITE, __NV_E4M3);
    const float sf = __half2float(__half(__nv_cvt_fp8_to_halfraw(sf8, __NV_E4M3)));
    return {static_cast<uint32_t>(sf8), sf != 0.0f ? __fdiv_rn(g, sf) : 0.0f};
}

}  // namespace nvfp4q
