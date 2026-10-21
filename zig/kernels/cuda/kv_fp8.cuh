// Shared E4M3 cache-row quantization: one thread per value, FP32 row scale, nearest-even saturated codes.
#pragma once
#include <cuda_fp8.h>

namespace tf_kv_fp8 {
template <int WIDTH, bool FINITE_SCALE = false>
__device__ __forceinline__ void quant_row(float x, int d, unsigned char* q, float* s, float* red) {
    static_assert(WIDTH % 32 == 0 && WIDTH <= 1024, "one whole-warp block owns a row");
    float amax = fabsf(x);
#pragma unroll
    for (int o = 16; o > 0; o /= 2) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    if ((d & 31) == 0) red[d >> 5] = amax;
    __syncthreads();
    amax = red[0];
#pragma unroll
    for (int w = 1; w < WIDTH / 32; w++) amax = fmaxf(amax, red[w]);
    __syncthreads();
    float inv = amax > 0.0f ? __fdiv_rn(448.0f, amax) : 1.0f;
    float scale = 1.0f;
    if constexpr (FINITE_SCALE) {
        scale = amax > 0.0f ? __fdiv_rn(amax, 448.0f) : 1.0f;
        if (scale < 0x1p-126f) { scale = 0x1p-126f; inv = __fdiv_rn(1.0f, scale); }
    }
    q[d] = __nv_cvt_float_to_fp8(__fmul_rn(x, inv), __NV_SATFINITE, __NV_E4M3);
    if (d == 0) {
        if constexpr (FINITE_SCALE) *s = scale;
        else *s = amax > 0.0f ? __fdiv_rn(amax, 448.0f) : 1.0f;
    }
}
}  // namespace tf_kv_fp8
