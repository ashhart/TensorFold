// W4A8 input conversion, shared by the CUDA entry and its CPU instruction shim.
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include "w4a8_normalize.h"
namespace tf_w4a8 {
// Within each 32-input step the int8 mma's slot s holds input P(s), the order the packed 4-bit words give each lane.
__host__ __device__ __forceinline__ int slot_input(int s) {
    const int q = (s & 15) >> 2, i = s & 3;
    return (s & 16) + 2 * q + (i & 1) + 8 * (i >> 1);
}

// Rows of x as int8 in slot order, a scale (amax / 127) and fp32 sum a 64-input group: a warp a group.
__device__ __forceinline__ void quantize_rows_body(const __nv_bfloat16* __restrict__ x, uint8_t* __restrict__ xq,
                                                     float* __restrict__ xa, float* __restrict__ xs, int K, int ldx, unsigned* __restrict__ exceptional) {
    const int row = blockIdx.x, lane = threadIdx.x & 31, g = blockIdx.y * 4 + (threadIdx.x >> 5);
    if (g * 64 >= K) return;
    const __nv_bfloat16* src = x + (size_t)row * ldx + g * 64;
    // lane holds slots 2 lane and 2 lane + 1 of the group (step lane / 16)
    const int s0 = (2 * lane) & 31, step = lane >> 4;
    const float v0 = __bfloat162float(src[step * 32 + slot_input(s0)]), v1 = __bfloat162float(src[step * 32 + slot_input(s0 + 1)]);
    if (__any_sync(0xffffffffu, !isfinite(v0) || !isfinite(v1))) {
        if (lane == 0) atomicOr(exceptional, 1u);
        return;
    }
    float amax = fmaxf(fabsf(v0), fabsf(v1)), sum = v0 + v1;
#pragma unroll
    for (int o = 16; o; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    }
    const float a = amax > 0.0f ? amax / 127.0f : 1.0f, inv = amax > 0.0f ? 127.0f / amax : 1.0f;
    if (__any_sync(0xffffffffu, !isfinite(a) || !isfinite(sum))) {
        if (lane == 0) atomicOr(exceptional, 1u);
        return;
    }
    const int q0 = __float2int_rn(tf_w4a8_normalized(v0, amax, inv)), q1 = __float2int_rn(tf_w4a8_normalized(v1, amax, inv));
    const uint16_t q = (uint16_t)((q0 & 0xFF) | ((q1 & 0xFF) << 8));
    reinterpret_cast<uint16_t*>(xq + (size_t)row * K + g * 64)[lane] = q;
    if (lane == 0) {
        xa[(size_t)row * (K / 64) + g] = a;
        xs[(size_t)row * (K / 64) + g] = sum;
    }
}

} // namespace tf_w4a8
extern "C" __global__ void __launch_bounds__(128) tf_w4a8_quantize(const __nv_bfloat16* __restrict__ x, uint8_t* __restrict__ xq,
                                                                  float* __restrict__ xa, float* __restrict__ xs, int K, int ldx, unsigned* __restrict__ exceptional) {
    tf_w4a8::quantize_rows_body(x, xq, xa, xs, K, ldx, exceptional);
}
