// Exact checked decomposition of FP32 state into three BF16 operands for TensorFold's chunked GDN experiment.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "gdn_chunk_layout.cuh"

namespace tf_gdn_state_parts {
__device__ __forceinline__ float value(uint16_t bits) {
    return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&bits));
}
__device__ __forceinline__ uint16_t rounded(float x) {
    const __nv_bfloat16 out = __float2bfloat16_rn(x);
    return *reinterpret_cast<const uint16_t*>(&out);
}
__device__ __forceinline__ bool split(float x, uint16_t (&parts)[3]) {
    parts[0] = rounded(x);
    const float first = __fsub_rn(x, value(parts[0]));
    parts[1] = rounded(first);
    const float second = __fsub_rn(first, value(parts[1]));
    parts[2] = rounded(second);
    const bool normal_parts = ((parts[0] & 0x7fff) == 0 || (parts[0] & 0x7f80) != 0) &&
        ((parts[1] & 0x7fff) == 0 || (parts[1] & 0x7f80) != 0) &&
        ((parts[2] & 0x7fff) == 0 || (parts[2] & 0x7f80) != 0);
    return normal_parts && isfinite(x) && isfinite(value(parts[0])) && __fsub_rn(second, value(parts[2])) == 0;
}
__device__ __forceinline__ float combine(const float (&v)[3][4], int at) {
    return __fadd_rn(__fadd_rn(v[0][at], v[1][at]), v[2][at]);
}
__device__ __forceinline__ uint32_t pair(uint16_t a, uint16_t b) {
    return static_cast<uint32_t>(a) | (static_cast<uint32_t>(b) << 16);
}
template<int V> struct alignas(16) Shared {
    union {
        uint16_t state[3][V][136];
        uint16_t scaled_delta[3][V][24];
        float exact_state[V][132];
        float exact_delta[V][24];
    } data;
    uint16_t q[16][136], k[16][136];
    float delta[V][17], history[V][17];
    tf_gdn_chunk::Coefficients coefficients;
    unsigned lost;
};
static_assert(sizeof(Shared<16>) == 26128);
static_assert(sizeof(Shared<32>) == 41360);
}
