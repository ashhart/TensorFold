#pragma once

// The torch ops between kernels, bit for bit: round where torch rounds, no contraction; kind 0 fp32, 1 fp16, 2 bf16.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace {

__device__ __forceinline__ float bf16_to_float(uint16_t h) { return __uint_as_float(static_cast<uint32_t>(h) << 16); }

// c10::BFloat16's round to nearest even, NaN to 0x7fc0.
__device__ __forceinline__ uint16_t float_to_bf16(float f) {
    if (f != f) return 0x7fc0;
    uint32_t u = __float_as_uint(f);
    uint32_t bias = ((u >> 16) & 1u) + 0x7fffu;
    return static_cast<uint16_t>((u + bias) >> 16);
}

__device__ __forceinline__ float load(const void* p, int kind, long long i) {
    if (kind == 0) return static_cast<const float*>(p)[i];
    if (kind == 1) return __half2float(static_cast<const __half*>(p)[i]);
    return bf16_to_float(static_cast<const uint16_t*>(p)[i]);
}

__device__ __forceinline__ void store(void* p, int kind, long long i, float v) {
    if (kind == 0) static_cast<float*>(p)[i] = v;
    else if (kind == 1) static_cast<__half*>(p)[i] = __float2half_rn(v);
    else static_cast<uint16_t*>(p)[i] = float_to_bf16(v);
}

// A value rounded to `kind` and widened again: what torch holds after an op that returns that dtype.
__device__ __forceinline__ float rounded(float v, int kind) {
    if (kind == 1) return __half2float(__float2half_rn(v));
    if (kind == 2) return bf16_to_float(float_to_bf16(v));
    return v;
}

// torch's silu and sigmoid on an fp32 operand: x / (1 + exp(-x)) and 1 / (1 + exp(-x)).
__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }
__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ long long gid() { return static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; }

}  // namespace

