// Reuses only our validated affine4 and qmmf transforms; no stored weight is re-quantized.
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace tf_lt_gpu {
__device__ __forceinline__ size_t scale_at(int outer, int group, int groups) {
    return (size_t(outer / 128) * ((groups + 3) / 4) + group / 4) * 512 +
        (outer % 32) * 16 + ((outer % 128) / 32) * 4 + group % 4;
}
__device__ __forceinline__ uint4 transpose(uint4 value, int chunk) {
    const uint32_t far0 = __shfl_xor_sync(0xffffffffu, chunk & 2 ? value.x : value.z, 2, 4);
    const uint32_t far1 = __shfl_xor_sync(0xffffffffu, chunk & 2 ? value.y : value.w, 2, 4);
    const uint4 half = chunk & 2 ? make_uint4(far0, far1, value.z, value.w)
        : make_uint4(value.x, value.y, far0, far1);
    const uint32_t near0 = __shfl_xor_sync(0xffffffffu, chunk & 1 ? half.x : half.y, 1, 4);
    const uint32_t near1 = __shfl_xor_sync(0xffffffffu, chunk & 1 ? half.z : half.w, 1, 4);
    return chunk & 1 ? make_uint4(near0, half.y, near1, half.w) : make_uint4(half.x, near0, half.z, near1);
}
__device__ __forceinline__ uint32_t pair4(uint32_t word, int nibble) {
    return ((word >> (4 * nibble)) & 15) | (((word >> (4 * (nibble + 4))) & 15) << 4) |
        (((word >> (4 * (nibble + 1))) & 15) << 8) | (((word >> (4 * (nibble + 5))) & 15) << 12);
}
__device__ __forceinline__ uint2 raw_codes(uint2 raw, int chunk) {
    const uint4 values = transpose(make_uint4(pair4(raw.x, 0), pair4(raw.x, 2), pair4(raw.y, 0), pair4(raw.y, 2)), chunk);
    const uint32_t first = values.x | (values.y << 16), second = values.z | (values.w << 16);
    return make_uint2(__byte_perm(first, second, 0x6420), __byte_perm(first, second, 0x7531));
}
__device__ __forceinline__ uint32_t affine_pair(uint32_t codes, int pair, uint32_t scale, uint32_t bias) {
    const __nv_bfloat162 values = __floats2bfloat162_rn(float((codes >> (pair * 8)) & 15),
        float((codes >> (pair * 8 + 4)) & 15));
    const uint32_t packed = *reinterpret_cast<const uint32_t*>(&values);
    uint32_t result;
    // Exactly the native BF16 weight rounding boundary, not an FP32 multiply followed by a separate addition.
    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(result) : "r"(packed), "r"(scale), "r"(bias));
    return result;
}
__global__ __launch_bounds__(256) void affine4(const uint32_t* source, const uint16_t* scales,
    const uint16_t* bias, uint16_t* output, int n, int k, int npad) {
    const int group = blockIdx.x, tile = blockIdx.y, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int row = tile * 64 + warp * 8 + lane / 4, chunk = lane & 3;
    const size_t at = ((size_t(tile) * (k / 64) + group) * 8 + warp) * 32 + lane;
    const uint2 codes = raw_codes(reinterpret_cast<const uint2*>(source)[at], chunk);
    const uint32_t scale = uint32_t(scales[size_t(group) * npad + row]) * 0x10001u;
    const uint32_t offset = uint32_t(bias[size_t(group) * npad + row]) * 0x10001u;
    if (row < n) {
        auto* target = reinterpret_cast<uint4*>(output + size_t(row) * k + group * 64 + chunk * 16);
        target[0] = make_uint4(affine_pair(codes.x, 0, scale, offset), affine_pair(codes.x, 1, scale, offset),
            affine_pair(codes.x, 2, scale, offset), affine_pair(codes.x, 3, scale, offset));
        target[1] = make_uint4(affine_pair(codes.y, 0, scale, offset), affine_pair(codes.y, 1, scale, offset),
            affine_pair(codes.y, 2, scale, offset), affine_pair(codes.y, 3, scale, offset));
    }
}
__global__ __launch_bounds__(256) void nvfp4_unpack(const uint8_t* source, const uint8_t* scales,
    uint8_t* output, uint8_t* output_scales, int n, int k) {
    const int group = blockIdx.x, tile = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int row = tile * 64 + warp * 8 + lane / 4, chunk = lane & 3;
    const size_t at = ((size_t(tile) * (k / 64) + group) * 8 + warp) * 32 + lane;
    const uint2 codes = raw_codes(reinterpret_cast<const uint2*>(source)[at], chunk);
    if (row < n) *reinterpret_cast<uint2*>(output + size_t(row) * (k / 2) + group * 32 + chunk * 8) = codes;
    const int scale_row = tile * 64 + tid / 4;
    if (scale_row < n) output_scales[scale_at(scale_row, group * 4 + tid % 4, k / 16)] =
        scales[(size_t(tile) * (k / 64) + group) * 256 + tid];
}
__global__ __launch_bounds__(256) void nvfp4_quant(const __nv_bfloat16* x, uint8_t* codes, uint8_t* scales,
    int m, int k, int ldx, float inverse) {
    const size_t group = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (group >= size_t(m) * (k / 16)) return;
    const int row = group / (k / 16), col = (group % (k / 16)) * 16;
    float values[16], maximum = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] = __bfloat162float(x[size_t(row) * ldx + col + i]);
        maximum = (isnan(values[i]) || isnan(maximum)) ? nanf("") : fmaxf(maximum, fabsf(values[i]));
    }
    // Matches Native NVFP4 for NaN, infinity, saturation, zero groups and each FP32 rounding point.
    const uint8_t scale = isfinite(maximum) ?
        __nv_cvt_float_to_fp8(__fdiv_rn(__fmul_rn(maximum, inverse), 6.0f), __NV_SATFINITE, __NV_E4M3) : 0x7f;
    scales[scale_at(row, col / 16, k / 16)] = scale;
    const float divisor = __half2float(__half(__nv_cvt_fp8_to_halfraw(scale, __NV_E4M3)));
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float2 pair = divisor == 0 ? make_float2(0, 0) :
            make_float2(__fdiv_rn(__fmul_rn(values[2 * i], inverse), divisor),
                __fdiv_rn(__fmul_rn(values[2 * i + 1], inverse), divisor));
        codes[group * 8 + i] = __nv_cvt_float2_to_fp4x2(pair, __NV_E2M1, cudaRoundNearest);
    }
}
} // namespace tf_lt_gpu
