#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "launch_shape.h"

extern "C" __global__ void tf_bf16_to_f32_kernel(const __nv_bfloat16* input, float* output, uint64_t count) {
    for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * blockDim.x) output[i] = __bfloat162float(input[i]);
}

extern "C" __global__ void tf_f32_to_bf16_kernel(const float* input, __nv_bfloat16* output, uint64_t count) {
    for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * blockDim.x) output[i] = __float2bfloat16_rn(input[i]);
}

extern "C" __global__ void tf_tap_add_kernel(const __nv_bfloat16* residual, const __nv_bfloat16* pending,
                                __nv_bfloat16* output, uint64_t count) {
    for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * blockDim.x) {
        const float sum = __fadd_rn(__bfloat162float(residual[i]), __bfloat162float(pending[i]));
        output[i] = __float2bfloat16_rn(sum);
    }
}



extern "C" cudaError_t tf_bf16_to_f32(const void* input, void* output, uint64_t count, cudaStream_t stream) {
    if (count == 0) return cudaSuccess;
    if (!input || !output || count > UINT64_MAX / sizeof(float)) return cudaErrorInvalidValue;
    tf_bf16_to_f32_kernel<<<tf_launch_blocks(count, 256), 256, 0, stream>>>(static_cast<const __nv_bfloat16*>(input),
        static_cast<float*>(output), count);
    return cudaGetLastError();
}

extern "C" cudaError_t tf_f32_to_bf16(const void* input, void* output, uint64_t count, cudaStream_t stream) {
    if (count == 0) return cudaSuccess;
    if (!input || !output || count > UINT64_MAX / sizeof(float)) return cudaErrorInvalidValue;
    tf_f32_to_bf16_kernel<<<tf_launch_blocks(count, 256), 256, 0, stream>>>(static_cast<const float*>(input),
        static_cast<__nv_bfloat16*>(output), count);
    return cudaGetLastError();
}

extern "C" cudaError_t tf_tap_add(const void* residual, const void* pending, void* output,
                                uint64_t count, cudaStream_t stream) {
    if (count == 0) return cudaSuccess;
    if (!residual || !pending || !output || count > UINT64_MAX / sizeof(float)) return cudaErrorInvalidValue;
    tf_tap_add_kernel<<<tf_launch_blocks(count, 256), 256, 0, stream>>>(static_cast<const __nv_bfloat16*>(residual),
        static_cast<const __nv_bfloat16*>(pending), static_cast<__nv_bfloat16*>(output), count);
    return cudaGetLastError();
}
