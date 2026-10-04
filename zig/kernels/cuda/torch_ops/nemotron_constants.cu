#include <cuda_runtime.h>
#include <stdint.h>
#include "launch_shape.h"
#include <math.h>

// Startup-only A_log preprocessing consumes native FP32 and preserves exp's FP32 output before negation.
extern "C" __global__ void tf_nemotron_mamba_a_f32_kernel(const float* logarithms, float* coefficients,
                                                        uint64_t count) {
    const uint64_t width = uint64_t(gridDim.x) * blockDim.x;
    for (uint64_t at = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; at < count; at += width) {
        const float magnitude = ::expf(logarithms[at]);
        coefficients[at] = __uint_as_float(__float_as_uint(magnitude) ^ 0x80000000u);
    }
}

// Development ABI uses dtype1 for FP32, raw device pointers, element count and a caller-owned stream.
extern "C" cudaError_t tf_nemotron_mamba_a_f32(const void* logarithms, void* coefficients,
                                             uint64_t count, int dtype, cudaStream_t stream) {
    if (count == 0) return cudaSuccess;
    const uintptr_t addresses = reinterpret_cast<uintptr_t>(logarithms) | reinterpret_cast<uintptr_t>(coefficients);
    if (dtype != 1 || !logarithms || !coefficients || (addresses & 3) || count > UINT64_MAX / sizeof(float))
        return cudaErrorInvalidValue;
    const uint32_t blocks = tf_launch_blocks(count, 128);
    tf_nemotron_mamba_a_f32_kernel<<<blocks, 128, 0, stream>>>(static_cast<const float*>(logarithms),
                                                           static_cast<float*>(coefficients), count);
    return cudaGetLastError();
}
