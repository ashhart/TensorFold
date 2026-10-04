#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "launch_shape.h"

extern "C" __global__ void tf_cast_f64_kernel(const void* input, double* output,
                                            uint64_t count, uint32_t dtype) {
    for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * blockDim.x) {
        const float value = dtype == 0 ? __bfloat162float(static_cast<const __nv_bfloat16*>(input)[i]) :
                                        static_cast<const float*>(input)[i];
        output[i] = double(value);
    }
}

extern "C" cudaError_t tf_cast_f64(const void* input, double* output, uint64_t count,
                                 uint32_t dtype, cudaStream_t stream) {
    if (count == 0) return cudaSuccess;
    if (!input || !output || dtype > 1 || count > UINT64_MAX / 8) return cudaErrorInvalidValue;
    tf_cast_f64_kernel<<<tf_launch_blocks(count, 256), 256, 0, stream>>>(input, output, count, dtype);
    return cudaGetLastError();
}
