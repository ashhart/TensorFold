#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "launch_shape.h"

extern "C" __global__ void tf_logprob_rows_kernel(const void* logits, const int64_t* columns,
                                                const float* lse, float* output, uint64_t rows,
                                                uint64_t vocab, uint64_t selected,
                                                uint32_t dtype, uint32_t* invalid) {
    const uint64_t count = rows * selected;
    for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * blockDim.x) {
        const uint64_t row = i / selected;
        const int64_t column = columns[i];
        if (column < 0 || uint64_t(column) >= vocab) {
            atomicExch(invalid, 1u);
            continue;
        }
        const uint64_t at = row * vocab + uint64_t(column);
        const float value = dtype == 0 ? __bfloat162float(static_cast<const __nv_bfloat16*>(logits)[at]) :
                                        static_cast<const float*>(logits)[at];
        output[i] = __fsub_rn(value, lse[row]);
    }
}

extern "C" __global__ void tf_logprob_keys_kernel(const float* logits, uint64_t* keys,
                                                uint64_t rows, uint64_t vocab) {
    for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < rows * vocab;
         i += uint64_t(gridDim.x) * blockDim.x) {
        uint32_t bits = __float_as_uint(logits[i]);
        if ((bits & 0x7fffffffu) == 0) bits = 0;
        const uint32_t monotone = (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
        const uint32_t signed_rank = monotone ^ 0x80000000u;
        keys[i] = (uint64_t(signed_rank) << 32) | (0xffffffffu - uint32_t(i % vocab));
    }
}



extern "C" cudaError_t tf_logprob_rows(const void* logits, const int64_t* columns,
                                     const float* lse, float* output, uint64_t rows,
                                     uint64_t vocab, uint64_t selected, uint32_t dtype,
                                     uint32_t* invalid, cudaStream_t stream) {
    if (rows == 0 || selected == 0) return cudaSuccess;
    if (!logits || !columns || !lse || !output || !invalid || vocab == 0 || dtype > 1 ||
        vocab > UINT64_MAX / rows || rows * vocab > UINT64_MAX / 4 ||
        selected > UINT64_MAX / rows || rows * selected > UINT64_MAX / 8)
        return cudaErrorInvalidValue;
    tf_logprob_rows_kernel<<<tf_launch_blocks(rows * selected, 256), 256, 0, stream>>>(logits, columns,
        lse, output, rows, vocab, selected, dtype, invalid);
    return cudaGetLastError();
}

extern "C" cudaError_t tf_logprob_keys(const float* logits, int64_t* keys,
                                     uint64_t rows, uint64_t vocab, cudaStream_t stream) {
    if (rows == 0) return cudaSuccess;
    if (!logits || !keys || vocab == 0 || vocab > UINT32_MAX ||
        vocab > UINT64_MAX / rows || rows * vocab > UINT64_MAX / 8)
        return cudaErrorInvalidValue;
    tf_logprob_keys_kernel<<<tf_launch_blocks(rows * vocab, 256), 256, 0, stream>>>(logits,
        reinterpret_cast<uint64_t*>(keys), rows, vocab);
    return cudaGetLastError();
}
