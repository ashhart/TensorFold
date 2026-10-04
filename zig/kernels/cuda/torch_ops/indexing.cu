#include <cuda_runtime.h>
#include <stdint.h>
#include "launch_shape.h"
#include <limits.h>

extern "C" __global__ void tf_widen_indices_kernel(const int32_t* input, int64_t* output, uint64_t count) {
    for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * blockDim.x) output[i] = int64_t(input[i]);
}

extern "C" __global__ void tf_select_axis_kernel(const uint8_t* input, uint8_t* output,
                                               const int64_t* indices, uint64_t outer,
                                               uint64_t input_middle, uint64_t selected,
                                               uint64_t inner_bytes, uint64_t index_stride, uint32_t* invalid) {
    const uint64_t group = blockIdx.x;
    if (group >= outer * selected) return;
    const uint64_t outer_row = group / selected;
    const int64_t pick = indices[outer_row * index_stride + group % selected];
    if (pick < 0 || uint64_t(pick) >= input_middle) {
        if (threadIdx.x == 0) atomicExch(invalid, 1u);
        return;
    }
    const uint64_t from = (outer_row * input_middle + uint64_t(pick)) * inner_bytes;
    const uint64_t to = group * inner_bytes;
    for (uint64_t i = threadIdx.x; i < inner_bytes; i += blockDim.x) output[to + i] = input[from + i];
}



extern "C" cudaError_t tf_widen_indices(const int32_t* input, int64_t* output,
                                      uint64_t count, cudaStream_t stream) {
    if (count == 0) return cudaSuccess;
    if (!input || !output || count > UINT64_MAX / sizeof(int64_t)) return cudaErrorInvalidValue;
    tf_widen_indices_kernel<<<tf_launch_blocks(count, 256), 256, 0, stream>>>(input, output, count);
    return cudaGetLastError();
}

extern "C" cudaError_t tf_select_axis(const void* input, void* output, const int64_t* indices,
                                    uint64_t outer, uint64_t input_middle, uint64_t selected,
                                    uint64_t inner_bytes, uint32_t* invalid, cudaStream_t stream) {
    if (outer == 0 || selected == 0) return cudaSuccess;
    if (!input || !output || !indices || !invalid || input_middle == 0 || inner_bytes == 0 ||
        selected > uint64_t(INT_MAX) / outer || input_middle > UINT64_MAX / outer ||
        outer * input_middle > UINT64_MAX / inner_bytes || outer * selected > UINT64_MAX / inner_bytes)
        return cudaErrorInvalidValue;
    tf_select_axis_kernel<<<uint32_t(outer * selected), 256, 0, stream>>>(static_cast<const uint8_t*>(input),
        static_cast<uint8_t*>(output), indices, outer, input_middle, selected, inner_bytes, 0, invalid);
    return cudaGetLastError();
}

extern "C" __global__ void tf_lookup_ids_kernel(const int64_t* vocabulary, uint64_t vocabulary_size,
                                              const int64_t* local, int64_t* global,
                                              uint64_t count, uint32_t* invalid) {
    for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * blockDim.x) {
        const int64_t index = local[i];
        if (index < 0 || uint64_t(index) >= vocabulary_size) atomicExch(invalid, 1u);
        else global[i] = vocabulary[index];
    }
}

extern "C" cudaError_t tf_lookup_ids(const int64_t* vocabulary, uint64_t vocabulary_size,
                                   const int64_t* local, int64_t* global, uint64_t count,
                                   uint32_t* invalid, cudaStream_t stream) {
    if (count == 0) return cudaSuccess;
    if (!vocabulary || !local || !global || !invalid || vocabulary_size == 0 ||
        count > UINT64_MAX / sizeof(int64_t) || vocabulary_size > UINT64_MAX / sizeof(int64_t))
        return cudaErrorInvalidValue;
    tf_lookup_ids_kernel<<<tf_launch_blocks(count, 256), 256, 0, stream>>>(vocabulary, vocabulary_size,
        local, global, count, invalid);
    return cudaGetLastError();
}


extern "C" cudaError_t tf_gather_columns(const void* input, void* output, const int64_t* indices,
                                       uint64_t rows, uint64_t columns, uint64_t selected,
                                       uint32_t element_bytes, uint32_t* invalid, cudaStream_t stream) {
    if (rows == 0 || selected == 0) return cudaSuccess;
    if (!input || !output || !indices || !invalid || columns == 0 ||
        (element_bytes != 2 && element_bytes != 4 && element_bytes != 8) ||
        selected > uint64_t(INT_MAX) / rows || columns > UINT64_MAX / rows ||
        rows * columns > UINT64_MAX / element_bytes || rows * selected > UINT64_MAX / 8)
        return cudaErrorInvalidValue;
    tf_select_axis_kernel<<<uint32_t(rows * selected), 256, 0, stream>>>(static_cast<const uint8_t*>(input),
        static_cast<uint8_t*>(output), indices, rows, columns, selected, element_bytes, selected, invalid);
    return cudaGetLastError();
}


extern "C" __global__ void tf_narrow_indices_kernel(const uint64_t* input, uint32_t* output, uint64_t count) {
    for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * blockDim.x) output[i] = uint32_t(input[i]);
}

extern "C" cudaError_t tf_narrow_indices(const int64_t* input, int32_t* output,
                                       uint64_t count, cudaStream_t stream) {
    if (count == 0) return cudaSuccess;
    if (!input || !output || count > UINT64_MAX / sizeof(int64_t)) return cudaErrorInvalidValue;
    tf_narrow_indices_kernel<<<tf_launch_blocks(count, 256), 256, 0, stream>>>(
        reinterpret_cast<const uint64_t*>(input), reinterpret_cast<uint32_t*>(output), count);
    return cudaGetLastError();
}
