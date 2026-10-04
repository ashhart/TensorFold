#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void tf_grammar_mask_kernel(void* logits, const uint8_t* packed,
                                               const int64_t* rows, uint64_t total_rows,
                                               uint64_t masked_rows, uint64_t columns,
                                               uint64_t packed_row_bytes, uint64_t token_offset,
                                               uint32_t element_bytes, uint32_t* invalid) {
    const uint64_t row = blockIdx.x;
    if (row >= masked_rows) return;
    const int64_t target = rows[row];
    if (target < 0 || uint64_t(target) >= total_rows) {
        if (threadIdx.x == 0) atomicExch(invalid, 1u);
        return;
    }
    for (uint64_t col = threadIdx.x; col < columns; col += blockDim.x) {
        const uint64_t token = token_offset + col;
        const uint64_t byte = token / 8;
        const bool allowed = byte < packed_row_bytes &&
            ((packed[row * packed_row_bytes + byte] >> (token % 8)) & 1u);
        if (allowed) continue;
        const uint64_t output = uint64_t(target) * columns + col;
        if (element_bytes == 2) static_cast<uint16_t*>(logits)[output] = 0xff80u;
        else static_cast<uint32_t*>(logits)[output] = 0xff800000u;
    }
}

extern "C" cudaError_t tf_grammar_mask(void* logits, const uint8_t* packed, const int64_t* rows,
                                     uint64_t total_rows, uint64_t masked_rows, uint64_t columns,
                                     uint64_t packed_row_bytes, uint64_t token_offset,
                                     uint32_t element_bytes, uint32_t* invalid, cudaStream_t stream) {
    if (masked_rows == 0 || columns == 0) return cudaSuccess;
    if (!logits || !packed || !rows || !invalid || total_rows == 0 || masked_rows > 2147483647u ||
        packed_row_bytes == 0 || (element_bytes != 2 && element_bytes != 4) ||
        columns > UINT64_MAX / total_rows || total_rows * columns > UINT64_MAX / element_bytes ||
        packed_row_bytes > UINT64_MAX / masked_rows || token_offset > UINT64_MAX - (columns - 1))
        return cudaErrorInvalidValue;
    tf_grammar_mask_kernel<<<uint32_t(masked_rows), 256, 0, stream>>>(logits, packed, rows, total_rows,
        masked_rows, columns, packed_row_bytes, token_offset, element_bytes, invalid);
    return cudaGetLastError();
}
