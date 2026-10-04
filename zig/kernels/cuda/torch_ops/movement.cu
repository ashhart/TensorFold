#include <cuda_runtime.h>
#include <stdint.h>
#include <limits.h>

struct TfCopySpan {
    const uint8_t* source;
    uint8_t* destination;
    uint64_t bytes;
};

extern "C" __global__ void tf_copy_spans_kernel(const TfCopySpan* spans, uint32_t count) {
    const uint32_t item = blockIdx.x;
    if (item >= count) return;
    const TfCopySpan span = spans[item];
    for (uint64_t i = threadIdx.x; i < span.bytes; i += blockDim.x)
        span.destination[i] = span.source[i];
}

extern "C" cudaError_t tf_copy_spans(const TfCopySpan* spans, uint32_t count, cudaStream_t stream) {
    if (count == 0) return cudaSuccess;
    if (!spans || count > INT_MAX) return cudaErrorInvalidValue;
    tf_copy_spans_kernel<<<count, 256, 0, stream>>>(spans, count);
    return cudaGetLastError();
}

extern "C" __global__ void tf_gather_rows_kernel(const uint8_t* source, uint8_t* destination,
                                     const int64_t* indices, uint64_t rows,
                                     uint64_t source_rows, uint64_t row_bytes, uint32_t* invalid) {
    const uint64_t row = blockIdx.x;
    if (row >= rows) return;
    const int64_t pick = indices[row];
    if (pick < 0 || uint64_t(pick) >= source_rows) {
        if (threadIdx.x == 0) atomicExch(invalid, 1u);
        return;
    }
    for (uint64_t i = threadIdx.x; i < row_bytes; i += blockDim.x)
        destination[row * row_bytes + i] = source[uint64_t(pick) * row_bytes + i];
}

extern "C" cudaError_t tf_gather_rows(const void* source, void* destination, const int64_t* indices,
                                    uint64_t rows, uint64_t source_rows, uint64_t row_bytes,
                                    uint32_t* invalid, cudaStream_t stream) {
    if (rows == 0) return cudaSuccess;
    if (!source || !destination || !indices || !invalid || row_bytes == 0 || rows > INT_MAX ||
        rows > UINT64_MAX / row_bytes || source_rows > UINT64_MAX / row_bytes)
        return cudaErrorInvalidValue;
    tf_gather_rows_kernel<<<uint32_t(rows), 256, 0, stream>>>(static_cast<const uint8_t*>(source),
        static_cast<uint8_t*>(destination), indices, rows, source_rows, row_bytes, invalid);
    return cudaGetLastError();
}

extern "C" __global__ void tf_strided_copy_kernel(const uint8_t* source, uint8_t* destination,
                                      uint64_t outer, uint64_t middle, uint64_t inner_bytes,
                                      uint64_t source_outer, uint64_t source_middle,
                                      uint64_t destination_outer, uint64_t destination_middle) {
    const uint64_t group = blockIdx.x;
    const uint64_t total = middle * inner_bytes;
    for (uint64_t i = threadIdx.x; group < outer && i < total; i += blockDim.x) {
        const uint64_t row = i / inner_bytes;
        const uint64_t byte = i % inner_bytes;
        destination[group * destination_outer + row * destination_middle + byte] =
            source[group * source_outer + row * source_middle + byte];
    }
}

static bool tf_span_fits(uint64_t outer, uint64_t middle, uint64_t inner,
                         uint64_t outer_stride, uint64_t middle_stride) {
    if (outer_stride && outer - 1 > UINT64_MAX / outer_stride) return false;
    if (middle_stride && middle - 1 > UINT64_MAX / middle_stride) return false;
    const uint64_t a = (outer - 1) * outer_stride;
    const uint64_t b = (middle - 1) * middle_stride;
    return b <= UINT64_MAX - a && inner <= UINT64_MAX - a - b;
}

extern "C" cudaError_t tf_strided_copy(const void* source, void* destination, uint64_t outer,
                                     uint64_t middle, uint64_t inner_bytes,
                                     uint64_t source_outer, uint64_t source_middle,
                                     uint64_t destination_outer, uint64_t destination_middle,
                                     cudaStream_t stream) {
    if (outer == 0 || middle == 0) return cudaSuccess;
    if (!source || !destination || inner_bytes == 0 || outer > INT_MAX ||
        middle > UINT64_MAX / inner_bytes || source_middle < inner_bytes ||
        destination_middle < inner_bytes ||
        !tf_span_fits(outer, middle, inner_bytes, source_outer, source_middle) ||
        !tf_span_fits(outer, middle, inner_bytes, destination_outer, destination_middle))
        return cudaErrorInvalidValue;
    tf_strided_copy_kernel<<<uint32_t(outer), 256, 0, stream>>>(static_cast<const uint8_t*>(source),
        static_cast<uint8_t*>(destination), outer, middle, inner_bytes, source_outer, source_middle,
        destination_outer, destination_middle);
    return cudaGetLastError();
}
