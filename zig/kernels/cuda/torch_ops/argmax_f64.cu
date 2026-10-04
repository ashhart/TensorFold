#include <cuda_runtime.h>
#include <stdint.h>
#include <limits.h>
#include "argmax_order.h"

struct TfDoubleMaximum {
    uint64_t key;
    uint64_t column;
    uint32_t nan;
};

__device__ __forceinline__ TfDoubleMaximum tf_double_empty() {
    return {0, UINT64_MAX, 0};
}

__device__ __forceinline__ TfDoubleMaximum tf_double_pick(TfDoubleMaximum a, TfDoubleMaximum b) {
    return tf_max_left_wins(a.key, a.nan, a.column, b.key, b.nan, b.column) ? a : b;
}

__device__ __forceinline__ uint64_t tf_shuffle_word(uint64_t word, int distance) {
    const uint32_t lo = __shfl_down_sync(0xffffffffu, uint32_t(word), distance);
    const uint32_t hi = __shfl_down_sync(0xffffffffu, uint32_t(word >> 32), distance);
    return (uint64_t(hi) << 32) | lo;
}

__device__ __forceinline__ TfDoubleMaximum tf_double_warp(TfDoubleMaximum value) {
    for (int distance = 16; distance; distance >>= 1) {
        TfDoubleMaximum next{tf_shuffle_word(value.key, distance),
                             tf_shuffle_word(value.column, distance),
                             __shfl_down_sync(0xffffffffu, value.nan, distance)};
        value = tf_double_pick(value, next);
    }
    return value;
}

extern "C" __global__ void tf_argmax_f64_kernel(const uint64_t* input, int64_t* output,
                                              uint64_t rows, uint64_t columns, uint64_t stride) {
    const uint64_t row = blockIdx.x;
    if (row >= rows) return;
    TfDoubleMaximum maximum = tf_double_empty();
    for (uint64_t col = threadIdx.x; col < columns; col += blockDim.x) {
        uint64_t bits = input[row * stride + col];
        const uint64_t magnitude = bits & UINT64_C(0x7fffffffffffffff);
        if (magnitude == 0) bits = 0;
        const uint64_t key = bits & UINT64_C(0x8000000000000000) ? ~bits : bits ^ UINT64_C(0x8000000000000000);
        const TfDoubleMaximum current{key, col, uint32_t(magnitude > UINT64_C(0x7ff0000000000000))};
        maximum = tf_double_pick(maximum, current);
    }
    maximum = tf_double_warp(maximum);
    __shared__ TfDoubleMaximum partials[8];
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5;
    if (lane == 0) partials[warp] = maximum;
    __syncthreads();
    if (warp == 0) {
        maximum = lane < 8 ? partials[lane] : tf_double_empty();
        maximum = tf_double_warp(maximum);
        if (lane == 0) output[row] = int64_t(maximum.column);
    }
}

extern "C" cudaError_t tf_argmax_f64(const double* input, int64_t* output, uint64_t rows,
                                   uint64_t columns, uint64_t stride, cudaStream_t stream) {
    if (rows == 0) return cudaSuccess;
    if (!input || !output || rows > INT_MAX || columns == 0 || stride < columns ||
        columns > INT64_MAX || columns > UINT64_MAX / 8 || (rows - 1 && stride > (UINT64_MAX / 8 - columns) / (rows - 1)) ||
        (reinterpret_cast<uintptr_t>(input) | reinterpret_cast<uintptr_t>(output)) & 7u)
        return cudaErrorInvalidValue;
    tf_argmax_f64_kernel<<<uint32_t(rows), 256, 0, stream>>>(reinterpret_cast<const uint64_t*>(input),
        output, rows, columns, stride);
    return cudaGetLastError();
}
