#include <cuda_runtime.h>
#include <stdint.h>
#include "launch_shape.h"

extern "C" __global__ void tf_nemotron_limits_kernel(const double* normalized, const double* ranked,
                                                   const double* parameters, int64_t* limits,
                                                   uint64_t rows, uint64_t columns, uint32_t minimum) {
    const uint64_t row = blockIdx.x;
    if (row >= rows) return;
    const double floor = __dadd_rn(ranked[row * columns], parameters[2]);
    uint64_t below = 0, above = 0;
    for (uint64_t column = threadIdx.x; column < columns; column += blockDim.x) {
        below += normalized[row * columns + column] < parameters[1];
        above += ranked[row * columns + column] >= floor;
    }
    __shared__ uint64_t counts[2][256];
    counts[0][threadIdx.x] = below;
    counts[1][threadIdx.x] = above;
    __syncthreads();
    for (uint32_t step = 128; step; step >>= 1) {
        if (threadIdx.x < step) {
            counts[0][threadIdx.x] += counts[0][threadIdx.x + step];
            counts[1][threadIdx.x] += counts[1][threadIdx.x + step];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const uint64_t cutoff = counts[0][0] + 1;
        limits[row] = int64_t(minimum && counts[1][0] < cutoff ? counts[1][0] : cutoff);
    }
}

extern "C" __global__ void tf_nemotron_probability_kernel(const double* probabilities,
                                                        const double* totals, const int64_t* selected,
                                                        float* output, uint64_t rows, uint64_t columns,
                                                        uint32_t* invalid) {
    for (uint64_t row = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; row < rows;
         row += uint64_t(gridDim.x) * blockDim.x) {
        const int64_t column = selected[row];
        if (column < 0 || uint64_t(column) >= columns) {
            atomicExch(invalid, 1u);
            continue;
        }
        const double ratio = __ddiv_rn(probabilities[row * columns + uint64_t(column)], totals[row]);
        output[row] = float(ratio);
    }
}

extern "C" cudaError_t tf_nemotron_limits(const double* normalized, const double* ranked,
                                        const double* parameters, int64_t* limits,
                                        uint64_t rows, uint64_t columns, uint32_t minimum, cudaStream_t stream) {
    if (rows == 0) return cudaSuccess;
    if (!normalized || !ranked || !parameters || !limits || rows > 2147483647u || minimum > 1 ||
        columns == 0 || columns >= INT64_MAX || columns > UINT64_MAX / rows || rows * columns > UINT64_MAX / 8)
        return cudaErrorInvalidValue;
    tf_nemotron_limits_kernel<<<uint32_t(rows), 256, 0, stream>>>(normalized, ranked, parameters,
        limits, rows, columns, minimum);
    return cudaGetLastError();
}

extern "C" cudaError_t tf_nemotron_probability(const double* probabilities, const double* totals,
                                             const int64_t* selected, float* output,
                                             uint64_t rows, uint64_t columns, uint32_t* invalid, cudaStream_t stream) {
    if (rows == 0) return cudaSuccess;
    if (!probabilities || !totals || !selected || !output || !invalid || columns == 0 ||
        columns > UINT64_MAX / rows || rows * columns > UINT64_MAX / 8) return cudaErrorInvalidValue;
    tf_nemotron_probability_kernel<<<tf_launch_blocks(rows, 256), 256, 0, stream>>>(
        probabilities, totals, selected, output, rows, columns, invalid);
    return cudaGetLastError();
}
