#include <cuda_runtime.h>
#include <stdint.h>
#include "launch_shape.h"
#include <math.h>

namespace {

__device__ __forceinline__ uint64_t sample_mix(uint64_t value) {
    value ^= value >> 30u;
    value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 27u;
    value *= UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31u);
}





}

extern "C" __global__ void tf_nemotron_scale_f64_kernel(
    const double* values, const double* parameters, double* output, uint64_t count) {
    const double temperature = parameters[0];
    for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * blockDim.x) {
        output[i] = __ddiv_rn(values[i], temperature);
    }
}

extern "C" __global__ void tf_nemotron_relative_exp_f64_kernel(
    const double* ranked, const double* parameters, double* output,
    uint64_t count, uint64_t columns, uint32_t confidence_temperature) {
    const double ratio = confidence_temperature ? __ddiv_rn(parameters[0], parameters[3]) : 1.0;
    for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * blockDim.x) {
        const uint64_t first = i / columns * columns;
        double relative = __dsub_rn(ranked[i], ranked[first]);
        if (confidence_temperature) relative = __dmul_rn(relative, ratio);
        output[i] = exp(relative);
    }
}

extern "C" __global__ void tf_nemotron_run_divide_f64_kernel(
    const double* cumulative, const double* row_totals, double* output,
    uint64_t count, uint64_t columns) {
    for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * blockDim.x) {
        output[i] = __ddiv_rn(cumulative[i], row_totals[i / columns]);
    }
}

extern "C" __global__ void tf_nemotron_uniform_f64_kernel(
    const int64_t* ranked_ids, const int64_t* seed, const int32_t* metadata,
    double* output, uint64_t count, uint64_t columns, int64_t offset) {
    const uint64_t base = sample_mix(uint64_t(seed[0]) + UINT64_C(0x9e3779b97f4a7c15));
    for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * blockDim.x) {
        const uint64_t position = uint64_t(int64_t(metadata[0])) + i / columns + 1u + uint64_t(offset);
        const uint64_t row_key = sample_mix(base ^ (position * UINT64_C(0xd1b54a32d192ed03)));
        const uint64_t token_key = sample_mix(row_key ^ uint64_t(ranked_ids[i]));
        const double unit = __dmul_rn(double(token_key >> 11u), 0x1p-53);
        output[i] = __dadd_rn(unit, 0x1p-54);
    }
}

extern "C" __global__ void tf_nemotron_gumbel_score_f64_kernel(
    const double* ranked, const double* uniform, const int64_t* limits,
    double* output, uint64_t count, uint64_t columns) {
    for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * blockDim.x) {
        const double inner = log(uniform[i]);
        const double noise = log(-inner);
        const double score = __dsub_rn(ranked[i], noise);
        const uint64_t row = i / columns;
        const bool kept = limits[row] > 0 && i % columns < uint64_t(limits[row]);
        output[i] = kept ? score : -INFINITY;
    }
}

extern "C" cudaError_t tf_nemotron_scale_f64(
    const double* values, const double* parameters, double* output, uint64_t count, cudaStream_t stream) {
    if (count == 0) return cudaSuccess;
    if (!values || !parameters || !output || count > UINT64_MAX / sizeof(double)) return cudaErrorInvalidValue;
    tf_nemotron_scale_f64_kernel<<<tf_launch_blocks(count, 256), 256, 0, stream>>>(values, parameters, output, count);
    return cudaGetLastError();
}

extern "C" cudaError_t tf_nemotron_relative_exp_f64(
    const double* ranked, const double* parameters, double* output, uint64_t rows, uint64_t columns,
    uint32_t confidence_temperature, cudaStream_t stream) {
    uint64_t count;
    if (!tf_matrix_count(rows, columns, sizeof(double), &count) || confidence_temperature > 1u) return cudaErrorInvalidValue;
    if (count == 0) return cudaSuccess;
    if (!ranked || !parameters || !output) return cudaErrorInvalidValue;
    tf_nemotron_relative_exp_f64_kernel<<<tf_launch_blocks(count, 256), 256, 0, stream>>>(
        ranked, parameters, output, count, columns, confidence_temperature);
    return cudaGetLastError();
}

extern "C" cudaError_t tf_nemotron_run_divide_f64(
    const double* cumulative, const double* row_totals, double* output,
    uint64_t rows, uint64_t columns, cudaStream_t stream) {
    uint64_t count;
    if (!tf_matrix_count(rows, columns, sizeof(double), &count)) return cudaErrorInvalidValue;
    if (count == 0) return cudaSuccess;
    if (!cumulative || !row_totals || !output) return cudaErrorInvalidValue;
    tf_nemotron_run_divide_f64_kernel<<<tf_launch_blocks(count, 256), 256, 0, stream>>>(
        cumulative, row_totals, output, count, columns);
    return cudaGetLastError();
}

extern "C" cudaError_t tf_nemotron_uniform_f64(
    const int64_t* ranked_ids, const int64_t* seed, const int32_t* metadata, double* output,
    uint64_t rows, uint64_t columns, int64_t offset, cudaStream_t stream) {
    uint64_t count;
    if (!tf_matrix_count(rows, columns, sizeof(double), &count)) return cudaErrorInvalidValue;
    if (count == 0) return cudaSuccess;
    if (!ranked_ids || !seed || !metadata || !output) return cudaErrorInvalidValue;
    tf_nemotron_uniform_f64_kernel<<<tf_launch_blocks(count, 256), 256, 0, stream>>>(
        ranked_ids, seed, metadata, output, count, columns, offset);
    return cudaGetLastError();
}

extern "C" cudaError_t tf_nemotron_gumbel_score_f64(
    const double* ranked, const double* uniform, const int64_t* limits, double* output,
    uint64_t rows, uint64_t columns, cudaStream_t stream) {
    uint64_t count;
    if (!tf_matrix_count(rows, columns, sizeof(double), &count)) return cudaErrorInvalidValue;
    if (count == 0) return cudaSuccess;
    if (!ranked || !uniform || !limits || !output) return cudaErrorInvalidValue;
    tf_nemotron_gumbel_score_f64_kernel<<<tf_launch_blocks(count, 256), 256, 0, stream>>>(
        ranked, uniform, limits, output, count, columns);
    return cudaGetLastError();
}
