#include <cuda_runtime.h>
#include <stdint.h>
#include "launch_shape.h"
#include <math.h>

extern "C" __global__ void tf_scale_logits_kernel(const float* input, double* output,
                                                uint64_t count, double inverse_temperature) {
    for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * blockDim.x) output[i] = __dmul_rn(double(input[i]), inverse_temperature);
}

extern "C" __global__ void tf_mass_kernel(const double* scaled, const double* maxima, int64_t* mass,
                                        uint64_t rows, uint64_t columns) {
    const uint64_t row = blockIdx.x;
    if (row >= rows) return;
    for (uint64_t col = threadIdx.x; col < columns; col += blockDim.x) {
        const double relative = __dsub_rn(scaled[row * columns + col], maxima[row]);
        const double probability = exp(relative);
        const double product = __dmul_rn(probability, 1099511627776.0);
        mass[row * columns + col] = __double2ll_rz(floor(product));
    }
}

extern "C" __global__ void tf_integer_row_sum_kernel(const uint64_t* input, uint64_t* output,
                                                   uint64_t rows, uint64_t columns) {
    __shared__ uint64_t partials[256];
    const uint64_t row = blockIdx.x;
    if (row >= rows) return;
    uint64_t sum = 0;
    for (uint64_t col = threadIdx.x; col < columns; col += blockDim.x) sum += input[row * columns + col];
    partials[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = 128; stride; stride >>= 1) {
        if (threadIdx.x < stride) partials[threadIdx.x] += partials[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) output[row] = partials[0];
}



extern "C" cudaError_t tf_scale_logits(const float* input, double* output, uint64_t count,
                                     double temperature, cudaStream_t stream) {
    if (count == 0) return cudaSuccess;
    if (!input || !output || !isfinite(temperature) || temperature < 1e-6 || count > UINT64_MAX / 8)
        return cudaErrorInvalidValue;
    const double inverse_temperature = 1.0 / temperature;
    tf_scale_logits_kernel<<<tf_launch_blocks(count, 256), 256, 0, stream>>>(input, output, count, inverse_temperature);
    return cudaGetLastError();
}

extern "C" cudaError_t tf_nucleus_mass(const double* scaled, const double* maxima, int64_t* mass,
                                     uint64_t rows, uint64_t columns, cudaStream_t stream) {
    if (rows == 0) return cudaSuccess;
    if (!scaled || !maxima || !mass || columns == 0 || rows > 2147483647u ||
        columns > UINT64_MAX / rows || rows * columns > UINT64_MAX / 8)
        return cudaErrorInvalidValue;
    tf_mass_kernel<<<uint32_t(rows), 256, 0, stream>>>(scaled, maxima, mass, rows, columns);
    return cudaGetLastError();
}

extern "C" cudaError_t tf_integer_row_sum(const int64_t* input, int64_t* output,
                                        uint64_t rows, uint64_t columns, cudaStream_t stream) {
    if (rows == 0) return cudaSuccess;
    if (!input || !output || columns == 0 || rows > 2147483647u ||
        columns > UINT64_MAX / rows || rows * columns > UINT64_MAX / 8)
        return cudaErrorInvalidValue;
    tf_integer_row_sum_kernel<<<uint32_t(rows), 256, 0, stream>>>(reinterpret_cast<const uint64_t*>(input),
        reinterpret_cast<uint64_t*>(output), rows, columns);
    return cudaGetLastError();
}
