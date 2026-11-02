#pragma once

#include "common.hpp"

extern "C" __global__ void __launch_bounds__(256) tf_columns_mul_bf16(
    const uint16_t* x, const uint16_t* weight, uint16_t* out, long long count, int width) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float product = bf16_to_float(x[i]) * bf16_to_float(weight[i % width]);
    out[i] = float_to_bf16(product);
}

extern "C" __global__ void __launch_bounds__(256) tf_columns_fill_bf16(
    const uint16_t* weight, uint16_t* out, long long count, int width) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) out[i] = weight[i % width];
}
