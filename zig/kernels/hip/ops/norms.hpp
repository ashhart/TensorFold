#pragma once

#include "common.hpp"

// RMS norm rows, a wave a row.

namespace {

// rms_kernel's arithmetic a wave a row, 8 rows a block, so short rows do not hold a 256-thread block each.
__device__ void rms_rows(const void* x, int kind, const float* weight, void* y, int rows, int width, float eps) {
    constexpr int kMost = 1024 / 32;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= rows) return;
    const long long at = static_cast<long long>(row) * width;
    float v[kMost];
    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < kMost; ++j) {
        const int i = lane + 32 * j;
        v[j] = i < width ? load(x, kind, at + i) : 0.f;
        if (i < width) sum += v[j] * v[j];
    }
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) sum += __shfl_xor(sum, mask, 32);
    const float inv = rsqrtf(__shfl(sum, 0, 32) / static_cast<float>(width) + eps);
#pragma unroll
    for (int j = 0; j < kMost; ++j) {
        const int i = lane + 32 * j;
        if (i >= width) break;
        float value = v[j] * inv;
        if (weight != nullptr) value *= weight[i];
        store(y, kind, at + i, value);
    }
}

}  // namespace

extern "C" __global__ void __launch_bounds__(256) tf_rms_rows(const void* x, int kind, const float* weight, void* y,
                                                              int rows, int width, float eps) {
    rms_rows(x, kind, weight, y, rows, width, eps);
}
