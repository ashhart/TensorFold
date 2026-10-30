#pragma once

#include "common.hpp"

// tf_qk_rope's two phases for one wave's (row, head), shared with Flash Next's QSA prep: RMS norm, then the rotation.

namespace {

constexpr int kQkRopeMost = 512 / 32;  // the widest head: 512 values

// src[at .. at + width) RMS-normed with weight, rounded, into the wave's shared row; a dead wave norms zeros.
__device__ __forceinline__ void qk_rope_norm(const void* src, int kind, long long at, const float* weight, float eps,
                                             int width, bool live, float* row) {
    const int lane = threadIdx.x & 31;
    float v[kQkRopeMost];
    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < kQkRopeMost; ++j) {
        const int i = lane + 32 * j;
        v[j] = live && i < width ? load(src, kind, at + i) : 0.f;
        if (i < width) sum += v[j] * v[j];
    }
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) sum += __shfl_xor(sum, mask, 32);
    const float inv = rsqrtf(__shfl(sum, 0, 32) / static_cast<float>(width) + eps);
#pragma unroll
    for (int j = 0; j < kQkRopeMost; ++j) {
        const int i = lane + 32 * j;
        if (i < width) row[i] = rounded(v[j] * inv * weight[i], kind);
    }
}

// The normed row rotated at position p over its first `rotary` values, rounded, into out (fp32) and/or the cache.
__device__ __forceinline__ void qk_rope_put(int kind, int width, int rotary, float theta, int p, const float* row,
                                            float* out, void* cache, long long slot) {
    const int lane = threadIdx.x & 31;
    const int half = rotary >> 1;
    auto put = [&](int i, float value) {
        value = rounded(value, kind);
        if (out) out[i] = value;
        if (cache) store(cache, kind, slot + i, value);
    };
    for (int i = lane; i < half; i += 32) {
        float freq = 1.f / powf(theta, static_cast<float>(i) / static_cast<float>(half));
        float ang = static_cast<float>(p) * freq;
        float c = cosf(ang);
        float sn = sinf(ang);
        float x1 = row[i];
        float x2 = row[half + i];
        put(i, x1 * c - x2 * sn);
        put(half + i, x1 * sn + x2 * c);
    }
    for (int i = rotary + lane; i < width; i += 32) put(i, row[i]);
}

}  // namespace
