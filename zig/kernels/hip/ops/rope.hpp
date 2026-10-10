#pragma once

#include "common.hpp"

// RoPE for a prefill and the window's fused q / k norm and rotation.

namespace {

// apply_rope's prefill formula as torch has it, rounded to kind, stored as out_kind at h * s_head + r * s_row + j.
extern "C" __global__ void tf_rope_prefill(const void* x, int kind, void* out, int out_kind, long long s_head, long long s_row,
                                    int len, int heads, int d, int rotary, int pos0, float theta) {
    long long i = gid();
    if (i >= static_cast<long long>(len) * heads * d) return;
    int j = static_cast<int>(i % d);
    int h = static_cast<int>((i / d) % heads);
    int r = static_cast<int>(i / (static_cast<long long>(d) * heads));
    long long base = (static_cast<long long>(r) * heads + h) * d;
    int half = rotary / 2;
    float y;
    if (j >= rotary) {
        y = load(x, kind, base + j);
    } else {
        int f = j < half ? j : j - half;
        float expo = static_cast<float>(f) / static_cast<float>(half);
        float freq = 1.0f / powf(theta, expo);
        float ang = static_cast<float>(pos0 + r) * freq;
        float c = cosf(ang), s = sinf(ang);
        float x1 = load(x, kind, base + f), x2 = load(x, kind, base + f + half);
        y = j < half ? x1 * c - x2 * s : x1 * s + x2 * c;
    }
    store(out, out_kind, h * s_head + r * s_row + j, rounded(y, kind));
}

// A window's q or k heads in one pass, a wave a (row, head): RMS norm, then rotation at its position, both rounded.
extern "C" __global__ void __launch_bounds__(256) tf_qk_rope(const void* src, int kind, long long s_row, int s_head,
                                                      const float* weight, float eps, int rows, int heads, int width,
                                                      int rotary, float theta, const int* pos, float* wide,
                                                      void* cache, int total) {
    constexpr int kMost = 512 / 32;
    __shared__ float row_v[8][512];
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const int g = blockIdx.x * 8 + wave;
    const bool live = g < rows * heads;
    const int r = live ? g / heads : 0, h = live ? g % heads : 0;
    const long long at = r * s_row + static_cast<long long>(h) * s_head;
    float v[kMost];
    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < kMost; ++j) {
        const int i = lane + 32 * j;
        v[j] = live && i < width ? load(src, kind, at + i) : 0.f;
        if (i < width) sum += v[j] * v[j];
    }
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) sum += __shfl_xor(sum, mask, 32);
    const float inv = rsqrtf(__shfl(sum, 0, 32) / static_cast<float>(width) + eps);
#pragma unroll
    for (int j = 0; j < kMost; ++j) {
        const int i = lane + 32 * j;
        if (i < width) row_v[wave][i] = rounded(v[j] * inv * weight[i], kind);
    }
    __syncthreads();
    if (!live) return;
    const int p = pos[r];
    const int half = rotary >> 1;
    float* out = wide ? wide + static_cast<long long>(g) * width : nullptr;
    const long long slot = (static_cast<long long>(h) * total + p) * width;
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
        float x1 = row_v[wave][i];
        float x2 = row_v[wave][half + i];
        put(i, x1 * c - x2 * sn);
        put(half + i, x1 * sn + x2 * c);
    }
    for (int i = rotary + lane; i < width; i += 32) put(i, row_v[wave][i]);
}

}  // namespace

