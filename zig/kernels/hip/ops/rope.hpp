#pragma once

#include "common.hpp"
#include "qk_rope_wave.hpp"

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
    __shared__ float row_v[8][512];
    const int wave = threadIdx.x >> 5;
    const int g = blockIdx.x * 8 + wave;
    const bool live = g < rows * heads;
    const int r = live ? g / heads : 0, h = live ? g % heads : 0;
    qk_rope_norm(src, kind, r * s_row + static_cast<long long>(h) * s_head, weight, eps, width, live, row_v[wave]);
    __syncthreads();
    if (!live) return;
    const int p = pos[r];
    qk_rope_put(kind, width, rotary, theta, p, row_v[wave], wide ? wide + static_cast<long long>(g) * width : nullptr,
                cache, (static_cast<long long>(h) * total + p) * width);
}

}  // namespace
