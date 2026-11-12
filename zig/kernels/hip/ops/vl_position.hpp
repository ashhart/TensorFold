#pragma once

#include "common.hpp"

// The learned field's antialiased resize in FP32, then content-normalized border sampling into BF16 patch rows.

namespace {

extern "C" __global__ __launch_bounds__(256)
void tf_vl_resize(const void* src, float* out, int ih, int iw, int oh, int ow, int width, int kind, int axis) {
    const long long i = gid();
    if (i >= (long long)oh * ow * width) return;
    const int ch = i % width, x = (i / width) % ow, y = i / (width * ow);
    if (axis == 2) {
        const int y0 = (y * ih) / oh, y1 = ((y + 1) * ih + oh - 1) / oh;
        const int x0 = (x * iw) / ow, x1 = ((x + 1) * iw + ow - 1) / ow;
        float sum = 0;
        for (int yy = y0; yy < y1; ++yy) for (int xx = x0; xx < x1; ++xx)
            sum += load(src, kind, ((long long)yy * iw + xx) * width + ch);
        out[i] = (sum / (y1 - y0)) / (x1 - x0);
        return;
    }
    const int in = axis == 0 ? iw : ih, size = axis == 0 ? ow : oh, index = axis == 0 ? x : y;
    if (in == size) {
        out[i] = load(src, kind, ((long long)y * iw + x) * width + ch);
        return;
    }
    const float scale = (float)in / size, support = scale > 1.0f ? scale : 1.0f;
    const float center = (index + 0.5f) * scale, inv = 1.0f / support;
    const int first = max((int)(center - support + 0.5f), 0);
    const int end = min((int)(center + support + 0.5f), in);
    float total = 0;
    for (int j = first; j < end; ++j) {
        float distance = fabsf((j - center + 0.5f) * inv);
        total += distance < 1.0f ? 1.0f - distance : 0.0f;
    }
    float result = 0;
    for (int j = first; j < end; ++j) {
        float distance = fabsf((j - center + 0.5f) * inv);
        float weight = (distance < 1.0f ? 1.0f - distance : 0.0f) / total;
        long long at = axis == 0 ? ((long long)y * iw + j) * width + ch : ((long long)j * iw + x) * width + ch;
        result += load(src, kind, at) * weight;
    }
    out[i] = result;
}

__device__ float vl_sample_coordinate(int index, int pixels, int size) {
    const float start = (float)(index * 16);
    const float extent = fminf(fmaxf((float)pixels - start, 0.0f), 16.0f);
    const float normalized = fminf(start + 0.5f * extent, (float)pixels) / pixels;
    const float grid = 2.0f * normalized - 1.0f;
    return fminf(fmaxf(((grid + 1.0f) * size - 1.0f) * 0.5f, 0.0f), (float)(size - 1));
}

extern "C" __global__ __launch_bounds__(256)
void tf_vl_position(const void* patches, const void* cls, const float* field, void* out,
                   int gh, int gw, int height, int width, int hidden, int prefix, int kind) {
    const long long i = gid();
    const int count = gh * gw + prefix;
    if (i >= (long long)count * hidden) return;
    const int row = i / hidden, ch = i % hidden;
    if (row < prefix) {
        store(out, kind, i, load(cls, kind, i));
        return;
    }
    const int r = row - prefix, oh = (height + 15) / 16, ow = (width + 15) / 16;
    const float y = vl_sample_coordinate(r / gw, height, oh), x = vl_sample_coordinate(r % gw, width, ow);
    const int y0 = (int)floorf(y), x0 = (int)floorf(x), y1 = min(y0 + 1, oh - 1), x1 = min(x0 + 1, ow - 1);
    const float sy = y - y0, sx = x - x0;
    float value = field[((long long)y0 * ow + x0) * hidden + ch] * ((1 - sx) * (1 - sy));
    value += field[((long long)y0 * ow + x1) * hidden + ch] * (sx * (1 - sy));
    value += field[((long long)y1 * ow + x0) * hidden + ch] * ((1 - sx) * sy);
    value += field[((long long)y1 * ow + x1) * hidden + ch] * (sx * sy);
    store(out, kind, i, load(patches, kind, (long long)r * hidden + ch) + rounded(value, kind));
}

}  // namespace
