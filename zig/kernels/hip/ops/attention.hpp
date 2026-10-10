#pragma once

#include "common.hpp"

// The gated attention's o input and the cache writes.

namespace {

// out[r, h * d + j] = att * sigmoid(gate) rounded to kind; att is (heads, len, d), or (len, heads, d) with rows_major.
extern "C" __global__ void tf_attn_gate(const float* att, const void* qg, void* out, int kind, int len, int heads, int d,
                                 int rows_major) {
    long long i = gid();
    if (i >= static_cast<long long>(len) * heads * d) return;
    int j = static_cast<int>(i % d);
    int h = static_cast<int>((i / d) % heads);
    int r = static_cast<int>(i / (static_cast<long long>(d) * heads));
    float a = rows_major ? att[i] : att[(static_cast<long long>(h) * len + r) * d + j];
    float g = load(qg, kind, (static_cast<long long>(r) * heads + h) * 2 * d + d + j);
    store(out, kind, i, a * sigmoid(g));
}

// keys or values (len, kv_heads, d) of kind into the cache (kv_heads, total, d) at slot pos0.
extern "C" __global__ void tf_kv_write(const void* src, void* cache, int kind, int len, int kv_heads, int d, int total,
                                int pos0) {
    long long i = gid();
    if (i >= static_cast<long long>(len) * kv_heads * d) return;
    int j = static_cast<int>(i % d);
    int h = static_cast<int>((i / d) % kv_heads);
    int r = static_cast<int>(i / (static_cast<long long>(d) * kv_heads));
    long long to = (static_cast<long long>(h) * total + pos0 + r) * d + j;
    if (kind == 0) static_cast<float*>(cache)[to] = static_cast<const float*>(src)[i];
    else static_cast<uint16_t*>(cache)[to] = static_cast<const uint16_t*>(src)[i];
}

// kv_write with the first slot read from the device (`pos`, one int32), so a captured graph replays at any position.
extern "C" __global__ void tf_kv_write_at(const void* src, void* cache, int kind, int len, int kv_heads, int d, int total,
                                   const int* pos) {
    long long i = gid();
    if (i >= static_cast<long long>(len) * kv_heads * d) return;
    int j = static_cast<int>(i % d);
    int h = static_cast<int>((i / d) % kv_heads);
    int r = static_cast<int>(i / (static_cast<long long>(d) * kv_heads));
    long long to = (static_cast<long long>(h) * total + *pos + r) * d + j;
    if (kind == 0) static_cast<float*>(cache)[to] = static_cast<const float*>(src)[i];
    else static_cast<uint16_t*>(cache)[to] = static_cast<const uint16_t*>(src)[i];
}

}  // namespace

