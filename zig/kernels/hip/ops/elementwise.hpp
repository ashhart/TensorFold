#pragma once

#include "common.hpp"

// Embedding rows, casts, sums, column copies and the draft projection.

namespace {

// An MLX code at index i of a packed row, including one that crosses two words (qwen_math._codes).
__device__ __forceinline__ uint32_t code_at(const uint32_t* row, int words, int bits, long long i) {
    long long bit = i * bits;
    int word = static_cast<int>(bit / 32), shift = static_cast<int>(bit % 32);
    uint64_t low = row[word];
    uint64_t high = (shift + bits > 32 && word + 1 < words) ? row[word + 1] : 0;
    uint64_t value = (low >> shift) | (high << ((32 - shift) & 31));
    return static_cast<uint32_t>(value & ((1u << bits) - 1u));
}

extern "C" __global__ void tf_embed_rows(const uint32_t* words, const void* scale, const void* bias, int scale_kind,
                                  const int* ids, int n, int bits, int group, int k, void* out, int out_kind) {
    long long i = gid();
    if (i >= static_cast<long long>(n) * k) return;
    int r = static_cast<int>(i / k), c = static_cast<int>(i % k);
    long long id = ids[r];
    int words_row = k * bits / 32, groups = k / group;
    float code = static_cast<float>(code_at(words + id * words_row, words_row, bits, c));
    float s = load(scale, scale_kind, id * groups + c / group);
    float b = load(bias, scale_kind, id * groups + c / group);
    float y = code * s;
    y = y + b;
    store(out, out_kind, i, y);
}

// The float gather: rows of an fp32 table (an embedding kept unquantized), written in the activation type.
extern "C" __global__ void tf_embed_dense(const float* table, const int* ids, int n, int k, void* out, int out_kind) {
    long long i = gid();
    if (i >= static_cast<long long>(n) * k) return;
    int r = static_cast<int>(i / k), c = static_cast<int>(i % k);
    store(out, out_kind, i, table[static_cast<long long>(ids[r]) * k + c]);
}

extern "C" __global__ void tf_cast(const void* src, int skind, void* dst, int dkind, long long n) {
    long long i = gid();
    if (i < n) store(dst, dkind, i, load(src, skind, i));
}

extern "C" __global__ void tf_silu_mul(const void* gate, const void* up, void* out, int kind, long long n) {
    long long i = gid();
    if (i >= n) return;
    float s = rounded(silu(load(gate, kind, i)), kind);
    store(out, kind, i, s * load(up, kind, i));
}

extern "C" __global__ void tf_add(const void* x, const void* y, void* out, int kind, long long n) {
    long long i = gid();
    if (i < n) store(out, kind, i, load(x, kind, i) + load(y, kind, i));
}

// rows of `cols` values from a source with row stride `stride` at column `offset`, kept in kind.
extern "C" __global__ void tf_copy_cols(const void* src, long long stride, int offset, void* dst, int kind, int rows,
                                 int cols) {
    long long i = gid();
    if (i >= static_cast<long long>(rows) * cols) return;
    long long r = i / cols;
    int c = static_cast<int>(i % cols);
    long long from = r * stride + offset + c;
    if (kind == 0) static_cast<float*>(dst)[i] = static_cast<const float*>(src)[from];
    else static_cast<uint16_t*>(dst)[i] = static_cast<const uint16_t*>(src)[from];
}

// y[r, n] = x[r, :] . w[n, :] over fp32 weights (MTP fc), a wave a value; drafts only, so the sum order is free.
extern "C" __global__ void tf_dense_rows(const void* x, int kind, const float* w, void* out, int rows, int n, int k) {
    int col = blockIdx.x, r = blockIdx.y;
    if (col >= n || r >= rows) return;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < k; i += 32) acc += load(x, kind, static_cast<long long>(r) * k + i) * w[static_cast<long long>(col) * k + i];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor(acc, off, 32);
    if (threadIdx.x == 0) store(out, kind, static_cast<long long>(r) * n + col, acc);
}

}  // namespace

