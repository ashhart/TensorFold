// Shared by qwen35.cu and qwen35_keep.cu: the widths (Geo), a stream's window (Seg), a row's log record, parent walks.

#pragma once
#include <cuda_bf16.h>
#include <stdint.h>

namespace tf_q35 {

constexpr int DK = 128;   // linear-attention key and value head width
constexpr int TAPS = 4;   // causal convolution taps
constexpr int HD = 256;   // attention head width
constexpr int MAXQ = 32;  // attention query heads at most
constexpr int MAXW = 16;  // a window's rows at most (the log's rows a stream)
constexpr int MAXCH = 64; // attention chunks a row at most (cuda_engine.zig's max_chunks)
constexpr int MAXR = 128;  // a forward's rows at most (cuda_engine.zig's max_rows): an image chunk's position rows

// The model's widths: hidden, linear key heads, linear value heads, attention query and key/value heads, linear layers.
struct Geo {
    int hid, nk, nv, qh, kvh, nl;
    __device__ int conv() const { return (2 * nk + nv) * DK; }  // q, k and v channels the convolution sees
    __device__ size_t rec() const { return (size_t)nv * DK * DK; }  // floats of one layer's recurrence
    __device__ size_t conv_rows() const { return (size_t)(TAPS - 1) * conv(); }
    // a row's record of a linear layer: decay, strength and write a value head, key a key head, the tail after it
    __device__ size_t log_bytes() const { return (size_t)nv * 4 + (size_t)nv * DK * 4 + (size_t)nk * DK * 2 + conv_rows() * 2; }
};

// One stream's window in this forward: its state blocks, cache length before the window, and rows.
struct Seg {
    unsigned long long rec;   // float recurrences, one rec() block a linear layer
    unsigned long long conv;  // bf16 convolution tails, conv_rows() a linear layer
    unsigned long long kv;    // bf16 keys then values, kvh * cap * HD each, an attention layer at a time
    unsigned long long tap;   // a drafter's ring of tapped rows (0: none): slot pos % tap_rows, taps * hid bf16 each
    unsigned long long log;   // the window's rows' records (MAXW rows of nl layers of log_bytes), for keep
    int cap, len, row0, rows, tap_rows, taps;
    int delta;                // an image prompt's rotary shift: rows past it rotate at position + delta
};

// A window row's record of one linear layer (Seg.log).
struct LogRow {
    float* g;              // [nv] decay
    float* delta;          // [nv][DK] the write's strength along dv
    __nv_bfloat16* k;      // [nk][DK] the key
    __nv_bfloat16* conv;   // [TAPS - 1][conv()] the tail after the row
};

__device__ __forceinline__ LogRow log_row(const Geo& geo, const Seg& sg, int layer, int row) {
    char* base = reinterpret_cast<char*>(sg.log) + ((size_t)row * geo.nl + layer) * geo.log_bytes();
    LogRow l;
    l.g = reinterpret_cast<float*>(base);
    l.delta = l.g + geo.nv;
    l.k = reinterpret_cast<__nv_bfloat16*>(l.delta + (size_t)geo.nv * DK);
    l.conv = l.k + (size_t)geo.nk * DK;
    return l;
}

// The rows of a window to keep, in order (row 0 first), passed by value.
struct Path {
    int n;
    int rows[MAXW];
};

// The row `steps` tokens before forward row `w` through rowpar: a forward row, or -1 - t for the stream's tail row t.
__device__ __forceinline__ int back(const int* __restrict__ rowpar, int w, int steps) {
    int a = w;
    for (int s = 1; s <= steps; s++) {
        a = rowpar[a];
        if (a < 0) return -1 - (TAPS - 2 - (steps - s));
    }
    return a;
}

__device__ __forceinline__ float bf(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float rbf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int m = 16; m; m >>= 1) x += __shfl_xor_sync(0xffffffffu, x, m);
    return x;
}

}  // namespace tf_q35
