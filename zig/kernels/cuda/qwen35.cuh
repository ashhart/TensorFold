// Shared by qwen35.cu and qwen35_keep.cu: the widths (Geo), a stream's window (Seg), a row's log record, parent walks.

#pragma once
#include <cuda_bf16.h>
#include <stdint.h>

namespace tf_q35 {

constexpr int DK = 128;   // linear-attention key and value head width
constexpr int TAPS = 4;   // causal convolution taps
constexpr int HD = 256;   // attention head width
constexpr int MAXQ = 32;  // attention query heads at most
constexpr int MAXW = 64;  // a window's rows at most (the log's rows a stream)
constexpr int MAXCH = 64; // attention chunks a row at most (cuda_engine.zig's max_chunks)
constexpr int MROPE = 3;  // an image row's t, h and w positions (cuda_round.zig's mrope_axes)

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
    unsigned long long before; // exact pre-round state for a one-row in-place update
    int cap, len, row0, rows, tap_rows, taps;
    int delta;                // an image prompt's rotary shift: rows past it rotate at position + delta
    unsigned long long pt;    // the stream's KV page table (int a page; 0: kv holds cap contiguous rows)
    unsigned long long ps;    // elements a KV page holds (bf16, or bytes for fp8): every attention layer's ptok rows
    int ptok;                 // positions a KV page holds
};

// Seg.kv's element of layer a's key (s 0) or value (s 1) at pos, head h: cap rows a head, or pos's page's row.
__device__ __forceinline__ size_t kv_row(const Seg& sg, int kvh, int a, int s, int h, int pos) {
    const size_t blk = ((size_t)a * 2 + s) * kvh + h;
    if (sg.pt == 0) return (blk * sg.cap + pos) * HD;
    const int* pt = reinterpret_cast<const int*>(sg.pt);
    return (size_t)pt[pos / sg.ptok] * sg.ps + (blk * sg.ptok + pos % sg.ptok) * HD;
}

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

// P (device counts): the row past the forward's rows, as qwen35.cu's other counts read it.
__device__ __forceinline__ bool past_rows(const int* P, int m) { return P != nullptr && m >= P[2]; }

// M-RoPE's interleaved sections (mrope_section 11, 11, 10): frequency i turns by the t, h or w position.
__device__ __forceinline__ int mrope_axis(int i) { return (i % 3 == 1 && i < 33) ? 1 : ((i % 3 == 2 && i < 30) ? 2 : 0); }

// RoPE on the first 64 of HD: pairs (i, i + 32) turned by pos[axis(i)] * theta^(-i / 32); text has one position.
__device__ __forceinline__ __nv_bfloat16 rotate(const __nv_bfloat16* x, int d, const int pos[3]) {
    if (d >= 64) return x[d];
    const int i = d % 32;
    const float angle = (float)pos[mrope_axis(i)] * powf(10000000.0f, -(float)i / 32.0f);
    float sn, cs;
    sincosf(angle, &sn, &cs);
    const float a = bf(x[i]), b = bf(x[i + 32]);
    return __float2bfloat16_rn(d < 32 ? a * cs - b * sn : b * cs + a * sn);
}

// A tree uses all 64 ancestor bits; prompt chains keep their depth separately, including rows past 64.
struct Ancestor { unsigned long long bits; unsigned depth, chain; };
__device__ __forceinline__ int rows_seen(Ancestor anc) { return anc.chain ? (int)anc.depth + 1 : __popcll(anc.bits); }

// A row's chunk width from its own logical keys alone, as its one-row step's: at most MAXCH chunks, 128-key multiples.
__device__ __forceinline__ int chunk_keys(int keys) { return max(128, (keys + MAXCH * 128 - 1) / (MAXCH * 128) * 128); }

// The window row holding a row's nth seen window key.
__device__ __forceinline__ int seen_row(Ancestor anc, int n) {
    if (anc.chain) return n;
    for (int j = 0; j < n; j++) anc.bits &= anc.bits - 1;
    return __ffsll(anc.bits) - 1;
}

// A row's t, h and w: an image chunk reads them from M (three axes a row); other rows are text.
__device__ __forceinline__ void row_positions(const Seg& sg, Ancestor anc, size_t m, const int* M, int pos[3]) {
    const int text = sg.len + rows_seen(anc) - 1 + sg.delta;
    for (int a = 0; a < MROPE; a++) pos[a] = M != nullptr ? M[m * MROPE + a] : text;
}

// The rope kernels' body, a block a row and head (y: query heads, then key heads): rotated in place, or stored.
template <typename Store>
__device__ __forceinline__ void rope_rows(const Geo& geo, __nv_bfloat16* __restrict__ Qn, const __nv_bfloat16* __restrict__ Kn,
                                          const __nv_bfloat16* __restrict__ Vr, const Seg* __restrict__ segs,
                                          const int* __restrict__ rowseg, const Ancestor* __restrict__ rowanc,
                                          const int* __restrict__ P, const int* __restrict__ M, Store store) {
    const int QH = geo.qh, KVH = geo.kvh;
    const int d = threadIdx.x;
    if (past_rows(P, blockIdx.x)) return;
    const size_t m = blockIdx.x;
    const Seg sg = segs[rowseg[m]];
    int pos[3];
    row_positions(sg, rowanc[m], m, M, pos);
    const int slot = sg.len + (int)m - sg.row0;
    const int h = blockIdx.y;
    if (h < QH) {
        __shared__ __nv_bfloat16 qs[HD];
        qs[d] = Qn[(m * QH + h) * HD + d];
        __syncthreads();
        Qn[(m * QH + h) * HD + d] = rotate(qs, d, pos);
    } else if (h < QH + KVH) {
        const int kh = h - QH;
        store(sg, kh, slot, d, rotate(Kn + (m * KVH + kh) * HD, d, pos), Vr[(m * KVH + kh) * HD + d]);
    }
}

}  // namespace tf_q35
