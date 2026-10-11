#pragma once

#include "common.hpp"

// The MoE plan and the router's logits.

namespace {

// Stable-sorts the pairs by expert into members and cuts items of at most `tile` pairs; one block, counted per segment.
extern "C" __global__ void tf_moe_route(const int* picks, int pairs, int experts, int tile, int* members, int* items,
                                 int capacity) {
    extern __shared__ int shared[];
    int* counts = shared;
    int* starts = shared + experts;
    int* ends = shared + 2 * experts;
    unsigned short* seg = reinterpret_cast<unsigned short*>(shared + 3 * experts);
    const int segs = experts <= 256 ? 64 : 16;
    const int span = (pairs + segs - 1) / segs;
    const int t = threadIdx.x;
    for (int i = t; i < segs * experts; i += blockDim.x) seg[i] = 0;
    __syncthreads();
    if (t < segs) {
        for (int p = t * span; p < pairs && p < (t + 1) * span; ++p) seg[t * experts + picks[p]] += 1;
    }
    __syncthreads();
    // an expert's total and each segment's first place for it (still relative to the expert's start)
    for (int e = t; e < experts; e += blockDim.x) {
        int run = 0;
        for (int g = 0; g < segs; ++g) {
            int c = seg[g * experts + e];
            seg[g * experts + e] = static_cast<unsigned short>(run);
            run += c;
        }
        counts[e] = run;
    }
    __syncthreads();
    if (t == 0) {
        int start = 0, end = 0;
        for (int e = 0; e < experts; ++e) {
            starts[e] = start;
            start += counts[e];
            end += (counts[e] + tile - 1) / tile;
            ends[e] = end;
        }
    }
    __syncthreads();
    if (t < segs) {
        for (int p = t * span; p < pairs && p < (t + 1) * span; ++p) {
            int e = picks[p];
            members[starts[e] + seg[t * experts + e]] = p;
            seg[t * experts + e] += 1;
        }
    }
    for (int slot = t; slot < capacity; slot += blockDim.x) {
        int lo = 0, hi = experts;
        while (lo < hi) {                       // searchsorted(ends, slot, right=True)
            int mid = (lo + hi) / 2;
            if (ends[mid] <= slot) lo = mid + 1;
            else hi = mid;
        }
        int e = lo < experts - 1 ? lo : experts - 1;
        int reps = (counts[e] + tile - 1) / tile;
        int within = slot - (ends[e] - reps);
        int count = counts[e] - within * tile;
        items[slot * 3] = e;
        items[slot * 3 + 1] = starts[e] + within * tile;
        items[slot * 3 + 2] = count < 0 ? 0 : (count > tile ? tile : count);
    }
}

// moe_router_kernel's logits with a wave's expert row read once for 8 token rows; each logit keeps its bits.
__device__ __forceinline__ float router_sum(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) x += __shfl_xor(x, mask, 32);
    return __shfl(x, 0, 32);
}

template <int KIND>
__device__ void router_rows(const void* x, const float* rows, float* logits, int r, int d, int e) {
    constexpr int kChunk = 1024;
    __shared__ float xs[8][kChunk];
    const int lane = threadIdx.x & 31;
    const int expert = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int row0 = blockIdx.y * 8;
    const float* w = rows + static_cast<long long>(expert < e ? expert : 0) * d;
    float acc[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) acc[k] = 0.f;
    for (int base = 0; base < d; base += kChunk) {
        const int n = d - base < kChunk ? d - base : kChunk;
        __syncthreads();
        for (int q = threadIdx.x; q < 8 * kChunk; q += blockDim.x) {
            const int k = q / kChunk, i = q % kChunk;
            xs[k][i] = (row0 + k < r && i < n) ? load(x, KIND, static_cast<long long>(row0 + k) * d + base + i) : 0.f;
        }
        __syncthreads();
        // lane l keeps adding i = l, l + 32, ... in order across the chunks
        for (int i = lane; i < n; i += 32) {
            const float wv = w[base + i];
#pragma unroll
            for (int k = 0; k < 8; ++k) acc[k] = fmaf(xs[k][i], wv, acc[k]);
        }
    }
    if (expert >= e) return;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const float sum = router_sum(acc[k]);
        if (lane == 0 && row0 + k < r) logits[static_cast<long long>(row0 + k) * e + expert] = sum;
    }
}

// The router's logits for many rows: a 64 x 64 tile of (row, expert) a block, 4 x 4 a thread, in fp32.
template <int KIND>
__device__ void router_tile(const void* x, const float* rows, float* logits, int r, int d, int e) {
    constexpr int kTile = 64, kStep = 32, kLd = kTile + 4;
    __shared__ alignas(16) float xs[kStep][kLd];
    __shared__ alignas(16) float ws[kStep][kLd];
    const int row0 = blockIdx.y * kTile, expert0 = blockIdx.x * kTile;
    const int ty = threadIdx.x / 16, tx = threadIdx.x % 16;
    // the staging: activation row ar at steps ak .. ak + 7, router rows wr and wr + 32 at steps wk .. wk + 3
    const int ar = threadIdx.x / 4, ak = (threadIdx.x % 4) * 8;
    const int wr = threadIdx.x / 8, wk = (threadIdx.x % 8) * 4;
    const uint16_t* xh = static_cast<const uint16_t*>(x);
    float acc[4][4] = {};
    for (int base = 0; base < d; base += kStep) {
        uint4 xv = {0, 0, 0, 0};
        if (row0 + ar < r) xv = *reinterpret_cast<const uint4*>(xh + static_cast<long long>(row0 + ar) * d + base + ak);
        float4 w0 = {0, 0, 0, 0}, w1 = {0, 0, 0, 0};
        if (expert0 + wr < e) w0 = *reinterpret_cast<const float4*>(rows + static_cast<long long>(expert0 + wr) * d + base + wk);
        if (expert0 + wr + 32 < e)
            w1 = *reinterpret_cast<const float4*>(rows + static_cast<long long>(expert0 + wr + 32) * d + base + wk);
        __syncthreads();
        const uint16_t* h = reinterpret_cast<const uint16_t*>(&xv);
#pragma unroll
        for (int j = 0; j < 8; ++j) xs[ak + j][ar] = KIND == 1 ? __half2float(__ushort_as_half(h[j])) : bf16_to_float(h[j]);
        const float a0[4] = {w0.x, w0.y, w0.z, w0.w}, a1[4] = {w1.x, w1.y, w1.z, w1.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) ws[wk + j][wr] = a0[j], ws[wk + j][wr + 32] = a1[j];
        __syncthreads();
#pragma unroll
        for (int i = 0; i < kStep; ++i) {
            const float4 a = *reinterpret_cast<const float4*>(&xs[i][4 * ty]);
            const float4 b = *reinterpret_cast<const float4*>(&ws[i][4 * tx]);
            const float av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
            for (int u = 0; u < 4; ++u)
#pragma unroll
                for (int w = 0; w < 4; ++w) acc[u][w] = fmaf(av[u], bv[w], acc[u][w]);
        }
    }
#pragma unroll
    for (int u = 0; u < 4; ++u) {
        const int row = row0 + 4 * ty + u;
        if (row >= r) continue;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const int expert = expert0 + 4 * tx + w;
            if (expert < e) logits[static_cast<long long>(row) * e + expert] = acc[u][w];
        }
    }
}

// router_tile's logits for up to 32 rows: a wave an expert, a lane a row, each logit one in-order fmaf chain.
template <int KIND>
__device__ void router_small(const void* x, const float* rows, float* logits, int r, int d, int e) {
    const int lane = threadIdx.x & 31;
    const int expert = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int row = blockIdx.y * 32 + lane;
    if (expert >= e) return;
    const uint16_t* xh = static_cast<const uint16_t*>(x) + static_cast<long long>(row < r ? row : 0) * d;
    const float* w = rows + static_cast<long long>(expert) * d;
    float acc = 0.f;
    for (int base = 0; base < d; base += 8) {
        const uint4 xv = *reinterpret_cast<const uint4*>(xh + base);
        const uint16_t* h = reinterpret_cast<const uint16_t*>(&xv);
        const float4 w0 = *reinterpret_cast<const float4*>(w + base);
        const float4 w1 = *reinterpret_cast<const float4*>(w + base + 4);
        const float wv[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            acc = fmaf(KIND == 1 ? __half2float(__ushort_as_half(h[j])) : bf16_to_float(h[j]), wv[j], acc);
        }
    }
    if (row < r) logits[static_cast<long long>(row) * e + expert] = acc;
}

}  // namespace

extern "C" __global__ void __launch_bounds__(256) tf_router_small_f16(const void* x, const float* rows, float* logits,
                                                                      int r, int d, int e) {
    router_small<1>(x, rows, logits, r, d, e);
}

extern "C" __global__ void __launch_bounds__(256) tf_router_small_bf16(const void* x, const float* rows, float* logits,
                                                                       int r, int d, int e) {
    router_small<2>(x, rows, logits, r, d, e);
}

extern "C" __global__ void __launch_bounds__(256) tf_router_tile_f16(const void* x, const float* rows, float* logits,
                                                                     int r, int d, int e) {
    router_tile<1>(x, rows, logits, r, d, e);
}

extern "C" __global__ void __launch_bounds__(256) tf_router_tile_bf16(const void* x, const float* rows, float* logits,
                                                                      int r, int d, int e) {
    router_tile<2>(x, rows, logits, r, d, e);
}

extern "C" __global__ void __launch_bounds__(256) tf_router_rows_f16(const void* x, const float* rows, float* logits,
                                                                     int r, int d, int e) {
    router_rows<1>(x, rows, logits, r, d, e);
}

extern "C" __global__ void __launch_bounds__(256) tf_router_rows_bf16(const void* x, const float* rows, float* logits,
                                                                      int r, int d, int e) {
    router_rows<2>(x, rows, logits, r, d, e);
}

