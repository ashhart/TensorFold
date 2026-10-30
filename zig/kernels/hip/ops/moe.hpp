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
    // starts: the counts' exclusive sums; ends: the tiles' inclusive sums; a thread's experts, then the block's scan
    int* scan_c = reinterpret_cast<int*>(seg + segs * experts + (segs * experts) % 2);
    int* scan_n = scan_c + blockDim.x;
    const int per = (experts + blockDim.x - 1) / blockDim.x;
    const int lo = min(experts, t * per), hi = min(experts, lo + per);
    int own_c = 0, own_n = 0;
    for (int e = lo; e < hi; ++e) {
        own_c += counts[e];
        own_n += (counts[e] + tile - 1) / tile;
    }
    scan_c[t] = own_c;
    scan_n[t] = own_n;
    __syncthreads();
    for (int off = 1; off < static_cast<int>(blockDim.x); off <<= 1) {
        const int vc = t >= off ? scan_c[t - off] : 0, vn = t >= off ? scan_n[t - off] : 0;
        __syncthreads();
        scan_c[t] += vc;
        scan_n[t] += vn;
        __syncthreads();
    }
    int start = scan_c[t] - own_c, end = scan_n[t] - own_n;
    for (int e = lo; e < hi; ++e) {
        starts[e] = start;
        start += counts[e];
        end += (counts[e] + tile - 1) / tile;
        ends[e] = end;
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

// tf_moe_route's plan of a prompt's many pairs in three launches (count, scan, scatter); `offs` is scratch. Same plan.
constexpr int kRouteSeg = 512;
constexpr int kRouteMaxExperts = 1024;

extern "C" __global__ void __launch_bounds__(256) tf_route_count(const int* picks, int pairs, int experts, int* offs) {
    __shared__ int counts[kRouteMaxExperts];
    for (int e = threadIdx.x; e < experts; e += blockDim.x) counts[e] = 0;
    __syncthreads();
    const int from = blockIdx.x * kRouteSeg;
    for (int p = from + threadIdx.x; p < pairs && p < from + kRouteSeg; p += blockDim.x) atomicAdd(&counts[picks[p]], 1);
    __syncthreads();
    for (int e = threadIdx.x; e < experts; e += blockDim.x) offs[static_cast<long long>(blockIdx.x) * experts + e] = counts[e];
}

extern "C" __global__ void __launch_bounds__(1024) tf_route_scan(int* offs, int segs, int experts, int tile, int* items,
                                                                 int capacity) {
    __shared__ int counts[kRouteMaxExperts];
    __shared__ int starts[kRouteMaxExperts];
    __shared__ int ends[kRouteMaxExperts];
    __shared__ int scan_c[1024];
    __shared__ int scan_n[1024];
    const int t = threadIdx.x;
    // an expert's segments in order: each one's place within the expert, and the expert's total
    if (t < experts) {
        int run = 0;
        for (int g = 0; g < segs; ++g) {
            int* at = offs + static_cast<long long>(g) * experts + t;
            const int c = *at;
            *at = run;
            run += c;
        }
        counts[t] = run;
    }
    // starts: the counts' exclusive sums; ends: the tiles' inclusive sums (an expert a thread)
    const int own_c = t < experts ? counts[t] : 0;
    const int own_n = t < experts ? (counts[t] + tile - 1) / tile : 0;
    scan_c[t] = own_c;
    scan_n[t] = own_n;
    __syncthreads();
    for (int off = 1; off < static_cast<int>(blockDim.x); off <<= 1) {
        const int vc = t >= off ? scan_c[t - off] : 0, vn = t >= off ? scan_n[t - off] : 0;
        __syncthreads();
        scan_c[t] += vc;
        scan_n[t] += vn;
        __syncthreads();
    }
    if (t < experts) {
        starts[t] = scan_c[t] - own_c;
        ends[t] = scan_n[t];
        offs[static_cast<long long>(segs) * experts + t] = starts[t];
    }
    __syncthreads();
    for (int slot = t; slot < capacity; slot += blockDim.x) {
        int lo = 0, hi = experts;
        while (lo < hi) {  // searchsorted(ends, slot, right=True)
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

extern "C" __global__ void __launch_bounds__(kRouteSeg) tf_route_scatter(const int* picks, int pairs, int segs, int experts,
                                                                         const int* offs, int* members) {
    __shared__ int pk[kRouteSeg];
    const int from = blockIdx.x * kRouteSeg;
    const int t = threadIdx.x;
    const int p = from + t;
    pk[t] = p < pairs ? picks[p] : -1;
    __syncthreads();
    if (p >= pairs) return;
    const int e = pk[t];
    int before = 0;
    for (int q = 0; q < t; ++q) before += pk[q] == e;
    members[offs[static_cast<long long>(segs) * experts + e] + offs[static_cast<long long>(blockIdx.x) * experts + e] + before] = p;
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

// The router's logits for many rows, 128 x 64 a block, 8 x 4 a thread: one fmaf chain over k from 0, so the same bits.
constexpr int kRouterTileRows = 128;
constexpr int kRouterTileExperts = 64;

template <int KIND>
__device__ void router_tile(const void* x, const float* rows, float* logits, int r, int d, int e) {
    constexpr int RT = 8, CT = 4, TR = 16 * RT, TC = 16 * CT, kStep = 32;
    static_assert(TR == kRouterTileRows && TC == kRouterTileExperts, "the launch's tile");
    __shared__ alignas(16) float xs[kStep][TR + 4];
    __shared__ alignas(16) float ws[kStep][TC + 4];
    const int row0 = blockIdx.y * TR, expert0 = blockIdx.x * TC;
    const int ty = threadIdx.x / 16, tx = threadIdx.x % 16;
    const uint16_t* xh = static_cast<const uint16_t*>(x);
    float acc[RT][CT] = {};
    for (int base = 0; base < d; base += kStep) {
        __syncthreads();
        // the staging: 8 activations of a row a load, 4 router weights of an expert a load
        for (int q = threadIdx.x; q < TR * kStep / 8; q += 256) {
            const int rr = q / (kStep / 8), kk = (q % (kStep / 8)) * 8;
            uint4 v = {0, 0, 0, 0};
            if (row0 + rr < r) v = *reinterpret_cast<const uint4*>(xh + static_cast<long long>(row0 + rr) * d + base + kk);
            const uint16_t* h = reinterpret_cast<const uint16_t*>(&v);
#pragma unroll
            for (int j = 0; j < 8; ++j) xs[kk + j][rr] = KIND == 1 ? __half2float(__ushort_as_half(h[j])) : bf16_to_float(h[j]);
        }
        for (int q = threadIdx.x; q < TC * kStep / 4; q += 256) {
            const int cc = q / (kStep / 4), kk = (q % (kStep / 4)) * 4;
            float4 v = {0, 0, 0, 0};
            if (expert0 + cc < e) v = *reinterpret_cast<const float4*>(rows + static_cast<long long>(expert0 + cc) * d + base + kk);
            ws[kk][cc] = v.x, ws[kk + 1][cc] = v.y, ws[kk + 2][cc] = v.z, ws[kk + 3][cc] = v.w;
        }
        __syncthreads();
#pragma unroll 4
        for (int i = 0; i < kStep; ++i) {
            float av[RT], bv[CT];
#pragma unroll
            for (int u = 0; u < RT; ++u) av[u] = xs[i][ty * RT + u];
#pragma unroll
            for (int w = 0; w < CT; ++w) bv[w] = ws[i][tx * CT + w];
#pragma unroll
            for (int u = 0; u < RT; ++u)
#pragma unroll
                for (int w = 0; w < CT; ++w) acc[u][w] = fmaf(av[u], bv[w], acc[u][w]);
        }
    }
#pragma unroll
    for (int u = 0; u < RT; ++u) {
        const int row = row0 + ty * RT + u;
        if (row >= r) continue;
#pragma unroll
        for (int w = 0; w < CT; ++w) {
            const int expert = expert0 + tx * CT + w;
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
