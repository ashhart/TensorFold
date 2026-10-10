#pragma once

#include "common.hpp"

// Argmax and top-k of logits rows.

namespace {

constexpr int kThreads = 1024;
constexpr int kWaves = kThreads / 32;

__device__ __forceinline__ float widen(uint16_t bits, int kind) {
    if (kind == 1) return __half2float(__ushort_as_half(bits));
    return __uint_as_float(static_cast<uint32_t>(bits) << 16);
}

// torch's argmax order: a NaN beats every value, then the larger value, then the lower index.
__device__ __forceinline__ bool better(float a, int ia, float b, int ib) {
    bool na = a != a, nb = b != b;
    if (na || nb) return na && (!nb || ia < ib);
    return a > b || (a == b && ia < ib);
}

extern "C" __global__ void tf_argmax(const uint16_t* logits, int kind, int n, int* out) {
    const uint16_t* row = logits + static_cast<long long>(blockIdx.x) * n;
    float top = 0.0f;
    int at = -1;
    for (int i = threadIdx.x; i < n; i += kThreads) {
        float v = widen(row[i], kind);
        if (at < 0 || better(v, i, top, at)) {
            top = v;
            at = i;
        }
    }
    __shared__ float sv[kThreads];
    __shared__ int si[kThreads];
    sv[threadIdx.x] = top;
    si[threadIdx.x] = at;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            int j = threadIdx.x + stride;
            if (si[j] >= 0 && (si[threadIdx.x] < 0 || better(sv[j], si[j], sv[threadIdx.x], si[threadIdx.x]))) {
                sv[threadIdx.x] = sv[j];
                si[threadIdx.x] = si[j];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = si[0];
}

// A monotone 16-bit key of a value's bits (-0 and +0 share one), so equal keys are equal values.
__device__ __forceinline__ uint32_t order_key(uint16_t bits) {
    if (bits == 0x8000) bits = 0;
    return (bits & 0x8000) ? (~bits & 0xffffu) : (bits | 0x8000u);
}

// Row r's ks[r] largest by (value desc, id asc): two histograms of the k-th key, then one ordered compaction.
extern "C" __global__ void tf_topk(const uint16_t* logits, int n, const int* ks, int stride, int* ids, uint16_t* values) {
    const int r = blockIdx.x, k = ks[r];
    if (k <= 0) return;
    const uint16_t* row = logits + static_cast<long long>(r) * n;
    __shared__ int hist[256];
    __shared__ int pick[4];  // high byte, low byte, count above, count of the k-th key to take
    __shared__ int wave_gt[kWaves], wave_eq[kWaves];
    for (int i = threadIdx.x; i < 256; i += kThreads) hist[i] = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += kThreads) atomicAdd(&hist[order_key(row[i]) >> 8], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int above = 0, b = 255;
        for (; b > 0 && above + hist[b] < k; --b) above += hist[b];
        pick[0] = b;
        pick[2] = above;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 256; i += kThreads) hist[i] = 0;
    __syncthreads();
    const uint32_t high = pick[0];
    for (int i = threadIdx.x; i < n; i += kThreads) {
        uint32_t key = order_key(row[i]);
        if ((key >> 8) == high) atomicAdd(&hist[key & 255u], 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int above = pick[2], b = 255;
        for (; b > 0 && above + hist[b] < k; --b) above += hist[b];
        pick[1] = b;
        pick[2] = above;       // the keys above the k-th
        pick[3] = k - above;   // how many of the k-th's to take, lowest ids first
    }
    __syncthreads();
    const uint32_t kth = (high << 8) | static_cast<uint32_t>(pick[1]);
    const int above = pick[2], take = pick[3];
    int seen_gt = 0, seen_eq = 0;
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    for (int base = 0; base < n; base += kThreads) {
        int i = base + threadIdx.x;
        uint16_t bits = i < n ? row[i] : 0;
        uint32_t key = order_key(bits);
        bool gt = i < n && key > kth, eq = i < n && key == kth;
        unsigned long long mg = __ballot(gt), me = __ballot(eq);
        unsigned long long below = (1ull << lane) - 1;
        if (lane == 0) {
            wave_gt[wave] = __popcll(mg);
            wave_eq[wave] = __popcll(me);
        }
        __syncthreads();
        int gt_before = seen_gt, eq_before = seen_eq;
        for (int w = 0; w < wave; ++w) {
            gt_before += wave_gt[w];
            eq_before += wave_eq[w];
        }
        if (gt) {
            int slot = gt_before + __popcll(mg & below);
            ids[r * stride + slot] = i;
            values[r * stride + slot] = bits;
        }
        if (eq) {
            int rank = eq_before + __popcll(me & below);
            if (rank < take) {
                ids[r * stride + above + rank] = i;
                values[r * stride + above + rank] = bits;
            }
        }
        for (int w = 0; w < kWaves; ++w) {
            seen_gt += wave_gt[w];
            seen_eq += wave_eq[w];
        }
        __syncthreads();
        if (seen_eq >= take && seen_gt >= above) break;
    }
}

}  // namespace

