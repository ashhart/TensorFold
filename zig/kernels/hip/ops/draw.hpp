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

constexpr int kPartThreads = 256;

// tf_argmax over part p of each row (`per` values): its best value and index; `better` is a total order.
extern "C" __global__ void __launch_bounds__(kPartThreads)
    tf_argmax_part(const uint16_t* logits, int kind, int n, int per, float* part_v, int* part_i) {
    const uint16_t* row = logits + static_cast<long long>(blockIdx.y) * n;
    const int from = blockIdx.x * per, to = from + per < n ? from + per : n;
    float top = 0.0f;
    int at = -1;
    for (int i = from + threadIdx.x; i < to; i += kPartThreads) {
        float v = widen(row[i], kind);
        if (at < 0 || better(v, i, top, at)) {
            top = v;
            at = i;
        }
    }
    __shared__ float sv[kPartThreads];
    __shared__ int si[kPartThreads];
    sv[threadIdx.x] = top;
    si[threadIdx.x] = at;
    __syncthreads();
    for (int stride = kPartThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            int j = threadIdx.x + stride;
            if (si[j] >= 0 && (si[threadIdx.x] < 0 || better(sv[j], si[j], sv[threadIdx.x], si[threadIdx.x]))) {
                sv[threadIdx.x] = sv[j];
                si[threadIdx.x] = si[j];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        part_v[blockIdx.y * gridDim.x + blockIdx.x] = sv[0];
        part_i[blockIdx.y * gridDim.x + blockIdx.x] = si[0];
    }
}

// The best of a row's `parts` candidates (tf_argmax_part's): the row's tf_argmax.
extern "C" __global__ void __launch_bounds__(kPartThreads)
    tf_argmax_join(const float* part_v, const int* part_i, int parts, int* out) {
    float top = 0.0f;
    int at = -1;
    for (int p = threadIdx.x; p < parts; p += kPartThreads) {
        const float v = part_v[blockIdx.x * parts + p];
        const int i = part_i[blockIdx.x * parts + p];
        if (i >= 0 && (at < 0 || better(v, i, top, at))) {
            top = v;
            at = i;
        }
    }
    __shared__ float sv[kPartThreads];
    __shared__ int si[kPartThreads];
    sv[threadIdx.x] = top;
    si[threadIdx.x] = at;
    __syncthreads();
    for (int stride = kPartThreads / 2; stride > 0; stride >>= 1) {
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

// A row's entries for the top-k: `has(i)` for i < n, their 16-bit values and ids.
struct RowEntries {
    const uint16_t* row;  // the slice's first value
    int base, n;          // the slice's first id and length
    __device__ bool has(int i) const { return i < n; }
    __device__ uint16_t bits(int i) const { return row[i]; }
    __device__ int id(int i) const { return base + i; }
};

// tf_topk_part's candidates of a row: `parts` lists of `stride` slots, an absent slot's id -1.
struct Candidates {
    const int* ids;
    const uint16_t* vals;
    int n;
    __device__ bool has(int i) const { return i < n && ids[i] >= 0; }
    __device__ uint16_t bits(int i) const { return vals[i]; }
    __device__ int id(int i) const { return ids[i]; }
};

// The k-th largest key by two 8-bit histograms: pick = {high byte, low byte, keys above it, how many of it to take}.
template <class E>
__device__ void topk_kth(const E& e, int k, int* hist, int* pick) {
    for (int i = threadIdx.x; i < 256; i += kThreads) hist[i] = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < e.n; i += kThreads)
        if (e.has(i)) atomicAdd(&hist[order_key(e.bits(i)) >> 8], 1);
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
    for (int i = threadIdx.x; i < e.n; i += kThreads) {
        if (!e.has(i)) continue;
        uint32_t key = order_key(e.bits(i));
        if ((key >> 8) == high) atomicAdd(&hist[key & 255u], 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int above = pick[2], b = 255;
        for (; b > 0 && above + hist[b] < k; --b) above += hist[b];
        pick[1] = b;
        pick[2] = above;      // the keys above the k-th
        pick[3] = k - above;  // how many of the k-th's to take, lowest ids first
    }
    __syncthreads();
}

// Row r's ks[r] largest by (value desc, id asc) in one ordered compaction: keys above the k-th, then its lowest ids.
__device__ void topk_slice(const RowEntries& e, int k, int* ids, uint16_t* values) {
    __shared__ int hist[256];
    __shared__ int pick[4];
    __shared__ int wave_gt[kWaves], wave_eq[kWaves];
    topk_kth(e, k, hist, pick);
    const uint32_t kth = (static_cast<uint32_t>(pick[0]) << 8) | static_cast<uint32_t>(pick[1]);
    const int above = pick[2], take = pick[3];
    int seen_gt = 0, seen_eq = 0;
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    for (int base = 0; base < e.n; base += kThreads) {
        int i = base + threadIdx.x;
        uint16_t bits = i < e.n ? e.bits(i) : 0;
        uint32_t key = order_key(bits);
        bool gt = i < e.n && key > kth, eq = i < e.n && key == kth;
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
            ids[slot] = e.id(i);
            values[slot] = bits;
        }
        if (eq) {
            int rank = eq_before + __popcll(me & below);
            if (rank < take) {
                ids[above + rank] = e.id(i);
                values[above + rank] = bits;
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

extern "C" __global__ void tf_topk(const uint16_t* logits, int n, const int* ks, int stride, int* ids, uint16_t* values) {
    const int r = blockIdx.x, k = ks[r];
    if (k <= 0) return;
    topk_slice({logits + static_cast<long long>(r) * n, 0, n}, k, ids + r * stride, values + r * stride);
}

// tf_topk over part blockIdx.x of row blockIdx.y: the part's own top min(ks[r], length) into `stride` slots (rest -1).
extern "C" __global__ void tf_topk_part(const uint16_t* logits, int n, const int* ks, int per, int stride, int* cand_ids,
                                        uint16_t* cand_vals) {
    const int r = blockIdx.y, p = blockIdx.x, k = ks[r];
    if (k <= 0) return;
    const int base = p * per, len = max(0, min(per, n - base)), own = min(k, len);
    const long long at = (static_cast<long long>(r) * gridDim.x + p) * stride;
    for (int i = own + threadIdx.x; i < stride; i += kThreads) cand_ids[at + i] = -1;
    if (own > 0) topk_slice({logits + static_cast<long long>(r) * n + base, base, len}, own, cand_ids + at, cand_vals + at);
}

// Candidates a row's join holds in shared memory at most (tf_topk_parts' parts * stride).
constexpr int kJoinMost = 4096;

// Row r's tf_topk from its parts' candidates: the same k-th key, then the same order, slots counted in shared memory.
extern "C" __global__ void tf_topk_join(const int* cand_ids, const uint16_t* cand_vals, int parts, const int* ks, int stride,
                                        int* ids, uint16_t* values) {
    const int r = blockIdx.x, k = ks[r];
    if (k <= 0) return;
    const long long at = static_cast<long long>(r) * parts * stride;
    const Candidates e{cand_ids + at, cand_vals + at, parts * stride};
    __shared__ int hist[256];
    __shared__ int pick[4];
    __shared__ int sid[kJoinMost];
    __shared__ uint32_t skey[kJoinMost];
    topk_kth(e, k, hist, pick);
    const uint32_t kth = (static_cast<uint32_t>(pick[0]) << 8) | static_cast<uint32_t>(pick[1]);
    const int above = pick[2], take = pick[3];
    // each candidate's id and class: 2 above the k-th, 1 the k-th's, 0 neither (or absent)
    for (int i = threadIdx.x; i < e.n; i += kThreads) {
        const uint32_t key = e.has(i) ? order_key(e.bits(i)) : 0u;
        sid[i] = e.ids[i];
        skey[i] = !e.has(i) ? 0u : key > kth ? 2u : key == kth ? 1u : 0u;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < e.n; i += kThreads) {
        const uint32_t cls = skey[i];
        if (cls == 0) continue;
        const bool gt = cls == 2;
        const int id = sid[i];
        int before = 0;  // the candidates of the same class with lower ids
        for (int j = 0; j < e.n; ++j) before += skey[j] == cls && sid[j] < id;
        if (gt) {
            ids[r * stride + before] = id;
            values[r * stride + before] = e.bits(i);
        } else if (before < take) {
            ids[r * stride + above + before] = id;
            values[r * stride + above + before] = e.bits(i);
        }
    }
}

}  // namespace
