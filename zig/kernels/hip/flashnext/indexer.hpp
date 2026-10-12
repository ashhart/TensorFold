#pragma once

#include "streams.hpp"

// The sparse attention's indexer: pooled block keys, each row's block scores, and the blocks a row attends.

namespace {

constexpr int kSelectThreads = 1024;
constexpr int kSelectWaves = kSelectThreads / 32;

// Block b: the mean of its ratio raw keys, RMS-normed with scale and rotated at its first position, as tf_qk_rope.
// With `pos`, launch x is a row: the block it completes (if any) goes to slot b of `pooled`; else block first + x to x.
extern "C" __global__ void __launch_bounds__(kFnThreads)
tf_fn_idx_pool(const void* raw, void* pooled, const float* scale, int kind, int first, const int* pos, int dims,
               int ratio, int rotary, float theta, float eps) {
    __shared__ float scratch[kFnWaves];
    __shared__ float normed[kFnThreads];
    const int end = pos ? pos[blockIdx.x] + 1 : 0;
    if (pos && end % ratio != 0) return;
    const int b = pos ? end / ratio - 1 : first + blockIdx.x, d = threadIdx.x;
    const long long slot = pos ? b : blockIdx.x;
    float x = 0.f;
    if (d < dims) {
        float m = 0.f;
        for (int k = 0; k < ratio; ++k) m += load(raw, kind, (static_cast<long long>(b) * ratio + k) * dims + d);
        x = rounded(m / static_cast<float>(ratio), kind);
    }
    const float rinv = rsqrtf(fn_block_sum(x * x, scratch) / static_cast<float>(dims) + eps);
    if (d < dims) normed[d] = rounded(x * rinv * scale[d], kind);
    __syncthreads();
    if (d >= dims) return;
    const int half = rotary >> 1;
    float y = normed[d];
    if (d < rotary) {
        const int f = d < half ? d : d - half;
        const float freq = 1.f / powf(theta, static_cast<float>(f) / static_cast<float>(half));
        const float ang = static_cast<float>(b * ratio) * freq;
        const float c = cosf(ang), s = sinf(ang);
        y = d < half ? normed[f] * c - normed[f + half] * s : normed[f] * s + normed[f + half] * c;
    }
    store(pooled, kind, slot * dims + d, rounded(y, kind));
}

// Wave w of block (x, y) scores block 8 x + w for rows 8 y ..: the sum over heads of relu(q . key) / sqrt(dims).
extern "C" __global__ void __launch_bounds__(kFnThreads)
tf_fn_idx_scores(const float* q, const void* pooled, const int* complete, float* scores, int kind, int rows, int heads,
                 int dims, int blocks, long long stride, int top) {
    constexpr int kMost = 8;
    const int lane = threadIdx.x & 31;
    const int b = blockIdx.x * kFnWaves + (threadIdx.x >> 5);
    const int r0 = blockIdx.y * 8, r1 = min(rows, r0 + 8);
    bool wanted = false;  // a graph launches over a bound of blocks: those no row scores read nothing
    for (int r = r0; r < r1; ++r) wanted = wanted || (complete[r] > top && b < complete[r]);
    if (b >= blocks || !wanted) return;
    float key[kMost];
#pragma unroll
    for (int i = 0; i < kMost; ++i) {
        const int d = lane + 32 * i;
        key[i] = d < dims ? load(pooled, kind, static_cast<long long>(b) * dims + d) : 0.f;
    }
    const float norm = sqrtf(static_cast<float>(dims));
    for (int r = r0; r < r1; ++r) {
        if (complete[r] <= top || b >= complete[r]) continue;
        float total = 0.f;
        for (int h = 0; h < heads; ++h) {
            const float* qh = q + (static_cast<long long>(r) * heads + h) * dims;
            float dot = 0.f;
#pragma unroll
            for (int i = 0; i < kMost; ++i) {
                const int d = lane + 32 * i;
                if (d < dims) dot += qh[d] * key[i];
            }
#pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) dot += __shfl_xor(dot, mask, 32);
            total += fmaxf(dot, 0.f);
        }
        if (lane == 0) scores[r * stride + b] = total / norm;
    }
}

// An order-keeping unsigned key of a float: larger floats, larger keys.
__device__ __forceinline__ unsigned fn_order(float v) {
    const unsigned u = __float_as_uint(v);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// The block's exclusive prefix sum of one count a thread, and the total through `total`.
__device__ int fn_scan(int v, int* waves, int* total) {
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    int x = v;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        const int y = __shfl_up(x, off, 32);
        if (lane >= off) x += y;
    }
    __syncthreads();
    if (lane == 31) waves[wave] = x;
    __syncthreads();
    if (wave == 0) {
        int w = lane < kSelectWaves ? waves[lane] : 0;
#pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const int y = __shfl_up(w, off, 32);
            if (lane >= off) w += y;
        }
        if (lane < kSelectWaves) waves[lane] = w;
    }
    __syncthreads();
    *total = waves[kSelectWaves - 1];
    return x - v + (wave > 0 ? waves[wave - 1] : 0);
}

// Block r: past top complete blocks, the top best (ties: lower id) in block order, then the tail; else keys < ends.
extern "C" __global__ void __launch_bounds__(kSelectThreads)
tf_fn_idx_select(const float* scores, long long stride, const int* complete, const int* ends, int* keys,
                 long long key_stride, int* count, int* sparse, int top, int ratio) {
    __shared__ int hist[256];
    __shared__ int waves[kSelectWaves];
    __shared__ unsigned cut_s;
    __shared__ int need_s;
    const int r = blockIdx.x, t = threadIdx.x;
    const int nb = complete[r], end = ends[r];
    if (nb <= top) {
        if (t == 0) {
            count[r] = end;
            sparse[r] = 0;
        }
        return;
    }
    const float* sc = scores + r * stride;
    unsigned prefix = 0, mask = 0;
    int need = top;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = t; i < 256; i += kSelectThreads) hist[i] = 0;
        __syncthreads();
        for (int b = t; b < nb; b += kSelectThreads) {
            const unsigned k = fn_order(sc[b]);
            if ((k & mask) == prefix) atomicAdd(&hist[(k >> shift) & 255u], 1);
        }
        __syncthreads();
        if (t == 0) {
            int above = 0, bin = 255;
            while (above + hist[bin] < need) above += hist[bin--];
            cut_s = prefix | (static_cast<unsigned>(bin) << shift);
            need_s = need - above;
        }
        __syncthreads();
        prefix = cut_s;
        need = need_s;
        mask |= 255u << shift;
    }
    const int chunk = (nb + kSelectThreads - 1) / kSelectThreads;
    const int lo = min(nb, t * chunk), hi = min(nb, lo + chunk);
    int above = 0, equal = 0;
    for (int b = lo; b < hi; ++b) {
        const unsigned k = fn_order(sc[b]);
        above += k > prefix;
        equal += k == prefix;
    }
    int above_total = 0, equal_total = 0;
    int pa = fn_scan(above, waves, &above_total);
    int pe = fn_scan(equal, waves, &equal_total);
    int out = pa + min(pe, need);
    int* row = keys + r * key_stride;
    for (int b = lo; b < hi; ++b) {
        const unsigned k = fn_order(sc[b]);
        bool take = k > prefix;
        if (k == prefix) take = pe++ < need;
        if (!take) continue;
        for (int j = 0; j < ratio; ++j) row[out * ratio + j] = b * ratio + j;
        ++out;
    }
    for (int k = nb * ratio + t; k < end; k += kSelectThreads) row[top * ratio + k - nb * ratio] = k;
    if (t == 0) {
        count[r] = top * ratio + end - nb * ratio;
        sparse[r] = 1;
    }
}

}  // namespace
