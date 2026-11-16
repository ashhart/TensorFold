#pragma once

#include "streams.hpp"

// The sparse attention's indexer: pooled block keys, each row's block scores, and the blocks a row attends.

namespace {

constexpr int kSelectThreads = 1024;
constexpr int kSelectWaves = kSelectThreads / 32;

// Block b: the mean of its ratio raw keys, RMS-normed with scale and rotated at its first position, as tf_qk_rope.
extern "C" __global__ void __launch_bounds__(kFnThreads)
tf_fn_idx_pool(const void* raw, void* pooled, const float* scale, int kind, int first, const int* pos, int dims,
               int ratio, int rotary, float theta, float eps) {
    __shared__ float scratch[kFnWaves];
    __shared__ float normed[kFnThreads];
    // With `pos`, launch x is a row whose completed block (if any) fills slot b; else block first + x fills slot x.
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
    // the block's rows of q, staged once for its eight waves when they fit
    constexpr int kStage = 512;
    __shared__ float qs[8 * kStage];
    const int lane = threadIdx.x & 31;
    const int b0 = blockIdx.x * kFnWaves, b = b0 + (threadIdx.x >> 5);
    const int r0 = blockIdx.y * 8, r1 = min(rows, r0 + 8);
    bool tile = false;  // a graph launches over a bound of blocks: those no row scores read nothing
    if (b0 < blocks)
        for (int r = r0; r < r1; ++r) tile = tile || (complete[r] > top && b0 < complete[r]);
    if (!tile) return;
    const int width = heads * dims;
    const bool staged = width <= kStage;
    if (staged)
        for (int i = threadIdx.x; i < (r1 - r0) * width; i += kFnThreads)
            qs[i] = q[static_cast<long long>(r0) * width + i];
    __syncthreads();
    bool wanted = false;
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
            const float* qh = staged ? qs + ((r - r0) * heads + h) * dims
                                     : q + (static_cast<long long>(r) * heads + h) * dims;
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

// Keys a select thread holds in registers (a row's blocks in contiguous runs): rows of more blocks take the global path.
constexpr int kSelectHeld = 32;

// tf_fn_idx_select's search for the cut, from a thread's `held` keys (runs of blocks [lo, lo + held)): each pass a
// histogram a wave (no shared atomic address across waves), summed a bin, and the bin found by a parallel suffix scan
// of the 256 bins. The same cut and need as the serial search: the same keys out.
__device__ __forceinline__ void fn_cut_held(const unsigned (&k)[kSelectHeld], int held, int top, int (*whist)[256],
                                            int* hist, unsigned* cut_s, int* need_s, unsigned& prefix, int& need) {
    const int t = threadIdx.x, lane = t & 31, wave = t >> 5;
    unsigned mask = 0;
    prefix = 0;
    need = top;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = lane; i < 256; i += 32) whist[wave][i] = 0;
        __syncthreads();
#pragma unroll
        for (int i = 0; i < kSelectHeld; ++i)
            if (i < held && (k[i] & mask) == prefix) atomicAdd(&whist[wave][(k[i] >> shift) & 255u], 1);
        __syncthreads();
        // bins on threads 0..255: the bin's count, then the count of bins above it (a suffix scan over 8 waves)
        int c = 0, x = 0;
        if (t < 256) {
#pragma unroll 4
            for (int w = 0; w < kSelectWaves; ++w) c += whist[w][255 - t];  // t's place in the scan from the top bin down
            x = c;
#pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                const int y = __shfl_up(x, off, 32);
                if (lane >= off) x += y;
            }
            if (lane == 31) hist[wave] = x;  // the 8 waves' totals
        }
        __syncthreads();
        if (t < 256) {
            int before = 0;
            for (int w = 0; w < wave; ++w) before += hist[w];
            const int incl = x + before, above = incl - c;  // blocks in bins above bin 255 - t, and with it
            if (incl >= need && above < need) {
                *cut_s = prefix | (static_cast<unsigned>(255 - t) << shift);
                *need_s = need - above;
            }
        }
        __syncthreads();
        prefix = *cut_s;
        need = *need_s;
        mask |= 255u << shift;
    }
}

// Block r: past top complete blocks, the top best (ties: lower id) in block order, then the tail; else keys < ends.
extern "C" __global__ void __launch_bounds__(kSelectThreads)
tf_fn_idx_select(const float* scores, long long stride, const int* complete, const int* ends, int* keys,
                 long long key_stride, int* count, int* sparse, int top, int ratio) {
    __shared__ int hist[256];
    __shared__ int waves[kSelectWaves];
    __shared__ int whist[kSelectWaves][256];
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
    const int chunk = (nb + kSelectThreads - 1) / kSelectThreads;
    const int lo = min(nb, t * chunk), hi = min(nb, lo + chunk);
    unsigned prefix = 0;
    int need = top;
    unsigned k[kSelectHeld];
    const bool held = chunk <= kSelectHeld;
    if (held) {
#pragma unroll
        for (int i = 0; i < kSelectHeld; ++i) k[i] = lo + i < hi ? fn_order(sc[lo + i]) : 0u;
        fn_cut_held(k, hi - lo, top, whist, hist, &cut_s, &need_s, prefix, need);
    } else {
        unsigned mask = 0;
        for (int shift = 24; shift >= 0; shift -= 8) {
            for (int i = t; i < 256; i += kSelectThreads) hist[i] = 0;
            __syncthreads();
            for (int b = t; b < nb; b += kSelectThreads) {
                const unsigned kb = fn_order(sc[b]);
                if ((kb & mask) == prefix) atomicAdd(&hist[(kb >> shift) & 255u], 1);
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
    }
    int above = 0, equal = 0;
    if (held) {
#pragma unroll
        for (int i = 0; i < kSelectHeld; ++i)
            if (lo + i < hi) { above += k[i] > prefix; equal += k[i] == prefix; }
    } else {
        for (int b = lo; b < hi; ++b) {
            const unsigned kb = fn_order(sc[b]);
            above += kb > prefix;
            equal += kb == prefix;
        }
    }
    int above_total = 0, equal_total = 0;
    int pa = fn_scan(above, waves, &above_total);
    int pe = fn_scan(equal, waves, &equal_total);
    int out = pa + min(pe, need);
    int* row = keys + r * key_stride;
    if (held) {
#pragma unroll
        for (int i = 0; i < kSelectHeld; ++i) {
            if (lo + i >= hi) break;
            bool take = k[i] > prefix;
            if (k[i] == prefix) take = pe++ < need;
            if (!take) continue;
            for (int j = 0; j < ratio; ++j) row[out * ratio + j] = (lo + i) * ratio + j;
            ++out;
        }
    } else {
        for (int b = lo; b < hi; ++b) {
            const unsigned kb = fn_order(sc[b]);
            bool take = kb > prefix;
            if (kb == prefix) take = pe++ < need;
            if (!take) continue;
            for (int j = 0; j < ratio; ++j) row[out * ratio + j] = b * ratio + j;
            ++out;
        }
    }
    for (int kk = nb * ratio + t; kk < end; kk += kSelectThreads) row[top * ratio + kk - nb * ratio] = kk;
    if (t == 0) {
        count[r] = top * ratio + end - nb * ratio;
        sparse[r] = 1;
    }
}

}  // namespace
