#pragma once

#include "streams.hpp"
#include "../ops/qk_rope_wave.hpp"
#include "../ops/common.hpp"
#include "common/wmma.hpp"

// The sparse attention's indexer: pooled block keys, each row's block scores, and the blocks a row attends.

namespace {

constexpr int kSelectThreads = 1024;
constexpr int kSelectWaves = kSelectThreads / 32;

// Block b (kFnThreads): the mean of its ratio raw keys, normed and rotated as tf_qk_rope, into slot `slot` of pooled.
__device__ __forceinline__ void fn_idx_pool_block(const void* raw, void* pooled, const float* scale, int kind, int b,
                                                  long long slot, int dims, int ratio, int rotary, float theta, float eps,
                                                  float* scratch, float* normed) {
    const int d = threadIdx.x;
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

// Block b: the mean of its ratio raw keys, RMS-normed with scale and rotated at its first position, as tf_qk_rope.
extern "C" __global__ void __launch_bounds__(kFnThreads)
tf_fn_idx_pool(const void* raw, void* pooled, const float* scale, int kind, int first, const int* pos, int dims,
               int ratio, int rotary, float theta, float eps) {
    __shared__ float scratch[kFnWaves];
    __shared__ float normed[kFnThreads];
    // With `pos`, launch x is a row whose completed block (if any) fills slot b; else block first + x fills slot x.
    const int end = pos ? pos[blockIdx.x] + 1 : 0;
    if (pos && end % ratio != 0) return;
    const int b = pos ? end / ratio - 1 : first + blockIdx.x;
    fn_idx_pool_block(raw, pooled, scale, kind, b, pos ? b : blockIdx.x, dims, ratio, rotary, theta, eps, scratch, normed);
}

// A QSA layer's decode prep in one launch: q, index q and k as tf_qk_rope, v and the raw index key cached, the pools.
struct QsaRope {
    const void* src;
    long long s_row;
    int s_head;
    int heads;
    int width;
    const float* weight;
    float* wide;   // fp32 rows (heads x width a row), or null
    void* cache;   // the cache's heads of `total` slots of width, or null
};
struct QsaPrep {
    QsaRope q, iq, k;
    const void* v;          // the segment's v rows (k.heads x v_width)
    void* v_cache;
    int v_width;
    const void* raw;        // the index projection rows; the raw key at raw_offset, rows raw_stride apart
    long long raw_stride;
    int raw_offset;
    void* raw_cache;        // one head of `total` slots of raw_width
    int raw_width;
    void* pooled;
    const float* pool_scale;
    int ratio;
    const int* pos;         // q and index q rows' positions (rows)
    const int* seg_pos;     // the segment's rows' positions (n)
    int rows;
    int n;
    int total;
    int kind;
    float eps;
    int rotary;
    float theta;
};

extern "C" __global__ void __launch_bounds__(kFnThreads) tf_fn_qsa_prep(QsaPrep a) {
    const int tq = a.q.src ? a.rows * a.q.heads : 0;
    const int tiq = a.q.src ? a.rows * a.iq.heads : 0;
    const int tasks = tq + tiq + a.n * a.k.heads;
    const int rope_blocks = (tasks + kFnWaves - 1) / kFnWaves;
    if (static_cast<int>(blockIdx.x) < rope_blocks) {
        __shared__ float row_v[kFnWaves][512];
        const int wave = threadIdx.x >> 5;
        const int g = blockIdx.x * kFnWaves + wave;
        const bool live = g < tasks;
        const int which = !live || g < tq ? 0 : g < tq + tiq ? 1 : 2;
        const QsaRope& t = which == 0 ? a.q : which == 1 ? a.iq : a.k;
        const int gi = !live ? 0 : which == 0 ? g : which == 1 ? g - tq : g - tq - tiq;
        const int r = gi / t.heads, h = gi % t.heads;
        qk_rope_norm(t.src, a.kind, r * t.s_row + static_cast<long long>(h) * t.s_head, t.weight, a.eps, t.width, live,
                     row_v[wave]);
        __syncthreads();
        if (!live) return;
        const int p = (which == 2 ? a.seg_pos : a.pos)[r];
        qk_rope_put(a.kind, t.width, a.rotary, a.theta, p, row_v[wave],
                    t.wide ? t.wide + static_cast<long long>(gi) * t.width : nullptr, t.cache,
                    (static_cast<long long>(h) * a.total + p) * t.width);
        return;
    }
    // the last block: v and the raw index key into their caches (16-bit or fp32 values copied), then the pools
    const int first = a.seg_pos[0];
    const long long nv = static_cast<long long>(a.n) * a.k.heads * a.v_width;
    for (long long i = threadIdx.x; i < nv; i += kFnThreads) {
        const int j = static_cast<int>(i % a.v_width);
        const int h = static_cast<int>((i / a.v_width) % a.k.heads);
        const int r = static_cast<int>(i / (static_cast<long long>(a.v_width) * a.k.heads));
        const long long to = (static_cast<long long>(h) * a.total + first + r) * a.v_width + j;
        if (a.kind == 0) static_cast<float*>(a.v_cache)[to] = static_cast<const float*>(a.v)[i];
        else static_cast<uint16_t*>(a.v_cache)[to] = static_cast<const uint16_t*>(a.v)[i];
    }
    const long long nr = static_cast<long long>(a.n) * a.raw_width;
    for (long long i = threadIdx.x; i < nr; i += kFnThreads) {
        const long long r = i / a.raw_width;
        const int c = static_cast<int>(i % a.raw_width);
        const long long from = r * a.raw_stride + a.raw_offset + c;
        const long long to = (first + r) * a.raw_width + c;
        if (a.kind == 0) static_cast<float*>(a.raw_cache)[to] = static_cast<const float*>(a.raw)[from];
        else static_cast<uint16_t*>(a.raw_cache)[to] = static_cast<const uint16_t*>(a.raw)[from];
    }
    __shared__ float scratch[kFnWaves];
    __shared__ float normed[kFnThreads];
    for (int r = 0; r < a.n; ++r) {
        __syncthreads();
        const int end = a.seg_pos[r] + 1;
        if (end % a.ratio != 0) continue;
        const int b = end / a.ratio - 1;
        fn_idx_pool_block(a.raw_cache, a.pooled, a.pool_scale, a.kind, b, b, a.raw_width, a.ratio, a.rotary, a.theta,
                          a.eps, scratch, normed);
    }
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

// tf_fn_idx_scores for 4 x 128 index heads at a window's rows: a wave's 8 rows' dots before the butterflies: same bits.
extern "C" __global__ void __launch_bounds__(kFnThreads)
tf_fn_idx_scores8(const float* q, const void* pooled, const int* complete, float* scores, int kind, int rows,
                  int blocks, long long stride, int top) {
    constexpr int R = 8, H = 4, D = 128, I = D / 32;
    const int lane = threadIdx.x & 31;
    const int b = blockIdx.x * kFnWaves + (threadIdx.x >> 5);
    const int r0 = blockIdx.y * R, r1 = min(rows, r0 + R);
    bool wanted = false;  // a graph launches over a bound of blocks: those no row scores read nothing
    for (int r = r0; r < r1; ++r) wanted = wanted || (complete[r] > top && b < complete[r]);
    if (b >= blocks || !wanted) return;
    float key[I];
#pragma unroll
    for (int i = 0; i < I; ++i) key[i] = load(pooled, kind, static_cast<long long>(b) * D + lane + 32 * i);
    float dot[R][H];
#pragma unroll
    for (int j = 0; j < R; ++j) {
        const int r = min(r0 + j, rows - 1);
#pragma unroll
        for (int h = 0; h < H; ++h) {
            const float* qh = q + (static_cast<long long>(r) * H + h) * D;
            float x = 0.f;
#pragma unroll
            for (int i = 0; i < I; ++i) x += qh[lane + 32 * i] * key[i];
            dot[j][h] = x;
        }
    }
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
#pragma unroll
        for (int j = 0; j < R; ++j)
#pragma unroll
            for (int h = 0; h < H; ++h) dot[j][h] += __shfl_xor(dot[j][h], mask, 32);
    const float norm = sqrtf(static_cast<float>(D));
#pragma unroll
    for (int j = 0; j < R; ++j) {
        const int r = r0 + j;
        if (r >= r1 || complete[r] <= top || b >= complete[r]) continue;
        float total = 0.f;
#pragma unroll
        for (int h = 0; h < H; ++h) total += fmaxf(dot[j][h], 0.f);
        if (lane == 0) scores[r * stride + b] = total / norm;
    }
}

// A prompt slice's scores for 4 heads of 128 dims, 4 lanes a score in tf_fn_idx_scores's sum order: the same bits.
constexpr int kQuadRows = 32, kQuadBlocks = 64, kQuadDims = 128, kQuadHeads = 4;

__device__ __forceinline__ float fn_quad_sum(const float (&part)[8]) {
    const float a0 = part[0] + part[4], a1 = part[1] + part[5], a2 = part[2] + part[6], a3 = part[3] + part[7];
    const float b0 = a0 + a2, b1 = a1 + a3;
    float c = b0 + b1;
    c = c + __shfl_xor(c, 2, 32);
    c = c + __shfl_xor(c, 1, 32);
    return c;
}

extern "C" __global__ void __launch_bounds__(128)
tf_fn_idx_scores_quad(const float* q, const void* pooled, const int* complete, float* scores, int kind, int rows,
                      int blocks, long long stride, int top) {
    constexpr int D = kQuadDims, H = kQuadHeads;
    __shared__ __attribute__((aligned(16))) float ks[kQuadBlocks][D];  // transposed: [blk][p * 32 + m] = key[p + 4 m]
    __shared__ int any;
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, p = lane & 3;
    const int r = blockIdx.y * kQuadRows + wave * 8 + (lane >> 2);
    const int b0 = blockIdx.x * kQuadBlocks;
    const int lim = r < rows && complete[r] > top ? complete[r] : 0;  // the row scores blocks below lim
    if (threadIdx.x == 0) any = 0;
    __syncthreads();
    if (lim > b0 && (lane & 3) == 0) atomicOr(&any, 1);
    __syncthreads();
    if (!any) return;  // no row of this tile scores a block of it
    for (int e = threadIdx.x; e < kQuadBlocks * D; e += 128) {
        const int blk = e / D, d = e - blk * D;
        ks[blk][(d & 3) * 32 + (d >> 2)] = b0 + blk < blocks ? load(pooled, kind, static_cast<long long>(b0 + blk) * D + d) : 0.f;
    }
    float qv[H][32];
#pragma unroll
    for (int h = 0; h < H; ++h)
#pragma unroll
        for (int m = 0; m < 32; ++m) qv[h][m] = r < rows ? q[(static_cast<long long>(r) * H + h) * D + p + 4 * m] : 0.f;
    __syncthreads();
    const float norm = sqrtf(static_cast<float>(D));
    const int n = min(kQuadBlocks, blocks - b0);
    for (int blk = 0; blk < n; ++blk) {
        float key[32];
        const float4* k4 = reinterpret_cast<const float4*>(&ks[blk][p * 32]);
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            const float4 a = k4[c];
            key[4 * c] = a.x, key[4 * c + 1] = a.y, key[4 * c + 2] = a.z, key[4 * c + 3] = a.w;
        }
        float total = 0.f;
#pragma unroll
        for (int h = 0; h < H; ++h) {
            float part[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) part[j] = 0.f;
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 8; ++j) part[j] = part[j] + qv[h][j + 8 * i] * key[j + 8 * i];
            total += fmaxf(fn_quad_sum(part), 0.f);
        }
        if (p == 0 && b0 + blk < lim) scores[r * stride + b0 + blk] = total / norm;
    }
}

// A prompt slice's index scores on the gfx11 matrix cores (numerics=inexact): their sum order, so last bits can differ.
constexpr int kMmaRows = 64, kMmaBlocks = 64;

extern "C" __global__ void __launch_bounds__(128)
tf_fn_idx_scores_mma(const float* q, const void* pooled, const int* complete, float* scores, int rows, int blocks,
                     long long stride, int top) {
#if TF_DEVICE_WMMA_GFX11
    using tf::rocm::bf16x16;
    using tf::rocm::f32x8;
    using tf::rocm::u32x8;
    using Mma = tf::rocm::WmmaBF16;
    constexpr int D = kQuadDims, H = kQuadHeads;
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, col = lane & 15, half = lane >> 4;
    const int tile_r0 = blockIdx.y * kMmaRows;
    const int b0 = blockIdx.x * kMmaBlocks;
    // a row scores its first complete blocks, and complete grows with the row: the tile's last row scores the most
    const int last = min(rows, tile_r0 + kMmaRows) - 1;
    if (last < tile_r0 || complete[last] <= top || b0 >= complete[last]) return;
    const int rb = tile_r0 + wave * 16;
    const int qr = rb + col;
    const bool qlive = qr < rows;
    const uint16_t* kp = static_cast<const uint16_t*>(pooled);
    f32x8 total[4];
#pragma unroll
    for (int t = 0; t < 4; ++t) total[t] = f32x8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int h = 0; h < H; ++h) {
        f32x8 acc[4];
#pragma unroll
        for (int t = 0; t < 4; ++t) acc[t] = f32x8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll 2
        for (int kk = 0; kk < D / 16; ++kk) {
            u32x8 aw;
            const float* src = q + (static_cast<long long>(qlive ? qr : 0) * H + h) * D + kk * 16;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float x0 = qlive ? src[2 * i] : 0.f, x1 = qlive ? src[2 * i + 1] : 0.f;
                aw[i] = static_cast<uint32_t>(float_to_bf16(x0)) | (static_cast<uint32_t>(float_to_bf16(x1)) << 16);
            }
            const bf16x16 a = __builtin_bit_cast(bf16x16, aw);
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const int kb = min(b0 + 16 * t + col, blocks - 1);
                const uint4* ks = reinterpret_cast<const uint4*>(kp + static_cast<long long>(kb) * D + kk * 16);
                const uint4 k0 = ks[0], k1 = ks[1];
                const u32x8 bw = {k0.x, k0.y, k0.z, k0.w, k1.x, k1.y, k1.z, k1.w};
                acc[t] = Mma::mma(a, __builtin_bit_cast(bf16x16, bw), acc[t]);
            }
        }
#pragma unroll
        for (int t = 0; t < 4; ++t)
#pragma unroll
            for (int i = 0; i < 8; ++i) total[t][i] += fmaxf(acc[t][i], 0.f);
    }
    const float norm = sqrtf(static_cast<float>(D));
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int r = rb + 2 * i + half;
        if (r >= rows) continue;
        const int lim = complete[r] > top ? complete[r] : 0;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int blk = b0 + 16 * t + col;
            if (blk < lim) scores[r * stride + blk] = total[t][i] / norm;
        }
    }
#endif
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
    __shared__ int suf[256];
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
        // suf[i]: the count in bins 255 - i .. 255 (8 waves of 32 bins: a wave's scan, then the waves above it)
        if (t < 256) {
            const int lane = t & 31;
            int x = hist[255 - t];
#pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                const int y = __shfl_up(x, off, 32);
                if (lane >= off) x += y;
            }
            if (lane == 31) waves[t >> 5] = x;
            suf[t] = x;
        }
        __syncthreads();
        if (t < 256) {
            int add = 0;
            for (int w = 0; w < (t >> 5); ++w) add += waves[w];
            suf[t] += add;
        }
        __syncthreads();
        // the cut: the one i with suf[i] >= need and the bins above it under need (the bins hold at least need)
        if (t < 256) {
            const int above = t > 0 ? suf[t - 1] : 0;
            if (suf[t] >= need && above < need) {
                cut_s = prefix | (static_cast<unsigned>(255 - t) << shift);
                need_s = need - above;
            }
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
