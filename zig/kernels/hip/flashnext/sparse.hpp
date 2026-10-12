#pragma once

#include "../ops/common.hpp"

// Attention over a row's listed keys (the indexer's), or its first n keys, in parts merged in part order.

namespace {

constexpr int kSparseWaves = 8;
constexpr int kSparseMostDims = 256;
constexpr int kSparsePer = kSparseMostDims / 32;

// Keys a block stages at a time, and the padded row of the staged keys (rows read by distinct lanes, no conflicts).
constexpr int kSparseTile = 32;
constexpr int kSparseRow = kSparseMostDims + 1;

// Block (g, r, p): kv head g's query heads (G at most), row r, part p of its keys, 32 keys staged at a time.
template <int kSparseGroup>
__device__ void sparse_parts(const float* q, const void* k, const void* v, int cache_kind, const int* ids,
                             long long id_stride, const int* count, const int* sparse, float* parts_o, float* parts_ms,
                             int heads, int kv_heads, int dims, long long cap, float scale, int parts) {
    __shared__ float qs[kSparseGroup][kSparseMostDims];
    __shared__ float kt[kSparseTile][kSparseRow];
    __shared__ float ps[kSparseGroup][kSparseTile];
    __shared__ float fs[kSparseGroup], m[kSparseGroup], l[kSparseGroup];
    __shared__ long long keys[kSparseTile];
    const int g = blockIdx.x, r = blockIdx.y, part = blockIdx.z;
    const int group = heads / kv_heads;
    const int t = threadIdx.x, lane = t & 31, wave = t >> 5;
    const int n = count[r];
    const bool listed = ids != nullptr && sparse != nullptr && sparse[r] != 0;
    const int lo = static_cast<int>(static_cast<long long>(part) * n / parts);
    const int hi = static_cast<int>(static_cast<long long>(part + 1) * n / parts);
    const long long head_base = static_cast<long long>(g) * cap * dims;
    for (int i = t; i < group * dims; i += kSparseWaves * 32) {
        const int h = i / dims, d = i % dims;
        qs[h][d] = q[(static_cast<long long>(r) * heads + g * group + h) * dims + d] * scale;
    }
    if (t < kSparseGroup) {
        m[t] = -INFINITY;
        l[t] = 0.f;
    }
    float o[kSparseGroup];
#pragma unroll
    for (int h = 0; h < kSparseGroup; ++h) o[h] = 0.f;
    for (int t0 = lo; t0 < hi; t0 += kSparseTile) {
        const int tn = min(kSparseTile, hi - t0);
        __syncthreads();
        if (t < tn) keys[t] = listed ? ids[r * id_stride + t0 + t] : t0 + t;
        __syncthreads();
        for (int i = t; i < tn * dims; i += kSparseWaves * 32) {
            const int j = i / dims, d = i % dims;
            kt[j][d] = load(k, cache_kind, head_base + keys[j] * dims + d);
        }
        __syncthreads();
        // scores: a lane a key, a wave a head; then that head's softmax over the tile, folded into its running max
        for (int pair = t; pair < kSparseTile * group; pair += kSparseWaves * 32) {
            const int j = pair % kSparseTile, h = pair / kSparseTile;
            float sc = 0.f;
            for (int d = 0; d < dims; ++d) sc += qs[h][d] * kt[j][d];
            const bool live = j < tn;
            float top = live ? sc : -INFINITY;
#pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) top = fmaxf(top, __shfl_xor(top, mask, 32));
            top = fmaxf(top, m[h]);
            const float e = live ? expf(sc - top) : 0.f;
            float sum = e;
#pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) sum += __shfl_xor(sum, mask, 32);
            ps[h][j] = e;
            if (lane == 0) {
                const float f = expf(m[h] - top);
                fs[h] = f;
                l[h] = l[h] * f + sum;
                m[h] = top;
            }
        }
        __syncthreads();
        // values: a thread a dim, every head
        for (int d = t; d < dims; d += kSparseWaves * 32) {
#pragma unroll
            for (int h = 0; h < kSparseGroup; ++h) o[h] *= h < group ? fs[h] : 1.f;
            for (int j = 0; j < tn; ++j) {
                const float vd = load(v, cache_kind, head_base + keys[j] * dims + d);
#pragma unroll
                for (int h = 0; h < kSparseGroup; ++h) {
                    if (h < group) o[h] += ps[h][j] * vd;
                }
            }
        }
    }
    __syncthreads();
    for (int h = 0; h < group; ++h) {
        const long long out = (static_cast<long long>(r) * heads + g * group + h) * parts + part;
        for (int d = t; d < dims; d += kSparseWaves * 32) parts_o[out * dims + d] = o[h];
        if (t == 0) {
            parts_ms[out * 2] = m[h];
            parts_ms[out * 2 + 1] = l[h];
        }
    }
}

// The parts kernel for groups of up to 6 query heads (a tp-4 rank's) and of up to 12 (the whole model's).
extern "C" __global__ void __launch_bounds__(kSparseWaves * 32)
tf_fn_sparse_parts6(const float* q, const void* k, const void* v, int cache_kind, const int* ids, long long id_stride,
                    const int* count, const int* sparse, float* parts_o, float* parts_ms, int heads, int kv_heads,
                    int dims, long long cap, float scale, int parts) {
    sparse_parts<6>(q, k, v, cache_kind, ids, id_stride, count, sparse, parts_o, parts_ms, heads, kv_heads, dims, cap,
                    scale, parts);
}

extern "C" __global__ void __launch_bounds__(kSparseWaves * 32)
tf_fn_sparse_parts12(const float* q, const void* k, const void* v, int cache_kind, const int* ids, long long id_stride,
                     const int* count, const int* sparse, float* parts_o, float* parts_ms, int heads, int kv_heads,
                     int dims, long long cap, float scale, int parts) {
    sparse_parts<12>(q, k, v, cache_kind, ids, id_stride, count, sparse, parts_o, parts_ms, heads, kv_heads, dims, cap,
                     scale, parts);
}

// Block (h, r), a thread a dim: the parts of head h of row r combined in part order, fp32 rows-major out.
extern "C" __global__ void tf_fn_sparse_merge(const float* parts_o, const float* parts_ms, float* out, int heads,
                                              int dims, int parts) {
    const int h = blockIdx.x, r = blockIdx.y;
    const long long at = (static_cast<long long>(r) * heads + h) * parts;
    float top = -INFINITY;
    for (int p = 0; p < parts; ++p) top = fmaxf(top, parts_ms[(at + p) * 2]);
    for (int d = threadIdx.x; d < dims; d += blockDim.x) {
        float total = 0.f, acc = 0.f;
        for (int p = 0; p < parts; ++p) {
            const float mp = parts_ms[(at + p) * 2];
            const float w = mp == -INFINITY ? 0.f : expf(mp - top);
            total += parts_ms[(at + p) * 2 + 1] * w;
            acc += parts_o[(at + p) * dims + d] * w;
        }
        out[(static_cast<long long>(r) * heads + h) * dims + d] = total > 0.f ? acc / total : 0.f;
    }
}

}  // namespace
