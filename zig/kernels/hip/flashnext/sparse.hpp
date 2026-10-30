#pragma once

#include "../ops/common.hpp"
#include "common/wmma.hpp"

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

// ---- A prompt's sparse attention on the gfx11 matrix cores, Flash Next only (24 q heads of 256, 2 kv heads, bf16 KV).

// After Strata's gfx11 prompt attention (github.com/Niko1221/Strata, MIT) and mlx-serve's qsa_nax.metal (rows = heads).
constexpr int kPromptHeads = 24;
constexpr int kPromptGroup = 12;
constexpr int kPromptDims = 256;
constexpr int kPromptThreads = 128;
constexpr int kPromptChunk = 32;
constexpr int kPromptPRow = kPromptChunk + 4;  // fp32 p row: 16-byte aligned, banks spread
constexpr int kPromptVRow = kPromptChunk + 8;  // bf16 transposed value row (a dim): 80 bytes

struct alignas(16) PromptSmem {
    float part[4][16][kPromptChunk + 1];
    float p[16][kPromptPRow];
    uint16_t vt[4][64][kPromptVRow];
    float m[16], l[16], alpha[16];
};

__device__ __forceinline__ uint32_t prompt_pack(uint16_t lo, uint16_t hi) {
    return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}

// PARTS: part p of gridDim.z of the row's keys, in tf_fn_sparse_parts' layout; LO: p v also over p's bf16 remainder.
template <bool SPLIT, bool PARTS, bool LO = true>
__device__ void sparse_prompt(const float* __restrict__ q, const uint16_t* __restrict__ k,
                              const uint16_t* __restrict__ v, const int* __restrict__ ids, long long id_stride,
                              const int* __restrict__ count, const int* __restrict__ sparse, const uint16_t* qg,
                              void* out, float* parts_ms, long long cap, float scale) {
#if TF_DEVICE_WMMA_GFX11
    using tf::rocm::bf16x16;
    using tf::rocm::f32x8;
    using tf::rocm::u32x8;
    using Mma = tf::rocm::WmmaBF16;
    __shared__ PromptSmem S;
    const int r = blockIdx.x, g = blockIdx.y;
    const int t = threadIdx.x, lane = t & 31, wave = t >> 5;
    const int col = lane & 15, half = lane >> 4;
    const int dim0 = wave * 64;
    const int n_row = count[r];
    const int part = PARTS ? static_cast<int>(blockIdx.z) : 0, parts = PARTS ? static_cast<int>(gridDim.z) : 1;
    const int lo = static_cast<int>(static_cast<long long>(part) * n_row / parts);
    const int n = static_cast<int>(static_cast<long long>(part + 1) * n_row / parts);
    const bool listed = ids != nullptr && sparse != nullptr && sparse[r] != 0;
    const int* row_ids = ids + r * id_stride;
    const long long head_base = static_cast<long long>(g) * cap * kPromptDims;

    // q fragments of this wave's 64 dims: row col (heads 12..15 zero), k = dim0 + 16 kk + 0..15
    bf16x16 qh[4], ql[4];
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
        u32x8 hw, lw;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            float x0 = 0.f, x1 = 0.f;
            if (col < kPromptGroup) {
                const float* src = q + (static_cast<long long>(r) * kPromptHeads + g * kPromptGroup + col) * kPromptDims +
                                   dim0 + kk * 16 + 2 * i;
                x0 = src[0] * scale;
                x1 = src[1] * scale;
            }
            const uint16_t h0 = float_to_bf16(x0), h1 = float_to_bf16(x1);
            hw[i] = prompt_pack(h0, h1);
            if constexpr (SPLIT) lw[i] = prompt_pack(float_to_bf16(x0 - bf16_to_float(h0)), float_to_bf16(x1 - bf16_to_float(h1)));
        }
        qh[kk] = __builtin_bit_cast(bf16x16, hw);
        if constexpr (SPLIT) ql[kk] = __builtin_bit_cast(bf16x16, lw);
    }
    if (t < 16) {
        S.m[t] = -INFINITY;
        S.l[t] = 0.f;
    }
    f32x8 acc[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[j] = f32x8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int c0 = lo; c0 < n; c0 += kPromptChunk) {
        // this lane's cell (lane = cell): its key, or key 0 past the row's keys (finite values under p = 0)
        const int cell = c0 + lane;
        const bool live = cell < n;
        const int key = live ? (listed ? row_ids[cell] : cell) : 0;
        // every read of the chunk first: the keys of the two 16-cell tiles (this wave's dims), the lane's value row
        uint4 kx[2][4][2];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int kr = __shfl(key, nt * 16 + col, 32);
            const uint4* src = reinterpret_cast<const uint4*>(k + head_base + static_cast<long long>(kr) * kPromptDims + dim0);
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                kx[nt][kk][0] = src[2 * kk];
                kx[nt][kk][1] = src[2 * kk + 1];
            }
        }
        uint4 vx[8];
        {
            const uint4* src = reinterpret_cast<const uint4*>(v + head_base + static_cast<long long>(key) * kPromptDims + dim0);
#pragma unroll
            for (int i = 0; i < 8; ++i) vx[i] = src[i];
        }
        // partial scores of this wave's dims, rows 2 i + half, cell nt * 16 + col
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            f32x8 s = f32x8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                const u32x8 w = {kx[nt][kk][0].x, kx[nt][kk][0].y, kx[nt][kk][0].z, kx[nt][kk][0].w,
                                 kx[nt][kk][1].x, kx[nt][kk][1].y, kx[nt][kk][1].z, kx[nt][kk][1].w};
                const bf16x16 b = __builtin_bit_cast(bf16x16, w);
                s = Mma::mma(qh[kk], b, s);
                if constexpr (SPLIT) s = Mma::mma(ql[kk], b, s);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) S.part[wave][2 * i + half][nt * 16 + col] = s[i];
        }
        // the wave's value rows, transposed: vt[wave][dim][cell]
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t w4[4] = {vx[i].x, vx[i].y, vx[i].z, vx[i].w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                S.vt[wave][i * 8 + 2 * j][lane] = static_cast<uint16_t>(w4[j] & 0xffffu);
                S.vt[wave][i * 8 + 2 * j + 1][lane] = static_cast<uint16_t>(w4[j] >> 16);
            }
        }
        __syncthreads();
        // online softmax: row t / 8, cells 4 (t % 8) ..; the four waves' partials in wave order
        {
            const int row = t >> 3, sub = t & 7;
            float x[4], mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = sub * 4 + j;
                const float sc = ((S.part[0][row][c] + S.part[1][row][c]) + S.part[2][row][c]) + S.part[3][row][c];
                x[j] = c0 + c < n ? sc : -INFINITY;
                mx = fmaxf(mx, x[j]);
            }
#pragma unroll
            for (int mask = 1; mask < 8; mask <<= 1) mx = fmaxf(mx, __shfl_xor(mx, mask, 32));
            const float m_old = S.m[row];
            const float m_new = fmaxf(m_old, mx);
            float sum = 0.f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float e = x[j] == -INFINITY ? 0.f : expf(x[j] - m_new);
                S.p[row][sub * 4 + j] = e;
                sum += e;
            }
#pragma unroll
            for (int mask = 1; mask < 8; mask <<= 1) sum += __shfl_xor(sum, mask, 32);
            if (sub == 0) {
                const float f = expf(m_old - m_new);
                S.alpha[row] = f;
                S.l[row] = S.l[row] * f + sum;
                S.m[row] = m_new;
            }
        }
        __syncthreads();
        // o = o alpha + p v over this wave's dims: 4 tiles of 16 dims, 2 steps of 16 cells
        float a[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) a[i] = S.alpha[2 * i + half];
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[j][i] *= a[i];
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            u32x8 hw, lw;
            const float* pr = &S.p[col][ks * 16];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float p0 = pr[2 * i], p1 = pr[2 * i + 1];
                const uint16_t h0 = float_to_bf16(p0), h1 = float_to_bf16(p1);
                hw[i] = prompt_pack(h0, h1);
                if constexpr (LO) lw[i] = prompt_pack(float_to_bf16(p0 - bf16_to_float(h0)), float_to_bf16(p1 - bf16_to_float(h1)));
            }
            const bf16x16 ph = __builtin_bit_cast(bf16x16, hw), pl = __builtin_bit_cast(bf16x16, lw);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint4* src = reinterpret_cast<const uint4*>(&S.vt[wave][j * 16 + col][ks * 16]);
                const uint4 lo = src[0], hi = src[1];
                const u32x8 w = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
                const bf16x16 b = __builtin_bit_cast(bf16x16, w);
                acc[j] = Mma::mma(ph, b, acc[j]);
                if constexpr (LO) acc[j] = Mma::mma(pl, b, acc[j]);
            }
        }
        // no barrier: only its own wave rewrites part and vt before the next first barrier, p and alpha after it
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int h = 2 * i + half;
        if (h >= kPromptGroup) continue;
        if constexpr (PARTS) {
            const long long row = (static_cast<long long>(r) * kPromptHeads + g * kPromptGroup + h) * parts + part;
#pragma unroll
            for (int j = 0; j < 4; ++j) static_cast<float*>(out)[row * kPromptDims + dim0 + j * 16 + col] = acc[j][i];
            if (wave == 0 && col == 0) {
                parts_ms[row * 2] = S.m[h];
                parts_ms[row * 2 + 1] = S.l[h];
            }
            continue;
        }
        const float total = S.l[h];
        const long long at = (static_cast<long long>(r) * kPromptHeads + g * kPromptGroup + h) * kPromptDims;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int d = dim0 + j * 16 + col;
            const float o = total > 0.f ? acc[j][i] / total : 0.f;
            if (qg == nullptr) {
                static_cast<float*>(out)[at + d] = o;
            } else {
                const float gate = bf16_to_float(qg[at * 2 + kPromptDims + d]);
                static_cast<uint16_t*>(out)[at + d] = float_to_bf16(o * sigmoid(gate));
            }
        }
    }
#else
    __builtin_trap();
#endif
}

extern "C" __global__ void __launch_bounds__(kPromptThreads)
tf_fn_sparse_prompt(const float* q, const void* k, const void* v, const int* ids, long long id_stride,
                    const int* count, const int* sparse, const void* qg, void* out, long long cap, float scale) {
    sparse_prompt<false, false>(q, static_cast<const uint16_t*>(k), static_cast<const uint16_t*>(v), ids, id_stride,
                                count, sparse, static_cast<const uint16_t*>(qg), out, nullptr, cap, scale);
}

// The same with p in bf16 once (no lo pass over its rounding remainder): numerics=inexact's prompt tile.
extern "C" __global__ void __launch_bounds__(kPromptThreads)
tf_fn_sparse_prompt_hi(const float* q, const void* k, const void* v, const int* ids, long long id_stride,
                       const int* count, const int* sparse, const void* qg, void* out, long long cap, float scale) {
    sparse_prompt<false, false, false>(q, static_cast<const uint16_t*>(k), static_cast<const uint16_t*>(v), ids, id_stride,
                                       count, sparse, static_cast<const uint16_t*>(qg), out, nullptr, cap, scale);
}

// The same over an fp32 q that need not be bf16-exact: q as bf16 hi + lo.
extern "C" __global__ void __launch_bounds__(kPromptThreads)
tf_fn_sparse_prompt_split(const float* q, const void* k, const void* v, const int* ids, long long id_stride,
                          const int* count, const int* sparse, const void* qg, void* out, long long cap, float scale) {
    sparse_prompt<true, false>(q, static_cast<const uint16_t*>(k), static_cast<const uint16_t*>(v), ids, id_stride,
                               count, sparse, static_cast<const uint16_t*>(qg), out, nullptr, cap, scale);
}

// The decode and window rows: the same tile over part blockIdx.z of each row's keys, into tf_fn_sparse_parts' parts.
extern "C" __global__ void __launch_bounds__(kPromptThreads)
tf_fn_sparse_tile_parts(const float* q, const void* k, const void* v, const int* ids, long long id_stride,
                        const int* count, const int* sparse, float* parts_o, float* parts_ms, long long cap,
                        float scale) {
    sparse_prompt<false, true>(q, static_cast<const uint16_t*>(k), static_cast<const uint16_t*>(v), ids, id_stride,
                               count, sparse, nullptr, parts_o, parts_ms, cap, scale);
}
extern "C" __global__ void __launch_bounds__(kPromptThreads)
tf_fn_sparse_tile_parts_split(const float* q, const void* k, const void* v, const int* ids, long long id_stride,
                              const int* count, const int* sparse, float* parts_o, float* parts_ms, long long cap,
                              float scale) {
    sparse_prompt<true, true>(q, static_cast<const uint16_t*>(k), static_cast<const uint16_t*>(v), ids, id_stride,
                              count, sparse, nullptr, parts_o, parts_ms, cap, scale);
}

// Block (h, r), a thread a dim: head h's parts of row r combined in part order; with qg, gated as tf_attn_gate does.
__device__ __forceinline__ void sparse_merge(const float* parts_o, const float* parts_ms, const void* qg, int kind,
                                             void* out, int heads, int dims, int parts) {
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
        const float o = total > 0.f ? acc / total : 0.f;
        const long long i = (static_cast<long long>(r) * heads + h) * dims + d;
        if (qg == nullptr) static_cast<float*>(out)[i] = o;
        else store(out, kind, i, o * sigmoid(load(qg, kind, (i - d) * 2 + dims + d)));
    }
}

extern "C" __global__ void tf_fn_sparse_merge(const float* parts_o, const float* parts_ms, float* out, int heads,
                                              int dims, int parts) {
    sparse_merge(parts_o, parts_ms, nullptr, 0, out, heads, dims, parts);
}

extern "C" __global__ void tf_fn_sparse_merge_gate(const float* parts_o, const float* parts_ms, const void* qg,
                                                   int kind, void* out, int heads, int dims, int parts) {
    sparse_merge(parts_o, parts_ms, qg, kind, out, heads, dims, parts);
}

}  // namespace
