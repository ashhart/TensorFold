// Prompt-only native E4M3 score products; existing cache bytes and BF16 value boundaries are preserved.
#pragma once
#include "qmm_frag.cuh"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <limits.h>
#include <stdint.h>

namespace tf_fp8_attention {
constexpr int D = 256, WARPS = 4, HPC = 2, ROWS = 32, QROWS = 64, KEYS = 64;
constexpr float LOG2E = 1.4426950408889634f;
template <int TERMS>
constexpr int shared_bytes =
    ((QROWS * D * TERMS + KEYS * D + (QROWS * TERMS + KEYS) * 4 + QROWS + 4 + 127) / 128) * 128;

struct Inputs {
    const __nv_bfloat16 *q;
    const uint8_t *k, *v;
    const float *ks, *vs;
    __nv_bfloat16 *out;
    int prefix, rows, heads, kv_heads, group, capacity;
    float scale;
    const int *pages;
    int64_t page_bytes;
    int page_tokens;
};

__device__ __forceinline__ int word_at(int row, int word) { return row * (D / 4) + (word ^ ((row & 7) * 4)); }
__device__ __forceinline__ int value_at(int row, int chunk) { return row * (D / 8) + (chunk ^ (row & 7)); }
__device__ __forceinline__ float decode(uint8_t v) {
    return __half2float(__half(__nv_cvt_fp8_to_halfraw(v, __NV_E4M3)));
}
__device__ __forceinline__ float bf(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ __forceinline__ uint32_t pair(float a, float b) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const uint32_t *>(&v);
}
__device__ __forceinline__ float exponential(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(__fmul_rn(x, LOG2E)));
    return y;
}
__device__ __forceinline__ float divide(float a, float b) {
    float y;
    asm("div.full.f32 %0,%1,%2;" : "=f"(y) : "f"(a), "f"(b));
    return y;
}
__device__ __forceinline__ void mma8(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#if __CUDA_ARCH__ >= 890
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    __trap();
#endif
}

template <bool PAGED> __device__ __forceinline__ int64_t cache_row(const Inputs &p, int head, int key) {
    return static_cast<int64_t>(head) * (PAGED ? p.page_tokens : p.capacity) + (PAGED ? key % p.page_tokens : key);
}
template <bool PAGED> __device__ __forceinline__ int64_t page_offset(const Inputs &p, int key) {
    if constexpr (PAGED) return static_cast<int64_t>(p.pages[key / p.page_tokens]) * p.page_bytes;
    return 0;
}
template <bool PAGED>
__device__ __forceinline__ float cache_value(const Inputs &p, bool value, int head, int key, int d) {
    const int64_t row = cache_row<PAGED>(p, head, key), page = page_offset<PAGED>(p, key);
    const uint8_t *codes = (value ? p.v : p.k) + page + row * D;
    const float s = *reinterpret_cast<const float *>(reinterpret_cast<const uint8_t *>(value ? p.vs : p.ks) + page +
                                                     row * sizeof(float));
    return bf(__fmul_rn(decode(codes[d]), s));
}

// Only exceptional rows use this path; masked keys are never read by the independent FP64 fallback.
template <bool PAGED> __device__ void exceptional_row(const Inputs &p, int row, int head, int kv_head) {
    const int lane = threadIdx.x & 31, end = p.prefix + row + 1;
    double q[8], out[8] = {}, maximum = -INFINITY, denominator = 0.0;
#pragma unroll
    for (int d = 0; d < 8; ++d)
        q[d] = __bfloat162float(p.q[(static_cast<int64_t>(row) * p.heads + head) * D + lane + 32 * d]);
    bool nan = false;
    for (int key = 0; key < end; ++key) {
        double dot = 0.0;
#pragma unroll
        for (int d = 0; d < 8; ++d)
            dot += q[d] * static_cast<double>(cache_value<PAGED>(p, false, kv_head, key, lane + 32 * d));
#pragma unroll
        for (int shift = 16; shift > 0; shift >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, shift);
        const double score = dot * static_cast<double>(p.scale);
        nan = nan || isnan(score);
        maximum = fmax(maximum, score);
    }
    if (nan || !isfinite(maximum)) {
#pragma unroll
        for (int d = 0; d < 8; ++d)
            p.out[(static_cast<int64_t>(row) * p.heads + head) * D + lane + 32 * d] = __float2bfloat16_rn(NAN);
        return;
    }
    for (int key = 0; key < end; ++key) {
        double dot = 0.0;
#pragma unroll
        for (int d = 0; d < 8; ++d)
            dot += q[d] * static_cast<double>(cache_value<PAGED>(p, false, kv_head, key, lane + 32 * d));
#pragma unroll
        for (int shift = 16; shift > 0; shift >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, shift);
        const double probability = exp(dot * static_cast<double>(p.scale) - maximum);
        denominator += probability;
#pragma unroll
        for (int d = 0; d < 8; ++d) out[d] += probability * cache_value<PAGED>(p, true, kv_head, key, lane + 32 * d);
    }
#pragma unroll
    for (int d = 0; d < 8; ++d)
        p.out[(static_cast<int64_t>(row) * p.heads + head) * D + lane + 32 * d] =
            __float2bfloat16_rn(static_cast<float>(out[d] / denominator));
}

template <int TERMS>
__device__ void quantize_queries(const Inputs &p, uint32_t *hi, uint32_t *lo, float *scales, uint8_t *bad) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int group_count = p.group / HPC, kh = blockIdx.y / group_count;
    const int head = kh * p.group + (blockIdx.y % group_count) * HPC + warp % HPC;
    const int first = blockIdx.x * ROWS + (warp / HPC) * 16;
    for (int r = 0; r < 16; ++r) {
        const int row = first + r, slot = warp * 16 + r;
        float x[8], error[8], maximum = 0.0f;
        bool nonfinite = false;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            x[i] = row < p.rows
                       ? __bfloat162float(p.q[(static_cast<int64_t>(row) * p.heads + head) * D + lane + 32 * i])
                       : 0.0f;
            nonfinite |= !isfinite(x[i]);
            maximum = fmaxf(maximum, fabsf(x[i]));
        }
        nonfinite = __any_sync(0xffffffffu, nonfinite);
#pragma unroll
        for (int shift = 16; shift > 0; shift >>= 1)
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, shift));
        const float scale = maximum > 0.0f && !nonfinite ? fmaxf(__fdiv_rn(maximum, 448.0f), FLT_MIN) : 1.0f;
        const float inverse = __fdiv_rn(1.0f, scale);
        uint8_t codes[8];
        float error_max = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            codes[i] = nonfinite ? 0 : __nv_cvt_float_to_fp8(__fmul_rn(x[i], inverse), __NV_SATFINITE, __NV_E4M3);
            error[i] = nonfinite ? 0.0f : __fsub_rn(x[i], __fmul_rn(decode(codes[i]), scale));
            error_max = fmaxf(error_max, fabsf(error[i]));
            reinterpret_cast<uint8_t *>(hi)[word_at(slot, (lane + 32 * i) / 4) * 4 + (lane & 3)] = codes[i];
        }
        if (lane == 0) {
            scales[slot] = scale;
            bad[slot] = static_cast<uint8_t>(nonfinite);
        }
        if constexpr (TERMS == 2) {
#pragma unroll
            for (int shift = 16; shift > 0; shift >>= 1)
                error_max = fmaxf(error_max, __shfl_xor_sync(0xffffffffu, error_max, shift));
            const float residual_scale = error_max > 0.0f ? fmaxf(__fdiv_rn(error_max, 448.0f), FLT_MIN) : 1.0f;
            const float residual_inverse = __fdiv_rn(1.0f, residual_scale);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const uint8_t c =
                    __nv_cvt_float_to_fp8(__fmul_rn(error[i], residual_inverse), __NV_SATFINITE, __NV_E4M3);
                reinterpret_cast<uint8_t *>(lo)[word_at(slot, (lane + 32 * i) / 4) * 4 + (lane & 3)] = c;
            }
            if (lane == 0) scales[QROWS + slot] = residual_scale;
        }
    }
}

template <bool PAGED>
__device__ void load_keys(const Inputs &p, int kh, int start, int end, uint32_t *tile, float *scales, int *bad) {
    for (int i = threadIdx.x; i < KEYS * D / 16; i += WARPS * 32) {
        const int r = i / (D / 16), chunk = i % (D / 16), key = start + r;
        uint4 codes = make_uint4(0, 0, 0, 0);
        if (key < end) {
            const int64_t row = cache_row<PAGED>(p, kh, key), page = page_offset<PAGED>(p, key);
            codes = *reinterpret_cast<const uint4 *>(p.k + page + row * D + chunk * 16);
            uint32_t *words = reinterpret_cast<uint32_t *>(&codes);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint32_t marks = ((words[j] & 0x7f7f7f7fu) + 0x01010101u) & 0x80808080u;
                if (marks) {
                    atomicMin(bad, key);
                    uint8_t *bytes = reinterpret_cast<uint8_t *>(&words[j]);
#pragma unroll
                    for (int b = 0; b < 4; ++b)
                        if ((bytes[b] & 127u) == 127u) bytes[b] = 0;
                }
                tile[word_at(r, chunk * 4 + j)] = words[j];
            }
        } else {
            const uint32_t *words = reinterpret_cast<const uint32_t *>(&codes);
#pragma unroll
            for (int j = 0; j < 4; ++j) tile[word_at(r, chunk * 4 + j)] = words[j];
        }
    }
    if (threadIdx.x < KEYS) {
        const int key = start + threadIdx.x;
        float s = 0.0f;
        if (key < end) {
            const int64_t row = cache_row<PAGED>(p, kh, key), page = page_offset<PAGED>(p, key);
            s = *reinterpret_cast<const float *>(reinterpret_cast<const uint8_t *>(p.ks) + page + row * 4);
            if (!isfinite(s)) {
                atomicMin(bad, key);
                s = 0.0f;
            }
        }
        scales[threadIdx.x] = s;
    }
}

template <bool PAGED> __device__ void load_values(const Inputs &p, int kh, int start, int end, uint4 *tile, int *bad) {
    for (int i = threadIdx.x; i < 32 * D / 8; i += WARPS * 32) {
        const int r = i / (D / 8), chunk = i % (D / 8), key = start + r;
        uint4 result = make_uint4(0, 0, 0, 0);
        if (key < end) {
            const int64_t row = cache_row<PAGED>(p, kh, key), page = page_offset<PAGED>(p, key);
            const uint2 codes = *reinterpret_cast<const uint2 *>(p.v + page + row * D + chunk * 8);
            const uint8_t *bytes = reinterpret_cast<const uint8_t *>(&codes);
            const float s = *reinterpret_cast<const float *>(reinterpret_cast<const uint8_t *>(p.vs) + page + row * 4);
            uint32_t *pairs = reinterpret_cast<uint32_t *>(&result);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float a = __fmul_rn(decode(bytes[2 * j]), s), b = __fmul_rn(decode(bytes[2 * j + 1]), s);
                if (!isfinite(a) || !isfinite(b)) {
                    atomicMin(bad, key);
                    a = isfinite(a) ? a : 0.0f;
                    b = isfinite(b) ? b : 0.0f;
                }
                pairs[j] = pair(a, b);
                if ((pairs[j] & 0x7f80u) == 0x7f80u) {
                    atomicMin(bad, key);
                    pairs[j] &= 0xffff0000u;
                }
                if ((pairs[j] & 0x7f800000u) == 0x7f800000u) {
                    atomicMin(bad, key);
                    pairs[j] &= 0x0000ffffu;
                }
            }
        }
        tile[value_at(r, chunk)] = result;
    }
}

__device__ __forceinline__ float row_sum(const float (&v)[8]) {
    float w[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float x = __fadd_rn(v[j], __shfl_xor_sync(0xffffffffu, v[j], 2));
        w[j] = __fadd_rn(x, __shfl_xor_sync(0xffffffffu, x, 1));
    }
    return __fadd_rn(__fadd_rn(__fadd_rn(w[0], w[4]), __fadd_rn(w[2], w[6])),
                     __fadd_rn(__fadd_rn(w[1], w[5]), __fadd_rn(w[3], w[7])));
}

__device__ __forceinline__ void values(float (&out)[32][4], const uint32_t (&a0)[4], const uint32_t (&a1)[4],
                                       const uint4 *tile) {
    const int lane = threadIdx.x & 31, mt = lane / 8;
#pragma unroll
    for (int n = 0; n < 32; n += 2) {
        uint32_t b[4];
        qmm_frag::ldmatrix4t(b, tile + value_at(8 * (mt % 2) + lane % 8, n + mt / 2));
        qmm_frag::mma(out[n], a0, b[0], b[1]);
        qmm_frag::mma(out[n + 1], a0, b[2], b[3]);
    }
#pragma unroll
    for (int n = 0; n < 32; n += 2) {
        uint32_t b[4];
        qmm_frag::ldmatrix4t(b, tile + value_at(16 + 8 * (mt % 2) + lane % 8, n + mt / 2));
        qmm_frag::mma(out[n], a1, b[0], b[1]);
        qmm_frag::mma(out[n + 1], a1, b[2], b[3]);
    }
}

template <int TERMS, bool PAGED, bool DUMP = false, bool NARROW = false>
__device__ void run(const Inputs &p, float *dumped = nullptr) {
    extern __shared__ __align__(16) uint8_t memory[];
    uint32_t *qhi = reinterpret_cast<uint32_t *>(memory);
    uint32_t *qlo = qhi + QROWS * (D / 4);
    uint32_t *tile = qhi + TERMS * QROWS * (D / 4);
    float *qs = reinterpret_cast<float *>(tile + KEYS * (D / 4));
    float *ks = qs + QROWS * TERMS;
    uint8_t *qbad = reinterpret_cast<uint8_t *>(ks + KEYS);
    int *bad = reinterpret_cast<int *>(qbad + QROWS);
    if (threadIdx.x == 0) *bad = INT_MAX;
    quantize_queries<TERMS>(p, qhi, qlo, qs, qbad);
    __syncthreads();
    const int warp = threadIdx.x / 32, lane = threadIdx.x & 31, g = lane / 4, qd = lane % 4;
    const int groups = p.group / HPC, kh = blockIdx.y / groups;
    const int head = kh * p.group + (blockIdx.y % groups) * HPC + warp % HPC;
    const int first = blockIdx.x * ROWS, wrow = first + (warp / HPC) * 16;
    const int end = p.prefix + min(first + ROWS, p.rows), pos0 = p.prefix + wrow + g, pos1 = pos0 + 8;
    const int qr = warp * 16 + g;
    float out[32][4] = {}, maximum[2] = {-INFINITY, -INFINITY}, denominator[2] = {};
    for (int start = 0; start < end; start += KEYS) {
        load_keys<PAGED>(p, kh, start, end, tile, ks, bad);
        __syncthreads();
        float scores[8][4] = {}, residual[TERMS == 2 ? 8 : 1][4] = {};
#pragma unroll
        for (int k = 0; k < D / 32; ++k) {
            uint32_t a[4] = {qhi[word_at(qr, k * 8 + qd)], qhi[word_at(qr + 8, k * 8 + qd)],
                             qhi[word_at(qr, k * 8 + qd + 4)], qhi[word_at(qr + 8, k * 8 + qd + 4)]};
            uint32_t low[4];
            if constexpr (TERMS == 2) {
                low[0] = qlo[word_at(qr, k * 8 + qd)];
                low[1] = qlo[word_at(qr + 8, k * 8 + qd)];
                low[2] = qlo[word_at(qr, k * 8 + qd + 4)];
                low[3] = qlo[word_at(qr + 8, k * 8 + qd + 4)];
            }
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                const int kr = n * 8 + g;
                const uint32_t b0 = tile[word_at(kr, k * 8 + qd)], b1 = tile[word_at(kr, k * 8 + qd + 4)];
                mma8(scores[n], a, b0, b1);
                if constexpr (TERMS == 2) mma8(residual[n], low, b0, b1);
            }
            if constexpr (NARROW) {
                for (int n = 0; n < 8; ++n)
                    for (int i = 0; i < 4; ++i) scores[n][i] = bf(scores[n][i]);
            }
        }
        float probability[8][4];
#pragma unroll
        for (int side = 0; side < 2; ++side) {
            const int position = side ? pos1 : pos0, query = qr + side * 8;
            float tile_max = -INFINITY;
#pragma unroll
            for (int n = 0; n < 8; ++n)
                for (int pair_index = 0; pair_index < 2; ++pair_index) {
                    const int column = n * 8 + qd * 2 + pair_index, index = side * 2 + pair_index;
                    float s = __fmul_rn(scores[n][index], qs[query]);
                    if constexpr (TERMS == 2) s = __fmaf_rn(residual[n][index], qs[QROWS + query], s);
                    s = __fmul_rn(__fmul_rn(s, ks[column]), p.scale);
                    if (start + column <= position && !isfinite(s)) atomicMin(bad, start + column);
                    if constexpr (DUMP) {
                        const int output_row = wrow + g + side * 8;
                        if (output_row < p.rows && start + column < end)
                            dumped[(static_cast<int64_t>(output_row) * p.heads + head) * p.capacity + start + column] =
                                s;
                    }
                    probability[n][index] = start + column <= position && start + column < end ? s : -INFINITY;
                    tile_max = fmaxf(tile_max, probability[n][index]);
                }
            tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, 1));
            tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, 2));
            const bool active = tile_max != -INFINITY;
            const float next = active ? fmaxf(maximum[side], tile_max) : maximum[side];
            const float alpha =
                active ? (maximum[side] == -INFINITY ? 0.0f : exponential(__fsub_rn(maximum[side], next))) : 1.0f;
            float sums[8];
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                const int column = start + n * 8 + qd * 2;
                probability[n][side * 2] = active && column <= position && column < end
                                               ? exponential(__fsub_rn(probability[n][side * 2], next))
                                               : 0.0f;
                probability[n][side * 2 + 1] = active && column + 1 <= position && column + 1 < end
                                                   ? exponential(__fsub_rn(probability[n][side * 2 + 1], next))
                                                   : 0.0f;
                sums[n] = __fadd_rn(probability[n][side * 2], probability[n][side * 2 + 1]);
            }
            denominator[side] = __fmaf_rn(denominator[side], alpha, row_sum(sums));
            maximum[side] = next;
#pragma unroll
            for (int n = 0; n < 32; ++n) {
                out[n][side * 2] = __fmul_rn(out[n][side * 2], alpha);
                out[n][side * 2 + 1] = __fmul_rn(out[n][side * 2 + 1], alpha);
            }
        }
        uint32_t probabilities[4][4];
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            probabilities[n][0] = pair(probability[2 * n][0], probability[2 * n][1]);
            probabilities[n][1] = pair(probability[2 * n][2], probability[2 * n][3]);
            probabilities[n][2] = pair(probability[2 * n + 1][0], probability[2 * n + 1][1]);
            probabilities[n][3] = pair(probability[2 * n + 1][2], probability[2 * n + 1][3]);
        }
        __syncthreads();
        load_values<PAGED>(p, kh, start, end, reinterpret_cast<uint4 *>(tile), bad);
        __syncthreads();
        values(out, probabilities[0], probabilities[1], reinterpret_cast<uint4 *>(tile));
        __syncthreads();
        load_values<PAGED>(p, kh, start + 32, end, reinterpret_cast<uint4 *>(tile), bad);
        __syncthreads();
        values(out, probabilities[2], probabilities[3], reinterpret_cast<uint4 *>(tile));
        __syncthreads();
    }
#pragma unroll
    for (int n = 0; n < D / 8; ++n) {
        const int d = 8 * n + 2 * qd;
        if (wrow + g < p.rows)
            *reinterpret_cast<uint32_t *>(p.out + (static_cast<int64_t>(wrow + g) * p.heads + head) * D + d) =
                pair(divide(out[n][0], denominator[0]), divide(out[n][1], denominator[0]));
        if (wrow + g + 8 < p.rows)
            *reinterpret_cast<uint32_t *>(p.out + (static_cast<int64_t>(wrow + g + 8) * p.heads + head) * D + d) =
                pair(divide(out[n][2], denominator[1]), divide(out[n][3], denominator[1]));
    }
    __syncthreads();
    for (int r = 0; r < 16; ++r) {
        const int row = wrow + r;
        if (row < p.rows && (qbad[warp * 16 + r] || p.prefix + row >= *bad)) exceptional_row<PAGED>(p, row, head, kh);
    }
}
} // namespace tf_fp8_attention
