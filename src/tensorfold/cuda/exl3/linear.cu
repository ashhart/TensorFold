// EXL3 linear, any codebook and width, 1-128 rows: a row's bits depend only on it (mma keeps rows apart, K ranges fixed by (K, N), fixed-order sums).

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <type_traits>

#include "decode.cuh"

using namespace tf_exl3;

namespace {

enum DType : int { F16 = 0, BF16 = 1, F32 = 2 };

__device__ __forceinline__ void load4(const void* p, int dtype, size_t i, float (&v)[4]) {
    if (dtype == F32) {
        const float4 u = *reinterpret_cast<const float4*>(static_cast<const float*>(p) + i);
        v[0] = u.x; v[1] = u.y; v[2] = u.z; v[3] = u.w;
    } else if (dtype == BF16) {
        const uint2 u = *reinterpret_cast<const uint2*>(static_cast<const __nv_bfloat16*>(p) + i);
        const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
        const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
        v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
    } else {
        const uint2 u = *reinterpret_cast<const uint2*>(static_cast<const half*>(p) + i);
        const float2 a = __half22float2(*reinterpret_cast<const half2*>(&u.x));
        const float2 b = __half22float2(*reinterpret_cast<const half2*>(&u.y));
        v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
    }
}

__device__ __forceinline__ void store4(void* p, int dtype, size_t i, const float (&v)[4]) {
    if (dtype == F32) {
        *reinterpret_cast<float4*>(static_cast<float*>(p) + i) = make_float4(v[0], v[1], v[2], v[3]);
    } else if (dtype == BF16) {
        __nv_bfloat162 a = __floats2bfloat162_rn(v[0], v[1]), b = __floats2bfloat162_rn(v[2], v[3]);
        uint2 u;
        u.x = *reinterpret_cast<uint32_t*>(&a);
        u.y = *reinterpret_cast<uint32_t*>(&b);
        *reinterpret_cast<uint2*>(static_cast<__nv_bfloat16*>(p) + i) = u;
    } else {
        half2 a = __floats2half2_rn(v[0], v[1]), b = __floats2half2_rn(v[2], v[3]);
        uint2 u;
        u.x = *reinterpret_cast<uint32_t*>(&a);
        u.y = *reinterpret_cast<uint32_t*>(&b);
        *reinterpret_cast<uint2*>(static_cast<half*>(p) + i) = u;
    }
}

// The finished outputs of one row's 128 columns from their fp32 sums (4 a lane): H / sqrt(128), * svh, + bias.
__device__ __forceinline__ void finish(float (&v)[4], int lane, const half* svh, const half* bias, int col) {
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j] = v[j] * HAD_SCALE * __half2float(__ldg(svh + col + j));
        if (bias) v[j] += __half2float(__ldg(bias + col + j));
    }
}

// A lane's words of one k step: at 1 and 2 bits the warp loads the step together and each lane takes its words by shuffle.
template <int K2>
__host__ __device__ constexpr bool step_shuffled() {
    return K2 == 2 || K2 == 4;
}

template <int K2>
__host__ __device__ constexpr int step_regs() {
    return step_shuffled<K2>() ? tile_words<K2>() / 4 : 8 * lane_words<K2>();
}

template <int K2>
__device__ __forceinline__ void load_step(const uint32_t* step, int lane, uint32_t (&raw)[step_regs<K2>()]) {
    constexpr int TW = tile_words<K2>(), LW = lane_words<K2>();
    if constexpr (step_shuffled<K2>()) {
#pragma unroll
        for (int c = 0; c < step_regs<K2>(); ++c) raw[c] = __ldg(step + c * 32 + lane);
    } else {
        int word, offset;
        lane_start<K2>(lane, word, offset);
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int q = 0; q < LW; ++q) raw[j * LW + q] = __ldg(step + j * TW + (word + q) % TW);
    }
}

// Tile j's lane words from a loaded step; prev: the lane's first window starts in the word before its own.
template <int K2>
__device__ __forceinline__ void step_lane_words(const uint32_t (&raw)[step_regs<K2>()], int j, int lane, bool prev,
                                                uint32_t (&w)[lane_words<K2>()]) {
    constexpr int TW = tile_words<K2>(), LW = lane_words<K2>();
    if constexpr (step_shuffled<K2>()) {
        static_assert(LW == 2, "a lane's windows span two words at 1 and 2 bits");
        constexpr int LPW = 8 / K2;                  // lanes whose windows end in the same word
        const uint32_t r = raw[j * TW / 32];
        const int base = (j * TW) % 32;
        const uint32_t own = __shfl_sync(0xffffffffu, r, base + lane / LPW);
        const uint32_t before = __shfl_sync(0xffffffffu, r, base + (lane / LPW + TW - 1) % TW);
        w[0] = prev ? before : own;
        w[1] = prev ? own : 0u;                      // a window inside one word never reads w[1]
    } else {
#pragma unroll
        for (int q = 0; q < LW; ++q) w[q] = raw[j * LW + q];
    }
}

__global__ void __launch_bounds__(128) rot_in_kernel(const void* __restrict__ x, int x_dtype,
                                                     const half* __restrict__ suh, half* __restrict__ xh, int K) {
    const int blk = blockIdx.x * 4 + (threadIdx.x >> 5), row = blockIdx.y, lane = threadIdx.x & 31;
    if (blk * 128 >= K) return;
    const int k = blk * 128 + 4 * lane;
    float v[4], s[4];
    load4(x, x_dtype, (size_t)row * K + k, v);
    load4(suh, F16, k, s);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= s[j];
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= HAD_SCALE;
    store4(xh, F16, (size_t)row * K + k, v);
}

template <int K2, int CB, int WK>
__global__ void __launch_bounds__(WK * 32) linear_kernel(
    const half* __restrict__ xh, const uint32_t* __restrict__ T, long long stride_k, long long stride_nb,
    const half* __restrict__ svh, const half* __restrict__ bias, void* __restrict__ y, int y_dtype,
    float* __restrict__ Z, int* __restrict__ counters, int M, int K, int N, int SK) {
    constexpr int TW = tile_words<K2>();
    constexpr int LW = lane_words<K2>();
    extern __shared__ __align__(16) float red[];              // WK * RH * 128 floats
    __shared__ int last;
    const int RH = min(M, 8);                                 // rows of red a warp

    const int nb = blockIdx.x, split = blockIdx.y, NB = gridDim.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int per_warp = (K >> 4) / SK / WK;
    const int kt0 = split * (per_warp * WK) + warp * per_warp;
    const uint32_t* tiles = T + nb * stride_nb;
    const int col0 = nb * 128;
    bool prev = false;
    if constexpr (step_shuffled<K2>()) {
        int word, offset;
        lane_start<K2>(lane, word, offset);
        prev = word != lane / (8 / K2);
    }

    for (int m0 = 0, pass = 0; m0 < M; m0 += 16, ++pass) {
        const int R = min(16, M - m0);

        float acc[8][2][4];
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

        // the walk's rows for the mma, clamped so rows past the pass read inside the buffer (their outputs are dropped)
        const int r0 = m0 + (g < R ? g : R - 1), r1 = m0 + (g + 8 < R ? g + 8 : R - 1);
        const half* x0 = xh + (size_t)r0 * K;
        const half* x1 = xh + (size_t)r1 * K;
        const uint32_t* tile = tiles + (size_t)kt0 * stride_k;
        // up to 6 bits the next k step's words are loaded while this one is decoded; 7 and 8 bits load per tile
        constexpr bool PF = K2 <= 12;
        constexpr int SR = step_regs<K2>();
        uint32_t cur[PF ? SR : 1], nxt[PF ? SR : 1];
        if constexpr (PF) load_step<K2>(tile, lane, cur);
#pragma unroll 1
        for (int i = 0; i < per_warp; ++i) {
            const int kt = kt0 + i;
            uint32_t a[4];
            a[0] = __ldg(reinterpret_cast<const uint32_t*>(x0 + kt * 16 + 2 * t));
            a[1] = __ldg(reinterpret_cast<const uint32_t*>(x1 + kt * 16 + 2 * t));
            a[2] = __ldg(reinterpret_cast<const uint32_t*>(x0 + kt * 16 + 2 * t + 8));
            a[3] = __ldg(reinterpret_cast<const uint32_t*>(x1 + kt * 16 + 2 * t + 8));
            if constexpr (PF) {
                if (i + 1 < per_warp) load_step<K2>(tile + (size_t)(i + 1) * stride_k, lane, nxt);
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                uint32_t w[LW];
                if constexpr (PF) step_lane_words<K2>(cur, j, lane, prev, w);
                else ldg_lane_words<K2>(tile + (size_t)i * stride_k + j * TW, lane, w);
                uint32_t b0[2], b1[2];
                decode_lane<K2, CB>(w, lane, b0, b1);
                mma16816(acc[j][0], a, b0);
                mma16816(acc[j][1], a, b1);
            }
            if constexpr (PF) {
#pragma unroll
                for (int q = 0; q < SR; ++q) cur[q] = nxt[q];
            }
        }

        // the warps' sums, added in warp order, rows 0-7 of the pass and then rows 8-15
        for (int rlo = 0; rlo < R; rlo += 8) {
            const int rn = min(R - rlo, 8);
            __syncthreads();                         // red is reused by every half and pass
            if (g < RH) {
#pragma unroll
                for (int i = 0; i < 8; ++i)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int col = i * 16 + h * 8 + 2 * t;
                        *reinterpret_cast<float2*>(red + (warp * RH + g) * 128 + col) =
                            rlo ? make_float2(acc[i][h][2], acc[i][h][3]) : make_float2(acc[i][h][0], acc[i][h][1]);
                    }
            }
            __syncthreads();

            if (SK == 1) {
                for (int r = warp; r < rn; r += WK) {
                    float v[4];
                    const float4 u = *reinterpret_cast<const float4*>(red + r * 128 + 4 * lane);
                    v[0] = u.x; v[1] = u.y; v[2] = u.z; v[3] = u.w;
#pragma unroll
                    for (int w = 1; w < WK; ++w) {
                        const float4 q = *reinterpret_cast<const float4*>(red + (w * RH + r) * 128 + 4 * lane);
                        v[0] += q.x; v[1] += q.y; v[2] += q.z; v[3] += q.w;
                    }
                    finish(v, lane, svh, bias, col0 + 4 * lane);
                    store4(y, y_dtype, (size_t)(m0 + rlo + r) * N + col0 + 4 * lane, v);
                }
            } else {
                for (int idx = threadIdx.x; idx < rn * 32; idx += WK * 32) {
                    const int r = idx >> 5, c = 4 * (idx & 31);
                    float4 s = *reinterpret_cast<const float4*>(red + r * 128 + c);
#pragma unroll
                    for (int w = 1; w < WK; ++w) {
                        const float4 q = *reinterpret_cast<const float4*>(red + (w * RH + r) * 128 + c);
                        s.x += q.x; s.y += q.y; s.z += q.z; s.w += q.w;
                    }
                    *reinterpret_cast<float4*>(Z + ((size_t)split * M + m0 + rlo + r) * N + col0 + c) = s;
                }
            }
        }
        if (SK > 1) {
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0) last = atomicAdd(counters + pass * NB + nb, 1) == SK - 1;
            __syncthreads();
            if (last) {
                __threadfence();
                for (int r = warp; r < R; r += WK) {
                    const size_t at = ((size_t)m0 + r) * N + col0 + 4 * lane;
                    float4 s = __ldcg(reinterpret_cast<const float4*>(Z + at));
                    for (int q = 1; q < SK; ++q) {
                        const float4 u = __ldcg(reinterpret_cast<const float4*>(Z + (size_t)q * M * N + at));
                        s.x += u.x; s.y += u.y; s.z += u.z; s.w += u.w;
                    }
                    float v[4] = {s.x, s.y, s.z, s.w};
                    finish(v, lane, svh, bias, col0 + 4 * lane);
                    store4(y, y_dtype, (size_t)(m0 + r) * N + col0 + 4 * lane, v);
                }
                if (threadIdx.x == 0) counters[pass * NB + nb] = 0;   // every program of the block has arrived
            }
        }
        __syncthreads();                             // red is reused in the next pass
    }
}

// 17-128 rows (the mid-M kernels): each CTA takes one 16 P-row group of one column block (G = ceil(M / 16 P) CTAs a
// block, side by side in the grid, so a block's CTAs read its words from L2 together) and decodes every k step's tiles
// once for its P 16-row passes. The warps keep linear_kernel's K ranges, mma chain and warp-order sums, and the split
// fold is the same: every output bit equals linear_kernel's at any row count.
template <int K2, int CB, int WK, int P>
__global__ void __launch_bounds__(WK * 32) linear_mpg_kernel(
    const half* __restrict__ xh, const uint32_t* __restrict__ T, long long stride_k, long long stride_nb,
    const half* __restrict__ svh, const half* __restrict__ bias, void* __restrict__ y, int y_dtype,
    float* __restrict__ Z, int* __restrict__ counters, int M, int K, int N, int SK) {
    constexpr int TW = tile_words<K2>();
    constexpr int LW = lane_words<K2>();
    extern __shared__ __align__(16) float red[];              // WK * RH * 128 floats
    __shared__ int last;
    const int RH = min(M, 8);                                 // rows of red a warp

    const int G = (M + 16 * P - 1) / (16 * P), grp = blockIdx.x % G, nb = blockIdx.x / G, split = blockIdx.y, NB = gridDim.x / G;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int per_warp = (K >> 4) / SK / WK;
    const int kt0 = split * (per_warp * WK) + warp * per_warp;
    const uint32_t* tiles = T + nb * stride_nb;
    const int col0 = nb * 128;
    bool prev = false;
    if constexpr (step_shuffled<K2>()) {
        int word, offset;
        lane_start<K2>(lane, word, offset);
        prev = word != lane / (8 / K2);
    }

    for (int mg = grp * 16 * P, pg = grp * P; mg < M && mg < (grp + 1) * 16 * P; mg += 16 * P, pg += P) {
        float acc[P][8][2][4];
#pragma unroll
        for (int q = 0; q < P; ++q)
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int c = 0; c < 4; ++c) acc[q][i][h][c] = 0.f;

        // each pass's rows for the mma, clamped inside the buffer (rows past M are computed and dropped)
        const half* x0[P];
        const half* x1[P];
#pragma unroll
        for (int q = 0; q < P; ++q) {
            const int mq = min(mg + 16 * q, M - 1), Rq = max(1, min(16, M - mq));
            const int r0 = mq + (g < Rq ? g : Rq - 1), r1 = mq + (g + 8 < Rq ? g + 8 : Rq - 1);
            x0[q] = xh + (size_t)r0 * K;
            x1[q] = xh + (size_t)r1 * K;
        }
        const uint32_t* tile = tiles + (size_t)kt0 * stride_k;
        // up to 6 bits the next k step's words are loaded while this one is decoded; 7 and 8 bits load per tile
        constexpr bool PF = K2 <= 12;
        constexpr int SR = step_regs<K2>();
        uint32_t cur[PF ? SR : 1], nxt[PF ? SR : 1];
        if constexpr (PF) load_step<K2>(tile, lane, cur);
#pragma unroll 1
        for (int i = 0; i < per_warp; ++i) {
            const int kt = kt0 + i;
            uint32_t a[P][4];
#pragma unroll
            for (int q = 0; q < P; ++q) {
                a[q][0] = __ldg(reinterpret_cast<const uint32_t*>(x0[q] + kt * 16 + 2 * t));
                a[q][1] = __ldg(reinterpret_cast<const uint32_t*>(x1[q] + kt * 16 + 2 * t));
                a[q][2] = __ldg(reinterpret_cast<const uint32_t*>(x0[q] + kt * 16 + 2 * t + 8));
                a[q][3] = __ldg(reinterpret_cast<const uint32_t*>(x1[q] + kt * 16 + 2 * t + 8));
            }
            if constexpr (PF) {
                if (i + 1 < per_warp) load_step<K2>(tile + (size_t)(i + 1) * stride_k, lane, nxt);
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                uint32_t w[LW];
                if constexpr (PF) step_lane_words<K2>(cur, j, lane, prev, w);
                else ldg_lane_words<K2>(tile + (size_t)i * stride_k + j * TW, lane, w);
                uint32_t b0[2], b1[2];
                decode_lane<K2, CB>(w, lane, b0, b1);
#pragma unroll
                for (int q = 0; q < P; ++q) {
                    mma16816(acc[q][j][0], a[q], b0);
                    mma16816(acc[q][j][1], a[q], b1);
                }
            }
            if constexpr (PF) {
#pragma unroll
                for (int q = 0; q < SR; ++q) cur[q] = nxt[q];
            }
        }

#pragma unroll
        for (int q = 0; q < P; ++q) {
        const int m0 = mg + 16 * q, pass = pg + q;
        if (m0 >= M) break;
        const int R = min(16, M - m0);
        // the warps' sums, added in warp order, rows 0-7 of the pass and then rows 8-15
        for (int rlo = 0; rlo < R; rlo += 8) {
            const int rn = min(R - rlo, 8);
            __syncthreads();                         // red is reused by every half and pass
            if (g < RH) {
#pragma unroll
                for (int i = 0; i < 8; ++i)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int col = i * 16 + h * 8 + 2 * t;
                        *reinterpret_cast<float2*>(red + (warp * RH + g) * 128 + col) =
                            rlo ? make_float2(acc[q][i][h][2], acc[q][i][h][3]) : make_float2(acc[q][i][h][0], acc[q][i][h][1]);
                    }
            }
            __syncthreads();

            if (SK == 1) {
                for (int r = warp; r < rn; r += WK) {
                    float v[4];
                    const float4 u = *reinterpret_cast<const float4*>(red + r * 128 + 4 * lane);
                    v[0] = u.x; v[1] = u.y; v[2] = u.z; v[3] = u.w;
#pragma unroll
                    for (int w = 1; w < WK; ++w) {
                        const float4 q = *reinterpret_cast<const float4*>(red + (w * RH + r) * 128 + 4 * lane);
                        v[0] += q.x; v[1] += q.y; v[2] += q.z; v[3] += q.w;
                    }
                    finish(v, lane, svh, bias, col0 + 4 * lane);
                    store4(y, y_dtype, (size_t)(m0 + rlo + r) * N + col0 + 4 * lane, v);
                }
            } else {
                for (int idx = threadIdx.x; idx < rn * 32; idx += WK * 32) {
                    const int r = idx >> 5, c = 4 * (idx & 31);
                    float4 s = *reinterpret_cast<const float4*>(red + r * 128 + c);
#pragma unroll
                    for (int w = 1; w < WK; ++w) {
                        const float4 q = *reinterpret_cast<const float4*>(red + (w * RH + r) * 128 + c);
                        s.x += q.x; s.y += q.y; s.z += q.z; s.w += q.w;
                    }
                    *reinterpret_cast<float4*>(Z + ((size_t)split * M + m0 + rlo + r) * N + col0 + c) = s;
                }
            }
        }
        if (SK > 1) {
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0) last = atomicAdd(counters + pass * NB + nb, 1) == SK - 1;
            __syncthreads();
            if (last) {
                __threadfence();
                for (int r = warp; r < R; r += WK) {
                    const size_t at = ((size_t)m0 + r) * N + col0 + 4 * lane;
                    float4 s = __ldcg(reinterpret_cast<const float4*>(Z + at));
                    for (int q = 1; q < SK; ++q) {
                        const float4 u = __ldcg(reinterpret_cast<const float4*>(Z + (size_t)q * M * N + at));
                        s.x += u.x; s.y += u.y; s.z += u.z; s.w += u.w;
                    }
                    float v[4] = {s.x, s.y, s.z, s.w};
                    finish(v, lane, svh, bias, col0 + 4 * lane);
                    store4(y, y_dtype, (size_t)(m0 + r) * N + col0 + 4 * lane, v);
                }
                if (threadIdx.x == 0) counters[pass * NB + nb] = 0;   // every program of the block has arrived
            }
        }
        }
        __syncthreads();                             // red is reused in the next pass
    }
}

// W_q [K, N] fp16 from the trellis words (tile (kt, nt) at kt * stride_k + (nt / 8) * stride_nb): one warp per tile.
template <int K2, int CB>
__global__ void __launch_bounds__(32) unpack_kernel(const uint32_t* __restrict__ T, half* __restrict__ W, int N,
                                                    int64_t stride_k, int64_t stride_nb) {
    const int kt = blockIdx.y, nt = blockIdx.x, lane = threadIdx.x;
    uint32_t w[lane_words<K2>()];
    ldg_lane_words<K2>(T + kt * stride_k + (nt >> 3) * stride_nb + (nt & 7) * tile_words<K2>(), lane, w);
    uint32_t b[2][2];
    decode_lane<K2, CB>(w, lane, b[0], b[1]);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint32_t v = b[j >> 2][(j >> 1) & 1];
        const unsigned short h = (j & 1) ? (unsigned short)(v >> 16) : (unsigned short)(v & 0xffffu);
        W[(size_t)(kt * 16 + value_row(lane, j)) * N + nt * 16 + value_col(lane, j)] = __ushort_as_half(h);
    }
}

// W'' [K, N] fp16 or bf16 = diag(suh) Hk W_q Hn / 128 for one 128 x 128 block (K block y, N block x): 8 warps decode the
// block's 64 trellis tiles into fp32 shared memory, then fwht128 runs along every row (N side) and every column (K side,
// times suh), and the block is stored with one rounding. A lane holds indices lane + 32 q (q = 0..3), a bit
// permutation of fwht128's 4 lane + q: H is invariant under it, so loads and stores use the same map (conflict-free).
template <int K2, int CB, typename OutT>
__global__ void __launch_bounds__(256) unpack_fold_kernel(const uint32_t* __restrict__ T, const half* __restrict__ suh,
                                                          OutT* __restrict__ W, int N, int64_t stride_k,
                                                          int64_t stride_nb) {
    extern __shared__ float fold_tile[];
    constexpr int LD = 129;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int bx = blockIdx.x, by = blockIdx.y, kt = by * 8 + warp;
#pragma unroll 1
    for (int j = 0; j < 8; ++j) {
        uint32_t w[lane_words<K2>()];
        ldg_lane_words<K2>(T + (int64_t)kt * stride_k + (int64_t)bx * stride_nb + j * tile_words<K2>(), lane, w);
        uint32_t b[2][2];
        decode_lane<K2, CB>(w, lane, b[0], b[1]);
#pragma unroll
        for (int jj = 0; jj < 8; ++jj) {
            const uint32_t v = b[jj >> 2][(jj >> 1) & 1];
            const unsigned short h = (jj & 1) ? (unsigned short)(v >> 16) : (unsigned short)(v & 0xffffu);
            fold_tile[(warp * 16 + value_row(lane, jj)) * LD + j * 16 + value_col(lane, jj)] =
                __half2float(__ushort_as_half(h));
        }
    }
    __syncthreads();
#pragma unroll 1
    for (int r = warp * 16; r < warp * 16 + 16; ++r) {           // N side: W_q Hn
        float v[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) v[q] = fold_tile[r * LD + lane + 32 * q];
        fwht128(v, lane);
#pragma unroll
        for (int q = 0; q < 4; ++q) fold_tile[r * LD + lane + 32 * q] = v[q];
    }
    __syncthreads();
#pragma unroll 1
    for (int c = warp * 16; c < warp * 16 + 16; ++c) {           // K side: diag(suh) Hk, and both 1 / sqrt(128)
        float v[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) v[q] = fold_tile[(lane + 32 * q) * LD + c];
        fwht128(v, lane);
#pragma unroll
        for (int q = 0; q < 4; ++q)
            fold_tile[(lane + 32 * q) * LD + c] =
                v[q] * (HAD_SCALE * HAD_SCALE) * __half2float(suh[by * 128 + lane + 32 * q]);
    }
    __syncthreads();
#pragma unroll 1
    for (int r = warp * 16; r < warp * 16 + 16; ++r) {
        OutT* dst = W + (int64_t)(by * 128 + r) * N + bx * 128;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const float f = fold_tile[r * LD + lane + 32 * q];
            if constexpr (sizeof(OutT) == 2 && std::is_same<OutT, half>::value) dst[lane + 32 * q] = __float2half_rn(f);
            else dst[lane + 32 * q] = __float2bfloat16_rn(f);
        }
    }
}

// unpack_fold2_kernel: the same W'' = diag(suh) Hk W_q Hn / 128 as unpack_fold_kernel, bit for bit, faster. The N-side
// fwht runs in registers on the decoded fragments (each warp holds 16 whole rows), only the K side goes through shared
// memory, in two 64-column halves (32 KB fp32 + 16 KB bf16 staging, XOR-swizzled, conflict-free), so 2 blocks fit an SM
// and the 8 tiles' words of a warp are all in flight at once; the output leaves in 16-byte row stores.
// Same values in the same butterfly order as fwht128 on its lane + 32 q map (N side bits c5 c6 c0..c4, K side r5 r6
// r0..r4; x0 + x1 / x0 - x1 at every stage), same scale and suh products, one rounding: identical bits.
template <int K2, int CB, typename OutT>
__global__ void __launch_bounds__(256, 2) unpack_fold2_kernel(const uint32_t* __restrict__ T, const half* __restrict__ suh,
                                                              OutT* __restrict__ W, int N, int64_t stride_k,
                                                              int64_t stride_nb) {
    __shared__ float s32[128 * 64];
    __shared__ uint32_t s16[128 * 32];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int bx = blockIdx.x, by = blockIdx.y, kt = by * 8 + warp;
    float v[8][8];                       // [tile j][value jj]: row 16 warp + value_row(lane, jj), column 16 j + value_col
    {
        uint32_t w[8][lane_words<K2>()];
#pragma unroll
        for (int j = 0; j < 8; ++j)
            ldg_lane_words<K2>(T + (int64_t)kt * stride_k + (int64_t)bx * stride_nb + j * tile_words<K2>(), lane, w[j]);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            uint32_t b[2][2];
            decode_lane<K2, CB>(w[j], lane, b[0], b[1]);
#pragma unroll
            for (int jj = 0; jj < 8; ++jj) {
                const uint32_t u = b[jj >> 2][(jj >> 1) & 1];
                const unsigned short h = (jj & 1) ? (unsigned short)(u >> 16) : (unsigned short)(u & 0xffffu);
                v[j][jj] = __half2float(__ushort_as_half(h));
            }
        }
    }
    // N side. Column bits: c0..c2 = lane bits 2..4, c3 = jj bit 2, c4..c6 = j bits 0..2.
#define BF(a, b) { const float x0 = (a), x1 = (b); (a) = x0 + x1; (b) = x0 - x1; }
#pragma unroll
    for (int jj = 0; jj < 8; ++jj) {
#pragma unroll
        for (int j = 0; j < 8; ++j) if (!(j & 2)) BF(v[j][jj], v[j | 2][jj]);        // c5
#pragma unroll
        for (int j = 0; j < 4; ++j) BF(v[j][jj], v[j | 4][jj]);                      // c6
#pragma unroll
        for (int m = 4; m <= 16; m <<= 1)                                            // c0, c1, c2
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float o = __shfl_xor_sync(0xffffffffu, v[j][jj], m);
                v[j][jj] = (lane & m) ? o - v[j][jj] : v[j][jj] + o;
            }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) BF(v[j][jj], v[j][jj | 4]);                   // c3
#pragma unroll
    for (int j = 0; j < 8; j += 2)
#pragma unroll
        for (int jj = 0; jj < 8; ++jj) BF(v[j][jj], v[j | 1][jj]);                   // c4
    // K side per 64-column half (c6 = half). s32 [row][col ^ 8 g(row)], g = (r1 ^ r3) + 2 (r2 ^ r4).
    const int c = lane & 7, r3 = (lane >> 3) & 1, r4 = lane >> 4;
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
#pragma unroll
        for (int t = 0; t < 4; ++t)
#pragma unroll
            for (int jj = 0; jj < 8; ++jj) {
                const int row = warp * 16 + value_row(lane, jj), col = 16 * t + value_col(lane, jj);
                const int g = (((row >> 1) ^ (row >> 3)) & 1) | ((((row >> 2) ^ (row >> 4)) & 1) << 1);
                s32[row * 64 + (col ^ (8 * g))] = v[4 * hf + t][jj];
            }
        __syncthreads();
        float u[32];                     // i bits: r5 r6 r0 r1 r2; column 8 warp + c, row bits r3 r4 from the lane
        const int col = warp * 8 + c;
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            const int row = ((i >> 2) & 7) + 8 * r3 + 16 * r4 + 32 * (i & 1) + 64 * ((i >> 1) & 1);
            const int g = (((row >> 1) ^ (row >> 3)) & 1) | ((((row >> 2) ^ (row >> 4)) & 1) << 1);
            u[i] = s32[row * 64 + (col ^ (8 * g))];
        }
#pragma unroll
        for (int bit = 1; bit < 32; bit <<= 1)                                       // r5, r6, r0, r1, r2
#pragma unroll
            for (int i = 0; i < 32; ++i) if (!(i & bit)) BF(u[i], u[i | bit]);
#pragma unroll
        for (int m = 8; m <= 16; m <<= 1)                                            // r3, r4
#pragma unroll
            for (int i = 0; i < 32; ++i) {
                const float o = __shfl_xor_sync(0xffffffffu, u[i], m);
                u[i] = (lane & m) ? o - u[i] : u[i] + o;
            }
        unsigned short* s16h = reinterpret_cast<unsigned short*>(s16);
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            const int row = ((i >> 2) & 7) + 8 * r3 + 16 * r4 + 32 * (i & 1) + 64 * ((i >> 1) & 1);
            const float f = u[i] * (HAD_SCALE * HAD_SCALE) * __half2float(suh[by * 128 + row]);
            unsigned short h;
            if constexpr (std::is_same<OutT, half>::value) h = __half_as_ushort(__float2half_rn(f));
            else h = __bfloat16_as_ushort(__float2bfloat16_rn(f));
            const int word = (col >> 1) ^ (4 * (r3 + 2 * r4));                       // g'(row) = r3 + 2 r4
            s16h[row * 64 + 2 * word + (col & 1)] = h;
        }
        __syncthreads();
#pragma unroll
        for (int it = 0; it < 4; ++it) {
            const int id = it * 256 + threadIdx.x, row = id >> 3, vec = id & 7;
            const uint4 q = *reinterpret_cast<const uint4*>(s16 + row * 32 + 4 * (vec ^ ((row >> 3) & 3)));
            *reinterpret_cast<uint4*>(W + (int64_t)(by * 128 + row) * N + bx * 128 + 64 * hf + 8 * vec) = q;
        }
    }
#undef BF
}

// fdirect_kernel: out [M, N] bf16 = (x [M, K] bf16 @ W'') * svh + bias for M <= 16 MT rows, W'' never written: each
// program owns one 128-column block and walks K in 128-row blocks in order, rebuilding each W'' block exactly as
// unpack_fold2_kernel does (same decode, same butterflies, same rounding, the same 64-column halves in s16), and
// feeds it straight into one mma.m16n8k16 bf16 chain over K in k16 order from zero, the chain tl.dot runs in
// _gemm_fold (a plain chain reproduced its bits on every 27B shape at 1-128 rows), so the output equals unpack_fold2 +
// _gemm_fold bit for bit.
template <int K2, int CB, int MT>
__global__ void __launch_bounds__(256, 1) fdirect_kernel(const __nv_bfloat16* __restrict__ x, int ldx,
                                                         const uint32_t* __restrict__ T, const half* __restrict__ suh,
                                                         const half* __restrict__ svh, const half* __restrict__ bias,
                                                         __nv_bfloat16* __restrict__ out, int ldo, int M, int K, int N,
                                                         int64_t stride_k, int64_t stride_nb) {
    __shared__ float s32[128 * 64];
    __shared__ uint32_t s16[128 * 32];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int bx = blockIdx.x;
    const int g8 = lane >> 2, t4 = lane & 3;
    const int c = lane & 7, r3 = (lane >> 3) & 1, r4 = lane >> 4;
    const unsigned short* s16h = reinterpret_cast<const unsigned short*>(s16);
    float acc[2][MT][4];
#pragma unroll
    for (int hf = 0; hf < 2; ++hf)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[hf][mt][i] = 0.f;
#define BF(a, b) { const float x0 = (a), x1 = (b); (a) = x0 + x1; (b) = x0 - x1; }
    uint32_t w[8][lane_words<K2>()];             // this warp's words of the next block, in flight during this one
#pragma unroll
    for (int j = 0; j < 8; ++j)
        ldg_lane_words<K2>(T + (int64_t)warp * stride_k + (int64_t)bx * stride_nb + j * tile_words<K2>(), lane, w[j]);
#pragma unroll 1
    for (int by = 0; by < K / 128; ++by) {
        float v[8][8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            uint32_t b[2][2];
            decode_lane<K2, CB>(w[j], lane, b[0], b[1]);
#pragma unroll
            for (int jj = 0; jj < 8; ++jj) {
                const uint32_t u = b[jj >> 2][(jj >> 1) & 1];
                const unsigned short h = (jj & 1) ? (unsigned short)(u >> 16) : (unsigned short)(u & 0xffffu);
                v[j][jj] = __half2float(__ushort_as_half(h));
            }
        }
        if (by + 1 < K / 128) {
            const int kt = (by + 1) * 8 + warp;
#pragma unroll
            for (int j = 0; j < 8; ++j)
                ldg_lane_words<K2>(T + (int64_t)kt * stride_k + (int64_t)bx * stride_nb + j * tile_words<K2>(), lane,
                                   w[j]);
        }
#pragma unroll
        for (int jj = 0; jj < 8; ++jj) {
#pragma unroll
            for (int j = 0; j < 8; ++j) if (!(j & 2)) BF(v[j][jj], v[j | 2][jj]);
#pragma unroll
            for (int j = 0; j < 4; ++j) BF(v[j][jj], v[j | 4][jj]);
#pragma unroll
            for (int m = 4; m <= 16; m <<= 1)
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float o = __shfl_xor_sync(0xffffffffu, v[j][jj], m);
                    v[j][jj] = (lane & m) ? o - v[j][jj] : v[j][jj] + o;
                }
        }
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) BF(v[j][jj], v[j][jj | 4]);
#pragma unroll
        for (int j = 0; j < 8; j += 2)
#pragma unroll
            for (int jj = 0; jj < 8; ++jj) BF(v[j][jj], v[j | 1][jj]);
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
#pragma unroll
            for (int t = 0; t < 4; ++t)
#pragma unroll
                for (int jj = 0; jj < 8; ++jj) {
                    const int row = warp * 16 + value_row(lane, jj), col = 16 * t + value_col(lane, jj);
                    const int g = (((row >> 1) ^ (row >> 3)) & 1) | ((((row >> 2) ^ (row >> 4)) & 1) << 1);
                    s32[row * 64 + (col ^ (8 * g))] = v[4 * hf + t][jj];
                }
            __syncthreads();
            {
                float u[32];
                const int col = warp * 8 + c;
#pragma unroll
                for (int i = 0; i < 32; ++i) {
                    const int row = ((i >> 2) & 7) + 8 * r3 + 16 * r4 + 32 * (i & 1) + 64 * ((i >> 1) & 1);
                    const int g = (((row >> 1) ^ (row >> 3)) & 1) | ((((row >> 2) ^ (row >> 4)) & 1) << 1);
                    u[i] = s32[row * 64 + (col ^ (8 * g))];
                }
#pragma unroll
                for (int bit = 1; bit < 32; bit <<= 1)
#pragma unroll
                    for (int i = 0; i < 32; ++i) if (!(i & bit)) BF(u[i], u[i | bit]);
#pragma unroll
                for (int m = 8; m <= 16; m <<= 1)
#pragma unroll
                    for (int i = 0; i < 32; ++i) {
                        const float o = __shfl_xor_sync(0xffffffffu, u[i], m);
                        u[i] = (lane & m) ? o - u[i] : u[i] + o;
                    }
                unsigned short* w16 = reinterpret_cast<unsigned short*>(s16);
#pragma unroll
                for (int i = 0; i < 32; ++i) {
                    const int row = ((i >> 2) & 7) + 8 * r3 + 16 * r4 + 32 * (i & 1) + 64 * ((i >> 1) & 1);
                    const float f = u[i] * (HAD_SCALE * HAD_SCALE) * __half2float(suh[by * 128 + row]);
                    const int word = (col >> 1) ^ (4 * (r3 + 2 * r4));
                    w16[row * 64 + 2 * word + (col & 1)] = __bfloat16_as_ushort(__float2bfloat16_rn(f));
                }
            }
            __syncthreads();
            // this half's W'' block [128 k][64 n] is in s16: the warp's n8 tile (columns 8 warp .. 8 warp + 7)
            const int ncol = 8 * warp + g8;
#pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
                const int k0 = 16 * kk + 2 * t4;
                auto wv = [&](int row) -> uint32_t {
                    const int word = (ncol >> 1) ^ (4 * ((row >> 3) & 3));
                    return s16h[row * 64 + 2 * word + (ncol & 1)];
                };
                uint32_t b[2] = {wv(k0) | (wv(k0 + 1) << 16), wv(k0 + 8) | (wv(k0 + 9) << 16)};
                const int kg = by * 128 + k0;
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    const int ra = mt * 16 + g8, rb = ra + 8;
                    uint32_t a[4];
                    a[0] = ra < M ? __ldg(reinterpret_cast<const uint32_t*>(x + (int64_t)ra * ldx + kg)) : 0u;
                    a[1] = rb < M ? __ldg(reinterpret_cast<const uint32_t*>(x + (int64_t)rb * ldx + kg)) : 0u;
                    a[2] = ra < M ? __ldg(reinterpret_cast<const uint32_t*>(x + (int64_t)ra * ldx + kg + 8)) : 0u;
                    a[3] = rb < M ? __ldg(reinterpret_cast<const uint32_t*>(x + (int64_t)rb * ldx + kg + 8)) : 0u;
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
                                 "{%8,%9}, {%0,%1,%2,%3};\n"
                                 : "+f"(acc[hf][mt][0]), "+f"(acc[hf][mt][1]), "+f"(acc[hf][mt][2]), "+f"(acc[hf][mt][3])
                                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
                }
            }
        }
    }
#undef BF
#pragma unroll
    for (int hf = 0; hf < 2; ++hf)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int r = mt * 16 + g8 + 8 * (i >> 1), col = bx * 128 + 64 * hf + 8 * warp + 2 * t4 + (i & 1);
                if (r >= M) continue;
                float y = __fmul_rn(acc[hf][mt][i], __half2float(svh[col]));
                if (bias) y = __fadd_rn(y, __half2float(bias[col]));
                out[(int64_t)r * ldo + col] = __float2bfloat16_rn(y);
            }
}

int dtype_of(const at::Tensor& t) {
    return t.scalar_type() == at::kFloat ? F32 : t.scalar_type() == at::kBFloat16 ? BF16 : F16;
}

}  // namespace

#define TF_EXL3_WIDTHS(X, CB) X(2, CB) X(4, CB) X(6, CB) X(8, CB) X(10, CB) X(12, CB) X(14, CB) X(16, CB)
#define TF_EXL3_ALL(X) TF_EXL3_WIDTHS(X, 0) TF_EXL3_WIDTHS(X, 1) TF_EXL3_WIDTHS(X, 2) X(3, 2) X(5, 2) X(7, 2)

void exl3_rot_in_cuda(const at::Tensor& x, const at::Tensor& suh, at::Tensor& xh) {
    const int M = (int)x.size(0), K = (int)x.size(1);
    dim3 grid((unsigned)((K / 128 + 3) / 4), (unsigned)M);
    rot_in_kernel<<<grid, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr(), dtype_of(x), reinterpret_cast<const half*>(suh.data_ptr()),
        reinterpret_cast<half*>(xh.data_ptr()), K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#include "linear_wc.cuh"

void exl3_linear_cuda(const at::Tensor& xh, const at::Tensor& T, int64_t stride_k, int64_t stride_nb,
                      const at::Tensor& svh, const c10::optional<at::Tensor>& bias, at::Tensor& y,
                      const c10::optional<at::Tensor>& Z, at::Tensor& counters, int64_t K2, int64_t cb, int64_t SK,
                      int64_t WK, int64_t mode) {
    const int M = (int)xh.size(0), K = (int)xh.size(1), N = (int)y.size(1);
    TORCH_CHECK(WK == 2 || WK == 4 || WK == 8, "WK must be 2, 4 or 8");
    TORCH_CHECK((K / 16) % (SK * WK) == 0, "K / 16 must split evenly over SK * WK warps");
    dim3 grid((unsigned)(N / 128), (unsigned)SK);
    auto stream = at::cuda::getCurrentCUDAStream();
    const half* bptr = bias ? reinterpret_cast<const half*>(bias->data_ptr()) : nullptr;
    float* zptr = Z ? Z->data_ptr<float>() : nullptr;
    TORCH_CHECK(SK == 1 || zptr, "Z is needed with more than one split");
    constexpr int MIDM = 6, WC = 7;    // mode: 0 linear_kernel at every row count, 6 the mid-M kernels, 7 linear_wc
    if (mode >= WC && cb == CB_MUL1 &&
        (K2 == 8 ? dispatch_wc<8>(xh, T, stride_k, stride_nb, svh, bptr, y, zptr, counters, M, K, N, (int)SK, (int)WK)
                 : K2 == 12 &&
                       dispatch_wc<12>(xh, T, stride_k, stride_nb, svh, bptr, y, zptr, counters, M, K, N, (int)SK, (int)WK)))
        return;                                   // else the mid-M kernels take it
#define TF_LAUNCH(K2_, CB_)                                                                                        \
    if (K2 == K2_ && cb == CB_) {                                                                               \
        auto kernel = WK == 2 ? linear_kernel<K2_, CB_, 2>                                                      \
                              : WK == 4 ? linear_kernel<K2_, CB_, 4> : linear_kernel<K2_, CB_, 8>;              \
        const int smem = (int)(WK * std::min(M, 8) * 128 * sizeof(float));                                \
        if (mode >= MIDM && M > 16) {          /* 17-48 rows: 16-row groups; 49-128: 32-row groups */          \
            const bool p2 = M > 48;                                                                              \
            kernel = p2 ? (WK == 2 ? linear_mpg_kernel<K2_, CB_, 2, 2>                                           \
                                   : WK == 4 ? linear_mpg_kernel<K2_, CB_, 4, 2> : linear_mpg_kernel<K2_, CB_, 8, 2>) \
                        : (WK == 2 ? linear_mpg_kernel<K2_, CB_, 2, 1>                                           \
                                   : WK == 4 ? linear_mpg_kernel<K2_, CB_, 4, 1> : linear_mpg_kernel<K2_, CB_, 8, 1>); \
            grid.x = (unsigned)((N / 128) * ((M + (p2 ? 31 : 15)) / (p2 ? 32 : 16)));                           \
        }                                                                                                        \
        if (smem > 48 * 1024) cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
        kernel<<<grid, (unsigned)(WK * 32), smem, stream>>>(                                              \
            reinterpret_cast<const half*>(xh.data_ptr()), reinterpret_cast<const uint32_t*>(T.data_ptr()),      \
            stride_k, stride_nb, reinterpret_cast<const half*>(svh.data_ptr()), bptr, y.data_ptr(), dtype_of(y),\
            zptr, counters.data_ptr<int>(), M, K, N, (int)SK);                                                  \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                         \
        return;                                                                                                 \
    }
    TF_EXL3_ALL(TF_LAUNCH)
#undef TF_LAUNCH
    TORCH_CHECK(false, "unsupported EXL3 width/codebook: K2=", K2, " codebook=", cb);
}

void exl3_unpack_cuda(const at::Tensor& T, at::Tensor& W, int64_t stride_k, int64_t stride_nb, int64_t K2,
                      int64_t cb) {
    const int K = (int)W.size(0), N = (int)W.size(1);
    dim3 grid((unsigned)(N / 16), (unsigned)(K / 16));
    auto stream = at::cuda::getCurrentCUDAStream();
#define TF_LAUNCH(K2_, CB_)                                                                                         \
    if (K2 == K2_ && cb == CB_) {                                                                                \
        unpack_kernel<K2_, CB_><<<grid, 32, 0, stream>>>(reinterpret_cast<const uint32_t*>(T.data_ptr()),         \
                                                        reinterpret_cast<half*>(W.data_ptr()), N, stride_k,      \
                                                        stride_nb);                                              \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                          \
        return;                                                                                                  \
    }
    TF_EXL3_ALL(TF_LAUNCH)
#undef TF_LAUNCH
    TORCH_CHECK(false, "unsupported EXL3 width/codebook: K2=", K2, " codebook=", cb);
}

void exl3_unpack_fold_cuda(const at::Tensor& T, const at::Tensor& suh, at::Tensor& W, int64_t stride_k,
                           int64_t stride_nb, int64_t K2, int64_t cb) {
    const int K = (int)W.size(0), N = (int)W.size(1);
    dim3 grid((unsigned)(N / 128), (unsigned)(K / 128));
    auto stream = at::cuda::getCurrentCUDAStream();
    constexpr int SMEM = 128 * 129 * 4;
#define TF_LAUNCH(K2_, CB_)                                                                                         \
    if (K2 == K2_ && cb == CB_) {                                                                                \
        if (W.scalar_type() == at::kBFloat16) {                                                                  \
            auto kernel = unpack_fold_kernel<K2_, CB_, __nv_bfloat16>;                                           \
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);                     \
            kernel<<<grid, 256, SMEM, stream>>>(reinterpret_cast<const uint32_t*>(T.data_ptr()),                 \
                                                reinterpret_cast<const half*>(suh.data_ptr()),                   \
                                                reinterpret_cast<__nv_bfloat16*>(W.data_ptr()), N, stride_k,     \
                                                stride_nb);                                                      \
        } else {                                                                                                 \
            auto kernel = unpack_fold_kernel<K2_, CB_, half>;                                                    \
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);                     \
            kernel<<<grid, 256, SMEM, stream>>>(reinterpret_cast<const uint32_t*>(T.data_ptr()),                 \
                                                reinterpret_cast<const half*>(suh.data_ptr()),                   \
                                                reinterpret_cast<half*>(W.data_ptr()), N, stride_k, stride_nb);  \
        }                                                                                                        \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                          \
        return;                                                                                                  \
    }
    TF_EXL3_ALL(TF_LAUNCH)
#undef TF_LAUNCH
    TORCH_CHECK(false, "unsupported EXL3 width/codebook: K2=", K2, " codebook=", cb);
}

void exl3_unpack_fold2_cuda(const at::Tensor& T, const at::Tensor& suh, at::Tensor& W, int64_t stride_k,
                            int64_t stride_nb, int64_t K2, int64_t cb) {
    const int K = (int)W.size(0), N = (int)W.size(1);
    dim3 grid((unsigned)(N / 128), (unsigned)(K / 128));
    auto stream = at::cuda::getCurrentCUDAStream();
#define TF_LAUNCH(K2_, CB_)                                                                                         \
    if (K2 == K2_ && cb == CB_) {                                                                                \
        if (W.scalar_type() == at::kBFloat16)                                                                    \
            unpack_fold2_kernel<K2_, CB_, __nv_bfloat16><<<grid, 256, 0, stream>>>(                              \
                reinterpret_cast<const uint32_t*>(T.data_ptr()), reinterpret_cast<const half*>(suh.data_ptr()),  \
                reinterpret_cast<__nv_bfloat16*>(W.data_ptr()), N, stride_k, stride_nb);                         \
        else                                                                                                     \
            unpack_fold2_kernel<K2_, CB_, half><<<grid, 256, 0, stream>>>(                                       \
                reinterpret_cast<const uint32_t*>(T.data_ptr()), reinterpret_cast<const half*>(suh.data_ptr()),  \
                reinterpret_cast<half*>(W.data_ptr()), N, stride_k, stride_nb);                                  \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                          \
        return;                                                                                                  \
    }
    TF_EXL3_ALL(TF_LAUNCH)
#undef TF_LAUNCH
    TORCH_CHECK(false, "unsupported EXL3 width/codebook: K2=", K2, " codebook=", cb);
}

void exl3_fdirect_cuda(const at::Tensor& x, const at::Tensor& T, int64_t stride_k, int64_t stride_nb,
                       const at::Tensor& suh, const at::Tensor& svh, const c10::optional<at::Tensor>& bias,
                       at::Tensor& out, int64_t K2, int64_t cb) {
    const int M = (int)x.size(0), K = (int)x.size(1), N = (int)out.size(1);
    auto stream = at::cuda::getCurrentCUDAStream();
    const half* bptr = bias ? reinterpret_cast<const half*>(bias->data_ptr()) : nullptr;
    TORCH_CHECK(K2 == 8, "fdirect: 4-bit tiles only");
    for (int r0 = 0; r0 < M; r0 += 128) {            // rows are independent: 128-row slices keep their bits
        const int m = std::min(128, M - r0);
        const auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()) + (int64_t)r0 * x.stride(0);
        auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr()) + (int64_t)r0 * out.stride(0);
#define TF_FD(CB_, MT_)                                                                                         \
        if (cb == CB_ && (m + 15) / 16 <= MT_) {                                                                \
            fdirect_kernel<8, CB_, MT_><<<dim3((unsigned)(N / 128)), 256, 0, stream>>>(                         \
                xp, (int)x.stride(0), reinterpret_cast<const uint32_t*>(T.data_ptr()),                          \
                reinterpret_cast<const half*>(suh.data_ptr()), reinterpret_cast<const half*>(svh.data_ptr()),  \
                bptr, op, (int)out.stride(0), m, K, N, stride_k, stride_nb);                                    \
            C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                     \
            continue;                                                                                           \
        }
#define TF_FD_CB(CB_) TF_FD(CB_, 1) TF_FD(CB_, 2) TF_FD(CB_, 3) TF_FD(CB_, 4) TF_FD(CB_, 8)
        TF_FD_CB(0) TF_FD_CB(1) TF_FD_CB(2)
#undef TF_FD_CB
#undef TF_FD
        TORCH_CHECK(false, "fdirect: unsupported codebook ", cb);
    }
}
