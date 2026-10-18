// EXL3 decode-row linear: any codebook, width, 1-128 rows, a row's bits only its own; port of tensorfold's linear.cu.
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "exl3_decode.cuh"

namespace tf_exl3 {

enum DType : int { F16 = 0, BF16 = 1, F32 = 2 };

__device__ __forceinline__ int imin(int a, int b) { return a < b ? a : b; }

struct LinearArgs {
    const void* xh;          // fp16 [m, k], the hadamard-rotated inputs
    const uint32_t* t;       // trellis words
    long long stride_k;      // words between k tiles
    long long stride_nb;     // words between 128-column blocks
    const half* svh;         // fp16 [n], output hadamard scales
    const half* bias;        // fp16 [n] or null
    void* y;                 // output
    float* z;                // fp32 split partials [sk, m, n] or null
    int* counters;           // per (pass, block) arrival counts, zeroed before the launch
    int y_dtype;
    int n, sk, m, k;
};

struct RotInArgs {
    const void* x;           // f16 / bf16 / f32 [m, k]
    const half* suh;         // fp16 [k], input hadamard scales
    half* xh;                // fp16 [m, k] out
    int x_dtype;
    int m, k;
};

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
        // Speculative loads must have a valid address even when the split-K result has no bias.
        const half bv = __ldg((bias ? bias : svh) + col + j);
        if (bias) v[j] += __half2float(bv);
    }
}

// A k step's words for one lane: the step is 8 tiles back to back, each lane takes the same slice of every tile.
template <int K2>
__device__ __forceinline__ void load_step(const uint32_t* step, int lane, uint32_t (&raw)[8 * lane_words<K2>()]) {
    constexpr int TW = tile_words<K2>(), LW = lane_words<K2>();
    int word, offset;
    lane_start<K2>(lane, word, offset);
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int q = 0; q < LW; ++q) raw[j * LW + q] = __ldg(step + j * TW + (word + q) % TW);
}

// Tile j's lane words from a loaded step.
template <int K2>
__device__ __forceinline__ void step_lane_words(const uint32_t (&raw)[8 * lane_words<K2>()], int j,
                                                uint32_t (&w)[lane_words<K2>()]) {
#pragma unroll
    for (int q = 0; q < lane_words<K2>(); ++q) w[q] = raw[j * lane_words<K2>() + q];
}

// xh = fp16(((x * suh) @ H) / sqrt(128)): one warp per 128 columns, one block of 4 warps per 512.
extern "C" __global__ void __launch_bounds__(128) tf_exl3_rot_in(RotInArgs a) {
    const int blk = blockIdx.x * 4 + (threadIdx.x >> 5), row = blockIdx.y, lane = threadIdx.x & 31;
    if (blk * 128 >= a.k) return;
    const int k = blk * 128 + 4 * lane;
    float v[4], s[4];
    load4(a.x, a.x_dtype, (size_t)row * a.k + k, v);
    load4(a.suh, F16, k, s);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= s[j];
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= HAD_SCALE;
    store4(a.xh, F16, (size_t)row * a.k + k, v);
}


// One program: (N / 128) * SK blocks, split-major like a (N / 128, SK) grid, WK warps each.
template <int K2, int CB, int WK>
__device__ __forceinline__ void linear_body(LinearArgs a) {
    constexpr int TW = tile_words<K2>(), LW = lane_words<K2>();
    constexpr bool PF = K2 <= 12;                             // the next k step's words load while this one decodes
    constexpr int SR = 8 * LW;
    extern __shared__ __align__(16) float red[];              // WK * RH * 128 floats
    __shared__ int last;
    const int M = a.m, K = a.k, N = a.n, SK = a.sk, NB = N >> 7;
    const int RH = imin(M, 8);                                // rows of red a warp
    const int nb = (int)blockIdx.x % NB, split = (int)blockIdx.x / NB;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int per_warp = (K >> 4) / SK / WK;
    const int kt0 = split * (per_warp * WK) + warp * per_warp;
    const uint32_t* tiles = a.t + nb * a.stride_nb;
    const int col0 = nb * 128;

    for (int m0 = 0, pass = 0; m0 < M; m0 += 16, ++pass) {
        const int R = imin(16, M - m0);

        float acc[8][2][4];
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

        // the walk's rows for the mma, clamped so rows past the pass read inside the buffer (their outputs are dropped)
        const int r0 = m0 + (g < R ? g : R - 1), r1 = m0 + (g + 8 < R ? g + 8 : R - 1);
        const half* x0 = static_cast<const half*>(a.xh) + (size_t)r0 * K;
        const half* x1 = static_cast<const half*>(a.xh) + (size_t)r1 * K;
        const uint32_t* tile = tiles + (size_t)kt0 * a.stride_k;
        uint32_t cur[PF ? SR : 1], nxt[PF ? SR : 1];
        if constexpr (PF) load_step<K2>(tile, lane, cur);     // the weights do not wait for the previous kernel

        // x's fragments of k step kt: rows g (and g + 8): [0] [2] k 2t.., 2t + 8..; [1] [3] the same of row g + 8
        auto load_x = [&](int kt, uint32_t (&u)[4]) {
            u[0] = __ldg(reinterpret_cast<const uint32_t*>(x0 + kt * 16 + 2 * t));
            u[2] = __ldg(reinterpret_cast<const uint32_t*>(x0 + kt * 16 + 2 * t + 8));
            u[1] = __ldg(reinterpret_cast<const uint32_t*>(x1 + kt * 16 + 2 * t));
            u[3] = __ldg(reinterpret_cast<const uint32_t*>(x1 + kt * 16 + 2 * t + 8));
        };
#pragma unroll 1
        for (int i = 0; i < per_warp; ++i) {
            uint32_t u[4];
            load_x(kt0 + i, u);
            if constexpr (PF) {
                if (i + 1 < per_warp) load_step<K2>(tile + (size_t)(i + 1) * a.stride_k, lane, nxt);
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                uint32_t w[LW];
                if constexpr (PF) step_lane_words<K2>(cur, j, w);
                else ldg_lane_words<K2>(tile + (size_t)i * a.stride_k + j * TW, lane, w);
                uint32_t b0[2], b1[2];
                decode_lane<K2, CB>(w, lane, b0, b1);
                mma16816(acc[j][0], u, b0);
                mma16816(acc[j][1], u, b1);
            }
            if constexpr (PF) {
#pragma unroll
                for (int q = 0; q < SR; ++q) cur[q] = nxt[q];
            }
        }

        // the warps' sums, added in warp order, rows 0-7 of the pass and then rows 8-15
        for (int rlo = 0; rlo < R; rlo += 8) {
            const int rn = imin(R - rlo, 8);
            __syncthreads();                         // red is reused by every half and pass
            if (g < RH) {
#pragma unroll
                for (int i = 0; i < 8; ++i)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int col = i * 16 + h * 8 + 2 * t;
                        *reinterpret_cast<float2*>(red + (warp * RH + g) * 128 + col) =
                            rlo ? make_float2(acc[i][h][2], acc[i][h][3])
                                : make_float2(acc[i][h][0], acc[i][h][1]);
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
                    finish(v, lane, a.svh, a.bias, col0 + 4 * lane);
                    store4(a.y, a.y_dtype, (size_t)(m0 + rlo + r) * N + col0 + 4 * lane, v);
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
                    *reinterpret_cast<float4*>(a.z + ((size_t)split * M + m0 + rlo + r) * N + col0 + c) = s;
                }
            }
        }
        if (SK > 1) {
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0) last = atomicAdd(a.counters + pass * NB + nb, 1) == SK - 1;
            __syncthreads();
            if (last) {
                __threadfence();
                for (int r = warp; r < R; r += WK) {
                    const size_t at = ((size_t)m0 + r) * N + col0 + 4 * lane;
                    float4 s = __ldcg(reinterpret_cast<const float4*>(a.z + at));
                    for (int q = 1; q < SK; ++q) {
                        const float4 u = __ldcg(reinterpret_cast<const float4*>(a.z + (size_t)q * M * N + at));
                        s.x += u.x; s.y += u.y; s.z += u.z; s.w += u.w;
                    }
                    float v[4] = {s.x, s.y, s.z, s.w};
                    finish(v, lane, a.svh, a.bias, col0 + 4 * lane);
                    store4(a.y, a.y_dtype, (size_t)(m0 + r) * N + col0 + 4 * lane, v);
                }
                if (threadIdx.x == 0) a.counters[pass * NB + nb] = 0;   // every program of the block has arrived
            }
        }
        __syncthreads();                             // red is reused in the next pass
    }
}

// One extern "C" kernel per instantiation: the fatbin carries __global__ functions, the host resolves them by name.
#define TF_EXL3_LINEAR(K2, CB, WK)                                                       \
    extern "C" __global__ void __launch_bounds__((WK) * 32)                              \
        tf_exl3_linear_##K2##_##CB##_##WK(tf_exl3::LinearArgs a) {                       \
        tf_exl3::linear_body<K2, CB, WK>(a);                                             \
    }

// Every width the format serves (linear.cu TF_EXL3_ALL plus 4.5-6.5): whole bits, all codebooks; 7.5 is not a width.
#define TF_EXL3_WIDTHS(K2, CB) TF_EXL3_LINEAR(K2, CB, 4) TF_EXL3_LINEAR(K2, CB, 8)
#define TF_EXL3_CODEBOOK(CB)                                                                                \
    TF_EXL3_WIDTHS(2, CB) TF_EXL3_WIDTHS(4, CB) TF_EXL3_WIDTHS(6, CB) TF_EXL3_WIDTHS(8, CB)                 \
    TF_EXL3_WIDTHS(10, CB) TF_EXL3_WIDTHS(12, CB) TF_EXL3_WIDTHS(14, CB) TF_EXL3_WIDTHS(16, CB)
TF_EXL3_CODEBOOK(2) // mul1
TF_EXL3_WIDTHS(3, 2) TF_EXL3_WIDTHS(5, 2) TF_EXL3_WIDTHS(7, 2) TF_EXL3_WIDTHS(9, 2)
TF_EXL3_WIDTHS(11, 2) TF_EXL3_WIDTHS(13, 2)
TF_EXL3_CODEBOOK(1) // mcg
TF_EXL3_CODEBOOK(0) // inst3
#undef TF_EXL3_CODEBOOK
#undef TF_EXL3_WIDTHS
}  // namespace tf_exl3
