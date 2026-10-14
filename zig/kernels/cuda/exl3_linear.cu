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

// 16 bytes of trellis words, read-only and not kept in L1.
__device__ __forceinline__ uint4 ldg_nc_v4(const uint32_t* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
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
    constexpr bool ST = TW % 8 == 0;                          // even widths: staged 16-byte loads, PD steps in flight
    constexpr int PD = 2;
    constexpr bool PF = !ST && K2 <= 12;                      // odd widths: the next step's lane words in registers
    constexpr int SR = 8 * LW;
    extern __shared__ __align__(16) float red[];              // WK * RH * 128 floats; even widths stage steps here
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
        constexpr int NV = (8 * TW + 127) / 128;              // a staged step's 16-byte loads a lane
        uint4 rv[ST ? PD : 1][ST ? NV : 1];
        if constexpr (ST) {
#pragma unroll
            for (int d = 0; d < PD; ++d)
                if (d < per_warp)
#pragma unroll
                    for (int v = 0; v < NV; ++v)
                        if (128 * v + 4 * lane < 8 * TW)
                            rv[d][v] = ldg_nc_v4(tile + d * a.stride_k + 128 * v + 4 * lane);
        }

        // x's fragments of k step kt: rows g (and g + 8): [0] [2] k 2t.., 2t + 8..; [1] [3] the same of row g + 8
        auto load_x = [&](int kt, uint32_t (&u)[4]) {
            u[0] = __ldg(reinterpret_cast<const uint32_t*>(x0 + kt * 16 + 2 * t));
            u[2] = __ldg(reinterpret_cast<const uint32_t*>(x0 + kt * 16 + 2 * t + 8));
            u[1] = __ldg(reinterpret_cast<const uint32_t*>(x1 + kt * 16 + 2 * t));
            u[3] = __ldg(reinterpret_cast<const uint32_t*>(x1 + kt * 16 + 2 * t + 8));
        };
        if constexpr (ST) {
            uint32_t* stage = reinterpret_cast<uint32_t*>(red) + warp * 8 * 64; // a warp's step: 8 tiles, 64 words most
            int word, offset;
            lane_start<K2>(lane, word, offset);
#pragma unroll 1
            for (int ib = 0; ib < per_warp; ib += PD) {
#pragma unroll
                for (int d = 0; d < PD; ++d) {
                    const int i = ib + d;
                    if (i >= per_warp) break;
                    __syncwarp(); // every lane has read the previous step back
#pragma unroll
                    for (int v = 0; v < NV; ++v)
                        if (128 * v + 4 * lane < 8 * TW)
                            *reinterpret_cast<uint4*>(stage + 128 * v + 4 * lane) = rv[d][v];
                    if (i + PD < per_warp)
#pragma unroll
                        for (int v = 0; v < NV; ++v)
                            if (128 * v + 4 * lane < 8 * TW)
                                rv[d][v] = ldg_nc_v4(tile + (size_t)(i + PD) * a.stride_k + 128 * v + 4 * lane);
                    __syncwarp(); // the step is staged before any lane reads it
                    uint32_t u[4];
                    load_x(kt0 + i, u);
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        uint32_t w[LW];
#pragma unroll
                        for (int q = 0; q < LW; ++q) w[q] = stage[j * TW + (word + q) % TW];
                        uint32_t b0[2], b1[2];
                        decode_lane<K2, CB>(w, lane, b0, b1);
                        mma16816(acc[j][0], u, b0);
                        mma16816(acc[j][1], u, b1);
                    }
                }
            }
        }
#pragma unroll 1
        for (int i = 0; i < (ST ? 0 : per_warp); ++i) {
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

// A prompt block of 64 rows (mul1): warp j owns column tile j for every K range, so x's tiles are staged once in shared
constexpr int PR = 64; // rows a block
constexpr int PS = 132; // psum's row stride (floats): rows apart in banks, 16-byte aligned
constexpr int PKC = 2; // k tiles a stage
constexpr int PXS = 24; // a staged x row's stride (halves): ldmatrix rows on distinct banks
constexpr int PROMPT_SHARED = PR * PS * 4 + 2 * PKC * PR * PXS * 2;

__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"((unsigned)__cvta_generic_to_shared(dst)), "l"(src));
}

// The same partial sums per K range as linear_body, added into psum in range order: its bits, each tile decoded once.
template <int K2, int WK>
__device__ __forceinline__ void prompt_body(LinearArgs a) {
    constexpr int TW = tile_words<K2>(), LW = lane_words<K2>();
    extern __shared__ __align__(16) float psum[]; // [PR][PS], then x stages [2][PKC][PR][PXS] fp16
    half* xs = reinterpret_cast<half*>(psum + PR * PS);
    __shared__ int last;
    const int M = a.m, K = a.k, N = a.n, SK = a.sk, NB = N >> 7;
    const int row0 = blockIdx.y * PR, MB = imin(PR, M - row0), MT = (MB + 15) >> 4;
    const int nb = (int)blockIdx.x % NB, split = (int)blockIdx.x / NB;
    const int j = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int per_warp = (K >> 4) / SK / WK, nkt = per_warp * WK, kt0 = split * nkt;
    const uint32_t* tiles = a.t + nb * a.stride_nb + j * TW;
    const half* xh = static_cast<const half*>(a.xh);
    const int col0 = nb * 128;
    auto stage = [&](int c) { // k tiles c * PKC.. into buffer c & 1: one 16-byte piece a thread
        const int kk = threadIdx.x >> 7, row = (threadIdx.x >> 1) & (PR - 1), part = threadIdx.x & 1;
        const half* src = xh + (size_t)(row0 + imin(row, MB - 1)) * K + (kt0 + c * PKC + kk) * 16 + part * 8;
        cp_async16(xs + (((c & 1) * PKC + kk) * PR + row) * PXS + part * 8, src);
        asm volatile("cp.async.commit_group;");
    };
    float acc[4][2][4];
    uint32_t w[LW], wn[LW];
    ldg_lane_words<K2>(tiles + (size_t)kt0 * a.stride_k, lane, w);
    stage(0);
    const int chunks = nkt / PKC;
#pragma unroll 1
    for (int c = 0; c < chunks; ++c) {
        if (c + 1 < chunks) {
            stage(c + 1);
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < PKC; ++kk) {
            const int i = c * PKC + kk;
            if (i % per_warp == 0) {
#pragma unroll
                for (int q = 0; q < 4; ++q)
#pragma unroll
                    for (int h = 0; h < 2; ++h)
#pragma unroll
                        for (int e = 0; e < 4; ++e) acc[q][h][e] = 0.f;
            }
            if (i + 1 < nkt) ldg_lane_words<K2>(tiles + (size_t)(kt0 + i + 1) * a.stride_k, lane, wn);
            uint32_t b0[2], b1[2];
            decode_lane<K2, 2>(w, lane, b0, b1);
            const half* xb = xs + (((c & 1) * PKC + kk) * PR + (lane & 15)) * PXS + (lane >> 4) * 8;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                if (q < MT) {
                    uint32_t u[4];
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                                 : "=r"(u[0]), "=r"(u[1]), "=r"(u[2]), "=r"(u[3])
                                 : "r"((unsigned)__cvta_generic_to_shared(xb + q * 16 * PXS)));
                    mma16816(acc[q][0], u, b0);
                    mma16816(acc[q][1], u, b1);
                }
            }
#pragma unroll
            for (int q = 0; q < LW; ++q) w[q] = wn[q];
            if ((i + 1) % per_warp == 0) { // a K range done: into psum, the first range stored, the rest added
                const bool first = i + 1 == per_warp;
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    if (q >= MT) continue;
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        float* p0 = psum + (16 * q + g) * PS + j * 16 + h * 8 + 2 * t;
                        float* p1 = p0 + 8 * PS;
                        p0[0] = first ? acc[q][h][0] : p0[0] + acc[q][h][0];
                        p0[1] = first ? acc[q][h][1] : p0[1] + acc[q][h][1];
                        p1[0] = first ? acc[q][h][2] : p1[0] + acc[q][h][2];
                        p1[1] = first ? acc[q][h][3] : p1[1] + acc[q][h][3];
                    }
                }
            }
        }
        __syncthreads(); // the stage is refilled two chunks on
    }
    if (SK == 1) {
        for (int r = j; r < MB; r += 8) {
            const float4 u = *reinterpret_cast<const float4*>(psum + r * PS + 4 * lane);
            float v[4] = {u.x, u.y, u.z, u.w};
            finish(v, lane, a.svh, a.bias, col0 + 4 * lane);
            store4(a.y, a.y_dtype, (size_t)(row0 + r) * N + col0 + 4 * lane, v);
        }
        return;
    }
    for (int r = j; r < MB; r += 8)
        *reinterpret_cast<float4*>(a.z + ((size_t)split * M + row0 + r) * N + col0 + 4 * lane) =
            *reinterpret_cast<const float4*>(psum + r * PS + 4 * lane);
    __threadfence();
    __syncthreads();
    int* counter = a.counters + blockIdx.y * NB + nb;
    if (threadIdx.x == 0) last = atomicAdd(counter, 1) == SK - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    for (int r = j; r < MB; r += 8) {
        const size_t at = (size_t)(row0 + r) * N + col0 + 4 * lane;
        float4 s = __ldcg(reinterpret_cast<const float4*>(a.z + at));
        for (int q = 1; q < SK; ++q) {
            const float4 u = __ldcg(reinterpret_cast<const float4*>(a.z + (size_t)q * M * N + at));
            s.x += u.x; s.y += u.y; s.z += u.z; s.w += u.w;
        }
        float v[4] = {s.x, s.y, s.z, s.w};
        finish(v, lane, a.svh, a.bias, col0 + 4 * lane);
        store4(a.y, a.y_dtype, at, v);
    }
    if (threadIdx.x == 0) *counter = 0; // every program of the block has arrived
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

// The prompt kernels: mul1, every width, 4 and 8 warps' K ranges, 8 warps a block.
#define TF_EXL3_PROMPT(K2)                                                                                  \
    extern "C" __global__ void __launch_bounds__(256, 2) tf_exl3_prompt_##K2##_4(tf_exl3::LinearArgs a) {   \
        tf_exl3::prompt_body<K2, 4>(a);                                                                     \
    }                                                                                                       \
    extern "C" __global__ void __launch_bounds__(256, 2) tf_exl3_prompt_##K2##_8(tf_exl3::LinearArgs a) {   \
        tf_exl3::prompt_body<K2, 8>(a);                                                                     \
    }
TF_EXL3_PROMPT(2) TF_EXL3_PROMPT(3) TF_EXL3_PROMPT(4) TF_EXL3_PROMPT(5) TF_EXL3_PROMPT(6) TF_EXL3_PROMPT(7)
TF_EXL3_PROMPT(8) TF_EXL3_PROMPT(9) TF_EXL3_PROMPT(10) TF_EXL3_PROMPT(11) TF_EXL3_PROMPT(12) TF_EXL3_PROMPT(13)
TF_EXL3_PROMPT(14) TF_EXL3_PROMPT(16)
#undef TF_EXL3_PROMPT
}  // namespace tf_exl3
