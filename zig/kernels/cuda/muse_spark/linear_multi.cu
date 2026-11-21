// Muse Spark's EXL3 launch forms on exl3_decode.cuh: q/k/v/gate in one launch and 32-row passes, exl3_linear.cu's bits.

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "../exl3_decode.cuh"

namespace muse_linear {

using namespace tf_exl3;

enum DType : int { F16 = 0, BF16 = 1, F32 = 2 };

__device__ __forceinline__ void load4(const void* p, int dtype, size_t i, float (&v)[4]) {
    if (dtype == F32) {
        const float4 u = *reinterpret_cast<const float4*>(static_cast<const float*>(p) + i);
        v[0] = u.x; v[1] = u.y; v[2] = u.z; v[3] = u.w;
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
    } else {
        half2 a = __floats2half2_rn(v[0], v[1]), b = __floats2half2_rn(v[2], v[3]);
        uint2 u;
        u.x = *reinterpret_cast<uint32_t*>(&a);
        u.y = *reinterpret_cast<uint32_t*>(&b);
        *reinterpret_cast<uint2*>(static_cast<half*>(p) + i) = u;
    }
}

// exl3_linear.cu's finish: H / sqrt(128), * svh, + bias.
__device__ __forceinline__ void finish(float (&v)[4], int lane, const half* svh, const half* bias, int col) {
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j] = v[j] * HAD_SCALE * __half2float(__ldg(svh + col + j));
        const half bv = __ldg((bias ? bias : svh) + col + j);
        if (bias) v[j] += __half2float(bv);
    }
}

// exl3_linear.cu's k step loads: 8 tiles back to back, each lane the same slice of every tile.
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

template <int K2>
__device__ __forceinline__ void step_lane_words(const uint32_t (&raw)[8 * lane_words<K2>()], int j,
                                                uint32_t (&w)[lane_words<K2>()]) {
#pragma unroll
    for (int q = 0; q < lane_words<K2>(); ++q) w[q] = raw[j * lane_words<K2>() + q];
}

template <int K2, int CB, int WK>
__device__ __forceinline__ void linear_body(
    const half* __restrict__ xh, const uint32_t* __restrict__ T, long long stride_k, long long stride_nb,
    const half* __restrict__ svh, const half* __restrict__ bias, void* __restrict__ y, int y_dtype,
    float* __restrict__ Z, int* __restrict__ counters, int M, int K, int N, int SK, const int nb, const int split,
    const int NB) {
    constexpr int TW = tile_words<K2>();
    constexpr int LW = lane_words<K2>();
    extern __shared__ __align__(16) float red[];              // WK * RH * 128 floats
    __shared__ int last;
    const int RH = min(M, 8);                                 // rows of red a warp

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int per_warp = (K >> 4) / SK / WK;
    const int kt0 = split * (per_warp * WK) + warp * per_warp;
    const uint32_t* tiles = T + nb * stride_nb;
    const int col0 = nb * 128;

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
        constexpr int SR = 8 * LW;
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
                if constexpr (PF) step_lane_words<K2>(cur, j, w);
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


// linear_body with 32-row passes: weights decoded once for two 16-row groups, each group linear_body's bits.
template <int K2, int CB, int WK>
__device__ __forceinline__ void linear_body2(
    const half* __restrict__ xh, const uint32_t* __restrict__ T, long long stride_k, long long stride_nb,
    const half* __restrict__ svh, const half* __restrict__ bias, void* __restrict__ y, int y_dtype,
    float* __restrict__ Z, int* __restrict__ counters, int M, int K, int N, int SK, const int nb, const int split,
    const int NB) {
    constexpr int TW = tile_words<K2>();
    constexpr int LW = lane_words<K2>();
    extern __shared__ __align__(16) float red[];              // WK * RH * 128 floats
    __shared__ int last;
    const int RH = min(M, 8);                                 // rows of red a warp

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int per_warp = (K >> 4) / SK / WK;
    const int kt0 = split * (per_warp * WK) + warp * per_warp;
    const uint32_t* tiles = T + nb * stride_nb;
    const int col0 = nb * 128;

    for (int mb = 0; mb < M; mb += 32) {
        const int ng = M - mb > 16 ? 2 : 1;                   // 16-row groups in this pass
        float acc[2][8][2][4];
#pragma unroll
        for (int q = 0; q < 2; ++q)
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int c = 0; c < 4; ++c) acc[q][i][h][c] = 0.f;
        const half* x0[2];
        const half* x1[2];
#pragma unroll
        for (int q = 0; q < 2; ++q) {   // linear_body's clamped rows; an empty second group reads row M - 1 (unused)
            const int m0 = mb + 16 * q;
            int r0 = M - 1, r1 = M - 1;
            if (m0 < M) {
                const int R = min(16, M - m0);
                r0 = m0 + (g < R ? g : R - 1);
                r1 = m0 + (g + 8 < R ? g + 8 : R - 1);
            }
            x0[q] = xh + (size_t)r0 * K;
            x1[q] = xh + (size_t)r1 * K;
        }
        const uint32_t* tile = tiles + (size_t)kt0 * stride_k;
        constexpr bool PF = K2 <= 12;
        constexpr int SR = 8 * LW;
        uint32_t cur[PF ? SR : 1], nxt[PF ? SR : 1];
        if constexpr (PF) load_step<K2>(tile, lane, cur);
#pragma unroll 1
        for (int i = 0; i < per_warp; ++i) {
            const int kt = kt0 + i;
            uint32_t a[2][4];
#pragma unroll
            for (int q = 0; q < 2; ++q) {
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
                if constexpr (PF) step_lane_words<K2>(cur, j, w);
                else ldg_lane_words<K2>(tile + (size_t)i * stride_k + j * TW, lane, w);
                uint32_t b0[2], b1[2];
                decode_lane<K2, CB>(w, lane, b0, b1);
                mma16816(acc[0][j][0], a[0], b0);
                mma16816(acc[0][j][1], a[0], b1);
                if (ng == 2) {
                    mma16816(acc[1][j][0], a[1], b0);
                    mma16816(acc[1][j][1], a[1], b1);
                }
            }
            if constexpr (PF) {
#pragma unroll
                for (int q = 0; q < SR; ++q) cur[q] = nxt[q];
            }
        }

#pragma unroll
        for (int q = 0; q < 2; ++q) {
            if (q >= ng) break;
            const int m0 = mb + 16 * q, pass = m0 >> 4;
            const int R = min(16, M - m0);
            // the warps' sums, added in warp order, rows 0-7 of the group and then rows 8-15 (linear_body's)
            for (int rlo = 0; rlo < R; rlo += 8) {
                const int rn = min(R - rlo, 8);
                __syncthreads();
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
                            const float4 qq = *reinterpret_cast<const float4*>(red + (w * RH + r) * 128 + 4 * lane);
                            v[0] += qq.x; v[1] += qq.y; v[2] += qq.z; v[3] += qq.w;
                        }
                        finish(v, lane, svh, bias, col0 + 4 * lane);
                        store4(y, y_dtype, (size_t)(m0 + rlo + r) * N + col0 + 4 * lane, v);
                    }
                } else {
                    for (int idx = threadIdx.x; idx < rn * 32; idx += WK * 32) {
                        const int r = idx >> 5, c = 4 * (idx & 31);
                        float4 sm = *reinterpret_cast<const float4*>(red + r * 128 + c);
#pragma unroll
                        for (int w = 1; w < WK; ++w) {
                            const float4 qq = *reinterpret_cast<const float4*>(red + (w * RH + r) * 128 + c);
                            sm.x += qq.x; sm.y += qq.y; sm.z += qq.z; sm.w += qq.w;
                        }
                        *reinterpret_cast<float4*>(Z + ((size_t)split * M + m0 + rlo + r) * N + col0 + c) = sm;
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
                        float4 sm = __ldcg(reinterpret_cast<const float4*>(Z + at));
                        for (int qq = 1; qq < SK; ++qq) {
                            const float4 u = __ldcg(reinterpret_cast<const float4*>(Z + (size_t)qq * M * N + at));
                            sm.x += u.x; sm.y += u.y; sm.z += u.z; sm.w += u.w;
                        }
                        float v[4] = {sm.x, sm.y, sm.z, sm.w};
                        finish(v, lane, svh, bias, col0 + 4 * lane);
                        store4(y, y_dtype, (size_t)(m0 + r) * N + col0 + 4 * lane, v);
                    }
                    if (threadIdx.x == 0) counters[pass * NB + nb] = 0;
                }
            }
            __syncthreads();
        }
    }
}

struct Job {
    const half* xh;
    const uint32_t* T;
    long long stride_k, stride_nb;
    const half* svh;
    const half* bias;
    void* y;
    float* Z;
    int* counters;
    int y_dtype, K, N, SK, b0, pad;
};
struct Batch {
    Job j[4];
    int n, M, pad[2];
};
struct Rot {
    const half* suh[4];
    half* xh[4];
};

}  // namespace muse_linear

// One launch of up to 4 jobs: block b runs the job whose block range holds it, as exl3_linear's grid would.
#define MUSE_MULTI(K2)                                                                                            \
    extern "C" __global__ void __launch_bounds__(128) muse_linear_multi_k##K2(const muse_linear::Batch b) {      \
        int ji = 0;                                                                                               \
        while (ji + 1 < b.n && (int)blockIdx.x >= b.j[ji + 1].b0) ++ji;                                           \
        const muse_linear::Job& J = b.j[ji];                                                                      \
        const int NB = J.N / 128, local = blockIdx.x - J.b0;                                                      \
        muse_linear::linear_body<K2, 2, 4>(J.xh, J.T, J.stride_k, J.stride_nb, J.svh, J.bias, J.y, J.y_dtype,     \
                                           J.Z, J.counters, b.M, J.K, J.N, J.SK, local % NB, local / NB, NB);    \
    }
#define MUSE_LIN2(K2, WK)                                                                                         \
    extern "C" __global__ void __launch_bounds__(WK * 32) muse_linear2_k##K2##_w##WK(                             \
        const half* __restrict__ xh, const uint32_t* __restrict__ T, long long stride_k, long long stride_nb,     \
        const half* __restrict__ svh, const half* __restrict__ bias, void* __restrict__ y, int y_dtype,          \
        float* __restrict__ Z, int* __restrict__ counters, int M, int K, int N, int SK) {                        \
        muse_linear::linear_body2<K2, 2, WK>(xh, T, stride_k, stride_nb, svh, bias, y, y_dtype, Z, counters, M, \
                                             K, N, SK, blockIdx.x, blockIdx.y, gridDim.x);                       \
    }
#define MUSE_WIDTH(K2) MUSE_MULTI(K2) MUSE_LIN2(K2, 4) MUSE_LIN2(K2, 8)
MUSE_WIDTH(2) MUSE_WIDTH(3) MUSE_WIDTH(4) MUSE_WIDTH(5) MUSE_WIDTH(6) MUSE_WIDTH(8)
#undef MUSE_WIDTH
#undef MUSE_LIN2
#undef MUSE_MULTI

// exl3_linear.cu's tf_exl3_rot_in for 4 suh / xh pairs (grid z).
extern "C" __global__ void __launch_bounds__(128) muse_rot_multi(const void* __restrict__ x, int x_dtype,
                                                                const muse_linear::Rot r, int K) {
    const int blk = blockIdx.x * 4 + (threadIdx.x >> 5), row = blockIdx.y, lane = threadIdx.x & 31;
    if (blk * 128 >= K) return;
    const half* suh = r.suh[blockIdx.z];
    half* xh = r.xh[blockIdx.z];
    const int k = blk * 128 + 4 * lane;
    float v[4], s[4];
    muse_linear::load4(x, x_dtype, (size_t)row * K + k, v);
    muse_linear::load4(suh, muse_linear::F16, k, s);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= s[j];
    tf_exl3::fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= tf_exl3::HAD_SCALE;
    muse_linear::store4(xh, muse_linear::F16, (size_t)row * K + k, v);
}
