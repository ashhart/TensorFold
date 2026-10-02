// EXL3 linear, any codebook and width, 1-128 rows: a row's bits depend only on it (mma keeps rows apart, K ranges fixed by (K, N), fixed-order sums).

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>

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

// Programmatic dependent launch (a no-op when the launch did not ask for it): wait for the previous kernel's writes;
// let the next PDL kernel launch (it still waits for all of this one before it reads anything this one writes).
__device__ __forceinline__ void griddep_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void griddep_launch() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

constexpr int MAX_JOBS = 3;    // layers of one input in one launch (rot_in_group / linear_group)

struct RotJobs {
    const half* suh[MAX_JOBS];
    half* xh[MAX_JOBS];
};

// xh_j = fp16(((x * suh_j) @ H) / sqrt(128)) for job j = blockIdx.z (the layers of one input share x).
__global__ void __launch_bounds__(128) rot_in_kernel(const void* __restrict__ x, int x_dtype, RotJobs jobs, int K) {
    griddep_wait();                                  // x is the previous kernel's, and xh may be read by one still
    griddep_launch();
    const int blk = blockIdx.x * 4 + (threadIdx.x >> 5), row = blockIdx.y, lane = threadIdx.x & 31;
    if (blk * 128 >= K) return;
    const half* suh = jobs.suh[0];
    half* xh = jobs.xh[0];
    if (blockIdx.z == 1) suh = jobs.suh[1], xh = jobs.xh[1];
    if (blockIdx.z == 2) suh = jobs.suh[2], xh = jobs.xh[2];
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

// One layer of a launch: its blocks are [start, start + (N / 128) * SK) of the grid, split-major like a (N / 128, SK)
// grid. All layers of a launch share M, K, the width, codebook and warps a block; each keeps its own K splits.
struct Job {
    const half* xh;
    const uint32_t* T;
    long long stride_k, stride_nb;
    const half* svh;
    const half* bias;
    void* y;
    float* Z;
    int* counters;
    int y_dtype, N, SK, start;
};

struct Jobs {
    Job j[MAX_JOBS];
    int n;
};

// G = 0: the original walk, the window's rows as the mma's A (m16: rows g and g + 8 of 16, so 2 mma a tile and 64
// accumulators whatever the rows) and the decoded tile as B. G = 1, 2 ("transposed"): the decoded tile as A (its 16
// columns as m16; the B fragments decode_lane makes are the A fragment of W^T as they stand) and G groups of 8 rows as
// B (n8): one mma a tile and 32 accumulators for up to 8 rows. Every output is the same dot product over the same k
// steps; mma.m16n8k16 gives D[i][j] of A @ B and (B^T @ A^T)[j][i] the same bits (checked on sm_121 by
// tests/cuda/test_exl3_linear_loads.py against G = 0), and the sums after it are unchanged, so G only moves work.
// G > 0 also loads the next k step's rows of x a step ahead.
// V (G > 0, 3 to 6 bits): a k step's words (contiguous in both layouts) come as 16-byte non-coherent loads (whole
// 128-byte lines a warp instruction, instead of a lane's 32-bit words), V steps ahead, through a per-warp staging area in
// shared memory the lanes read their words of each tile back from (only V = 1 is launched: deeper measured no faster).
template <int K2>
__host__ __device__ constexpr bool vec_ok() {
    return !step_shuffled<K2>() && K2 <= 12;
}
template <int K2>
__host__ __device__ constexpr int step_words() {
    return 8 * tile_words<K2>();
}

__device__ __forceinline__ uint4 ldg_nc_v4(const uint32_t* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

template <int K2, int CB, int WK, int G, int V>
__global__ void __launch_bounds__(WK * 32) linear_kernel(
    const __grid_constant__ Jobs jobs, int M, int K) {
    constexpr int TW = tile_words<K2>();
    constexpr int LW = lane_words<K2>();
    constexpr int NH = G == 0 ? 2 : G;                        // mma a tile
    extern __shared__ __align__(16) float red[];              // WK * RH * 128 floats
    __shared__ int last;
    const int RH = min(M, 8);                                 // rows of red a warp
    griddep_launch();                                         // a PDL successor may launch (it waits for all of this)

    int ji = 0;
#pragma unroll
    for (int q = 1; q < MAX_JOBS; ++q)
        if (q < jobs.n && (int)blockIdx.x >= jobs.j[q].start) ji = q;
    const Job& J = jobs.j[ji];
    const half* __restrict__ xh = J.xh;
    const uint32_t* __restrict__ T = J.T;
    const long long stride_k = J.stride_k, stride_nb = J.stride_nb;
    const half* __restrict__ svh = J.svh;
    const half* __restrict__ bias = J.bias;
    void* __restrict__ y = J.y;
    float* __restrict__ Z = J.Z;
    int* __restrict__ counters = J.counters;
    const int y_dtype = J.y_dtype, N = J.N, SK = J.SK, NB = N >> 7;
    const int local = (int)blockIdx.x - J.start, nb = local % NB, split = local / NB;
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

        float acc[8][NH][4];
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int h = 0; h < NH; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

        // the walk's rows for the mma, clamped so rows past the pass read inside the buffer (their outputs are dropped)
        const int r0 = m0 + (g < R ? g : R - 1), r1 = m0 + (g + 8 < R ? g + 8 : R - 1);
        const half* x0 = xh + (size_t)r0 * K;
        const half* x1 = xh + (size_t)r1 * K;
        const uint32_t* tile = tiles + (size_t)kt0 * stride_k;
        // up to 6 bits the next k step's words are loaded while this one is decoded; 7 and 8 bits load per tile
        constexpr bool PF = K2 <= 12 && V == 0;
        constexpr int SR = step_regs<K2>();
        uint32_t cur[PF ? SR : 1], nxt[PF ? SR : 1];
        if constexpr (PF) load_step<K2>(tile, lane, cur);   // the weights do not wait for the previous kernel
        constexpr int SW = step_words<K2>(), CH = SW / 4, NV = V ? (CH + 31) / 32 : 1;   // 16-byte pieces of a step
        uint4 vq[V + 1][NV];                         // V > 0: steps i .. i + V, step i in vq[0]
        uint32_t* stage = reinterpret_cast<uint32_t*>(red) + warp * SW;
        auto load_vec = [&](const uint32_t* step, uint4 (&v)[NV]) {
#pragma unroll
            for (int c = 0; c < NV; ++c)
                if (CH % 32 == 0 || c * 32 + lane < CH) v[c] = ldg_nc_v4(step + 4 * (c * 32 + lane));
        };
        if constexpr (V > 0) {
#pragma unroll
            for (int d = 0; d < V; ++d)
                if (d < per_warp) load_vec(tile + (size_t)d * stride_k, vq[d]);
        }
        if (pass == 0) griddep_wait();                       // x (and Z, y, counters) may be the previous kernel's

        // x's fragments of k step kt: rows g (and g + 8): [0] [2] k 2t.., 2t + 8..; [1] [3] the same of row g + 8
        auto load_x = [&](int kt, uint32_t (&a)[4]) {
            a[0] = __ldg(reinterpret_cast<const uint32_t*>(x0 + kt * 16 + 2 * t));
            a[2] = __ldg(reinterpret_cast<const uint32_t*>(x0 + kt * 16 + 2 * t + 8));
            if (G != 1) {
                a[1] = __ldg(reinterpret_cast<const uint32_t*>(x1 + kt * 16 + 2 * t));
                a[3] = __ldg(reinterpret_cast<const uint32_t*>(x1 + kt * 16 + 2 * t + 8));
            } else {
                a[1] = a[3] = 0u;
            }
        };
        uint32_t an[4];
        if constexpr (G > 0) load_x(kt0, an);
#pragma unroll 1
        for (int i = 0; i < per_warp; ++i) {
            const int kt = kt0 + i;
            uint32_t a[4];
            if constexpr (G > 0) {
#pragma unroll
                for (int c = 0; c < 4; ++c) a[c] = an[c];
                load_x(i + 1 < per_warp ? kt + 1 : kt, an);
            } else {
                load_x(kt, a);
            }
            if constexpr (PF) {
                if (i + 1 < per_warp) load_step<K2>(tile + (size_t)(i + 1) * stride_k, lane, nxt);
            }
            if constexpr (V > 0) {
                __syncwarp();                        // every lane has read the previous step back
#pragma unroll
                for (int c = 0; c < NV; ++c)
                    if (CH % 32 == 0 || c * 32 + lane < CH)
                        *reinterpret_cast<uint4*>(stage + 4 * (c * 32 + lane)) = vq[0][c];
                if (i + V < per_warp) load_vec(tile + (size_t)(i + V) * stride_k, vq[V]);
                __syncwarp();
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                uint32_t w[LW];
                if constexpr (V > 0) load_lane_words<K2>(stage + j * TW, lane, w);
                else if constexpr (PF) step_lane_words<K2>(cur, j, lane, prev, w);
                else ldg_lane_words<K2>(tile + (size_t)i * stride_k + j * TW, lane, w);
                uint32_t b0[2], b1[2];
                decode_lane<K2, CB>(w, lane, b0, b1);
                if constexpr (G == 0) {
                    mma16816(acc[j][0], a, b0);
                    mma16816(acc[j][1], a, b1);
                } else {
                    const uint32_t wa[4] = {b0[0], b1[0], b0[1], b1[1]};   // W^T: columns g, g + 8; k 2t.., 2t + 8..
#pragma unroll
                    for (int q = 0; q < G; ++q) {
                        const uint32_t xb[2] = {a[q], a[q + 2]};             // rows 8q + g; k 2t.., 2t + 8..
                        mma16816(acc[j][q], wa, xb);
                    }
                }
            }
            if constexpr (PF) {
#pragma unroll
                for (int q = 0; q < SR; ++q) cur[q] = nxt[q];
            }
            if constexpr (V > 0) {
#pragma unroll
                for (int d = 0; d < V; ++d)
#pragma unroll
                    for (int c = 0; c < NV; ++c) vq[d][c] = vq[d + 1][c];
            }
        }

        // the warps' sums, added in warp order, rows 0-7 of the pass and then rows 8-15
        for (int rlo = 0; rlo < R; rlo += 8) {
            const int rn = min(R - rlo, 8);
            __syncthreads();                         // red is reused by every half and pass
            if constexpr (G == 0) {
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
            } else {
                // acc[i][q]: column 16 i + g (+ 8 in [2] [3]), rows 8q + 2t ([0] [2]) and 8q + 2t + 1 ([1] [3])
#pragma unroll
                for (int q = 0; q < G; ++q) {
                    if (q * 8 != rlo) continue;
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        const int r = 2 * t + e;
                        if (r < RH) {
#pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                red[(warp * RH + r) * 128 + i * 16 + g] = acc[i][q][e];
                                red[(warp * RH + r) * 128 + i * 16 + g + 8] = acc[i][q][2 + e];
                            }
                        }
                    }
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

int dtype_of(const at::Tensor& t) {
    return t.scalar_type() == at::kFloat ? F32 : t.scalar_type() == at::kBFloat16 ? BF16 : F16;
}

}  // namespace

#define TF_EXL3_WIDTHS(X, CB) X(2, CB) X(4, CB) X(6, CB) X(8, CB) X(10, CB) X(12, CB) X(14, CB) X(16, CB)
#define TF_EXL3_ALL(X) TF_EXL3_WIDTHS(X, 0) TF_EXL3_WIDTHS(X, 1) TF_EXL3_WIDTHS(X, 2) X(3, 2) X(5, 2) X(7, 2)

static void launch_pdl(cudaLaunchConfig_t& config, bool pdl, cudaLaunchAttribute (&attr)[1]) {
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    config.attrs = attr;
    config.numAttrs = 1;
}

void exl3_rot_in_cuda(const at::Tensor& x, const std::vector<at::Tensor>& suh, std::vector<at::Tensor>& xh,
                      int64_t pdl) {
    const int M = (int)x.size(0), K = (int)x.size(1), n = (int)suh.size();
    TORCH_CHECK(n >= 1 && n <= MAX_JOBS && (int)xh.size() == n, "rot_in: 1 to 3 (suh, xh) pairs");
    RotJobs jobs = {};
    for (int i = 0; i < n; ++i) {
        jobs.suh[i] = reinterpret_cast<const half*>(suh[i].data_ptr());
        jobs.xh[i] = reinterpret_cast<half*>(xh[i].data_ptr());
    }
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3((unsigned)((K / 128 + 3) / 4), (unsigned)M, (unsigned)n);
    config.blockDim = dim3(128);
    config.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr[1];
    launch_pdl(config, pdl, attr);
    C10_CUDA_CHECK(cudaLaunchKernelEx(&config, rot_in_kernel, x.data_ptr(), (int)dtype_of(x), jobs, K));
}

namespace {

template <int K2, int CB, int WK, int G, int V>
void launch_wk(const Jobs& jobs, int blocks, cudaStream_t stream, bool pdl, int M, int K) {
    auto kernel = linear_kernel<K2, CB, WK, G, V>;
    const int smem = (int)std::max(WK * std::min(M, 8) * 128 * sizeof(float),
                                   V > 0 ? WK * step_words<K2>() * sizeof(uint32_t) : (size_t)0);
    if (smem > 48 * 1024) cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3((unsigned)blocks);
    config.blockDim = dim3((unsigned)(WK * 32));
    config.dynamicSmemBytes = smem;
    config.stream = stream;
    cudaLaunchAttribute attr[1];
    launch_pdl(config, pdl, attr);
    C10_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, jobs, M, K));
}

template <int K2, int CB, int G, int V = 0>
void launch_g(const Jobs& jobs, int blocks, cudaStream_t stream, bool pdl, int M, int K, int WK) {
    if (WK == 2) launch_wk<K2, CB, 2, G, V>(jobs, blocks, stream, pdl, M, K);
    else if (WK == 4) launch_wk<K2, CB, 4, G, V>(jobs, blocks, stream, pdl, M, K);
    else launch_wk<K2, CB, 8, G, V>(jobs, blocks, stream, pdl, M, K);
}

// loads & 1: the transposed walk (G = 1 up to 8 rows, else 2), else the original (G = 0); loads & 2 (with 1): 16-byte
// weight loads a step ahead (V = 1, 3 to 6 bits); loads & 32: PDL launch. No choice changes a bit of any output.
template <int K2, int CB>
void launch(const Jobs& jobs, int blocks, cudaStream_t stream, int M, int K, int WK, int loads) {
    const bool pdl = loads & 32;
    if (!(loads & 1))
        launch_g<K2, CB, 0>(jobs, blocks, stream, pdl, M, K, WK);
    else if (vec_ok<K2>() && (loads & 2)) {
        if (M <= 8)
            launch_g<K2, CB, 1, vec_ok<K2>()>(jobs, blocks, stream, pdl, M, K, WK);
        else
            launch_g<K2, CB, 2, vec_ok<K2>()>(jobs, blocks, stream, pdl, M, K, WK);
    } else if (M <= 8)
        launch_g<K2, CB, 1>(jobs, blocks, stream, pdl, M, K, WK);
    else
        launch_g<K2, CB, 2>(jobs, blocks, stream, pdl, M, K, WK);
}

}  // namespace

void exl3_linear_cuda(const std::vector<at::Tensor>& xh, const std::vector<at::Tensor>& T,
                      const std::vector<int64_t>& stride_k, const std::vector<int64_t>& stride_nb,
                      const std::vector<at::Tensor>& svh, const std::vector<at::Tensor>& bias,
                      std::vector<at::Tensor>& y, const std::vector<at::Tensor>& Z, std::vector<at::Tensor>& counters,
                      int64_t K2, int64_t cb, const std::vector<int64_t>& SK, int64_t WK, int64_t loads) {
    const int n = (int)xh.size();
    const int M = (int)xh[0].size(0), K = (int)xh[0].size(1);
    TORCH_CHECK(WK == 2 || WK == 4 || WK == 8, "WK must be 2, 4 or 8");
    Jobs jobs = {};
    jobs.n = n;
    int blocks = 0;
    for (int i = 0; i < n; ++i) {
        const int N = (int)y[i].size(1);
        TORCH_CHECK((K / 16) % (SK[i] * WK) == 0, "K / 16 must split evenly over SK * WK warps");
        TORCH_CHECK(SK[i] == 1 || Z[i].numel() > 0, "Z is needed with more than one split");
        Job& J = jobs.j[i];
        J.xh = reinterpret_cast<const half*>(xh[i].data_ptr());
        J.T = reinterpret_cast<const uint32_t*>(T[i].data_ptr());
        J.stride_k = stride_k[i];
        J.stride_nb = stride_nb[i];
        J.svh = reinterpret_cast<const half*>(svh[i].data_ptr());
        J.bias = bias[i].numel() ? reinterpret_cast<const half*>(bias[i].data_ptr()) : nullptr;
        J.y = y[i].data_ptr();
        J.y_dtype = dtype_of(y[i]);
        J.Z = Z[i].numel() ? Z[i].data_ptr<float>() : nullptr;
        J.counters = counters[i].data_ptr<int>();
        J.N = N;
        J.SK = (int)SK[i];
        J.start = blocks;
        blocks += (N / 128) * (int)SK[i];
    }
    auto stream = at::cuda::getCurrentCUDAStream();
#define TF_LAUNCH(K2_, CB_)                                                     \
    if (K2 == K2_ && cb == CB_) {                                            \
        launch<K2_, CB_>(jobs, blocks, stream, M, K, (int)WK, (int)loads);   \
        return;                                                              \
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
