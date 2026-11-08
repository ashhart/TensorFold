// Router logits on mma tiles for Flash Next and GLM: each logit (or K slice's partial) one mma chain from zero.
#pragma once

#include <cuda_bf16.h>
#include <stdint.h>

#include "common/qmm_frag.cuh"

namespace tf_router_mma {

// OUT[m, e] = x[m] . w[e], 8 experts a warp, RT tiles of 16 rows, one mma chain over K, STAGES of KC (row sums fixed).
template <int WARPS, int STAGES, int KC, int RT = 1, bool EFAST = false, bool SLICED = false>  // EFAST: x walks the expert blocks
__device__ __forceinline__ void logits(const __nv_bfloat16* __restrict__ X, int x_ld, const __nv_bfloat16* __restrict__ W,
                                       float* __restrict__ OUT, int M, int D, int E, int SK = 1) {
    constexpr int LD = KC + 8, EB = 8 * WARPS, XR = 16 * RT, ROWS = XR + EB;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto tile = reinterpret_cast<__nv_bfloat16 (*)[ROWS][LD]>(smem_raw);  // [STAGES] XR input rows, then EB expert rows
    const int K = SLICED ? D / SK : D;  // SLICED: block z sums K slice z into OUT[z][m][e]; w rows stay D apart
    if (SLICED) {
        X += static_cast<long long>(blockIdx.z) * K;
        W += static_cast<long long>(blockIdx.z) * K;
        OUT += static_cast<long long>(blockIdx.z) * M * E;
    }
    const int m0 = (EFAST ? blockIdx.y : blockIdx.x) * XR, e0 = (EFAST ? blockIdx.x : blockIdx.y) * EB;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int stages = (K + KC - 1) / KC;
    const int xr = min(XR, M - m0);  // input rows past M stay zero: written once, never staged
    for (int v = threadIdx.x; v < STAGES * (XR - xr) * (KC / 8); v += blockDim.x) {
        const int per = (XR - xr) * (KC / 8);
        *reinterpret_cast<uint4*>(&tile[v / per][xr + (v % per) / (KC / 8)][8 * (v % (KC / 8))]) = make_uint4(0, 0, 0, 0);
    }
    auto stage = [&](int st) {
        const int k0 = st * KC, buf = st % STAGES;
        for (int v = threadIdx.x; v < (xr + EB) * (KC / 8); v += blockDim.x) {
            const int i = v / (KC / 8), k = k0 + 8 * (v % (KC / 8));
            const bool x_row = i < xr;
            const int row = x_row ? i : XR + i - xr;
            const int r = x_row ? m0 + i : e0 + i - xr;
            const bool read = k < K && (x_row || r < E);
            const __nv_bfloat16* src = x_row ? X + static_cast<long long>(r) * x_ld + (read ? k : 0)
                                             : W + static_cast<long long>(read ? r : 0) * D + (read ? k : 0);
            qmm_frag::cp16z(&tile[buf][row][8 * (v % (KC / 8))], src, read);
        }
    };
    float acc[RT][4] = {};
    for (int st = 0; st < STAGES - 1; ++st) {
        if (st < stages) stage(st);
        qmm_frag::commit();
    }
    for (int st = 0; st < stages; ++st) {
        if (st + STAGES - 1 < stages) stage(st + STAGES - 1);
        qmm_frag::commit();
        qmm_frag::wait<STAGES - 1>();
        __syncthreads();
        const int buf = st % STAGES;
        const int steps = min(KC, K - st * KC) / 16;
        for (int kk = 0; kk < steps; ++kk) {
            uint32_t b[2];
            qmm_frag::ldmatrix2(b, &tile[buf][XR + warp * 8 + lane % 8][kk * 16 + ((lane / 8) % 2) * 8]);
#pragma unroll
            for (int i = 0; i < RT; ++i) {
                uint32_t a[4];
                qmm_frag::ldmatrix4(a, &tile[buf][i * 16 + lane % 16][kk * 16 + (lane / 16) * 8]);
                qmm_frag::mma(acc[i], a, b[0], b[1]);
            }
        }
        __syncthreads();
    }
    const int e = e0 + warp * 8 + 2 * t;
#pragma unroll
    for (int i = 0; i < RT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = m0 + i * 16 + g + 8 * h;
            if (r >= M) continue;
            if (e < E) OUT[static_cast<long long>(r) * E + e] = acc[i][2 * h];
            if (e + 1 < E) OUT[static_cast<long long>(r) * E + e + 1] = acc[i][2 * h + 1];
        }
}

}  // namespace tf_router_mma
