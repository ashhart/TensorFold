#pragma once

// A frozen copy of the gfx12 WMMA dense tile's big block as served before #2264 (main cde4fe7d9,
// tiles/wmma12_gemm.hpp, W12Geo<128, 128, 32, 2, 4>): 32-deep steps, rows padded to 40 halves, __syncthreads().
// Only the contract's controls object includes it: the faster block replaces this one and must equal its bytes.

#include "common/arch.hpp"

namespace tf {
namespace rocm {

typedef __bf16 ref_bf16x8 __attribute__((ext_vector_type(8)));
typedef float ref_f32x8 __attribute__((ext_vector_type(8)));
typedef unsigned ref_u32x4 __attribute__((ext_vector_type(4)));

// Block (x, y) in launch order takes tile (bx, by): bands of 4 row blocks share each column block's weights.
__device__ __forceinline__ void k32_ref_band(int& bx, int& by) {
    const int cols = gridDim.x, rows = gridDim.y;
    const int b = blockIdx.y * cols + blockIdx.x;
    const int first = b / (4 * cols) * 4;
    const int in_band = b % (4 * cols);
    const int height = min(4, rows - first);
    by = first + in_band % height;
    bx = in_band / height;
}

template <class Act, class Epi, class Args>
__device__ __forceinline__ void wmma12_k32_ref_tile(const Args& a) {
#if TF_DEVICE_WMMA_GFX12
    constexpr int kW12M = 128, kW12N = 128, kW12K = 32, kW12Ld = 40;
    constexpr int kW12WavesM = 2, kW12WavesN = 4, kW12Threads = 256;
    constexpr int TM = kW12M / kW12WavesM / 16, TN = kW12N / kW12WavesN / 16;  // 16 x 16 products a wave
    constexpr int XP = kW12M * kW12K / 8, WP = kW12N * kW12K / 8;               // 16-byte pieces a stage
    constexpr int XPT = (XP + kW12Threads - 1) / kW12Threads, WPT = (WP + kW12Threads - 1) / kW12Threads;
    constexpr int PR = kW12K / 8;                                                // pieces a staged row
    __shared__ __attribute__((aligned(16))) unsigned short xs[2][kW12M * kW12Ld];
    __shared__ __attribute__((aligned(16))) unsigned short ws[2][kW12N * kW12Ld];

    int bx, by;
    k32_ref_band(bx, by);
    const int m0 = by * kW12M, n0 = bx * kW12N;
    if (m0 >= a.m) return;
    const int tid = threadIdx.x, lane = tid & 31;
    const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
    const int wr = wave / kW12WavesN, wc = wave % kW12WavesN;

    // Each thread's 16-byte pieces of a stage: row r, piece c of x; column r, piece c of W.
    const ref_u32x4* xsrc[XPT];
    const ref_u32x4* wsrc[WPT];
#pragma unroll
    for (int q = 0; q < XPT; ++q) {
        const int p = min(tid + q * kW12Threads, XP - 1);
        const int r = m0 + p / PR;
        xsrc[q] = reinterpret_cast<const ref_u32x4*>(Act::row(a, r < a.m ? r : 0)) + p % PR;
    }
#pragma unroll
    for (int q = 0; q < WPT; ++q) {
        const int p = min(tid + q * kW12Threads, WP - 1);
        const int c = n0 + p / PR;
        wsrc[q] = reinterpret_cast<const ref_u32x4*>(a.words + static_cast<long long>(c < a.n ? c : a.n - 1) * (a.k / 2)) + p % PR;
    }
    ref_u32x4 rx[XPT], rw[WPT];
    auto fetch = [&](int s) {
#pragma unroll
        for (int q = 0; q < XPT; ++q) rx[q] = xsrc[q][s * PR];
#pragma unroll
        for (int q = 0; q < WPT; ++q) rw[q] = wsrc[q][s * PR];
    };
    auto stash = [&](int buf) {
#pragma unroll
        for (int q = 0; q < XPT; ++q) {
            const int p = tid + q * kW12Threads;
            if (XP % kW12Threads == 0 || p < XP) *reinterpret_cast<ref_u32x4*>(&xs[buf][(p / PR) * kW12Ld + (p % PR) * 8]) = rx[q];
        }
#pragma unroll
        for (int q = 0; q < WPT; ++q) {
            const int p = tid + q * kW12Threads;
            if (WP % kW12Threads == 0 || p < WP) *reinterpret_cast<ref_u32x4*>(&ws[buf][(p / PR) * kW12Ld + (p % PR) * 8]) = rw[q];
        }
    };

    ref_f32x8 acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = ref_f32x8{};
    const int steps = a.k / kW12K;
    const int arow = wr * (kW12M / kW12WavesM) + (lane & 15), bcol = wc * (kW12N / kW12WavesN) + (lane & 15);
    const int koff = 8 * (lane >> 4);
    fetch(0);
    stash(0);
    __syncthreads();
    for (int s = 0; s < steps; ++s) {
        const int buf = s & 1;
        if (s + 1 < steps) fetch(s + 1);
#pragma unroll
        for (int kk = 0; kk < kW12K; kk += 16) {
            ref_bf16x8 av[TM], bv[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) av[i] = *reinterpret_cast<const ref_bf16x8*>(&xs[buf][(arow + 16 * i) * kW12Ld + kk + koff]);
#pragma unroll
            for (int j = 0; j < TN; ++j) bv[j] = *reinterpret_cast<const ref_bf16x8*>(&ws[buf][(bcol + 16 * j) * kW12Ld + kk + koff]);
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(av[i], bv[j], acc[i][j]);
        }
        if (s + 1 < steps) stash(buf ^ 1);
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int col = n0 + wc * (kW12N / kW12WavesN) + 16 * j + (lane & 15);
            if (col >= a.n) continue;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int row = m0 + wr * (kW12M / kW12WavesM) + 16 * i + koff + e;
                if (row < a.m) Epi::store(a, row, col, acc[i][j][e]);
            }
        }
    }
#else
    (void)a;
    __builtin_trap();
#endif
}

}  // namespace rocm
}  // namespace tf
