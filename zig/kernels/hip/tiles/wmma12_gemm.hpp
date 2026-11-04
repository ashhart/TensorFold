#pragma once

// gfx12 WMMA BF16 GEMM tile: y = x . W^T for BF16 x (m x k) and BF16 W (n x k, as stored), fp32 sums.
// Two block geometries (W12Geo), picked by the product's size; whatever the block, each output adds its k products in k
// order, 16 a WMMA step into one fp32 accumulator, so a row's bytes depend on neither m nor the geometry.

#include "common/vec.hpp"
#include "common/wmma.hpp"

namespace tf {
namespace rocm {

// A block of BM x BN outputs, WM x WN waves of (BM / WM) x (BN / WN), K steps of BK through double-buffered LDS.
template <int BM_, int BN_, int BK_, int WM_, int WN_>
struct W12Geo {
    static constexpr int BM = BM_, BN = BN_, BK = BK_, WM = WM_, WN = WN_;
    static constexpr int Ld = BK + 8;  // halves a staged row: 16-byte aligned, the pad spreads the readers' banks
    static constexpr int Threads = 32 * WM * WN;
    static_assert(BM % (16 * WM) == 0 && BN % (16 * WN) == 0, "a wave's sub-tile is whole 16 x 16 products");
    static_assert(BK % 16 == 0, "a K step is whole 16-deep products");
    static_assert(2 * (BM + BN) * Ld * 2 <= 64 * 1024, "double-buffered x and W rows fit 64 KiB of LDS");
};

// Few rows or narrow outputs take the small block so the grid fills the GPU; launches/dense.zig mirrors both shapes.
using W12Small = W12Geo<16, 64, 64, 1, 4>;
using W12Big = W12Geo<128, 128, 32, 2, 4>;
constexpr int kW12Band = 4;  // row blocks a band of the launch order walks the column blocks with

typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));

// The served step, a 16-deep BF16 product into fp32 sums; the contract's control swaps in a less exact one.
struct W12Step {
#if TF_DEVICE_WMMA_GFX12
    __device__ static f32x8 mma(bf16x8 a, bf16x8 b, f32x8 c) {
        return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
    }
#endif
};

// Block (x, y) in launch order takes tile (bx, by): bands of kW12Band row blocks share each column block's weights.
__device__ __forceinline__ void w12_band(int& bx, int& by) {
    const int cols = gridDim.x, rows = gridDim.y;
    const int b = blockIdx.y * cols + blockIdx.x;
    const int first = b / (kW12Band * cols) * kW12Band;
    const int in_band = b % (kW12Band * cols);
    const int height = min(kW12Band, rows - first);
    by = first + in_band % height;
    bx = in_band / height;
}

// One G::BM x G::BN block. gfx12 wave32 operands: A lane l holds row l % 16 with k = 8 (l / 16) + i, B lane l column
// l % 16 with the same k; D lane l, element i is row 8 (l / 16) + i, column l % 16.
// Rows past m stage row 0 and columns past n the last column; neither is stored.
template <class G, class Act, class Epi, class Step = W12Step, class Args>
__device__ __forceinline__ void wmma12_gemm_tile(const Args& a) {
#if TF_DEVICE_WMMA_GFX12
    constexpr int kW12M = G::BM, kW12N = G::BN, kW12K = G::BK, kW12Ld = G::Ld;
    constexpr int kW12WavesM = G::WM, kW12WavesN = G::WN, kW12Threads = G::Threads;
    constexpr int TM = kW12M / kW12WavesM / 16, TN = kW12N / kW12WavesN / 16;  // 16 x 16 products a wave
    constexpr int XP = kW12M * kW12K / 8, WP = kW12N * kW12K / 8;               // 16-byte pieces a stage
    constexpr int XPT = (XP + kW12Threads - 1) / kW12Threads, WPT = (WP + kW12Threads - 1) / kW12Threads;
    constexpr int PR = kW12K / 8;                                                // pieces a staged row
    __shared__ __attribute__((aligned(16))) unsigned short xs[2][kW12M * kW12Ld];
    __shared__ __attribute__((aligned(16))) unsigned short ws[2][kW12N * kW12Ld];

    int bx, by;
    w12_band(bx, by);
    const int m0 = by * kW12M, n0 = bx * kW12N;
    if (m0 >= a.m) return;
    const int tid = threadIdx.x, lane = tid & 31;
    const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
    const int wr = wave / kW12WavesN, wc = wave % kW12WavesN;

    // Each thread's 16-byte pieces of a stage: row r, piece c of x; column r, piece c of W.
    const u32x4* xsrc[XPT];
    const u32x4* wsrc[WPT];
#pragma unroll
    for (int q = 0; q < XPT; ++q) {
        const int p = min(tid + q * kW12Threads, XP - 1);
        const int r = m0 + p / PR;
        xsrc[q] = reinterpret_cast<const u32x4*>(Act::row(a, r < a.m ? r : 0)) + p % PR;
    }
#pragma unroll
    for (int q = 0; q < WPT; ++q) {
        const int p = min(tid + q * kW12Threads, WP - 1);
        const int c = n0 + p / PR;
        wsrc[q] = reinterpret_cast<const u32x4*>(a.words + static_cast<long long>(c < a.n ? c : a.n - 1) * (a.k / 2)) + p % PR;
    }
    u32x4 rx[XPT], rw[WPT];
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
            if (XP % kW12Threads == 0 || p < XP) *reinterpret_cast<u32x4*>(&xs[buf][(p / PR) * kW12Ld + (p % PR) * 8]) = rx[q];
        }
#pragma unroll
        for (int q = 0; q < WPT; ++q) {
            const int p = tid + q * kW12Threads;
            if (WP % kW12Threads == 0 || p < WP) *reinterpret_cast<u32x4*>(&ws[buf][(p / PR) * kW12Ld + (p % PR) * 8]) = rw[q];
        }
    };

    f32x8 acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = f32x8{};
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
            bf16x8 av[TM], bv[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) av[i] = *reinterpret_cast<const bf16x8*>(&xs[buf][(arow + 16 * i) * kW12Ld + kk + koff]);
#pragma unroll
            for (int j = 0; j < TN; ++j) bv[j] = *reinterpret_cast<const bf16x8*>(&ws[buf][(bcol + 16 * j) * kW12Ld + kk + koff]);
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = Step::mma(av[i], bv[j], acc[i][j]);
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
