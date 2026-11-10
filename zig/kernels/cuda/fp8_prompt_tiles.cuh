// Optional geometry variants of TensorFold native FP8 prompt products; arithmetic is unchanged.
#pragma once
#include "fp8_prefill_common.cuh"

namespace tf_fp8_tiles {
template<int BM, int BN, int WM, int WN>
constexpr int shared_bytes = 2 * (BM + BN) * 128;

template<int BM, int BN, int WM, int WN, bool F32, bool RowScaled>
__device__ __forceinline__ void product(const uint8_t* x, const uint8_t* weight, void* output,
    int m, int n, int k, int npad, float alpha, const float* row_scales, const float* weight_scales) {
    constexpr int BK = 128, MT = BM / WM / 16, NT = BN / WN / 8, NTILES = BN / 64, THREADS = 256;
    static_assert(WM * WN == 8 && BM % (WM * 16) == 0 && BN % (WN * 8) == 0 && BN % 64 == 0);
    struct Shared { alignas(16) uint8_t a[2][BM * BK], b[2][BN * BK]; };
    static_assert(sizeof(Shared) == shared_bytes<BM, BN, WM, WN>);
    extern __shared__ __align__(16) uint8_t storage[];
    Shared& shared = *reinterpret_cast<Shared*>(storage);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int row_tiles = (m + BM - 1) / BM;
    const int tile_id = blockIdx.y * gridDim.x + blockIdx.x;
    const int m0 = (tile_id % row_tiles) * BM, n0 = (tile_id / row_tiles) * BN;
    const int kgroups = k / 64, tiles = (k + BK - 1) / BK;
    auto load = [&](int stage, int tile) {
        for (int task = tid; task < BM * (BK / 16); task += THREADS) {
            const int row = task / (BK / 16), column = task % (BK / 16);
            const int source_k = tile * BK + column * 16;
            const bool valid = m0 + row < m && source_k < k;
            qmm_frag::cp16z(shared.a[stage] + row * BK + tf_fp8_fast::chunk(row, column) * 16,
                x + size_t(valid ? m0 + row : 0) * k + (valid ? source_k : 0), valid);
        }
        for (int task = tid; task < BN * BK / 16; task += THREADS) {
            const int square = task / 256, within = task % 256;
            const int half = square / NTILES, ntile = square % NTILES;
            const int column = n0 + ntile * 64, kg = tile * 2 + half;
            const bool valid = column < npad && kg < kgroups;
            const size_t index = (size_t(valid ? column / 64 : 0) * kgroups + (valid ? kg : 0)) * 4096 + within * 16;
            qmm_frag::cp16z(shared.b[stage] + task * 16, weight + index, valid);
        }
    };
    float sums[MT][NT][4] = {};
#pragma unroll
    for (int stage = 0; stage < 2; ++stage) {
        if (stage < tiles) load(stage, stage);
        qmm_frag::commit();
    }
    for (int tile = 0; tile < tiles; ++tile) {
        qmm_frag::wait<1>(); __syncthreads();
        const int stage = tile & 1;
        // These K32 steps and each accumulator's MMA sequence are identical to the original kernel.
#pragma unroll
        for (int step = 0; step < BK / 32; ++step) {
            uint32_t a[MT][4];
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                const int row = (warp / WN) * (BM / WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                qmm_frag::ldmatrix4(a[i], shared.a[stage] + row * BK + tf_fp8_fast::chunk(row, step * 2 + (lane >> 4)) * 16);
            }
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                const int ntile8 = (warp % WN) * NT + j;
                const int square = (step / 2) * NTILES + ntile8 / 8;
                const uint2 b = *reinterpret_cast<const uint2*>(shared.b[stage] + square * 4096 +
                    (ntile8 % 8) * 512 + lane * 16 + (step & 1) * 8);
#pragma unroll
                for (int i = 0; i < MT; ++i) tf_fp8_fast::mma(sums[i][j], a[i], b.x, b.y);
            }
        }
        __syncthreads();
        if (tile + 2 < tiles) load(stage, tile + 2);
        qmm_frag::commit();
    }
    qmm_frag::wait<0>();
    tf_fp8_fast::store_tile<BM, WN, MT, NT, F32, RowScaled>(sums, output, m0, n0, m, n, alpha,
        row_scales, weight_scales, k / 64);
}
} // namespace tf_fp8_tiles
