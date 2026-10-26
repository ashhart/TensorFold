// exl3_experts.cu's tf_exl3_grouped_2 with 32-member tiles: trellis words decoded once for two 16-member groups.

#include <cstdint>
#include <cuda_fp16.h>

#include "../exl3_decode.cuh"

namespace muse_x2 {

using namespace tf_exl3;

constexpr int NT = 8, W = 4;  // the shared grouped kernel's tile: 8 n tiles a block, 4 warps

__device__ __forceinline__ uint32_t pair(const half* x, bool ok) { return ok ? *reinterpret_cast<const uint32_t*>(x) : 0u; }

// One warp's k tiles [kt0, kt0 + nkt) for both groups: tf_exl3's warp_tiles, the second group's mma beside the first.
template <int K2>
__device__ __forceinline__ void warp_tiles2(const uint32_t* T, int ntiles, int kt0, int nkt, int nt0,
                                            const half* const (&x0)[2], const half* const (&x1)[2],
                                            const bool (&ok0)[2], const bool (&ok1)[2], bool two, int lane,
                                            float (&acc)[2][NT][2][4]) {
    constexpr int TW = tile_words<K2>(), LW = lane_words<K2>();
    const size_t kstride = (size_t)ntiles * TW;
    for (int i = 0; i < nkt; ++i) {
        const uint32_t* t0 = T + (size_t)(kt0 + i) * kstride + (size_t)nt0 * TW;
        uint32_t w[NT][LW];
#pragma unroll
        for (int j = 0; j < NT; ++j) ldg_lane_words<K2>(t0 + j * TW, lane, w[j]);
        const int k = (kt0 + i) * 16;
        uint32_t a[2][4];
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            a[q][0] = pair(x0[q] + k, ok0[q]);
            a[q][1] = pair(x1[q] + k, ok1[q]);
            a[q][2] = pair(x0[q] + k + 8, ok0[q]);
            a[q][3] = pair(x1[q] + k + 8, ok1[q]);
        }
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            uint32_t b0[2], b1[2];
            decode_lane<K2, 2>(w[j], lane, b0, b1);
            mma16816(acc[0][j][0], a[0], b0);
            mma16816(acc[0][j][1], a[0], b1);
            if (two) {
                mma16816(acc[1][j][0], a[1], b0);
                mma16816(acc[1][j][1], a[1], b1);
            }
        }
    }
}

}  // namespace muse_x2

// Grid (nexp_max, N / 128, mats * SK * MT2), MT2 = (maxm + 31) / 32; Z as tf_exl3_grouped_2 writes it.
extern "C" __global__ void __launch_bounds__(128) muse_grouped2(
    const half* __restrict__ X0, const half* __restrict__ X1, const int64_t* __restrict__ TP0,
    const int64_t* __restrict__ TP1, const int* __restrict__ K2_0, const int* __restrict__ K2_1,
    const int* __restrict__ uids, const int* __restrict__ ucount, const int* __restrict__ members,
    float* __restrict__ Z, int K, int N, int P, int SK, int maxm, int slots) {
    using namespace muse_x2;
    const int u = blockIdx.x;
    if (u >= ucount[0]) return;
    const int MT = (maxm + 31) / 32;
    const int mtile = blockIdx.z % MT;
    const int split = (blockIdx.z / MT) % SK;
    const int mat = blockIdx.z / MT / SK;
    const half* X = mat ? X1 : X0;
    const int e = uids[u];
    const uint32_t* T = reinterpret_cast<const uint32_t*>(mat ? TP1[e] : TP0[e]);
    const int k2 = mat ? K2_1[e] : K2_0[e];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;

    __shared__ int rows_sh[32];
    if (threadIdx.x < 32) {
        const int m = mtile * 32 + threadIdx.x;
        const int code = m < maxm ? members[u * maxm + m] : -1;
        rows_sh[threadIdx.x] = code >= 0 ? (code >> 5) * slots + (code & 31) : -1;
    }
    __syncthreads();
    if (rows_sh[0] < 0) return;  // members come first, so this tile is empty
    const bool two = rows_sh[16] >= 0;
    const half* x0[2];
    const half* x1[2];
    bool ok0[2], ok1[2];
#pragma unroll
    for (int q = 0; q < 2; ++q) {
        const int r0 = rows_sh[16 * q + g], r1 = rows_sh[16 * q + g + 8];
        x0[q] = X + (size_t)(r0 < 0 ? 0 : r0) * K + 2 * t;
        x1[q] = X + (size_t)(r1 < 0 ? 0 : r1) * K + 2 * t;
        ok0[q] = r0 >= 0;
        ok1[q] = r1 >= 0;
    }
    const int per_warp = (K >> 4) / SK / W;
    const int kt0 = split * (per_warp * W) + warp * per_warp;
    const int nt0 = blockIdx.y * NT;

    float acc[2][NT][2][4];
#pragma unroll
    for (int q = 0; q < 2; ++q)
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[q][i][h][c] = 0.f;

    switch (k2) {
#define MUSE_X2_CASE(K2_) \
    case K2_: warp_tiles2<K2_>(T, N >> 4, kt0, per_warp, nt0, x0, x1, ok0, ok1, two, lane, acc); break;
        MUSE_X2_CASE(2) MUSE_X2_CASE(3) MUSE_X2_CASE(4) MUSE_X2_CASE(5) MUSE_X2_CASE(6) MUSE_X2_CASE(7)
        MUSE_X2_CASE(8) MUSE_X2_CASE(9) MUSE_X2_CASE(10) MUSE_X2_CASE(11) MUSE_X2_CASE(12) MUSE_X2_CASE(13)
        MUSE_X2_CASE(14) MUSE_X2_CASE(15) MUSE_X2_CASE(16)
#undef MUSE_X2_CASE
        default: __trap();
    }

    // each group's warps' sums through shared memory, added in warp order (tf_exl3_grouped_2's)
    __shared__ __align__(16) float red[W][16][NT * 16];
#pragma unroll
    for (int q = 0; q < 2; ++q) {
        if (q == 1 && !two) break;
        if (q) __syncthreads();
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int col = i * 16 + h * 8 + 2 * t;
                red[warp][g][col] = acc[q][i][h][0];
                red[warp][g][col + 1] = acc[q][i][h][1];
                red[warp][g + 8][col] = acc[q][i][h][2];
                red[warp][g + 8][col + 1] = acc[q][i][h][3];
            }
        __syncthreads();
        for (int idx = threadIdx.x; idx < 16 * NT * 16; idx += W * 32) {
            const int row = idx / (NT * 16), col = idx % (NT * 16);
            const int r = rows_sh[16 * q + row];
            if (r < 0) continue;
            float s = red[0][row][col];
#pragma unroll
            for (int w = 1; w < W; ++w) s += red[w][row][col];
            Z[(((size_t)mat * SK + split) * P + r) * N + nt0 * 16 + col] = s;
        }
    }
}
