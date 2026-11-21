// The grouped EXL3 experts' projection (tf_exl3_grouped_*), shared with the contract's baseline and planted faults.
#pragma once

#include <cstdint>
#include <cuda_fp16.h>

#include "exl3_decode.cuh"

namespace tf_exl3 {

// One projection's grouped arguments; the fields must stay in launch.zig add order.
struct GroupedArgs {
    const half* x0; // fp16 [P, K], the rotated gate (or down) inputs
    const half* x1; // fp16 [P, K], the rotated up inputs
    const long long* tp0; // int64 [E], one trellis pointer per expert
    const long long* tp1;
    const int* k2_0; // int32 [E], half-bits a value
    const int* k2_1;
    const int* uids; // int32 [nexp_max], distinct routed experts in id order
    const int* ucount; // int32 [1]
    const int* members; // int32 [nexp_max, maxm], row * 32 + slot, -1 after the last member
    float* z; // fp32 partials, [(mat * sk + split) * P + p] * N + col
    int k, n, p, sk, maxm, slots, nexp_max, mats, ns, standard, skip;
};

// How a warp reads its k tiles: main's loop, or a tile ahead (sm_120 up); SKIP and STALE are the contract's faults.
enum Load : int { PLAIN = 0, AHEAD = 1, SKIP = 2, STALE = 3 };

// One warp's k tiles [kt0, kt0 + nkt) of an expert matrix for MT 16-member tiles: per-lane loads, the shared decode.
template <int CB, int K2, int NT, int LOAD, int MT = 1>
__device__ __forceinline__ void warp_tiles(const uint32_t* T, int ntiles, int kt0, int nkt, int nt0,
                                           const half* (&x0)[MT], const half* (&x1)[MT],
                                           const int (&r0)[MT], const int (&r1)[MT], int lane,
                                           float (&acc)[MT][NT][2][4]) {
    constexpr int TW = tile_words<K2>(), LW = lane_words<K2>();
    bool ok0[MT], ok1[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        ok0[m] = r0[m] >= 0;
        ok1[m] = r1[m] >= 0;
    }
    const size_t kstride = (size_t)ntiles * TW;
    const uint32_t* tb = T + (size_t)kt0 * kstride + (size_t)nt0 * TW;
    uint32_t nx[NT][LW];  // the next tile's words, in flight while this one decodes
    if (LOAD != PLAIN && nkt > 0)
#pragma unroll
        for (int j = 0; j < NT; ++j) ldg_lane_words<K2>(tb + j * TW, lane, nx[j]);
    for (int i = 0; i < nkt; ++i) {
        uint32_t w[NT][LW];
        if constexpr (LOAD == PLAIN) {
            const uint32_t* t0 = T + (size_t)(kt0 + i) * kstride + (size_t)nt0 * TW;
#pragma unroll
            for (int j = 0; j < NT; ++j) ldg_lane_words<K2>(t0 + j * TW, lane, w[j]);
        } else {
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int l = 0; l < LW; ++l) w[j][l] = nx[j][l];
            const int next = LOAD == SKIP ? (i + 2) % nkt : i + 1;  // SKIP: tile i + 2, in the warp's range
            const int last = LOAD == STALE ? nkt - 1 : nkt;  // STALE: the last tile decodes the one before's words
            if (i + 1 < last)
#pragma unroll
                for (int j = 0; j < NT; ++j) ldg_lane_words<K2>(tb + (size_t)next * kstride + j * TW, lane, nx[j]);
        }
        const int k = (kt0 + i) * 16;
        uint32_t a[MT][4];
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            a[m][0] = ok0[m] ? *reinterpret_cast<const uint32_t*>(x0[m] + k) : 0u;
            a[m][1] = ok1[m] ? *reinterpret_cast<const uint32_t*>(x1[m] + k) : 0u;
            a[m][2] = ok0[m] ? *reinterpret_cast<const uint32_t*>(x0[m] + k + 8) : 0u;
            a[m][3] = ok1[m] ? *reinterpret_cast<const uint32_t*>(x1[m] + k + 8) : 0u;
        }
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            uint32_t b0[2], b1[2];
            decode_lane<K2, CB>(w[j], lane, b0, b1);
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                mma16816(acc[m][j][0], a[m], b0);
                mma16816(acc[m][j][1], a[m], b1);
            }
        }
    }
}

// Program (expert u, n block, split, member tile): 16 MT members times W_q over the split's K range, in warp order.
template <int CB, int NT, int W, int LOAD, int MT = 1>
__device__ __forceinline__ void grouped_body(const GroupedArgs a, int u, int nblock, int zz) {
    if (u >= a.ucount[0]) return;
    constexpr int M = 16 * MT;  // members a program
    const int TILES = (a.maxm + M - 1) / M;
    const int mtile = zz % TILES;
    const int split = (zz / TILES) % a.sk;
    const int mat = zz / TILES / a.sk;
    const half* X = mat ? a.x1 : a.x0;
    const int e = a.standard ? a.uids[3 * u] : a.uids[u];
    if (a.standard && e == a.skip) return;
    const uint32_t* T = reinterpret_cast<const uint32_t*>(mat ? a.tp1[e] : a.tp0[e]);
    const int k2 = mat ? a.k2_1[e] : a.k2_0[e];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int NTILES = a.ns > 0 ? a.ns : (a.n >> 4); // row stride of the trellis, in tiles

    __shared__ int rows_sh[M];
    if (threadIdx.x < M) {
        const int m = mtile * M + threadIdx.x;
        const int count = a.standard ? a.uids[3 * u + 2] : a.maxm;
        const int first = a.standard ? a.uids[3 * u + 1] : u * a.maxm;
        const int code = m < count ? a.members[first + m] : -1;
        rows_sh[threadIdx.x] = a.standard ? code : (code >= 0 ? (code >> 5) * a.slots + (code & 31) : -1);
    }
    __syncthreads();
    if (rows_sh[0] < 0) return; // members come first, so this tile is empty
    int r0[MT], r1[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        r0[m] = rows_sh[16 * m + g];
        r1[m] = rows_sh[16 * m + g + 8];
    }
    const half* x0[MT];
    const half* x1[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        x0[m] = X + (size_t)(r0[m] < 0 ? 0 : r0[m]) * a.k + 2 * t;
        x1[m] = X + (size_t)(r1[m] < 0 ? 0 : r1[m]) * a.k + 2 * t;
    }
    const int per_warp = (a.k >> 4) / a.sk / W;
    const int kt0 = split * (per_warp * W) + warp * per_warp;
    const int nt0 = nblock * NT;

    float acc[MT][NT][2][4];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[m][i][h][c] = 0.f;

    __shared__ __align__(16) float red[W][M][NT * 16];
    switch (k2) {
#define TF_EXL3X_CASE(K2_)                                                                                   \
    case K2_:                                                                                                \
        warp_tiles<CB, K2_, NT, LOAD, MT>(T, NTILES, kt0, per_warp, nt0, x0, x1, r0, r1, lane, acc);       \
        break;
        TF_EXL3X_CASE(2)
        TF_EXL3X_CASE(3)
        TF_EXL3X_CASE(4)
        TF_EXL3X_CASE(5)
        TF_EXL3X_CASE(6)
        TF_EXL3X_CASE(7)
        TF_EXL3X_CASE(8)
        TF_EXL3X_CASE(9)
        TF_EXL3X_CASE(10)
        TF_EXL3X_CASE(11)
        TF_EXL3X_CASE(12)
        TF_EXL3X_CASE(13)
        TF_EXL3X_CASE(14)
        TF_EXL3X_CASE(15)
        TF_EXL3X_CASE(16)
#undef TF_EXL3X_CASE
        default:
            __trap();
    }

#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int col = i * 16 + h * 8 + 2 * t;
                red[warp][16 * m + g][col] = acc[m][i][h][0];
                red[warp][16 * m + g][col + 1] = acc[m][i][h][1];
                red[warp][16 * m + g + 8][col] = acc[m][i][h][2];
                red[warp][16 * m + g + 8][col + 1] = acc[m][i][h][3];
            }
    __syncthreads();
    for (int idx = threadIdx.x; idx < M * NT * 16; idx += W * 32) {
        const int row = idx / (NT * 16), col = idx % (NT * 16);
        const int r = rows_sh[row];
        if (r < 0) continue;
        float s = red[0][row][col];
#pragma unroll
        for (int w = 1; w < W; ++w) s += red[w][row][col];
        a.z[(((size_t)mat * a.sk + split) * a.p + r) * a.n + nt0 * 16 + col] = s;
    }
}

}  // namespace tf_exl3
