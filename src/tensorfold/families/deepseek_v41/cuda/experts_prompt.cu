// Grouped EXL3 expert GEMM for prompt chunks: each program decodes a trellis tile once and multiplies it against up to
// MTP member tiles of 16 rows (the decode kernel re-decodes the tile for every 16 rows). Same Z layout as
// tensorfold/cuda/exl3/experts_grouped.cuh with one K split, so its epilogues apply unchanged.
#include <torch/extension.h>

#include "experts_grouped.cuh"

namespace tf_exl3x {

template <int CB, int K2, int NT, int MTP>
__device__ __forceinline__ void warp_tiles_multi(const uint32_t* __restrict__ T, int NTILES, int kt0, int nkt,
                                                 int nt0, const half* const (&x0)[MTP], const half* const (&x1)[MTP],
                                                 const bool (&ok0)[MTP], const bool (&ok1)[MTP], int lane,
                                                 float (&acc)[MTP][NT][2][4]) {
    constexpr int TW = Fmt<K2>::TW, LW = Fmt<K2>::LW;
    const LaneMap<K2> map(lane);
    const size_t kstride = (size_t)NTILES * TW;
    const uint32_t* tp = T + ((size_t)kt0 * NTILES + nt0) * TW + lane;
    uint32_t cur[NT][LW], nxt[NT][LW];
#pragma unroll
    for (int i = 0; i < NT; ++i) load_words<K2>(cur[i], tp + i * TW, lane);
    for (int it = 0; it < nkt; ++it) {
        if (it + 1 < nkt)
#pragma unroll
            for (int i = 0; i < NT; ++i) load_words<K2>(nxt[i], tp + (size_t)(it + 1) * kstride + i * TW, lane);
        uint32_t b0[NT][2], b1[NT][2];
#pragma unroll
        for (int i = 0; i < NT; ++i) decode_tile<CB, K2>(cur[i], map, lane, b0[i], b1[i]);
        const int k = (kt0 + it) * 16;
#pragma unroll
        for (int m = 0; m < MTP; ++m) {
            uint32_t a[4] = {load_pair(x0[m] + k, ok0[m]), load_pair(x1[m] + k, ok1[m]),
                             load_pair(x0[m] + k + 8, ok0[m]), load_pair(x1[m] + k + 8, ok1[m])};
#pragma unroll
            for (int i = 0; i < NT; ++i) {
                mma16816(acc[m][i][0], a, b0[i]);
                mma16816(acc[m][i][1], a, b1[i]);
            }
        }
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int l = 0; l < LW; ++l) cur[i][l] = nxt[i][l];
    }
}

// Program (expert u, n block of 16 * NT, member group of 16 * MTP rows, matrix); K split over the W warps only.
template <int CB, int NT, int W, int MTP>
__global__ void __launch_bounds__(W * 32) grouped_prompt_kernel(
    const half* __restrict__ X0, const half* __restrict__ X1, const int64_t* __restrict__ TP0,
    const int64_t* __restrict__ TP1, const int* __restrict__ K2_0, const int* __restrict__ K2_1,
    const int* __restrict__ uids, const int* __restrict__ ucount, const int* __restrict__ members,
    float* __restrict__ Z, int K, int N, int P, int maxm, int slots) {
    const int u = blockIdx.x;
    if (u >= ucount[0]) return;
    const int MG = (maxm + 16 * MTP - 1) / (16 * MTP);
    const int mgroup = blockIdx.z % MG;
    const int mat = blockIdx.z / MG;
    const half* X = mat ? X1 : X0;
    const int e = uids[u];
    const uint32_t* T = reinterpret_cast<const uint32_t*>(mat ? TP1[e] : TP0[e]);
    const int k2 = mat ? K2_1[e] : K2_0[e];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int KT = K >> 4, NTILES = N >> 4;

    __shared__ int rows_sh[16 * MTP];
    for (int i = threadIdx.x; i < 16 * MTP; i += W * 32) {
        const int m = mgroup * 16 * MTP + i;
        const int code = m < maxm ? members[u * maxm + m] : -1;
        rows_sh[i] = code >= 0 ? (code >> 5) * slots + (code & 31) : -1;
    }
    __syncthreads();
    if (rows_sh[0] < 0) return;                            // members come first, so this group is empty
    const half* x0[MTP];
    const half* x1[MTP];
    bool ok0[MTP], ok1[MTP];
#pragma unroll
    for (int m = 0; m < MTP; ++m) {
        const int r0 = rows_sh[m * 16 + g], r1 = rows_sh[m * 16 + g + 8];
        x0[m] = X + (size_t)(r0 < 0 ? 0 : r0) * K + 2 * t;
        x1[m] = X + (size_t)(r1 < 0 ? 0 : r1) * K + 2 * t;
        ok0[m] = r0 >= 0;
        ok1[m] = r1 >= 0;
    }
    const int per_warp = KT / W;
    const int kt0 = warp * per_warp;
    const int nt0 = blockIdx.y * NT;
    float acc[MTP][NT][2][4];
#pragma unroll
    for (int m = 0; m < MTP; ++m)
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[m][i][h][c] = 0.f;
    switch (k2) {
        case 4: warp_tiles_multi<CB, 4, NT, MTP>(T, NTILES, kt0, per_warp, nt0, x0, x1, ok0, ok1, lane, acc); break;
        case 5: warp_tiles_multi<CB, 5, NT, MTP>(T, NTILES, kt0, per_warp, nt0, x0, x1, ok0, ok1, lane, acc); break;
        case 6: warp_tiles_multi<CB, 6, NT, MTP>(T, NTILES, kt0, per_warp, nt0, x0, x1, ok0, ok1, lane, acc); break;
        case 8: warp_tiles_multi<CB, 8, NT, MTP>(T, NTILES, kt0, per_warp, nt0, x0, x1, ok0, ok1, lane, acc); break;
        default: __trap();
    }
    // warps' partial sums through shared memory, added in warp order, one member tile at a time
    __shared__ float red[W][16][NT * 16];
#pragma unroll
    for (int m = 0; m < MTP; ++m) {
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int col = i * 16 + h * 8 + 2 * t;
                red[warp][g][col] = acc[m][i][h][0];
                red[warp][g][col + 1] = acc[m][i][h][1];
                red[warp][g + 8][col] = acc[m][i][h][2];
                red[warp][g + 8][col + 1] = acc[m][i][h][3];
            }
        __syncthreads();
        for (int idx = threadIdx.x; idx < 16 * NT * 16; idx += W * 32) {
            const int row = idx / (NT * 16), col = idx % (NT * 16);
            const int r = rows_sh[m * 16 + row];
            if (r < 0) continue;
            float s = red[0][row][col];
#pragma unroll
            for (int w = 1; w < W; ++w) s += red[w][row][col];
            Z[((size_t)mat * P + r) * N + nt0 * 16 + col] = s;
        }
        __syncthreads();
    }
}


// v2: a program owns (expert, 16-member group, K slice of KC k tiles, matrix). The members' activations for the slice
// are staged in shared memory once; each warp then walks its own N blocks (NT tiles of 16 columns) over the slice,
// decoding each weight tile straight into MMA fragments. Slices write Z partials [mat, split, P, N] (epilogues sum).
template <int CB, int K2, int NT, int MTP>
__device__ __forceinline__ void warp_nblock(const uint32_t* __restrict__ T, int NTILES, int kt0, int nkt, int nt0,
                                            const half* xs, int xs_stride, int lane, float (&acc)[MTP][NT][2][4]) {
    constexpr int TW = Fmt<K2>::TW, LW = Fmt<K2>::LW;
    const LaneMap<K2> map(lane);
    const int g = lane >> 2, t = lane & 3;
    const size_t kstride = (size_t)NTILES * TW;
    const uint32_t* tp = T + ((size_t)kt0 * NTILES + nt0) * TW + lane;
    uint32_t cur[NT][LW], nxt[NT][LW], nx2[NT][LW];
#pragma unroll
    for (int i = 0; i < NT; ++i) load_words<K2>(cur[i], tp + i * TW, lane);
    if (nkt > 1)
#pragma unroll
        for (int i = 0; i < NT; ++i) load_words<K2>(nxt[i], tp + kstride + i * TW, lane);
    for (int it = 0; it < nkt; ++it) {
        if (it + 2 < nkt)
#pragma unroll
            for (int i = 0; i < NT; ++i) load_words<K2>(nx2[i], tp + (size_t)(it + 2) * kstride + i * TW, lane);
        uint32_t a[MTP][4];
#pragma unroll
        for (int m = 0; m < MTP; ++m) {
            const half* xk = xs + (m * 16) * xs_stride + it * 16 + 2 * t;
            a[m][0] = *reinterpret_cast<const uint32_t*>(xk + g * xs_stride);
            a[m][1] = *reinterpret_cast<const uint32_t*>(xk + (g + 8) * xs_stride);
            a[m][2] = *reinterpret_cast<const uint32_t*>(xk + g * xs_stride + 8);
            a[m][3] = *reinterpret_cast<const uint32_t*>(xk + (g + 8) * xs_stride + 8);
        }
#pragma unroll
        for (int i = 0; i < NT; ++i) {
            uint32_t b0[2], b1[2];
            decode_tile<CB, K2>(cur[i], map, lane, b0, b1);
#pragma unroll
            for (int m = 0; m < MTP; ++m) {
                mma16816(acc[m][i][0], a[m], b0);
                mma16816(acc[m][i][1], a[m], b1);
            }
        }
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int l = 0; l < LW; ++l) {
                cur[i][l] = nxt[i][l];
                nxt[i][l] = nx2[i][l];
            }
    }
}

template <int CB, int NT, int W, int MTP>
__global__ void __launch_bounds__(W * 32) grouped_prompt2_kernel(
    const half* __restrict__ X0, const half* __restrict__ X1, const int64_t* __restrict__ TP0,
    const int64_t* __restrict__ TP1, const int* __restrict__ K2_0, const int* __restrict__ K2_1,
    const int* __restrict__ uids, const int* __restrict__ ucount, const int* __restrict__ members,
    float* __restrict__ Z, int K, int N, int P, int maxm, int slots, int KC, int SK) {
    extern __shared__ __align__(16) half xs[];            // [16 * MTP][KC * 16 + 8]
    const int u = blockIdx.x;
    if (u >= ucount[0]) return;
    const int split = blockIdx.y;
    const int MG = (maxm + 16 * MTP - 1) / (16 * MTP);
    const int mgroup = blockIdx.z % MG;
    const int mat = blockIdx.z / MG;
    const half* X = mat ? X1 : X0;
    const int e = uids[u];
    const uint32_t* T = reinterpret_cast<const uint32_t*>(mat ? TP1[e] : TP0[e]);
    const int k2 = mat ? K2_1[e] : K2_0[e];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int NTILES = N >> 4;
    const int stride = KC * 16 + 8;                        // halves; +8 staggers rows across banks

    __shared__ int rows_sh[16 * MTP];
    for (int i = threadIdx.x; i < 16 * MTP; i += W * 32) {
        const int m = mgroup * 16 * MTP + i;
        const int code = m < maxm ? members[u * maxm + m] : -1;
        rows_sh[i] = code >= 0 ? (code >> 5) * slots + (code & 31) : -1;
    }
    __syncthreads();
    if (rows_sh[0] < 0) return;                            // members come first, so this group is empty
    const int k0 = split * KC * 16;
    // stage the 16 members' K slice: 16-byte vectors, zeros for absent members
    const int vecs = KC * 2;                               // 8 halves a vector, 16 halves a k tile
    for (int idx = threadIdx.x; idx < 16 * MTP * vecs; idx += W * 32) {
        const int row = idx / vecs, v = idx % vecs;
        const int r = rows_sh[row];
        uint4 val = make_uint4(0u, 0u, 0u, 0u);
        if (r >= 0) val = *reinterpret_cast<const uint4*>(X + (size_t)r * K + k0 + v * 8);
        *reinterpret_cast<uint4*>(xs + row * stride + v * 8) = val;
    }
    __syncthreads();
    const int nblocks = N / (16 * NT);
    for (int nb = warp; nb < nblocks; nb += W) {
        float acc[MTP][NT][2][4];
#pragma unroll
        for (int m = 0; m < MTP; ++m)
#pragma unroll
            for (int i = 0; i < NT; ++i)
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int c = 0; c < 4; ++c) acc[m][i][h][c] = 0.f;
        const int nt0 = nb * NT;
        switch (k2) {
            case 4: warp_nblock<CB, 4, NT, MTP>(T, NTILES, split * KC, KC, nt0, xs, stride, lane, acc); break;
            case 5: warp_nblock<CB, 5, NT, MTP>(T, NTILES, split * KC, KC, nt0, xs, stride, lane, acc); break;
            case 6: warp_nblock<CB, 6, NT, MTP>(T, NTILES, split * KC, KC, nt0, xs, stride, lane, acc); break;
            case 8: warp_nblock<CB, 8, NT, MTP>(T, NTILES, split * KC, KC, nt0, xs, stride, lane, acc); break;
            default: __trap();
        }
        float* zbase = Z + ((size_t)mat * SK + split) * P * N;
#pragma unroll
        for (int m = 0; m < MTP; ++m) {
            const int ra = rows_sh[m * 16 + g], rb = rows_sh[m * 16 + g + 8];
#pragma unroll
            for (int i = 0; i < NT; ++i)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int col = (nt0 + i) * 16 + h * 8 + 2 * t;
                    if (ra >= 0)
                        *reinterpret_cast<float2*>(zbase + (size_t)ra * N + col) =
                            make_float2(acc[m][i][h][0], acc[m][i][h][1]);
                    if (rb >= 0)
                        *reinterpret_cast<float2*>(zbase + (size_t)rb * N + col) =
                            make_float2(acc[m][i][h][2], acc[m][i][h][3]);
                }
        }
    }
}


// v3: a program owns (expert, 16 * MTP member rows, a group of W N blocks of 16 * NT columns); each warp keeps one
// N block's accumulators for the whole K range while K slices of the members' activations stream through double-
// buffered shared memory. Each decoded weight tile feeds 2 * MTP MMAs; no K split, so Z is written once.
template <int CB, int K2, int NT, int MTP>
__device__ __forceinline__ void slice_tiles(const uint32_t* __restrict__ T, int NTILES, int kt0, int nkt, int nt0,
                                            const half* xs, int stride, int lane, float (&acc)[MTP][NT][2][4]) {
    constexpr int TW = Fmt<K2>::TW, LW = Fmt<K2>::LW;
    const LaneMap<K2> map(lane);
    const int g = lane >> 2, t = lane & 3;
    const size_t kstride = (size_t)NTILES * TW;
    const uint32_t* tp = T + ((size_t)kt0 * NTILES + nt0) * TW + lane;
    uint32_t cur[NT][LW], nxt[NT][LW];
#pragma unroll
    for (int i = 0; i < NT; ++i) load_words<K2>(cur[i], tp + i * TW, lane);
    for (int it = 0; it < nkt; ++it) {
        if (it + 1 < nkt)
#pragma unroll
            for (int i = 0; i < NT; ++i) load_words<K2>(nxt[i], tp + (size_t)(it + 1) * kstride + i * TW, lane);
        uint32_t a[MTP][4];
#pragma unroll
        for (int m = 0; m < MTP; ++m) {
            const half* xk = xs + (m * 16) * stride + it * 16 + 2 * t;
            a[m][0] = *reinterpret_cast<const uint32_t*>(xk + g * stride);
            a[m][1] = *reinterpret_cast<const uint32_t*>(xk + (g + 8) * stride);
            a[m][2] = *reinterpret_cast<const uint32_t*>(xk + g * stride + 8);
            a[m][3] = *reinterpret_cast<const uint32_t*>(xk + (g + 8) * stride + 8);
        }
#pragma unroll
        for (int i = 0; i < NT; ++i) {
            uint32_t b0[2], b1[2];
            decode_tile<CB, K2>(cur[i], map, lane, b0, b1);
#pragma unroll
            for (int m = 0; m < MTP; ++m) {
                mma16816(acc[m][i][0], a[m], b0);
                mma16816(acc[m][i][1], a[m], b1);
            }
        }
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int l = 0; l < LW; ++l) cur[i][l] = nxt[i][l];
    }
}

template <int CB, int NT, int W, int MTP, int KC>
__global__ void __launch_bounds__(W * 32) grouped_prompt3_kernel(
    const half* __restrict__ X0, const half* __restrict__ X1, const int64_t* __restrict__ TP0,
    const int64_t* __restrict__ TP1, const int* __restrict__ K2_0, const int* __restrict__ K2_1,
    const int* __restrict__ uids, const int* __restrict__ ucount, const int* __restrict__ members,
    float* __restrict__ Z, int K, int N, int P, int maxm, int slots) {
    constexpr int ROWS = 16 * MTP, STRIDE = KC * 16 + 8;
    __shared__ __align__(16) half xs[2][ROWS * STRIDE];
    __shared__ int rows_sh[ROWS];
    const int u = blockIdx.x;
    if (u >= ucount[0]) return;
    const int MG = (maxm + ROWS - 1) / ROWS;
    const int mgroup = blockIdx.z % MG;
    const int mat = blockIdx.z / MG;
    const half* X = mat ? X1 : X0;
    const int e = uids[u];
    const uint32_t* T = reinterpret_cast<const uint32_t*>(mat ? TP1[e] : TP0[e]);
    const int k2 = mat ? K2_1[e] : K2_0[e];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int NTILES = N >> 4, KT = K >> 4;
    for (int i = threadIdx.x; i < ROWS; i += W * 32) {
        const int m = mgroup * ROWS + i;
        const int code = m < maxm ? members[u * maxm + m] : -1;
        rows_sh[i] = code >= 0 ? (code >> 5) * slots + (code & 31) : -1;
    }
    __syncthreads();
    if (rows_sh[0] < 0) return;                            // members come first, so this group is empty
    const int nt0 = (blockIdx.y * W + warp) * NT;
    const bool active = nt0 < NTILES;
    constexpr int VECS = KC * 2;                           // 16-byte vectors a row of one slice
    auto stage = [&](int buf, int slice) {
        const int k0 = slice * KC * 16;
        for (int idx = threadIdx.x; idx < ROWS * VECS; idx += W * 32) {
            const int row = idx / VECS, v = idx % VECS;
            const int r = rows_sh[row];
            uint4 val = make_uint4(0u, 0u, 0u, 0u);
            if (r >= 0) val = *reinterpret_cast<const uint4*>(X + (size_t)r * K + k0 + v * 8);
            *reinterpret_cast<uint4*>(&xs[buf][row * STRIDE + v * 8]) = val;
        }
    };
    float acc[MTP][NT][2][4];
#pragma unroll
    for (int m = 0; m < MTP; ++m)
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[m][i][h][c] = 0.f;
    const int slices = KT / KC;
    stage(0, 0);
    __syncthreads();
    for (int sl = 0; sl < slices; ++sl) {
        if (sl + 1 < slices) stage((sl + 1) & 1, sl + 1);
        if (active) {
            const half* cur = xs[sl & 1];
            switch (k2) {
                case 4: slice_tiles<CB, 4, NT, MTP>(T, NTILES, sl * KC, KC, nt0, cur, STRIDE, lane, acc); break;
                case 5: slice_tiles<CB, 5, NT, MTP>(T, NTILES, sl * KC, KC, nt0, cur, STRIDE, lane, acc); break;
                case 6: slice_tiles<CB, 6, NT, MTP>(T, NTILES, sl * KC, KC, nt0, cur, STRIDE, lane, acc); break;
                case 8: slice_tiles<CB, 8, NT, MTP>(T, NTILES, sl * KC, KC, nt0, cur, STRIDE, lane, acc); break;
                default: __trap();
            }
        }
        __syncthreads();
    }
    if (!active) return;
    float* zbase = Z + (size_t)mat * P * N;
#pragma unroll
    for (int m = 0; m < MTP; ++m) {
        const int ra = rows_sh[m * 16 + g], rb = rows_sh[m * 16 + g + 8];
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int col = (nt0 + i) * 16 + h * 8 + 2 * t;
                if (ra >= 0)
                    *reinterpret_cast<float2*>(zbase + (size_t)ra * N + col) = make_float2(acc[m][i][h][0], acc[m][i][h][1]);
                if (rb >= 0)
                    *reinterpret_cast<float2*>(zbase + (size_t)rb * N + col) = make_float2(acc[m][i][h][2], acc[m][i][h][3]);
            }
    }
}


// v4: v3's tiling with the stalls taken out — member activations staged by cp.async one slice ahead, and each warp's
// weight words prefetched PF k tiles ahead in a register ring that runs on across slice boundaries.
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool valid) {
    const uint32_t sa = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa), "l"(gmem), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N_>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N_)); }

template <int CB, int K2, int NT, int MTP, int KC, int PF>
__device__ __forceinline__ void prompt4_body(const uint32_t* __restrict__ T, int NTILES, int KT, int nt0, bool active,
                                             half (*xs)[16 * MTP * (KC * 16 + 8)], const half* __restrict__ X,
                                             const int* rows_sh, int K, float (&acc)[MTP][NT][2][4], int W32, int nm) {
    constexpr int TW = Fmt<K2>::TW, LW = Fmt<K2>::LW;
    constexpr int ROWS = 16 * MTP, STRIDE = KC * 16 + 8, VECS = KC * 2;
    const int lane = threadIdx.x & 31;
    const LaneMap<K2> map(lane);
    const int g = lane >> 2, t = lane & 3;
    const size_t kstride = (size_t)NTILES * TW;
    const uint32_t* tp = T + (size_t)nt0 * TW + lane;
    auto stage = [&](int buf, int slice) {
        const int k0 = slice * KC * 16;
        for (int idx = threadIdx.x; idx < nm * 16 * VECS; idx += W32) {
            const int row = idx / VECS, v = idx % VECS;
            const int r = rows_sh[row];
            cp_async16(&xs[buf][row * STRIDE + v * 8], X + (size_t)(r >= 0 ? r : 0) * K + k0 + v * 8, r >= 0);
        }
        cp_async_commit();
    };
    uint32_t pf[PF][NT][LW];
    if (active)
#pragma unroll
        for (int d = 0; d < PF; ++d)
#pragma unroll
            for (int i = 0; i < NT; ++i) load_words<K2>(pf[d][i], tp + d * kstride + i * TW, lane);
    const int slices = KT / KC;
    stage(0, 0);
    for (int sl = 0; sl < slices; ++sl) {
        if (sl + 1 < slices) {
            stage((sl + 1) & 1, sl + 1);
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();
        if (active) {
            const half* xb = xs[sl & 1];
#pragma unroll
            for (int j = 0; j < KC; ++j) {
                const int d = j % PF;
                const int it = sl * KC + j;
                uint32_t w[NT][LW];
#pragma unroll
                for (int i = 0; i < NT; ++i)
#pragma unroll
                    for (int l = 0; l < LW; ++l) w[i][l] = pf[d][i][l];
                if (it + PF < KT)
#pragma unroll
                    for (int i = 0; i < NT; ++i) load_words<K2>(pf[d][i], tp + (size_t)(it + PF) * kstride + i * TW, lane);
                uint32_t a[MTP][4];
#pragma unroll
                for (int m = 0; m < MTP; ++m) {
                    if (m >= nm) break;
                    const half* xk = xb + (m * 16) * STRIDE + j * 16 + 2 * t;
                    a[m][0] = *reinterpret_cast<const uint32_t*>(xk + g * STRIDE);
                    a[m][1] = *reinterpret_cast<const uint32_t*>(xk + (g + 8) * STRIDE);
                    a[m][2] = *reinterpret_cast<const uint32_t*>(xk + g * STRIDE + 8);
                    a[m][3] = *reinterpret_cast<const uint32_t*>(xk + (g + 8) * STRIDE + 8);
                }
#pragma unroll
                for (int i = 0; i < NT; ++i) {
                    uint32_t b0[2], b1[2];
                    decode_tile<CB, K2>(w[i], map, lane, b0, b1);
#pragma unroll
                    for (int m = 0; m < MTP; ++m) {
                        if (m >= nm) break;
                        mma16816(acc[m][i][0], a[m], b0);
                        mma16816(acc[m][i][1], a[m], b1);
                    }
                }
            }
        }
        __syncthreads();
    }
}

__device__ __forceinline__ void store2(float* z, float a, float b) { *reinterpret_cast<float2*>(z) = make_float2(a, b); }
__device__ __forceinline__ void store2(__nv_bfloat16* z, float a, float b) {
    *reinterpret_cast<__nv_bfloat162*>(z) = __floats2bfloat162_rn(a, b);
}
constexpr float ZH_SCALE = 1.f / 64.f;                   // fp16 Z holds acc / 64: range to 4M, 11-bit mantissa
__device__ __forceinline__ void store2(half* z, float a, float b) {
    *reinterpret_cast<half2*>(z) = __floats2half2_rn(a * ZH_SCALE, b * ZH_SCALE);
}

template <int CB, int NT, int W, int MTP, int KC, int PF, typename ZT>
__global__ void __launch_bounds__(W * 32) grouped_prompt4_kernel(
    const half* __restrict__ X0, const half* __restrict__ X1, const int64_t* __restrict__ TP0,
    const int64_t* __restrict__ TP1, const int* __restrict__ K2_0, const int* __restrict__ K2_1,
    const int* __restrict__ uids, const int* __restrict__ ucount, const int* __restrict__ members,
    ZT* __restrict__ Z, int K, int N, int P, int maxm, int slots) {
    constexpr int ROWS = 16 * MTP, STRIDE = KC * 16 + 8;
    __shared__ __align__(16) half xs[2][ROWS * STRIDE];
    __shared__ int rows_sh[ROWS];
    const int u = blockIdx.y;                              // grid: (n group, expert, mat x member group)
    if (u >= ucount[0]) return;
    const int MG = (maxm + ROWS - 1) / ROWS;
    const int mgroup = blockIdx.z % MG;
    const int mat = blockIdx.z / MG;
    const half* X = mat ? X1 : X0;
    const int e = uids[u];
    const uint32_t* T = reinterpret_cast<const uint32_t*>(mat ? TP1[e] : TP0[e]);
    const int k2 = mat ? K2_1[e] : K2_0[e];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int NTILES = N >> 4, KT = K >> 4;
    for (int i = threadIdx.x; i < ROWS; i += W * 32) {
        const int m = mgroup * ROWS + i;
        const int code = m < maxm ? members[u * maxm + m] : -1;
        rows_sh[i] = code >= 0 ? (code >> 5) * slots + (code & 31) : -1;
    }
    __syncthreads();
    if (rows_sh[0] < 0) return;
    const int nt0 = (blockIdx.x * W + warp) * NT;   // an expert's n groups run side by side: X shared in L2
    const bool active = nt0 < NTILES;
    int nm = 0;                                            // m tiles holding members (members come first)
#pragma unroll
    for (int m = 0; m < MTP; ++m) nm += rows_sh[m * 16] >= 0;
    float acc[MTP][NT][2][4];
#pragma unroll
    for (int m = 0; m < MTP; ++m)
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[m][i][h][c] = 0.f;
    switch (k2) {
        case 4: prompt4_body<CB, 4, NT, MTP, KC, PF>(T, NTILES, KT, nt0, active, xs, X, rows_sh, K, acc, W * 32, nm); break;
        case 5: prompt4_body<CB, 5, NT, MTP, KC, PF>(T, NTILES, KT, nt0, active, xs, X, rows_sh, K, acc, W * 32, nm); break;
        case 6: prompt4_body<CB, 6, NT, MTP, KC, PF>(T, NTILES, KT, nt0, active, xs, X, rows_sh, K, acc, W * 32, nm); break;
        case 8: prompt4_body<CB, 8, NT, MTP, KC, PF>(T, NTILES, KT, nt0, active, xs, X, rows_sh, K, acc, W * 32, nm); break;
        default: __trap();
    }
    if (!active) return;
    ZT* zbase = Z + (size_t)mat * P * N;
#pragma unroll
    for (int m = 0; m < MTP; ++m) {
        const int ra = rows_sh[m * 16 + g], rb = rows_sh[m * 16 + g + 8];
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int col = (nt0 + i) * 16 + h * 8 + 2 * t;
                if (ra >= 0) store2(zbase + (size_t)ra * N + col, acc[m][i][h][0], acc[m][i][h][1]);
                if (rb >= 0) store2(zbase + (size_t)rb * N + col, acc[m][i][h][2], acc[m][i][h][3]);
            }
    }
}


// Prompt epilogues over bf16 Z (one K pass): the act of gate and up into the down input, and the down rows rotated,
// scaled and combined in slot order (no per-member copy is kept).
constexpr float HAD_SCALE = 0.08838834764831845f;   // 1 / sqrt(128)

__device__ __forceinline__ void fwht128(float (&v)[4], int lane) {
    float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c; v[1] = b + d; v[2] = a - c; v[3] = b - d;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float o = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = (lane & m) ? o - v[j] : v[j] + o;
        }
    }
}

__device__ __forceinline__ void load4(const half* z, float (&v)[4]) {
    const uint2 raw = *reinterpret_cast<const uint2*>(z);
    const float2 lo = __half22float2(*reinterpret_cast<const half2*>(&raw.x));
    const float2 hi = __half22float2(*reinterpret_cast<const half2*>(&raw.y));
    v[0] = lo.x * 64.f; v[1] = lo.y * 64.f; v[2] = hi.x * 64.f; v[3] = hi.y * 64.f;
}

__device__ __forceinline__ void load4(const __nv_bfloat16* z, float (&v)[4]) {
    const uint2 raw = *reinterpret_cast<const uint2*>(z);
    const float2 lo = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.x));
    const float2 hi = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.y));
    v[0] = lo.x; v[1] = lo.y; v[2] = hi.x; v[3] = hi.y;
}

template <typename ZT>
__global__ void gateup_epilogue_b_kernel(const ZT* __restrict__ Z, const int* __restrict__ pick,
                                         const half* __restrict__ svh_g, const half* __restrict__ svh_u,
                                         const half* __restrict__ suh_d, half* __restrict__ xd, int P, int N, int E,
                                         float limit) {
    const int p = blockIdx.x, blk = blockIdx.y;
    const int e = pick[p];
    if (e < 0 || e >= E) return;
    const int lane = threadIdx.x;
    const int n = blk * 128 + 4 * lane;
    float gv[4], uv[4];
    load4(Z + (size_t)p * N + n, gv);
    load4(Z + ((size_t)P + p) * N + n, uv);
    fwht128(gv, lane);
    fwht128(uv, lane);
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float gg = fminf(gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j]), limit);
        const float uu = fminf(fmaxf(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j]), -limit), limit);
        v[j] = gg / (1.f + expf(-gg)) * uu * __half2float(suh_d[(size_t)e * N + n + j]);
    }
    fwht128(v, lane);
    half* o = xd + (size_t)p * N + n;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

template <typename ZT>
__global__ void down_combine_b_kernel(const ZT* __restrict__ Z, const int* __restrict__ pick,
                                      const half* __restrict__ svh_d, const float* __restrict__ wts,
                                      float* __restrict__ out, int D, int E, int slots) {
    __shared__ float4 part[32][32];
    const int r = blockIdx.x, blk = blockIdx.y;
    const int k = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n = blk * 128 + 4 * lane;
    const int p = r * slots + k;
    const int e = pick[p];
    float o[4] = {0.f, 0.f, 0.f, 0.f};
    if (e >= 0 && e < E) {
        float v[4];
        load4(Z + (size_t)p * D + n, v);
        fwht128(v, lane);
#pragma unroll
        for (int j = 0; j < 4; ++j) o[j] = v[j] * HAD_SCALE * __half2float(svh_d[(size_t)e * D + n + j]);
    }
    part[k][lane] = make_float4(o[0], o[1], o[2], o[3]);
    __syncthreads();
    if (k != 0) return;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int q = 0; q < slots; ++q) {
        const float w = wts[r * slots + q];
        const float4 u = part[q][lane];
        acc[0] = fmaf(w, u.x, acc[0]);
        acc[1] = fmaf(w, u.y, acc[1]);
        acc[2] = fmaf(w, u.z, acc[2]);
        acc[3] = fmaf(w, u.w, acc[3]);
    }
    *reinterpret_cast<float4*>(out + (size_t)r * D + n) = make_float4(acc[0], acc[1], acc[2], acc[3]);
}

}  // namespace tf_exl3x

template <int NT, int W, int MTP>
static void launch(const at::Tensor& X0, const at::Tensor& X1, const at::Tensor& TP0, const at::Tensor& TP1,
                   const at::Tensor& K2_0, const at::Tensor& K2_1, const at::Tensor& uids, const at::Tensor& ucount,
                   const at::Tensor& members, at::Tensor& Z, int64_t mats, int64_t K, int64_t N, int64_t P,
                   int64_t slots) {
    TORCH_CHECK((K / 16) % W == 0 && N % (16 * NT) == 0, "prompt expert kernel: shape does not tile");
    const int maxm = (int)members.size(1);
    const int MG = (maxm + 16 * MTP - 1) / (16 * MTP);
    dim3 grid((unsigned)uids.numel(), (unsigned)(N / (16 * NT)), (unsigned)(mats * MG));
    tf_exl3x::grouped_prompt_kernel<2, NT, W, MTP><<<grid, W * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(X0.data_ptr()), reinterpret_cast<const half*>(X1.data_ptr()),
        TP0.data_ptr<int64_t>(), TP1.data_ptr<int64_t>(), K2_0.data_ptr<int>(), K2_1.data_ptr<int>(),
        uids.data_ptr<int>(), ucount.data_ptr<int>(), members.data_ptr<int>(), Z.data_ptr<float>(), (int)K, (int)N,
        (int)P, maxm, (int)slots);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// config: 0 NT8/MTP1, 1 NT8/MTP2, 2 NT4/MTP2, 3 NT4/MTP4, 4 NT2/MTP4, 5 NT8/MTP4
void grouped_prompt(const at::Tensor& X0, const at::Tensor& X1, const at::Tensor& TP0, const at::Tensor& TP1,
                    const at::Tensor& K2_0, const at::Tensor& K2_1, const at::Tensor& uids, const at::Tensor& ucount,
                    const at::Tensor& members, at::Tensor& Z, int64_t mats, int64_t K, int64_t N, int64_t P,
                    int64_t slots, int64_t cb, int64_t config) {
    TORCH_CHECK(cb == 2, "prompt expert kernel: mul1 codebook only");
#define TF_CFG(ID, NT_, MTP_) \
    case ID: launch<NT_, 4, MTP_>(X0, X1, TP0, TP1, K2_0, K2_1, uids, ucount, members, Z, mats, K, N, P, slots); break;
    switch (config) {
        TF_CFG(0, 8, 1)
        TF_CFG(1, 8, 2)
        TF_CFG(2, 4, 2)
        TF_CFG(3, 4, 4)
        TF_CFG(4, 2, 4)
        TF_CFG(5, 8, 4)
        default: TORCH_CHECK(false, "unknown prompt expert config");
    }
#undef TF_CFG
}

// v2: K split in slices of KC k tiles staged in shared memory; returns the split count the epilogues must sum.
int64_t grouped_prompt2(const at::Tensor& X0, const at::Tensor& X1, const at::Tensor& TP0, const at::Tensor& TP1,
                        const at::Tensor& K2_0, const at::Tensor& K2_1, const at::Tensor& uids,
                        const at::Tensor& ucount, const at::Tensor& members, at::Tensor& Z, int64_t mats, int64_t K,
                        int64_t N, int64_t P, int64_t slots, int64_t cb, int64_t KC, int64_t mtp, int64_t warps) {
    TORCH_CHECK(cb == 2, "prompt expert kernel: mul1 codebook only");
    TORCH_CHECK(mtp == 1 || mtp == 2, "member tiles a program: 1 or 2");
    TORCH_CHECK(warps == 4 || warps == 8, "warps a program: 4 or 8");
    constexpr int NT = 8;
    const int KT = (int)(K / 16);
    TORCH_CHECK(KT % KC == 0 && N % (16 * NT) == 0, "prompt expert kernel v2: shape does not tile");
    const int SK = KT / (int)KC;
    TORCH_CHECK(Z.numel() >= mats * SK * P * N, "Z too small for the K splits");
    const int maxm = (int)members.size(1);
    const int MG = (maxm + 16 * (int)mtp - 1) / (16 * (int)mtp);
    const size_t smem = (size_t)16 * mtp * (KC * 16 + 8) * sizeof(half);
    auto kern = warps == 8 ? (mtp == 2 ? tf_exl3x::grouped_prompt2_kernel<2, NT, 8, 2>
                                       : tf_exl3x::grouped_prompt2_kernel<2, NT, 8, 1>)
                           : (mtp == 2 ? tf_exl3x::grouped_prompt2_kernel<2, NT, 4, 2>
                                       : tf_exl3x::grouped_prompt2_kernel<2, NT, 4, 1>);
    if (smem > 48 * 1024) cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    dim3 grid((unsigned)uids.numel(), (unsigned)SK, (unsigned)(mats * MG));
    kern<<<grid, (unsigned)(warps * 32), smem, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(X0.data_ptr()), reinterpret_cast<const half*>(X1.data_ptr()),
        TP0.data_ptr<int64_t>(), TP1.data_ptr<int64_t>(), K2_0.data_ptr<int>(), K2_1.data_ptr<int>(),
        uids.data_ptr<int>(), ucount.data_ptr<int>(), members.data_ptr<int>(), Z.data_ptr<float>(), (int)K, (int)N,
        (int)P, maxm, (int)slots, (int)KC, SK);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return SK;
}

// v3: one K pass per program (Z written once, the epilogues read SK = 1); config 0: NT4/W8/MTP4, 1: NT2/W8/MTP4,
// 2: NT4/W4/MTP4, 3: NT4/W8/MTP2 — all with 8-k-tile slices.
void grouped_prompt3(const at::Tensor& X0, const at::Tensor& X1, const at::Tensor& TP0, const at::Tensor& TP1,
                     const at::Tensor& K2_0, const at::Tensor& K2_1, const at::Tensor& uids, const at::Tensor& ucount,
                     const at::Tensor& members, at::Tensor& Z, int64_t mats, int64_t K, int64_t N, int64_t P,
                     int64_t slots, int64_t cb, int64_t config) {
    TORCH_CHECK(cb == 2, "prompt expert kernel: mul1 codebook only");
    TORCH_CHECK((K / 16) % 8 == 0, "prompt expert kernel v3: K/16 must be a multiple of 8");
    TORCH_CHECK(Z.numel() >= mats * P * N, "Z too small");
    const int maxm = (int)members.size(1);
#define TF_V3(ID, NT_, W_, MTP_, KC_)                                                                                 \
    case ID: {                                                                                                     \
        const int MG = (maxm + 16 * MTP_ - 1) / (16 * MTP_);                                                        \
        const int ngroups = (int)((N / 16 + NT_ * W_ - 1) / (NT_ * W_));                                            \
        dim3 grid((unsigned)uids.numel(), (unsigned)ngroups, (unsigned)(mats * MG));                                \
        tf_exl3x::grouped_prompt3_kernel<2, NT_, W_, MTP_, KC_><<<grid, W_ * 32, 0, at::cuda::getCurrentCUDAStream()>>>( \
            reinterpret_cast<const half*>(X0.data_ptr()), reinterpret_cast<const half*>(X1.data_ptr()),             \
            TP0.data_ptr<int64_t>(), TP1.data_ptr<int64_t>(), K2_0.data_ptr<int>(), K2_1.data_ptr<int>(),           \
            uids.data_ptr<int>(), ucount.data_ptr<int>(), members.data_ptr<int>(), Z.data_ptr<float>(), (int)K,     \
            (int)N, (int)P, maxm, (int)slots);                                                                      \
        break;                                                                                                     \
    }
#define TF_V4(ID, NT_, W_, MTP_, KC_, PF_)                                                                         \
    case ID: {                                                                                                     \
        const int MG = (maxm + 16 * MTP_ - 1) / (16 * MTP_);                                                        \
        const int ngroups = (int)((N / 16 + NT_ * W_ - 1) / (NT_ * W_));                                            \
        dim3 grid((unsigned)ngroups, (unsigned)uids.numel(), (unsigned)(mats * MG));                                \
        tf_exl3x::grouped_prompt4_kernel<2, NT_, W_, MTP_, KC_, PF_, float><<<grid, W_ * 32, 0,                            \
                                                                      at::cuda::getCurrentCUDAStream()>>>(          \
            reinterpret_cast<const half*>(X0.data_ptr()), reinterpret_cast<const half*>(X1.data_ptr()),             \
            TP0.data_ptr<int64_t>(), TP1.data_ptr<int64_t>(), K2_0.data_ptr<int>(), K2_1.data_ptr<int>(),           \
            uids.data_ptr<int>(), ucount.data_ptr<int>(), members.data_ptr<int>(), Z.data_ptr<float>(), (int)K,     \
            (int)N, (int)P, maxm, (int)slots);                                                                      \
        break;                                                                                                     \
    }
    switch (config) {
        TF_V4(100, 2, 8, 4, 8, 2)
        TF_V4(101, 2, 8, 4, 8, 4)
        TF_V4(102, 2, 16, 4, 8, 2)
        TF_V4(103, 2, 16, 4, 8, 4)
        TF_V4(104, 4, 8, 2, 8, 2)
        TF_V4(105, 2, 8, 2, 8, 4)
        TF_V4(106, 1, 16, 4, 8, 4)
        TF_V4(107, 2, 16, 2, 8, 4)
        TF_V4(108, 2, 12, 4, 8, 4)
        TF_V4(109, 2, 18, 4, 8, 4)
        TF_V4(110, 1, 24, 4, 8, 4)
        TF_V4(111, 3, 8, 4, 8, 4)
        TF_V4(112, 2, 16, 4, 8, 8)
        TF_V4(113, 2, 20, 4, 8, 4)
        TF_V4(114, 1, 32, 4, 8, 4)
        TF_V4(115, 2, 18, 4, 8, 2)
        TF_V3(0, 4, 8, 4, 8)
        TF_V3(1, 2, 8, 4, 8)
        TF_V3(2, 4, 4, 4, 8)
        TF_V3(3, 4, 8, 2, 8)
        TF_V3(4, 2, 16, 2, 8)
        TF_V3(5, 1, 16, 4, 8)
        TF_V3(6, 2, 32, 4, 8)
        TF_V3(7, 2, 16, 4, 8)
        TF_V3(8, 4, 16, 2, 8)
        TF_V3(9, 1, 32, 4, 8)
        default: TORCH_CHECK(false, "unknown v3 config");
    }
#undef TF_V3
#undef TF_V4
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// v4 with fp16 (scaled) or bf16 Z (configs as grouped_prompt3's 100..) and its epilogues.
void grouped_prompt4(const at::Tensor& X0, const at::Tensor& X1, const at::Tensor& TP0, const at::Tensor& TP1,
                     const at::Tensor& K2_0, const at::Tensor& K2_1, const at::Tensor& uids, const at::Tensor& ucount,
                     const at::Tensor& members, at::Tensor& Z, int64_t mats, int64_t K, int64_t N, int64_t P,
                     int64_t slots, int64_t cb, int64_t config) {
    TORCH_CHECK(cb == 2, "prompt expert kernel: mul1 codebook only");
    TORCH_CHECK((K / 16) % 8 == 0, "prompt expert kernel v4: K/16 must be a multiple of 8");
    TORCH_CHECK(Z.numel() >= mats * P * N, "Z: mats x P x N");
    const bool zh = Z.scalar_type() == at::kHalf;
    TORCH_CHECK(zh || Z.scalar_type() == at::kBFloat16, "Z: fp16 or bf16");
    const int maxm = (int)members.size(1);
#define TF_V4B_LAUNCH(ZT_, NT_, W_, MTP_, KC_, PF_, grid)                                                             \
    tf_exl3x::grouped_prompt4_kernel<2, NT_, W_, MTP_, KC_, PF_, ZT_><<<grid, W_ * 32, 0,                          \
                                                                        at::cuda::getCurrentCUDAStream()>>>(        \
        reinterpret_cast<const half*>(X0.data_ptr()), reinterpret_cast<const half*>(X1.data_ptr()),                 \
        TP0.data_ptr<int64_t>(), TP1.data_ptr<int64_t>(), K2_0.data_ptr<int>(), K2_1.data_ptr<int>(),               \
        uids.data_ptr<int>(), ucount.data_ptr<int>(), members.data_ptr<int>(), reinterpret_cast<ZT_*>(Z.data_ptr()), \
        (int)K, (int)N, (int)P, maxm, (int)slots)
#define TF_V4B(ID, NT_, W_, MTP_, KC_, PF_)                                                                        \
    case ID: {                                                                                                     \
        const int MG = (maxm + 16 * MTP_ - 1) / (16 * MTP_);                                                        \
        const int ngroups = (int)((N / 16 + NT_ * W_ - 1) / (NT_ * W_));                                            \
        dim3 grid((unsigned)ngroups, (unsigned)uids.numel(), (unsigned)(mats * MG));                                \
        if (zh)                                                                                                    \
            TF_V4B_LAUNCH(half, NT_, W_, MTP_, KC_, PF_, grid);                                                    \
        else                                                                                                       \
            TF_V4B_LAUNCH(__nv_bfloat16, NT_, W_, MTP_, KC_, PF_, grid);                                           \
        break;                                                                                                     \
    }
    switch (config) {
        TF_V4B(102, 2, 16, 4, 8, 2)
        TF_V4B(103, 2, 16, 4, 8, 4)
        TF_V4B(108, 2, 12, 4, 8, 4)
        TF_V4B(112, 2, 16, 4, 8, 8)
        TF_V4B(101, 2, 8, 4, 8, 4)
        TF_V4B(105, 2, 8, 2, 8, 4)
        TF_V4B(107, 2, 16, 2, 8, 4)
        TF_V4B(109, 2, 18, 4, 8, 4)
        TF_V4B(113, 2, 20, 4, 8, 4)
        TF_V4B(116, 1, 16, 4, 8, 4)
        TF_V4B(117, 4, 8, 4, 8, 2)
        TF_V4B(118, 2, 12, 4, 8, 4)
        default: TORCH_CHECK(false, "unknown v4 config");
    }
#undef TF_V4B
#undef TF_V4B_LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gateup_epilogue_b(const at::Tensor& Z, const at::Tensor& pick, const at::Tensor& svh_g, const at::Tensor& svh_u,
                       const at::Tensor& suh_d, at::Tensor& xd, int64_t P, int64_t N, int64_t E, double limit) {
    dim3 grid((unsigned)P, (unsigned)(N / 128));
    auto run = [&](auto* z) {
        tf_exl3x::gateup_epilogue_b_kernel<<<grid, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
            z, pick.data_ptr<int>(), reinterpret_cast<const half*>(svh_g.data_ptr()),
            reinterpret_cast<const half*>(svh_u.data_ptr()), reinterpret_cast<const half*>(suh_d.data_ptr()),
            reinterpret_cast<half*>(xd.data_ptr()), (int)P, (int)N, (int)E, (float)limit);
    };
    if (Z.scalar_type() == at::kHalf)
        run(reinterpret_cast<const half*>(Z.data_ptr()));
    else
        run(reinterpret_cast<const __nv_bfloat16*>(Z.data_ptr()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void down_combine_b(const at::Tensor& Z, const at::Tensor& pick, const at::Tensor& svh_d, const at::Tensor& wts,
                    at::Tensor& out, int64_t rows, int64_t D, int64_t slots, int64_t E) {
    TORCH_CHECK(slots <= 32, "at most 32 slots a row");
    dim3 grid((unsigned)rows, (unsigned)(D / 128));
    auto run = [&](auto* z) {
        tf_exl3x::down_combine_b_kernel<<<grid, (unsigned)(32 * slots), 0, at::cuda::getCurrentCUDAStream()>>>(
            z, pick.data_ptr<int>(), reinterpret_cast<const half*>(svh_d.data_ptr()), wts.data_ptr<float>(),
            out.data_ptr<float>(), (int)D, (int)E, (int)slots);
    };
    if (Z.scalar_type() == at::kHalf)
        run(reinterpret_cast<const half*>(Z.data_ptr()));
    else
        run(reinterpret_cast<const __nv_bfloat16*>(Z.data_ptr()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grouped_prompt4", &grouped_prompt4);
    m.def("gateup_epilogue_b", &gateup_epilogue_b);
    m.def("down_combine_b", &down_combine_b);
    m.def("grouped_prompt3", &grouped_prompt3);
    m.def("grouped_prompt", &grouped_prompt);
    m.def("grouped_prompt2", &grouped_prompt2);
}
