// sm_12x lane matmul: up to four projections of one input and bit width (4, 5, 6 or 8) a launch, each column with
// qmm.cu's order and so its bits.

#include <ATen/ATen.h>
#include <algorithm>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <type_traits>
#include <vector>

#include "qmm_frag.cuh"

namespace {

using namespace qmm_frag;

constexpr int PARTS = 4;

// One projection: packed weights, scales and biases (rows npad apart), (M, n) output, K split, tiles, first cluster.
struct Part {
    const uint32_t* w;
    const __nv_bfloat16* scales;
    const __nv_bfloat16* biases;
    void* out;
    int n, npad, sk, tiles, first;
};

struct Parts {
    Part p[PARTS];
    int count;
};

// pair() with the nibble mask in a register: compilers rematerialize a literal mask at every use.
__device__ __forceinline__ uint32_t pairm(uint32_t w, int s, uint32_t mask) {
    const uint32_t t = ((w >> s) & mask) | 0x43004300u;
    uint32_t r;
#if __CUDA_ARCH__ >= 900
    asm("sub.rn.bf16x2 %0, %1, %2;\n" : "=r"(r) : "r"(t), "r"(0x43004300u));
#else
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(t), "r"(0x3F803F80u), "r"(0xC300C300u));
#endif
    return r;
}

template <int N>
using ic = std::integral_constant<int, N>;

// Clusters of C blocks along x, one part each: a cluster covers C / sk column tiles, each split in sk K slices.
// SWAP (8-row tiles): weights are the MMA's A operand (16 columns) and the rows its B (8), half the MMAs of 16 rows.
// SKIP: a warp's m16 tiles wholly past M skip their fragments, mmas and scaling (a part-filled row tile). SPREAD:
// every K slice sums its share of the tile's outputs (else slice 0 sums them all); the same adds in the same order.
template <int GS, int BM, int BN, int WM, int WN, int STAGES, bool F32, int BITS, bool SWAP = false,
          bool SKIP = false, bool SPREAD = false>
__global__ void __launch_bounds__(WM * WN * 32) group_kernel(
        const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs, const __grid_constant__ Parts parts,
        int M, int K, int ldx, int rows_t, int C) {
    using T = LaneTile<GS, BM, BN, WM, WN, STAGES, BITS>;
    static_assert(!SWAP || (BM == 8 && WM == 1 && T::NT % 2 == 0), "swapped tiles: 8 rows, column pairs a warp");
    static_assert(BITS != 5 || T::NT % 2 == 0, "5 bits: a warp holds both n8 tiles of each shared high word");
    constexpr int I = SWAP ? T::NT / 2 : T::MT;     // MMA m16 tiles a warp: column pairs (swapped) or row tiles
    constexpr int J = SWAP ? 1 : T::NT;             // MMA n8 tiles a warp: the 8 rows (swapped) or column tiles
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    const int KG = K / GS, cid = blockIdx.x / C, rank = blockIdx.x % C;
    int q = 0;
#pragma unroll
    for (int i = 1; i < PARTS; ++i)
        if (i < parts.count && cid >= parts.p[i].first) q = i;
    const Part P = parts.p[q];                   // in registers: the loads below index it every group
    const int sk = P.sk, lc = cid - P.first;
    const int tile = lc / rows_t * (C / sk) + rank / sk, slice = rank % sk;
    const int m0 = lc % rows_t * BM, n0 = tile * BN, per = KG / sk, g0 = slice * per;
    const bool live = tile < P.tiles;            // a cluster's spare blocks past the last tile only join its syncs

    auto stage = [&](int s) { return buf + s * T::STAGE; };
    const int rows = min(BM, M - m0);           // rows past M stay zero in every stage (set once below)
    // each thread's copies, fixed for the block: sources advance a group at a time
    constexpr int TILE_BYTES = 64 * GS * BITS / 8;  // one stored 64-column tile's group block
    constexpr int XC = (BM * T::CHUNKS + T::THREADS - 1) / T::THREADS, WC = (T::W / 16 + T::THREADS - 1) / T::THREADS;
    constexpr int SC = (2 * T::S / 16 + T::THREADS - 1) / T::THREADS;
    static_assert(BM <= T::THREADS, "one input sum a thread");
    const __nv_bfloat16* xsrc[XC];
    const unsigned char* wsrc[WC];
    const __nv_bfloat16* ssrc[SC];
    int xdst[XC], wdst[WC], sdst[SC];
    bool xok[XC], wok[WC], sok[SC];
#pragma unroll
    for (int j = 0; j < XC; ++j) {
        const int c = tid + j * T::THREADS, r = c / T::CHUNKS, ch = c % T::CHUNKS;
        xok[j] = c < BM * T::CHUNKS && r < rows;
        xsrc[j] = x + static_cast<size_t>(m0 + min(r, rows - 1)) * ldx + ch * 8;
        xdst[j] = r * T::ROW + swz<T::CHUNKS>(r, ch) * 16;
    }
#pragma unroll
    for (int j = 0; j < WC; ++j) {
        const int c = tid + j * T::THREADS, t = c / (TILE_BYTES / 16), off = c % (TILE_BYTES / 16);
        wok[j] = c < T::W / 16;
        wsrc[j] = reinterpret_cast<const unsigned char*>(P.w) + static_cast<size_t>(n0 / 64 + t) * KG * TILE_BYTES +
                  off * 16;
        wdst[j] = T::X + c * 16;
    }
#pragma unroll
    for (int j = 0; j < SC; ++j) {
        const int c = tid + j * T::THREADS, which = c / (T::S / 16), off = c % (T::S / 16);
        sok[j] = c < 2 * (T::S / 16);
        ssrc[j] = (which ? P.biases : P.scales) + n0 + off * 8;
        sdst[j] = T::X + T::W + which * T::S + off * 16;
    }
    const bool xsok = tid < rows;
    const float* xssrc = xs + static_cast<size_t>(m0 + min(tid, rows - 1)) * KG;
    // a group's inputs and input sums (the previous kernel's outputs) into stage s
    auto load_x = [&](int s, int g) {
        unsigned char* p = stage(s);
#pragma unroll
        for (int j = 0; j < XC; ++j)
            if (xok[j]) cp16(p + xdst[j], xsrc[j] + g * GS);
        if (xsok) cp4(p + T::X + T::W + 2 * T::S + tid * 4, xssrc + g);
    };
    // a group's weights, scales and biases (constant, so loadable before the previous kernel ends)
    auto load_w = [&](int s, int g) {
        unsigned char* p = stage(s);
#pragma unroll
        for (int j = 0; j < WC; ++j)
            if (wok[j]) cp16(p + wdst[j], wsrc[j] + static_cast<size_t>(g) * TILE_BYTES);
#pragma unroll
        for (int j = 0; j < SC; ++j)
            if (sok[j]) cp16(p + sdst[j], ssrc[j] + static_cast<size_t>(g) * P.npad);
    };
    uint32_t mask;                              // 0x000F000F, opaque to the compiler (kept in a register)
    asm volatile("mov.b32 %0, 0x000F000F;\n" : "=r"(mask));
    auto unpack = [mask](uint32_t w, int s) { return pairm(w, s, mask); };

    float acc[I][J][4];
#pragma unroll
    for (int i = 0; i < I; ++i)
#pragma unroll
        for (int j = 0; j < J; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;
    if (live) {
#pragma unroll
        for (int c = tid; c < BM * T::CHUNKS; c += T::THREADS)
            if (c / T::CHUNKS >= rows)
#pragma unroll
                for (int s = 0; s < STAGES; ++s)
                    *reinterpret_cast<uint4*>(stage(s) + c / T::CHUNKS * T::ROW + c % T::CHUNKS * 16) = uint4{};
        // weights before the previous kernel ends, inputs after it: wait_group's count still finds each stage whole
#pragma unroll
        for (int s = 0; s < STAGES - 1; ++s) {
            if (s < per) load_w(s, g0 + s);
            commit();
        }
        grid_wait();
#pragma unroll
        for (int s = 0; s < STAGES - 1; ++s) {
            if (s < per) load_x(s, g0 + s);
            commit();
        }
        grid_launch();
        auto steps = [&](auto live_c) {                   // LIVE: the warp's m16 tiles holding rows below M
            [[maybe_unused]] constexpr int LIVE = decltype(live_c)::value;
            for (int it = 0; it < per; ++it) {
                wait<STAGES - 2>();
                __syncthreads();
                const int next = it + STAGES - 1;
                if (next < per) {
                    load_x(next % STAGES, g0 + next);
                    load_w(next % STAGES, g0 + next);
                }
                commit();
                const unsigned char* p = stage(it % STAGES);
                const uint32_t* pw = reinterpret_cast<const uint32_t*>(p + T::X);
                const __nv_bfloat16* ps = reinterpret_cast<const __nv_bfloat16*>(p + T::X + T::W);
                const float* px = reinterpret_cast<const float*>(p + T::X + T::W + 2 * T::S);
                uint32_t words[T::NT][T::WORDS];
#pragma unroll
                for (int j = 0; j < T::NT; ++j) {
                    if constexpr (BITS == 5 || BITS == 6) {
                        high_words<GS, BITS>(words[j], pw, wn * T::NT + j, lane);
                    } else {
#pragma unroll
                        for (int v = 0; v < T::WORDS; ++v)
                            words[j][v] = pw[((wn * T::NT + j) * 32 + lane) * T::WORDS + v];
                    }
                }
                float d[I][J][4];
                if constexpr (SWAP) {
#pragma unroll
                    for (int kt = 0; kt < GS / 16; ++kt) {
                        uint32_t bx[2];                           // rows 0-7 at k lo, k hi: the B fragment
                        const int ch = kt * 2 + ((lane >> 3) & 1);
                        ldmatrix2(bx, p + (lane & 7) * T::ROW + swz<T::CHUNKS>(lane & 7, ch) * 16);
#pragma unroll
                        for (int i = 0; i < I; ++i) {             // A rows g, g + 8 are columns of n8 tiles 2i, 2i + 1
                            const uint32_t a[4] = {frag<BITS>(words[2 * i], kt, 0, unpack),
                                                   frag<BITS>(words[2 * i + 1], kt, 0, unpack),
                                                   frag<BITS>(words[2 * i], kt, 1, unpack),
                                                   frag<BITS>(words[2 * i + 1], kt, 1, unpack)};
                            if (kt == 0) mma0(d[i][0], a, bx[0], bx[1]);
                            else mma(d[i][0], a, bx[0], bx[1]);
                        }
                    }
#pragma unroll
                    for (int i = 0; i < I; ++i)
#pragma unroll
                        for (int e = 0; e < 4; ++e) {             // d: column g (+ 8 for e >= 2), rows 2t, 2t + 1
                            const int col = wn * (BN / WN) + i * 16 + (lane >> 2) + (e >> 1) * 8;
                            const float sv = __bfloat162float(ps[col]), bv = __bfloat162float(ps[BN + col]);
                            const float xv = px[(lane & 3) * 2 + (e & 1)];
                            acc[i][0][e] = __fmaf_rn(xv, bv, __fmaf_rn(d[i][0][e], sv, acc[i][0][e]));
                        }
                } else {
#pragma unroll
                    for (int kt = 0; kt < GS / 16; ++kt) {
                        uint32_t a[LIVE > 0 ? LIVE : 1][4];
#pragma unroll
                        for (int i = 0; i < LIVE; ++i) {
                            const int r = wm * (BM / WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                            const int ch = kt * 2 + (lane >> 4);
                            ldmatrix4(a[i], p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16);
                        }
#pragma unroll
                        for (int j = 0; j < T::NT; ++j) {
                            const uint32_t b0 = frag<BITS>(words[j], kt, 0, unpack);
                            const uint32_t b1 = frag<BITS>(words[j], kt, 1, unpack);
#pragma unroll
                            for (int i = 0; i < LIVE; ++i) {
                                if (kt == 0) mma0(d[i][j], a[i], b0, b1);
                                else mma(d[i][j], a[i], b0, b1);
                            }
                        }
                    }
#pragma unroll
                    for (int j = 0; j < T::NT; ++j) {
                        const int col = wn * (BN / WN) + j * 8 + (lane & 3) * 2;
                        const __nv_bfloat162 s2 = *reinterpret_cast<const __nv_bfloat162*>(ps + col);
                        const __nv_bfloat162 b2 = *reinterpret_cast<const __nv_bfloat162*>(ps + BN + col);
                        const float sv[2] = {__low2float(s2), __high2float(s2)};
                        const float bv[2] = {__low2float(b2), __high2float(b2)};
#pragma unroll
                        for (int i = 0; i < LIVE; ++i) {
                            const int row = wm * (BM / WM) + i * 16 + (lane >> 2);
                            const float xv[2] = {px[row], px[row + 8]};
#pragma unroll
                            for (int e = 0; e < 4; ++e)           // acc = fma(xs, b, fma(p, s, acc)): qmm.cu's order
                                acc[i][j][e] = __fmaf_rn(xv[e >> 1], bv[e & 1],
                                                         __fmaf_rn(d[i][j][e], sv[e & 1], acc[i][j][e]));
                        }
                    }
                }
            }
        };
        if constexpr (!SKIP || SWAP || T::MT == 1) {
            steps(ic<T::MT>());
        } else {
            const int live = min(T::MT, max(0, (M - m0 - wm * (BM / WM) + 15) / 16));
            if (live == 0) steps(ic<0>());
            else if (live == 1) steps(ic<1>());
            else if (T::MT > 2 && live == 2) steps(ic<(T::MT > 2 ? 2 : T::MT)>());
            else if (T::MT > 3 && live == 3) steps(ic<(T::MT > 3 ? 3 : T::MT)>());
            else if (T::MT > 4 && live <= 4) steps(ic<(T::MT > 4 ? 4 : T::MT)>());
            else if (T::MT > 6 && live <= 6) steps(ic<(T::MT > 6 ? 6 : T::MT)>());
            else steps(ic<T::MT>());
        }
        wait<0>();
        __syncthreads();
    } else {
        grid_launch();
    }
    const int N = P.n;
    auto put = [&](int row, int col, float v) {
        if (row >= M || col >= N) return;
        if (F32) reinterpret_cast<float*>(P.out)[static_cast<size_t>(row) * N + col] = v;
        else reinterpret_cast<__nv_bfloat16*>(P.out)[static_cast<size_t>(row) * N + col] = __float2bfloat16_rn(v);
    };
    auto emit = [&](int i, int j, int h, float v0, float v1) {      // entries 2h, 2h + 1 of fragment (i, j)
        if constexpr (SWAP) {                    // column g (+ 8 for h 1), rows 2t, 2t + 1
            const int col = n0 + wn * (BN / WN) + i * 16 + (lane >> 2) + h * 8;
            put(m0 + (lane & 3) * 2, col, v0);
            put(m0 + (lane & 3) * 2 + 1, col, v1);
        } else {
            const int col = n0 + wn * (BN / WN) + j * 8 + (lane & 3) * 2;
            const int row = m0 + wm * (BM / WM) + i * 16 + (lane >> 2) + h * 8;
            if (row >= M) return;
            if (!F32 && col + 1 < N && (N & 1) == 0) {
                auto* dst = reinterpret_cast<__nv_bfloat16*>(P.out) + static_cast<size_t>(row) * N + col;
                *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v0, v1);
            } else {
                put(row, col, v0);
                put(row, col + 1, v1);
            }
        }
    };
    if (sk > 1) {                                // uniform in a cluster: it holds one part
#if __CUDA_ARCH__ >= 900
        auto cluster = cooperative_groups::this_cluster();
        float* mine = reinterpret_cast<float*>(buf);
        if (SPREAD || slice != 0) {
#pragma unroll
            for (int i = 0; i < I; ++i)
#pragma unroll
                for (int j = 0; j < J; ++j)
#pragma unroll
                    for (int e = 0; e < 4; ++e) mine[((i * J + j) * 4 + e) * T::THREADS + tid] = acc[i][j][e];
        }
        cluster.sync();
        if constexpr (SPREAD) {                  // slice s adds pairs s, s + sk, .. over the tile's slices in order
            if (live) {
                for (int pr = slice; pr < I * J * 2; pr += sk) {
                    const float* first = cluster.map_shared_rank(mine, rank - slice);
                    float v0 = first[2 * pr * T::THREADS + tid], v1 = first[(2 * pr + 1) * T::THREADS + tid];
#pragma unroll 7
                    for (int peer = 1; peer < sk; ++peer) {
                        const float* theirs = cluster.map_shared_rank(mine, rank - slice + peer);
                        v0 = v0 + theirs[2 * pr * T::THREADS + tid];
                        v1 = v1 + theirs[(2 * pr + 1) * T::THREADS + tid];
                    }
                    emit(pr / (J * 2), pr / 2 % J, pr % 2, v0, v1);
                }
            }
            cluster.sync();                      // peers keep their memory until every slice has read it
            return;
        } else {
            if (slice == 0 && live) {            // slice 0 adds its tile's peers in slice order, as qmm.cu does
                for (int peer = 1; peer < sk; ++peer) {
                    const float* theirs = cluster.map_shared_rank(mine, rank + peer);
#pragma unroll
                    for (int i = 0; i < I; ++i)
#pragma unroll
                        for (int j = 0; j < J; ++j)
#pragma unroll
                            for (int e = 0; e < 4; ++e)
                                acc[i][j][e] = acc[i][j][e] + theirs[((i * J + j) * 4 + e) * T::THREADS + tid];
                }
            }
            cluster.sync();                      // peers keep their memory until slice 0 has read it
            if (slice != 0) return;
        }
#else
        __trap();
        return;
#endif
    }
    if (!live) return;
#pragma unroll
    for (int i = 0; i < I; ++i)
#pragma unroll
        for (int j = 0; j < J; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h) emit(i, j, h, acc[i][j][2 * h], acc[i][j][2 * h + 1]);
}

template <int GS, int BM, int BN, int WM, int WN, int STAGES, bool F32, int BITS, bool SWAP = false,
          bool SKIP = false, bool SPREAD = false>
void launch(const at::Tensor& x, const at::Tensor& xs, Parts& parts, int C, bool pdl) {
    using T = LaneTile<GS, BM, BN, WM, WN, STAGES, BITS>;
    const int M = x.size(0), K = x.size(1), rows_t = (M + BM - 1) / BM;
    int clusters = 0;
    for (int i = 0; i < parts.count; ++i) {
        Part& P = parts.p[i];
        P.tiles = (P.n + BN - 1) / BN;
        P.first = clusters;
        clusters += rows_t * ((P.tiles + C / P.sk - 1) / (C / P.sk));
    }
    auto kernel = group_kernel<GS, BM, BN, WM, WN, STAGES, F32, BITS, SWAP, SKIP, SPREAD>;
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM);
        configured = true;
    }
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(clusters * C);
    config.blockDim = dim3(T::THREADS);
    config.dynamicSmemBytes = T::SMEM;
    config.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr[2];
    int n = 0;
    if (C > 1) {
        attr[n].id = cudaLaunchAttributeClusterDimension;
        attr[n].val.clusterDim.x = C;
        attr[n].val.clusterDim.y = 1;
        attr[n].val.clusterDim.z = 1;
        ++n;
    }
    if (pdl) {                                   // may start while the previous kernel finishes: grid_wait guards x
        attr[n].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[n].val.programmaticStreamSerializationAllowed = 1;
        ++n;
    }
    config.attrs = attr;
    config.numAttrs = n;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
                                      xs.data_ptr<float>(), parts, M, K, M == 1 ? K : static_cast<int>(x.stride(0)),
                                      rows_t, C));
}

// Tiles never change bits; 0 picks by rows and chip: the PRO 6000, power-capped at one row, takes fewer MMAs a weight.
// 10-12 (``qmm.group_tile`` on SM 12.0 from 96 SMs) spread the slice sums and skip m16 tiles wholly past M. GB10
// streams 8-bit weights, twice the bytes a tile, best through 16 x 128 tiles, then the spread 64- and 128-row ones.
template <bool F32, int B>
void dispatch(int tile, int M, bool gb10, const at::Tensor& x, const at::Tensor& xs, Parts& parts, int C, bool pdl) {
    if (tile == 0 && gb10 && B == 8) tile = M <= 32 ? 8 : M <= 128 ? 11 : 10;
    if (tile == 0 && gb10) tile = M <= 16 ? 2 : M <= 32 ? 3 : M <= 64 ? 4 : 5;
    if (tile == 0) tile = M <= 8 ? 7 : M <= 16 ? 8 : M <= 32 ? 9 : M <= 64 ? 4 : 5;
    auto go = [&](auto full, auto skip, int bm) { M % bm ? skip() : full(); };
    switch (tile) {
        case 1: launch<64, 16, 64, 1, 4, 4, F32, B>(x, xs, parts, C, pdl); break;
        case 2: launch<64, 16, 64, 1, 4, 8, F32, B>(x, xs, parts, C, pdl); break;
        case 3: launch<64, 32, 64, 1, 4, 4, F32, B>(x, xs, parts, C, pdl); break;
        case 4: launch<64, 64, 64, 1, 4, 4, F32, B>(x, xs, parts, C, pdl); break;
        case 5: launch<64, 64, 128, 2, 4, 3, F32, B>(x, xs, parts, C, pdl); break;
        case 6: launch<64, 8, 64, 1, 4, 4, F32, B, true>(x, xs, parts, C, pdl); break;
        case 7: launch<64, 8, 128, 1, 4, 4, F32, B, true>(x, xs, parts, C, pdl); break;
        case 8: launch<64, 16, 128, 1, 8, 4, F32, B>(x, xs, parts, C, pdl); break;
        case 9: launch<64, 32, 128, 1, 8, 4, F32, B>(x, xs, parts, C, pdl); break;
        case 10: go([&] { launch<64, 128, 128, 2, 4, 2, F32, B, false, false, true>(x, xs, parts, C, pdl); },
                    [&] { launch<64, 128, 128, 2, 4, 2, F32, B, false, true, true>(x, xs, parts, C, pdl); }, 128);
                 break;
        case 11: go([&] { launch<64, 64, 128, 1, 8, 3, F32, B, false, false, true>(x, xs, parts, C, pdl); },
                    [&] { launch<64, 64, 128, 1, 8, 3, F32, B, false, true, true>(x, xs, parts, C, pdl); }, 64); break;
        case 12: go([&] { launch<64, 64, 128, 2, 4, 3, F32, B, false, false, true>(x, xs, parts, C, pdl); },
                    [&] { launch<64, 64, 128, 2, 4, 3, F32, B, false, true, true>(x, xs, parts, C, pdl); }, 64); break;
        default: TORCH_CHECK(false, "unknown group tile ", tile);
    }
}

} // namespace

void qmm_group_cuda(const at::Tensor& x, const at::Tensor& xs, const std::vector<at::Tensor>& ws,
                    const std::vector<at::Tensor>& scales, const std::vector<at::Tensor>& biases,
                    std::vector<at::Tensor>& outs, const std::vector<int64_t>& ns, const std::vector<int64_t>& sks,
                    bool f32, int tile, int pdl, int bits) {
    Parts parts = {};
    parts.count = static_cast<int>(ws.size());
    int C = 1;
    for (int i = 0; i < parts.count; ++i) {
        Part& P = parts.p[i];
        P.w = reinterpret_cast<const uint32_t*>(ws[i].data_ptr());
        P.scales = reinterpret_cast<const __nv_bfloat16*>(scales[i].data_ptr());
        P.biases = reinterpret_cast<const __nv_bfloat16*>(biases[i].data_ptr());
        P.out = outs[i].data_ptr();
        P.n = static_cast<int>(ns[i]);
        P.npad = static_cast<int>(scales[i].stride(0));
        P.sk = static_cast<int>(sks[i]);
        C = std::max(C, P.sk);
    }
    const auto* props = at::cuda::getCurrentDeviceProperties();
    const bool gb10 = props->major == 12 && props->minor == 1;
    const bool early = pdl < 0 ? gb10 : pdl > 0;           // overlapping launches pay on GB10, cost the capped PRO 6000
    if (bits == 8) {
        if (f32) dispatch<true, 8>(tile, x.size(0), gb10, x, xs, parts, C, early);
        else dispatch<false, 8>(tile, x.size(0), gb10, x, xs, parts, C, early);
    } else if (bits == 6) {
        if (f32) dispatch<true, 6>(tile, x.size(0), gb10, x, xs, parts, C, early);
        else dispatch<false, 6>(tile, x.size(0), gb10, x, xs, parts, C, early);
    } else if (bits == 5) {
        if (f32) dispatch<true, 5>(tile, x.size(0), gb10, x, xs, parts, C, early);
        else dispatch<false, 5>(tile, x.size(0), gb10, x, xs, parts, C, early);
    } else {
        if (f32) dispatch<true, 4>(tile, x.size(0), gb10, x, xs, parts, C, early);
        else dispatch<false, 4>(tile, x.size(0), gb10, x, xs, parts, C, early);
    }
}
