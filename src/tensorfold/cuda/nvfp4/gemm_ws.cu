// Prompt GEMM on bulk copies: each stage arrives by cp.async.bulk counted on an mbarrier, and the last of the eight
// mma warps done with a stage refills it, so the loop has no block barrier and no per-thread copy addressing. Rows come
// from the quantizers' tiled, swizzled layout. One K chain a row: a row's bits never depend on its chunk or the tile.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "../kernels/qmm_frag.cuh"
#include "mma4.cuh"

namespace {

using namespace qmm_frag;
using namespace mma4;

// Bulk copies and transaction-counted mbarriers are sm_90 on: older GPUs never launch this file's kernel
// (checkpoint.bulk_tile sends them to gemm_ck.cu) and build it as traps.
#if __CUDA_ARCH__ >= 900 || !defined(__CUDA_ARCH__)
#define WS_ASM(...) asm volatile(__VA_ARGS__)
#else
#define WS_ASM(...) __trap()
#endif

__device__ __forceinline__ void mbar_init(uint64_t* b, int count) {
    WS_ASM("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem(b)), "r"(count) : "memory");
}

__device__ __forceinline__ void mbar_expect(uint64_t* b, uint32_t bytes) {
    WS_ASM("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem(b)), "r"(bytes) : "memory");
}

__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {
    WS_ASM("{\n .reg .pred p;\n W_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
           " @!p bra W_%=;\n}\n" ::"r"(smem(b)), "r"(parity) : "memory");
}

// ``bytes`` from global into this block's shared memory, completing a transaction count on ``b``. The destination is
// shared::cta (PTX ISA 8.6, CUDA 12.8 on): a shared::cluster one makes ptxas guard every copy with a runtime call
// for remote blocks, and that call has the driver reserve 14.5 KB of stack a resident thread. Older toolkits (a
// CUDA 12.0-12.6 pip route on sm_90) keep the shared::cluster form and that stack.
#if (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 8)) && \
    !defined(TF_BULK_CLUSTER)
#define TF_BULK_DST "shared::cta"
#else
#define TF_BULK_DST "shared::cluster"
#endif
__device__ __forceinline__ void bulk(void* dst, const void* src, uint32_t bytes, uint64_t* b) {
    WS_ASM("cp.async.bulk." TF_BULK_DST ".global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
           ::"r"(smem(dst)), "l"(src), "r"(bytes), "r"(smem(b)) : "memory");
}

template <int MODE, int BM, int BN, int STAGES, int KS>
struct WS {
    static constexpr int CONS = 8, THREADS = CONS * 32;                // eight mma warps, 2 x 4 of 64 x BN / 4
    static constexpr int MT = BM / 2 / 16, NT = BN / 4 / 8, TILES = BN / 64;
    static constexpr int ROW = MODE == A4 ? 32 : 64, TILE = MODE == A4 ? 2048 : 4096, TS = MODE == A4 ? 256 : 0;
    static constexpr int X = BM * ROW, SX = MODE == A4 ? BM * 4 : 0;   // a step's rows and row scales
    static constexpr int W_AT = KS * (X + SX), SW_AT = W_AT + TILES * KS * TILE;
    static constexpr int STAGE = SW_AT + TILES * KS * TS;              // KS steps: rows, scales, then each tile's
    static constexpr int SMEM = STAGES * STAGE + STAGES * 8 + STAGES * 4;
    static constexpr int COPIES = MODE == A4 ? 2 + 2 * TILES : 1 + TILES;
};

// Stage ``s`` <- steps [kt KS, kt KS + KS): one bulk copy a lane (rows, row scales, each tile's weights and scales,
// each KS steps long), counted on ``full``. The whole warp calls it.
template <int MODE, int BM, int BN, int STAGES, int KS>
__device__ __forceinline__ void issue(unsigned char* buf, uint64_t* full, int s, int kt, const uint8_t* x,
                                      const uint8_t* xs, const uint8_t* w, const uint8_t* ws, int m0, int n0, int KG,
                                      int live, int lane) {
    using G = WS<MODE, BM, BN, STAGES, KS>;
    if (lane == 0) mbar_expect(full + s, KS * (G::X + G::SX + live * (G::TILE + G::TS)));
    __syncwarp();
    unsigned char* p = buf + s * G::STAGE;
    const size_t row = static_cast<size_t>(m0 / BM) * KG + kt * KS;     // tiled rows [mpad / BM][K/64][BM][...]
    if (lane == 0) {
        bulk(p, x + row * G::X, KS * G::X, full + s);
    } else if (MODE == A4 && lane == 1) {
        bulk(p + KS * G::X, xs + row * G::SX, KS * G::SX, full + s);
    } else {
        const int o = lane - (MODE == A4 ? 2 : 1), tl = o % G::TILES;
        const size_t wt = static_cast<size_t>(n0 / 64 + tl) * KG + kt * KS;
        if (o < live) bulk(p + G::W_AT + tl * KS * G::TILE, w + wt * G::TILE, KS * G::TILE, full + s);
        else if (MODE == A4 && o >= G::TILES && o < 2 * G::TILES && tl < live)
            bulk(p + G::SW_AT + tl * KS * G::TS, ws + wt * G::TS, KS * G::TS, full + s);
    }
}

// x: tiled rows [mpad / BM][K/64][BM][ROW], A4 scales [mpad / BM][K/64][BM][4]; w, ws: lane4.cu's words and scales.
template <int MODE, int BM, int BN, int STAGES, int KS, bool F32>
__global__ void __launch_bounds__(WS<MODE, BM, BN, STAGES, KS>::THREADS) ws_kernel(
        const uint8_t* __restrict__ x, const uint8_t* __restrict__ xs, const uint8_t* __restrict__ w,
        const uint8_t* __restrict__ ws, float alpha, void* __restrict__ out, int M, int N, int K, int mpad, int npad,
        int group) {
    using G = WS<MODE, BM, BN, STAGES, KS>;
    extern __shared__ __align__(128) unsigned char buf[];
    uint64_t* full = reinterpret_cast<uint64_t*>(buf + STAGES * G::STAGE);
    unsigned* done = reinterpret_cast<unsigned*>(full + STAGES);          // warps through each stage, ever
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int2 at = tile_of(blockIdx.x, mpad, npad, BM, BN, group);
    const int m0 = at.x, n0 = at.y, KG = K / 64, KT = KG / KS, live = min(G::TILES, npad / 64 - n0 / 64);
    if (tid == 0) {
        for (int s = 0; s < STAGES; ++s) {
            mbar_init(full + s, 1);
            done[s] = 0;
        }
        WS_ASM("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncwarp();
    if (warp == 0)
        for (int s = 0; s < STAGES && s < KT; ++s)
            issue<MODE, BM, BN, STAGES, KS>(buf, full, s, s, x, xs, w, ws, m0, n0, KG, live, lane);
    __syncthreads();
    const int wm = warp / 4, wn = warp % 4, g = lane >> 2, t = lane & 3, q = lane >> 3, rr = lane & 7;
    float acc[G::MT][G::NT][4];
#pragma unroll
    for (int i = 0; i < G::MT; ++i)
#pragma unroll
        for (int j = 0; j < G::NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;
    for (int kt = 0; kt < KT; ++kt) {
        const int s = kt % STAGES;
        mbar_wait(full + s, (kt / STAGES) & 1);
#pragma unroll
        for (int u = 0; u < KS; ++u) {
            const unsigned char* base = buf + s * G::STAGE;
            const unsigned char* p = base + u * G::X;
            if constexpr (MODE == A4) {
                const uint32_t* sx = reinterpret_cast<const uint32_t*>(base + KS * G::X + u * G::SX);
                uint32_t a[G::MT][4], sa[G::MT];
#pragma unroll
                for (int i = 0; i < G::MT; ++i) {
                    const int rb = wm * (BM / 2) + i * 16, r = rb + rr + (q & 1) * 8;
                    ldmatrix4(a[i], p + r * G::ROW + chunk<MODE>(r, q >> 1) * 16);
                    const uint32_t v = sx[rb + g + 8 * (t & 1)];
                    sa[i] = v;                         // lanes 2, 3 repeat rows g, g + 8: never read
                }
#pragma unroll
                for (int j = 0; j < G::NT; ++j) {
                    const int jj = wn * G::NT + j, tl = jj / 8, jt = jj % 8;
                    const uint2 b = reinterpret_cast<const uint2*>(base + G::W_AT + (tl * KS + u) * G::TILE)[jt * 32 +
                                                                                                           lane];
                    const uint32_t sb = reinterpret_cast<const uint32_t*>(base + G::SW_AT + (tl * KS + u) * G::TS)[
                        jt * 8 + g];                       // every lane its column's: lane 0's is read
#pragma unroll
                    for (int i = 0; i < G::MT; ++i) mma_fp4(acc[i][j], a[i], b.x, b.y, sa[i], sb);
                }
            } else {
                uint4 b[G::NT];
#pragma unroll
                for (int j = 0; j < G::NT; ++j) {
                    const int jj = wn * G::NT + j;
                    b[j] = reinterpret_cast<const uint4*>(base + G::W_AT + (jj / 8 * KS + u) * G::TILE)[jj % 8 * 32 +
                                                                                                       lane];
                }
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    uint32_t a[G::MT][4];
#pragma unroll
                    for (int i = 0; i < G::MT; ++i) {
                        const int r = wm * (BM / 2) + i * 16 + rr + (q & 1) * 8;
                        ldmatrix4(a[i], p + r * G::ROW + chunk<MODE>(r, 2 * h + (q >> 1)) * 16);
                    }
#pragma unroll
                    for (int j = 0; j < G::NT; ++j)
#pragma unroll
                        for (int i = 0; i < G::MT; ++i)
                            mma_fp8(acc[i][j], a[i], h ? b[j].z : b[j].x, h ? b[j].w : b[j].y);
                }
            }
        }
        __syncwarp();
        unsigned last = 0;
        if (lane == 0) {
            __threadfence_block();
            last = atomicAdd(done + s, 1u) % G::CONS == G::CONS - 1;
            __threadfence_block();
        }
        if (__shfl_sync(0xffffffffu, last, 0) && kt + STAGES < KT) {      // the stage's last warp refills it
            WS_ASM("fence.proxy.async.shared::cta;\n" ::: "memory");
            issue<MODE, BM, BN, STAGES, KS>(buf, full, s, kt + STAGES, x, xs, w, ws, m0, n0, KG, live, lane);
        }
    }
#pragma unroll
    for (int i = 0; i < G::MT; ++i)
#pragma unroll
        for (int j = 0; j < G::NT; ++j) {
            const int col = n0 + wn * (BN / 4) + j * 8 + t * 2;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + wm * (BM / 2) + i * 16 + g + h * 8;
                if (row >= M) continue;
                const float v0 = acc[i][j][2 * h] * alpha, v1 = acc[i][j][2 * h + 1] * alpha;
                if (F32) {
                    float* dst = reinterpret_cast<float*>(out) + static_cast<size_t>(row) * N + col;
                    if (col < N) dst[0] = v0;
                    if (col + 1 < N) dst[1] = v1;
                } else {
                    __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(out) + static_cast<size_t>(row) * N + col;
                    if (col + 1 < N && (N & 1) == 0) {
                        *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v0, v1);
                    } else {
                        if (col < N) dst[0] = __float2bfloat16_rn(v0);
                        if (col + 1 < N) dst[1] = __float2bfloat16_rn(v1);
                    }
                }
            }
        }
}

template <int MODE, int BM, int BN, int STAGES, int KS, bool F32>
void launch(const at::Tensor& x, const at::Tensor& xs, const at::Tensor& w, const at::Tensor& ws, double alpha,
            at::Tensor& out, int M, int N, int K, int mpad, int npad) {
    using G = WS<MODE, BM, BN, STAGES, KS>;
    TORCH_CHECK((K / 64) % KS == 0 && mpad % BM == 0, "K in whole stages, rows tiled by the GEMM's height");
    auto kernel = ws_kernel<MODE, BM, BN, STAGES, KS, F32>;
    static bool configured = false;
    if (!configured) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM));
        configured = true;
    }
    const int rows_t = mpad / BM, cols_t = (npad + BN - 1) / BN;
    const long long row_bytes = static_cast<long long>(BM) * (MODE == A4 ? K / 2 : K);
    const int group = std::max(1, std::min(rows_t, static_cast<int>((12LL << 20) / row_bytes)));
    kernel<<<rows_t * cols_t, G::THREADS, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint8_t*>(x.data_ptr()), xs.defined() ? reinterpret_cast<const uint8_t*>(xs.data_ptr())
        : nullptr, reinterpret_cast<const uint8_t*>(w.data_ptr()), ws.defined() ? reinterpret_cast<const uint8_t*>(
        ws.data_ptr()) : nullptr, static_cast<float>(alpha), out.data_ptr(), M, N, K, mpad, npad, group);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Tiles never change a row's bits (EXPERIMENT table): 1 128 x 256 a step a stage, 2 128 x 256 two steps a stage,
// 3 128 x 128 two steps a stage.
template <int MODE, bool F32>
void by_tile(int tile, const at::Tensor& x, const at::Tensor& xs, const at::Tensor& w, const at::Tensor& ws,
             double alpha, at::Tensor& out, int M, int N, int K, int mpad, int npad) {
    switch (tile) {
        case 2: launch<MODE, 128, 256, MODE == A4 ? 3 : 2, 2, F32>(x, xs, w, ws, alpha, out, M, N, K, mpad, npad);
            break;
        case 3: launch<MODE, 128, 128, 3, 2, F32>(x, xs, w, ws, alpha, out, M, N, K, mpad, npad); break;
        default: launch<MODE, 128, 256, MODE == A4 ? 4 : 3, 1, F32>(x, xs, w, ws, alpha, out, M, N, K, mpad, npad);
    }
}

}  // namespace

void gemm_ws_cuda(int64_t mode, const at::Tensor& x, const at::Tensor& xs, const at::Tensor& w, const at::Tensor& ws,
                  double alpha, at::Tensor& out, int64_t N, int64_t K, int64_t mpad, int64_t npad, int64_t tile,
                  bool f32) {
    const int M = static_cast<int>(out.size(0)), n = static_cast<int>(N), k = static_cast<int>(K);
    const int mp = static_cast<int>(mpad), np = static_cast<int>(npad), tl = static_cast<int>(tile);
    if (mode == A4) {
        if (f32) by_tile<A4, true>(tl, x, xs, w, ws, alpha, out, M, n, k, mp, np);
        else by_tile<A4, false>(tl, x, xs, w, ws, alpha, out, M, n, k, mp, np);
    } else {
        if (f32) by_tile<A8, true>(tl, x, xs, w, ws, alpha, out, M, n, k, mp, np);
        else by_tile<A8, false>(tl, x, xs, w, ws, alpha, out, M, n, k, mp, np);
    }
}
