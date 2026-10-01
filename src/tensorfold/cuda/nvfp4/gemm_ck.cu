// Prompt GEMM in the checkpoint's own math: NVFP4 rows times NVFP4 weights (block-scaled FP4 mma) and FP8 rows times
// FP8 weights (e4m3 mma), in lane4.cu's layouts. One fp32 chain over K a row: a row's bits never depend on its chunk.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "../kernels/qmm_frag.cuh"
#include "mma4.cuh"
#include "swiglu4.cuh"

namespace {

using namespace qmm_frag;
using namespace mma4;

template <int MODE, int BM, int BN, int WM, int WN, int STAGES, int KS>
struct Gemm {
    static constexpr int THREADS = WM * WN * 32;
    static constexpr int MT = BM / WM / 16, NT = BN / WN / 8;          // m16 and n8 tiles a warp
    static constexpr int ROW = MODE == A4 ? 32 : 64;                     // bytes a row a step of 64 inputs
    static constexpr int TILE = MODE == A4 ? 2048 : 4096;                // a stored 64-column tile's step
    static constexpr int X = BM * ROW, SX = MODE == A4 ? BM * 4 : 0;     // a step's rows and row scales,
    static constexpr int W = BN / 64 * TILE, SW = MODE == A4 ? BN * 4 : 0;   // weights and their scales
    static constexpr int STEP = X + SX + W + SW;
    static constexpr int STAGE = (KS * STEP + 127) / 128 * 128;          // KS steps a stage
    static constexpr int SMEM = STAGES * STAGE;
};

// The fused gate|up launch's second weight (up), its factor, and down's input it writes: NVFP4 rows [M, kd/2] and
// scales [kd/64, mpad, 4] under down's global scale qg.
struct Up {
    const uint8_t* w;
    const uint8_t* ws;
    float alpha;
    uint8_t* codes;
    uint8_t* scales;
    float qg;
    int kd;
};

// x: A4 codes [M, K/2] and scales [K/64, mpad, 4]; A8 e4m3 [M, K] in fragment order. w: lane4.cu's words or bytes,
// ws its block scales [npad/64, K/64, 64, 4]. out (M, N) = alpha * the product, every row one K chain. EPI 1, 2:
// w is gate and up.w up, a block their same 128 columns, written as SiLU(gate) * up in NVFP4 (swiglu4.cuh).
template <int MODE, int BM, int BN, int WM, int WN, int STAGES, int KS, bool F32, int EPI = 0>
__global__ void __launch_bounds__(WM * WN * 32) gemm_kernel(
        const uint8_t* __restrict__ x, const uint8_t* __restrict__ xs, const uint8_t* __restrict__ w,
        const uint8_t* __restrict__ ws, float alpha, void* __restrict__ out, int M, int N, int K, int mpad, int npad,
        int group, Up up) {
    using G = Gemm<MODE, BM, BN, WM, WN, STAGES, KS>;
    constexpr bool GU = EPI > 0;
    static_assert(!GU || (MODE == A4 && BM == 128 && BN == 256 && WM == 2 && WN == 4), "128 x 256, 64 x 64 warps");
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp / WN, wn = warp % WN;
    const int g = lane >> 2, t = lane & 3, q = lane >> 3, rr = lane & 7;
    const int2 at = tile_of(blockIdx.x, M, npad, BM, GU ? BN / 2 : BN, group);
    const int m0 = at.x, n0 = at.y, KG = K / 64, tiles = npad / 64;
    constexpr int CH = G::ROW / 16;

    auto stage = [&](int s) { return buf + s * G::STAGE; };
    auto load = [&](unsigned char* p, int kg) {                     // one step of 64 inputs at p
#pragma unroll
        for (int c = tid; c < BM * CH; c += G::THREADS) {
            const int r = c / CH, ch = c % CH;
            const bool in = m0 + r < M;
            cp16z(p + r * G::ROW + chunk<MODE>(r, ch) * 16,
                  x + static_cast<size_t>(in ? m0 + r : 0) * (K / (MODE == A4 ? 2 : 1)) + kg * G::ROW + ch * 16, in);
        }
        if constexpr (MODE == A4) {
            for (int c = tid; c < BM / 4; c += G::THREADS)
                cp16z(p + G::X + c * 16, xs + (static_cast<size_t>(kg) * mpad + m0 + 4 * c) * 4, m0 + 4 * c < mpad);
        }
        unsigned char* pw = p + G::X + G::SX;
#pragma unroll
        for (int c = tid; c < G::W / 16; c += G::THREADS) {
            const int tl = c / (G::TILE / 16), off = c % (G::TILE / 16), wt = n0 / 64 + (GU ? tl % 2 : tl);
            const size_t src = (static_cast<size_t>(min(wt, tiles - 1)) * KG + kg) * G::TILE + off * 16;
            cp16z(pw + c * 16, (GU && tl >= 2 ? up.w : w) + src, wt < tiles);
        }
        if constexpr (MODE == A4) {
            for (int c = tid; c < G::SW / 16; c += G::THREADS) {
                const int tl = c / 16, off = c % 16, wt = n0 / 64 + (GU ? tl % 2 : tl);
                cp16z(pw + G::W + c * 16, (GU && tl >= 2 ? up.ws : ws) + (static_cast<size_t>(min(wt, tiles - 1)) *
                      KG + kg) * 256 + off * 16, wt < tiles);
            }
        }
    };

    float acc[G::MT][G::NT][4];
#pragma unroll
    for (int i = 0; i < G::MT; ++i)
#pragma unroll
        for (int j = 0; j < G::NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;
    const int KT = KG / KS;                                              // stages over K (KG a multiple of KS)
    auto fill = [&](int s, int kt) {
#pragma unroll
        for (int u = 0; u < KS; ++u) load(stage(s) + u * G::STEP, kt * KS + u);
    };
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KT) fill(s, s);
        commit();
    }
    for (int kt = 0; kt < KT; ++kt) {
        wait<STAGES - 2>();
        __syncthreads();
        if (kt + STAGES - 1 < KT) fill((kt + STAGES - 1) % STAGES, kt + STAGES - 1);
        commit();
#pragma unroll
        for (int u = 0; u < KS; ++u) {
        const unsigned char* p = stage(kt % STAGES) + u * G::STEP;
        const unsigned char* pw = p + G::X + G::SX;
        if constexpr (MODE == A4) {
            const uint32_t* sx = reinterpret_cast<const uint32_t*>(p + G::X);
            const uint32_t* sw = reinterpret_cast<const uint32_t*>(pw + G::W);
            uint32_t a[G::MT][4], sa[G::MT];
#pragma unroll
            for (int i = 0; i < G::MT; ++i) {
                const int base = wm * (BM / WM) + i * 16, r = base + rr + (q & 1) * 8;
                ldmatrix4(a[i], p + r * G::ROW + chunk<MODE>(r, q >> 1) * 16);
                const uint32_t v = sx[base + g + 8 * (t & 1)];   // row g (t 0), g + 8 (t 1): one load, no branch
                sa[i] = v;                         // lanes 2, 3 repeat rows g, g + 8: never read
            }
#pragma unroll
            for (int j = 0; j < G::NT; ++j) {
                const int jj = wn * G::NT + j;
                const uint2 b = reinterpret_cast<const uint2*>(pw)[jj * 32 + lane];
                const uint32_t sb = sw[jj * 8 + g];          // every lane its column's: lane 0's is read
#pragma unroll
                for (int i = 0; i < G::MT; ++i) mma_fp4(acc[i][j], a[i], b.x, b.y, sa[i], sb);
            }
        } else {
            uint4 b[G::NT];
#pragma unroll
            for (int j = 0; j < G::NT; ++j) b[j] = reinterpret_cast<const uint4*>(pw)[(wn * G::NT + j) * 32 + lane];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                uint32_t a[G::MT][4];
#pragma unroll
                for (int i = 0; i < G::MT; ++i) {
                    const int r = wm * (BM / WM) + i * 16 + rr + (q & 1) * 8;
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
    }
    if constexpr (GU) {                                    // row-major rows [M, kd/2], scales [kd/64, mpad, 4]
        wait<0>();
        auto put = [&](int r, int c, int b, uint32_t lo, uint32_t hi) {
            if (m0 + r < M)
                *reinterpret_cast<uint2*>(up.codes + static_cast<size_t>(m0 + r) * (up.kd / 2) + (n0 / 64 + c) * 32 +
                                          b * 8) = make_uint2(lo, hi);
        };
        auto scales = [&](int r, int c, uint32_t sw) {
            if (m0 + r < mpad)
                *reinterpret_cast<uint32_t*>(up.scales + ((static_cast<size_t>(n0 / 64 + c)) * mpad + m0 + r) * 4) =
                    sw;
        };
        swiglu4::epilogue<EPI>(acc, buf, wm, wn, lane, alpha, up.alpha, up.qg, min(2, tiles - n0 / 64), put, scales);
        return;
    }
#pragma unroll
    for (int i = 0; i < G::MT; ++i)
#pragma unroll
        for (int j = 0; j < G::NT; ++j) {
            const int col = n0 + wn * (BN / WN) + j * 8 + t * 2;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + wm * (BM / WM) + i * 16 + g + h * 8;
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

template <int MODE, int BM, int BN, int WM, int WN, int STAGES, int KS, bool F32, int EPI = 0>
void launch(const at::Tensor& x, const at::Tensor& xs, const at::Tensor& w, const at::Tensor& ws, double alpha,
            at::Tensor& out, int M, int N, int K, int mpad, int npad, Up up = {}) {
    using G = Gemm<MODE, BM, BN, WM, WN, STAGES, KS>;
    TORCH_CHECK((K / 64) % KS == 0, "K must hold whole stages");
    static_assert(EPI == 0 || G::SMEM >= swiglu4::XCH, "the stages hold the epilogue's hand-over");
    auto kernel = gemm_kernel<MODE, BM, BN, WM, WN, STAGES, KS, F32, EPI>;
    static bool configured = false;
    if (!configured) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM));
        configured = true;
    }
    const int rows_t = (M + BM - 1) / BM, cols_t = (npad + (EPI ? BN / 2 : BN) - 1) / (EPI ? BN / 2 : BN);
    const long long row_bytes = static_cast<long long>(BM) * (MODE == A4 ? K / 2 : K);
    const int group = std::max(1, std::min(rows_t, static_cast<int>((12LL << 20) / row_bytes)));   // rows near L2
    kernel<<<rows_t * cols_t, G::THREADS, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint8_t*>(x.data_ptr()), xs.defined() ? reinterpret_cast<const uint8_t*>(xs.data_ptr())
        : nullptr, reinterpret_cast<const uint8_t*>(w.data_ptr()), ws.defined() ? reinterpret_cast<const uint8_t*>(
        ws.data_ptr()) : nullptr, static_cast<float>(alpha), out.defined() ? out.data_ptr() : nullptr, M, N, K, mpad,
        npad, group, up);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Tiles never change a row's bits: 0 picks by chip and rows, 1 128 x 128 (64 x 32 warps, a step a stage), 2 128 x 256
// (64 x 64 warps, two steps a stage), 3 64 x 128 (32 x 32 warps).
template <int MODE, bool F32>
void by_tile(int tile, bool gb10, const at::Tensor& x, const at::Tensor& xs, const at::Tensor& w, const at::Tensor& ws,
             double alpha, at::Tensor& out, int M, int N, int K, int mpad, int npad) {
    if (tile == 0) tile = !gb10 && M >= 512 && (K / 64) % 2 == 0 ? 2 : 1;
    switch (tile) {
        case 2: launch<MODE, 128, 256, 2, 4, 2, 2, F32>(x, xs, w, ws, alpha, out, M, N, K, mpad, npad); break;
        case 3: launch<MODE, 64, 128, 2, 4, 4, 1, F32>(x, xs, w, ws, alpha, out, M, N, K, mpad, npad); break;
        default: launch<MODE, 128, 128, 2, 4, MODE == A4 ? 4 : 3, 1, F32>(x, xs, w, ws, alpha, out, M, N, K, mpad,
                                                                              npad);
    }
}

}  // namespace

void gemm_cuda(int64_t mode, const at::Tensor& x, const at::Tensor& xs, const at::Tensor& w, const at::Tensor& ws,
               double alpha, at::Tensor& out, int64_t N, int64_t K, int64_t npad, int64_t tile, bool f32) {
    const int M = static_cast<int>(out.size(0)), n = static_cast<int>(N), k = static_cast<int>(K);
    const int mpad = mode == A4 ? static_cast<int>(xs.size(1)) : 0, np = static_cast<int>(npad);
    const int tl = static_cast<int>(tile);
    const auto* props = at::cuda::getCurrentDeviceProperties();
    const bool gb10 = props->major == 12 && props->minor == 1;
    if (mode == A4) {
        if (f32) by_tile<A4, true>(tl, gb10, x, xs, w, ws, alpha, out, M, n, k, mpad, np);
        else by_tile<A4, false>(tl, gb10, x, xs, w, ws, alpha, out, M, n, k, mpad, np);
    } else {
        if (f32) by_tile<A8, true>(tl, gb10, x, xs, w, ws, alpha, out, M, n, k, mpad, np);
        else by_tile<A8, false>(tl, gb10, x, xs, w, ws, alpha, out, M, n, k, mpad, np);
    }
}

// gate|up of the same NVFP4 rows (``quant4``'s row-major layout) -> SiLU(gate) * up -> down's input as NVFP4 rows
// ``codes`` (M, kd/2) and ``scales`` (kd/64, mpad, 4) under down's global scale ``qg`` (``fp32``: no bf16 rounding).
void gemm_gu_ck_cuda(const at::Tensor& x, const at::Tensor& xs, const at::Tensor& wg, const at::Tensor& wsg,
                     const at::Tensor& wu, const at::Tensor& wsu, double alpha_g, double alpha_u, int64_t M,
                     int64_t npad, int64_t K, double qg, at::Tensor& codes, at::Tensor& scales, bool fp32) {
    const Up up = {reinterpret_cast<const uint8_t*>(wu.data_ptr()), reinterpret_cast<const uint8_t*>(wsu.data_ptr()),
                   static_cast<float>(alpha_u), codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),
                   static_cast<float>(qg), static_cast<int>(npad)};
    at::Tensor none;
    const int m = static_cast<int>(M), k = static_cast<int>(K), mpad = static_cast<int>(xs.size(1));
    const int np = static_cast<int>(npad);
    if (fp32) launch<A4, 128, 256, 2, 4, 3, 2, false, 2>(x, xs, wg, wsg, alpha_g, none, m, np, k, mpad, np, up);
    else launch<A4, 128, 256, 2, 4, 3, 2, false, 1>(x, xs, wg, wsg, alpha_g, none, m, np, k, mpad, np, up);
}
