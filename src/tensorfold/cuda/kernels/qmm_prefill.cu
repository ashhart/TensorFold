// Prefill: w = bf16(fma(q, s, b)), one fp32 chain over K; chunk-invariant bits, not decode's (replies prefill again).

#include <ATen/ATen.h>
#include <algorithm>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "qmm_frag.cuh"

namespace {

using namespace qmm_frag;

// Tile shapes: BM rows by BN columns a block, WM x WN warps, each warp (BM / WM) x (BN / WN).
template <int GS, int BM, int BN, int WM, int WN, int STAGES>
struct Tile {
    static constexpr int THREADS = WM * WN * 32;
    static constexpr int MT = BM / WM / 16;               // m16 tiles a warp
    static constexpr int NT = BN / WN / 8;                // n8 tiles a warp
    static constexpr int ROW = GS * 2;                    // bytes of one input row a group
    static constexpr int CHUNKS = ROW / 16;
    static constexpr int X = BM * ROW;                    // stage bytes: inputs,
    static constexpr int W = BN * GS / 2;                 // weights,
    static constexpr int S = BN * 2;                      // scales and biases (bf16)
    static constexpr int STAGE = X + W + 2 * S;
    static constexpr int SMEM = STAGES * STAGE;
};

__device__ __forceinline__ uint32_t fma2(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

template <int GS, int BM, int BN, int WM, int WN, int STAGES, bool F32>
__global__ void __launch_bounds__(WM * WN * 32) prefill_kernel(
        const __nv_bfloat16* __restrict__ x, const uint32_t* __restrict__ w, const __nv_bfloat16* __restrict__ scales,
        const __nv_bfloat16* __restrict__ biases, void* __restrict__ out, int M, int N, int K, int npad, int ldx,
        int group) {
    using T = Tile<GS, BM, BN, WM, WN, STAGES>;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    const int KG = K / GS, per = KG, g0 = 0;
    const int2 at = tile_of(blockIdx.x, M, N, BM, BN, group);
    const int m0 = at.x, n0 = at.y;

    auto stage = [&](int s) { return buf + s * T::STAGE; };
    // one group's inputs, weights, scales and biases into stage s
    auto load = [&](int s, int g) {
        unsigned char* p = stage(s);
        for (int c = tid; c < BM * T::CHUNKS; c += T::THREADS) {
            const int r = c / T::CHUNKS, ch = c % T::CHUNKS;
            const int row = min(m0 + r, M - 1);
            cp16z(p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16, x + static_cast<size_t>(row) * ldx + g * GS + ch * 8,
                  m0 + r < M);
        }
        unsigned char* pw = p + T::X;
        constexpr int TILE_BYTES = 64 * GS / 2;           // one stored 64-column tile's group block
        for (int c = tid; c < T::W / 16; c += T::THREADS) {
            const int t = c / (TILE_BYTES / 16), off = c % (TILE_BYTES / 16);
            const size_t tile = static_cast<size_t>(n0 / 64 + t) * KG + g;
            cp16(pw + c * 16, reinterpret_cast<const unsigned char*>(w) + tile * TILE_BYTES + off * 16);
        }
        unsigned char* ps = pw + T::W;
        for (int c = tid; c < 2 * (T::S / 16); c += T::THREADS) {
            const int which = c / (T::S / 16), off = c % (T::S / 16);
            const __nv_bfloat16* src = (which ? biases : scales) + static_cast<size_t>(g) * npad + n0 + off * 8;
            cp16(ps + which * T::S + off * 16, src);
        }
    };

    float acc[T::MT][T::NT][4];
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < T::NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < per) load(s, g0 + s);
        commit();
    }
    for (int it = 0; it < per; ++it) {
        wait<STAGES - 2>();
        __syncthreads();
        const int next = it + STAGES - 1;
        if (next < per) load(next % STAGES, g0 + next);
        commit();
        const unsigned char* p = stage(it % STAGES);
        const uint32_t* pw = reinterpret_cast<const uint32_t*>(p + T::X);
        const uint16_t* ps = reinterpret_cast<const uint16_t*>(p + T::X + T::W);
        uint32_t words[T::NT][GS / 32], sv[T::NT], bv[T::NT];
#pragma unroll
        for (int j = 0; j < T::NT; ++j) {
#pragma unroll
            for (int v = 0; v < GS / 32; ++v) words[j][v] = pw[((wn * T::NT + j) * 32 + lane) * (GS / 32) + v];
            const int col = wn * (BN / WN) + j * 8 + (lane >> 2);             // this lane's B-fragment column
            sv[j] = ps[col] * 0x10001u;                                       // (s, s) and (b, b) as bf16 pairs
            bv[j] = ps[BN + col] * 0x10001u;
        }
#pragma unroll
        for (int kt = 0; kt < GS / 16; ++kt) {
            uint32_t a[T::MT][4];
#pragma unroll
            for (int i = 0; i < T::MT; ++i) {
                const int r = wm * (BM / WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                const int ch = kt * 2 + (lane >> 4);
                ldmatrix4(a[i], p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16);
            }
#pragma unroll
            for (int j = 0; j < T::NT; ++j) {
                const uint32_t b0 = fma2(pair(words[j][kt / 2], (kt & 1) * 8), sv[j], bv[j]);
                const uint32_t b1 = fma2(pair(words[j][kt / 2], (kt & 1) * 8 + 4), sv[j], bv[j]);
#pragma unroll
                for (int i = 0; i < T::MT; ++i) mma(acc[i][j], a[i], b0, b1);
            }
        }
    }
    wait<0>();
    __syncthreads();
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < T::NT; ++j) {
            const int col = n0 + wn * (BN / WN) + j * 8 + (lane & 3) * 2;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + wm * (BM / WM) + i * 16 + (lane >> 2) + h * 8;
                if (row >= M) continue;
                const float v0 = acc[i][j][2 * h], v1 = acc[i][j][2 * h + 1];
                if (F32) {
                    float* dst = reinterpret_cast<float*>(out) + static_cast<size_t>(row) * N + col;
                    if (col < N) dst[0] = v0;
                    if (col + 1 < N) dst[1] = v1;
                } else {
                    __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(out) + static_cast<size_t>(row) * N + col;
                    if (col + 1 < N && (N & 1) == 0)
                        *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v0, v1);
                    else {
                        if (col < N) dst[0] = __float2bfloat16_rn(v0);
                        if (col + 1 < N) dst[1] = __float2bfloat16_rn(v1);
                    }
                }
            }
        }
}

template <int GS, int BM, int BN, int WM, int WN, int STAGES, bool F32>
void launch(const at::Tensor& x, const at::Tensor& w, const at::Tensor& scales, const at::Tensor& biases,
            at::Tensor& out, int N) {
    using T = Tile<GS, BM, BN, WM, WN, STAGES>;
    const int M = x.size(0), K = x.size(1);
    auto kernel = prefill_kernel<GS, BM, BN, WM, WN, STAGES, F32>;
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM);
        configured = true;
    }
    const int rows_t = (M + BM - 1) / BM;
    // a group's inputs stay near 12 MB of L2 while its blocks sweep the column tiles
    const int group = std::max(1, std::min(rows_t, static_cast<int>((12LL << 20) / (static_cast<long long>(BM) * K * 2))));
    kernel<<<rows_t * ((N + BN - 1) / BN), T::THREADS, T::SMEM, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const uint32_t*>(w.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(scales.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(biases.data_ptr()),
        out.data_ptr(), M, N, K, static_cast<int>(scales.size(1)), M == 1 ? K : static_cast<int>(x.stride(0)), group);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int GS, bool F32>
void dispatch(int tile, const at::Tensor& x, const at::Tensor& w, const at::Tensor& s, const at::Tensor& b,
              at::Tensor& out, int N) {
    switch (tile) {
        case 0: launch<GS, 128, 128, 2, 4, 3, F32>(x, w, s, b, out, N); break;
        case 1: launch<GS, 64, 128, 1, 4, 4, F32>(x, w, s, b, out, N); break;
        case 2: launch<GS, 128, 64, 2, 2, 4, F32>(x, w, s, b, out, N); break;
        case 3: launch<GS, 64, 64, 1, 4, 4, F32>(x, w, s, b, out, N); break;
        default: launch<GS, 128, 128, 2, 4, 4, F32>(x, w, s, b, out, N); break;
    }
}

} // namespace

// ``tile`` (0: 128x128, 3 stages; 1: 64x128; 2: 128x64; 3: 64x64; 4: 128x128, 4 stages) never changes a row's bits.
void qmm_prefill_cuda(const at::Tensor& x, const at::Tensor& w, const at::Tensor& scales, const at::Tensor& biases,
                      at::Tensor& out, int N, int gs, bool f32, int tile) {
    if (gs == 64) { if (f32) dispatch<64, true>(tile, x, w, scales, biases, out, N); else dispatch<64, false>(tile, x, w, scales, biases, out, N); }
    else { if (f32) dispatch<32, true>(tile, x, w, scales, biases, out, N); else dispatch<32, false>(tile, x, w, scales, biases, out, N); }
}
