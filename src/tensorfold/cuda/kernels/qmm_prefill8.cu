// FP8 prefill on exact e4m3 weights (4-bit words, or staged e4m3 bytes with W8): acc = fma(P, s, acc) per group,
// bias MMAs for 4-bit words, row scale; no row affects another.

#include <ATen/ATen.h>
#include <algorithm>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "qmm_frag.cuh"

namespace {

using namespace qmm_frag;

__device__ __forceinline__ void mma8(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma8z(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    const float z = 0.0f;
    asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%10, %10, %10, %10};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(z));
}

// e4m3 of the nibbles at bits [s, s + 4) (low byte) and [16 + s, 20 + s) (high byte): f16 1024 + q, minus 1024, cvt.
__device__ __forceinline__ uint32_t e4m3_pair(uint32_t w, int s) {
    const uint32_t t = ((w >> s) & 0x000F000Fu) | 0x64006400u;
    uint32_t h;
    asm("sub.rn.f16x2 %0, %1, %2;\n" : "=r"(h) : "r"(t), "r"(0x64006400u));
    uint16_t r;
    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;\n" : "=h"(r) : "r"(h));
    return r;
}

// One k32 step's weight fragment from a lane's word: nibble slots (0, 4, 1, 5) and (2, 6, 3, 7) as e4m3.
__device__ __forceinline__ void weights8(uint32_t w, uint32_t& b0, uint32_t& b1) {
    b0 = __byte_perm(e4m3_pair(w, 0), e4m3_pair(w, 4), 0x5410);
    b1 = __byte_perm(e4m3_pair(w, 8), e4m3_pair(w, 12), 0x5410);
}

template <int GS, int BM, int BN, int WM, int WN, int STAGES, bool W8 = false>
struct Tile {
    static constexpr int THREADS = WM * WN * 32;
    static constexpr int MT = BM / WM / 16;
    static constexpr int NT = BN / WN / 8;
    static constexpr int KS = GS / 32;                    // k32 steps a group
    static constexpr int ROW = GS;                        // bytes of one 8-bit input row a group
    static constexpr int CHUNKS = ROW / 16;
    static constexpr int PER128 = 128 / ROW;              // rows in 128 bytes of shared memory
    static constexpr int X = BM * ROW;
    static constexpr int W = W8 ? BN * GS : BN * GS / 2;   // e4m3 bytes, or 4-bit words
    static constexpr int S = BN * 2;
    static constexpr int STAGE = X + W + S;
    static constexpr int SMEM = STAGES * STAGE;
};

// 16-byte chunk c of row r sits at c ^ ((r / PER128) % CHUNKS): ldmatrix's eight rows cover all 32 banks
template <int CHUNKS, int PER128>
__device__ __forceinline__ int swz8(int r, int c) { return c ^ ((r / PER128) % CHUNKS); }

template <int GS, int BM, int BN, int WM, int WN, int STAGES, bool F32, bool W8>
__global__ void __launch_bounds__(WM * WN * 32, 2) prefill8_kernel(
        const uint8_t* __restrict__ x, const __nv_bfloat16* __restrict__ xs, const float* __restrict__ scale,
        const uint32_t* __restrict__ w, const __nv_bfloat16* __restrict__ scales,
        const __nv_bfloat16* __restrict__ biases, void* __restrict__ out, int M, int N, int K, int npad, int group) {
    using T = Tile<GS, BM, BN, WM, WN, STAGES, W8>;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    const int KG = K / GS;
    const int2 at = tile_of(blockIdx.x, M, N, BM, BN, group);
    const int m0 = at.x, n0 = at.y;

    auto stage = [&](int s) { return buf + s * T::STAGE; };
    auto load = [&](int s, int g) {
        unsigned char* p = stage(s);
#pragma unroll
        for (int c = tid; c < BM * T::CHUNKS; c += T::THREADS) {
            const int r = c / T::CHUNKS, ch = c % T::CHUNKS;
            cp16z(p + r * T::ROW + swz8<T::CHUNKS, T::PER128>(r, ch) * 16,
                  x + static_cast<size_t>(min(m0 + r, M - 1)) * K + g * GS + ch * 16, m0 + r < M);
        }
        unsigned char* pw = p + T::X;
        constexpr int TILE_BYTES = W8 ? 64 * GS : 64 * GS / 2;
#pragma unroll
        for (int c = tid; c < T::W / 16; c += T::THREADS) {
            const int t = c / (TILE_BYTES / 16), off = c % (TILE_BYTES / 16);
            cp16(pw + c * 16, reinterpret_cast<const unsigned char*>(w) +
                              (static_cast<size_t>(n0 / 64 + t) * KG + g) * TILE_BYTES + off * 16);
        }
        unsigned char* ps = pw + T::W;
        for (int c = tid; c < T::S / 16; c += T::THREADS)
            cp16(ps + c * 16, scales + static_cast<size_t>(g) * npad + n0 + c * 8);
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
        if (s < KG) load(s, s);
        commit();
    }
    // this lane's ldmatrix row and chunk base inside a stage, per m16 tile
    int arow[T::MT];
#pragma unroll
    for (int i = 0; i < T::MT; ++i) arow[i] = wm * (BM / WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
    const int wbase = (wn * T::NT * 32 + lane) * T::KS;
    const int scol = wn * (BN / WN) + (lane & 3) * 2;
    for (int g = 0; g < KG; ++g) {
        wait<STAGES - 2>();
        __syncthreads();
        if (g + STAGES - 1 < KG) load((g + STAGES - 1) % STAGES, g + STAGES - 1);
        commit();
        const unsigned char* p = stage(g % STAGES);
        const uint32_t* pw = reinterpret_cast<const uint32_t*>(p + T::X);
        const __nv_bfloat16* ps = reinterpret_cast<const __nv_bfloat16*>(p + T::X + T::W);
        uint32_t a[T::KS][T::MT][4];
#pragma unroll
        for (int ks = 0; ks < T::KS; ++ks)
#pragma unroll
            for (int i = 0; i < T::MT; ++i)
                ldmatrix4(a[ks][i], p + arow[i] * T::ROW +
                                        swz8<T::CHUNKS, T::PER128>(arow[i], ks * 2 + (lane >> 4)) * 16);
        // column tile j's MMAs, then the previous tile's scaled sum while they run
        float prev[T::MT][4];
#pragma unroll
        for (int j = 0; j <= T::NT; ++j) {
            float cur[T::MT][4];
            if (j < T::NT) {
#pragma unroll
                for (int ks = 0; ks < T::KS; ++ks) {
                    uint32_t b0, b1;
                    if constexpr (W8) {
                        const uint2 b = reinterpret_cast<const uint2*>(pw)[wbase + j * 32 * T::KS + ks];
                        b0 = b.x;
                        b1 = b.y;
                    } else {
                        weights8(pw[wbase + j * 32 * T::KS + ks], b0, b1);
                    }
#pragma unroll
                    for (int i = 0; i < T::MT; ++i) {
                        if (ks == 0) mma8z(cur[i], a[ks][i], b0, b1);
                        else mma8(cur[i], a[ks][i], b0, b1);
                    }
                }
            }
            if (j > 0) {
                const __nv_bfloat162 s2 = *reinterpret_cast<const __nv_bfloat162*>(ps + scol + (j - 1) * 8);
                const float s0 = __low2float(s2), s1 = __high2float(s2);
#pragma unroll
                for (int i = 0; i < T::MT; ++i)
#pragma unroll
                    for (int e = 0; e < 4; ++e) acc[i][j - 1][e] = __fmaf_rn(prev[i][e], (e & 1) ? s1 : s0, acc[i][j - 1][e]);
            }
            if (j < T::NT) {
#pragma unroll
                for (int i = 0; i < T::MT; ++i)
#pragma unroll
                    for (int e = 0; e < 4; ++e) prev[i][e] = cur[i][e];
            }
        }
    }
    wait<0>();
    // bias term: acc += xs (bf16, (M, KG)) times biases (bf16, (KG, npad)) as k16 MMAs over the groups (zeros past KG)
    if constexpr (!W8) {
        const int c = lane & 3, gr = lane >> 2;
        const unsigned short* xb = reinterpret_cast<const unsigned short*>(xs);
        const unsigned short* bb = reinterpret_cast<const unsigned short*>(biases);
        auto pair16 = [&](const unsigned short* p, size_t stride, int k) -> uint32_t {
            const uint32_t lo = k < KG ? p[static_cast<size_t>(k) * stride] : 0;
            const uint32_t hi = k + 1 < KG ? p[static_cast<size_t>(k + 1) * stride] : 0;
            return lo | (hi << 16);
        };
        for (int kb = 0; kb < KG; kb += 16) {
            uint32_t af[T::MT][4];
#pragma unroll
            for (int i = 0; i < T::MT; ++i) {
                const int r0 = min(m0 + wm * (BM / WM) + i * 16 + gr, M - 1), r1 = min(r0 + 8, M - 1);
                const unsigned short* x0 = xb + static_cast<size_t>(r0) * KG;
                const unsigned short* x1 = xb + static_cast<size_t>(r1) * KG;
                af[i][0] = pair16(x0, 1, kb + 2 * c);
                af[i][1] = pair16(x1, 1, kb + 2 * c);
                af[i][2] = pair16(x0, 1, kb + 8 + 2 * c);
                af[i][3] = pair16(x1, 1, kb + 8 + 2 * c);
            }
#pragma unroll
            for (int j = 0; j < T::NT; ++j) {
                const unsigned short* bn = bb + n0 + wn * (BN / WN) + j * 8 + gr;
                const uint32_t b0 = pair16(bn, npad, kb + 2 * c), b1 = pair16(bn, npad, kb + 8 + 2 * c);
#pragma unroll
                for (int i = 0; i < T::MT; ++i) mma(acc[i][j], af[i], b0, b1);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < T::MT; ++i) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = m0 + wm * (BM / WM) + i * 16 + (lane >> 2) + h * 8;
            if (row >= M) continue;
            const float a = scale[row];
#pragma unroll
            for (int j = 0; j < T::NT; ++j) {
                const int col = n0 + wn * (BN / WN) + j * 8 + (lane & 3) * 2;
                const float v0 = acc[i][j][2 * h] * a, v1 = acc[i][j][2 * h + 1] * a;
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
}

template <int GS, int BM, int BN, int WM, int WN, int STAGES, bool F32, bool W8 = false>
void launch(const at::Tensor& x, const at::Tensor& xs, const at::Tensor& scale, const at::Tensor& w,
            const at::Tensor& scales, const at::Tensor& biases, at::Tensor& out, int N) {
    using T = Tile<GS, BM, BN, WM, WN, STAGES, W8>;
    const int M = x.size(0), K = x.size(1);
    auto kernel = prefill8_kernel<GS, BM, BN, WM, WN, STAGES, F32, W8>;
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM);
        configured = true;
    }
    const int rows_t = (M + BM - 1) / BM;
    const int group = std::max(1, std::min(rows_t, static_cast<int>((12LL << 20) / (static_cast<long long>(BM) * K))));
    kernel<<<rows_t * ((N + BN - 1) / BN), T::THREADS, T::SMEM, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint8_t*>(x.data_ptr()),
        W8 ? nullptr : reinterpret_cast<const __nv_bfloat16*>(xs.data_ptr()), scale.data_ptr<float>(),
        reinterpret_cast<const uint32_t*>(w.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(scales.data_ptr()),
        W8 ? nullptr : reinterpret_cast<const __nv_bfloat16*>(biases.data_ptr()),
        out.data_ptr(), M, N, K, static_cast<int>(scales.size(1)), group);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int GS, bool F32>
void dispatch(int tile, const at::Tensor& x, const at::Tensor& xs, const at::Tensor& a, const at::Tensor& w,
              const at::Tensor& s, const at::Tensor& b, at::Tensor& out, int N) {
    switch (tile) {
        case 1: launch<GS, 64, 128, 1, 4, 4, F32>(x, xs, a, w, s, b, out, N); break;
        case 2: launch<GS, 128, 128, 2, 4, 3, F32>(x, xs, a, w, s, b, out, N); break;
        default: launch<GS, 128, 128, 2, 2, 3, F32>(x, xs, a, w, s, b, out, N); break;
    }
}

} // namespace

template <int GS, bool F32>
void dispatch_w8(int tile, const at::Tensor& x, const at::Tensor& a, const at::Tensor& w, const at::Tensor& s,
                 at::Tensor& out, int N) {
    switch (tile) {
        case 1: launch<GS, 64, 128, 1, 4, 4, F32, true>(x, x, a, w, s, s, out, N); break;
        case 2: launch<GS, 128, 128, 2, 4, 3, F32, true>(x, x, a, w, s, s, out, N); break;
        default: launch<GS, 128, 128, 2, 2, 3, F32, true>(x, x, a, w, s, s, out, N); break;
    }
}

// ``tile`` (0: 128x128, four 64x64 warps; 1: 64x128; 2: 128x128, eight 64x32 warps) never changes a row's bits.
void qmm_prefill8_cuda(const at::Tensor& x, const at::Tensor& xs, const at::Tensor& a, const at::Tensor& w,
                       const at::Tensor& scales, const at::Tensor& biases, at::Tensor& out, int N, int gs, bool f32,
                       int tile) {
    if (gs == 64) { if (f32) dispatch<64, true>(tile, x, xs, a, w, scales, biases, out, N); else dispatch<64, false>(tile, x, xs, a, w, scales, biases, out, N); }
    else { if (f32) dispatch<32, true>(tile, x, xs, a, w, scales, biases, out, N); else dispatch<32, false>(tile, x, xs, a, w, scales, biases, out, N); }
}

// The same over e4m3 weight bytes in fragment order with a bf16 scale per (group of 32 or 64 inputs, column), no bias.
void qmm_prefill8w_cuda(const at::Tensor& x, const at::Tensor& a, const at::Tensor& w, const at::Tensor& scales,
                        at::Tensor& out, int N, int gs, bool f32, int tile) {
    if (gs == 32) { if (f32) dispatch_w8<32, true>(tile, x, a, w, scales, out, N); else dispatch_w8<32, false>(tile, x, a, w, scales, out, N); }
    else { if (f32) dispatch_w8<64, true>(tile, x, a, w, scales, out, N); else dispatch_w8<64, false>(tile, x, a, w, scales, out, N); }
}
