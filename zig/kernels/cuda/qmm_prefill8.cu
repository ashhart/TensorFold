// TensorFold v0.6.0 qmm_prefill8 arithmetic with ATen launch wrappers replaced by the native Zig driver (Apache-2.0).
#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "qmm_frag.cuh"

namespace tf_qmm_prefill8_impl {
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

__device__ __forceinline__ uint32_t e4m3_pair(uint32_t w, int s) {
    const uint32_t t = ((w >> s) & 0x000F000Fu) | 0x64006400u;
    uint32_t h;
    asm("sub.rn.f16x2 %0, %1, %2;\n" : "=r"(h) : "r"(t), "r"(0x64006400u));
    uint16_t r;
    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;\n" : "=h"(r) : "r"(h));
    return r;
}

__device__ __forceinline__ void weights8(uint32_t w, uint32_t& b0, uint32_t& b1) {
    b0 = __byte_perm(e4m3_pair(w, 0), e4m3_pair(w, 4), 0x5410);
    b1 = __byte_perm(e4m3_pair(w, 8), e4m3_pair(w, 12), 0x5410);
}

template <int GS, int BM, int BN, int WM, int WN, int STAGES, bool W8 = false>
struct Tile {
    static constexpr int THREADS = WM * WN * 32;
    static constexpr int MT = BM / WM / 16;
    static constexpr int NT = BN / WN / 8;
    static constexpr int KS = GS / 32;
    static constexpr int ROW = GS;
    static constexpr int CHUNKS = ROW / 16;
    static constexpr int PER128 = 128 / ROW;
    static constexpr int X = BM * ROW;
    static constexpr int W = W8 ? BN * GS : BN * GS / 2;
    static constexpr int S = BN * 2;
    static constexpr int STAGE = X + W + S;
};

template <int CHUNKS, int PER128>
__device__ __forceinline__ int swz8(int r, int c) { return c ^ ((r / PER128) % CHUNKS); }

template <int GS, int BM, int BN, int WM, int WN, int STAGES, bool F32, bool W8, bool L64 = false>
__device__ __forceinline__ void prefill8_body(
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
        if constexpr (L64) {
#pragma unroll
            for (int c = tid; c < T::W / 8; c += T::THREADS) {
                const int t = c / 256, lane8 = c % 256;
                cp8(pw + c * 8, reinterpret_cast<const unsigned char*>(w) +
                                (static_cast<size_t>(n0 / 64 + t) * (KG / 2) + (g >> 1)) * 4096 + (lane8 * 2 + (g & 1)) * 8);
            }
        } else {
#pragma unroll
            for (int c = tid; c < T::W / 16; c += T::THREADS) {
                const int t = c / (TILE_BYTES / 16), off = c % (TILE_BYTES / 16);
                cp16(pw + c * 16, reinterpret_cast<const unsigned char*>(w) +
                                  (static_cast<size_t>(n0 / 64 + t) * KG + g) * TILE_BYTES + off * 16);
            }
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
    // The same BF16 bias-correction MMAs make a prompt-only W8 copy bitwise equivalent to expanding Q4 here.
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
} // namespace tf_qmm_prefill8_impl

// Expand eight Q4 codes to positive e4m3 once for prompt gate/up; decode and verification retain the packed Q4 words.
extern "C" __global__ void tf_qmm_prefill8_expand_w8(
    const uint32_t* __restrict__ w, uint2* __restrict__ out, int words) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= words) return;
    uint32_t b0, b1;
    tf_qmm_prefill8_impl::weights8(w[i], b0, b1);
    out[i] = make_uint2(b0, b1);
}

// MLX Q4 group-64, old tile 0, BF16 output. Decode and verify never call this.
extern "C" __global__ void __launch_bounds__(128, 2) tf_qmm_prefill8(
    const uint8_t* x, const __nv_bfloat16* xs, const float* scale,
    const uint32_t* w, const __nv_bfloat16* scales, const __nv_bfloat16* biases,
    void* out, int M, int N, int K, int npad, int group) {
    tf_qmm_prefill8_impl::prefill8_body<64, 128, 128, 2, 2, 3, false, false, false>(
        x, xs, scale, w, scales, biases, out, M, N, K, npad, group);
}

extern "C" __global__ void __launch_bounds__(128, 2) tf_qmm_prefill8_w8(
    const uint8_t* x, const __nv_bfloat16* xs, const float* scale,
    const uint32_t* w, const __nv_bfloat16* scales, const __nv_bfloat16* biases,
    void* out, int M, int N, int K, int npad, int group) {
    tf_qmm_prefill8_impl::prefill8_body<64, 128, 128, 2, 2, 3, false, true, false>(
        x, xs, scale, w, scales, biases, out, M, N, K, npad, group);
}
