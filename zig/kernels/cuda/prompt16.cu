// Device code of nvfp4/prompt.cu (lines 12-197, comments and ATen dropped), the FP4 and FP8 tile-4 instances.

#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "qmm_frag.cuh"

namespace tf_prompt16 {

using namespace qmm_frag;

enum Mode : int { FP4 = 0, FP8 = 1, MXFP8 = 2 };

constexpr int GS = 64;

__device__ __forceinline__ uint32_t fp4pair(uint32_t w, int s) {
    const uint32_t v = w >> s;
    const uint32_t t = ((v & 0x00070007u) << 6) | ((v & 0x00080008u) << 12);
    uint32_t r;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(t), "r"(0x7E807E80u), "r"(0x80008000u));
    return r;
}

__device__ __forceinline__ uint32_t fp8pair(uint32_t w, uint32_t unit) {
    const uint32_t x = (w & 0xFFu) | ((w & 0xFF00u) << 8);
    const uint32_t t = ((x & 0x007F007Fu) << 4) | ((x & 0x00800080u) << 8);
    uint32_t r;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(t), "r"(unit), "r"(0x80008000u));
    return r;
}

__device__ __forceinline__ uint32_t mul2(uint32_t a, uint32_t b) {
    uint32_t r;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(a), "r"(b), "r"(0x80008000u));
    return r;
}

__device__ __forceinline__ uint32_t comp(const uint4& v, int c) { return c == 0 ? v.x : c == 1 ? v.y : c == 2 ? v.z : v.w; }

template <int MODE, int BM, int BN, int WM, int WN, int STAGES>
struct Tile {
    static constexpr int THREADS = WM * WN * 32;
    static constexpr int MT = BM / WM / 16;
    static constexpr int NT = BN / WN / 8;
    static constexpr int ROW = GS * 2;
    static constexpr int CHUNKS = ROW / 16;
    static constexpr int X = BM * ROW;
    static constexpr int W = MODE == FP4 ? BN * GS / 2 : BN * GS;
    static constexpr int S = MODE == FP4 ? BN * 4 : MODE == MXFP8 ? BN * 2 : 0;
    static constexpr int STAGE = (X + W + S + 127) / 128 * 128;
    static constexpr int SMEM = STAGES * STAGE;
};

template <int MODE, int BM, int BN, int WM, int WN, int STAGES, bool F32>
__global__ void __launch_bounds__(WM * WN * 32) prompt_kernel(
        const __nv_bfloat16* __restrict__ x, const unsigned char* __restrict__ w, const uint8_t* __restrict__ bs,
        float scale, void* __restrict__ out, int M, int N, int K, int npad, int ldx, int group) {
    using T = Tile<MODE, BM, BN, WM, WN, STAGES>;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    const int KG = K / GS;
    const int2 at = tile_of(blockIdx.x, M, N, BM, BN, group);
    const int m0 = at.x, n0 = at.y;

    auto stage = [&](int s) { return buf + s * T::STAGE; };
    auto load = [&](int s, int g) {
        unsigned char* p = stage(s);
        for (int c = tid; c < BM * T::CHUNKS; c += T::THREADS) {
            const int r = c / T::CHUNKS, ch = c % T::CHUNKS;
            cp16z(p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16,
                  x + static_cast<size_t>(min(m0 + r, M - 1)) * ldx + g * GS + ch * 8, m0 + r < M);
        }
        unsigned char* pw = p + T::X;
        constexpr int TILE_BYTES = MODE == FP4 ? 64 * GS / 2 : 64 * GS;
        for (int c = tid; c < T::W / 16; c += T::THREADS) {
            const int t = c / (TILE_BYTES / 16), off = c % (TILE_BYTES / 16);
            if (n0 + t * 64 < npad)
                cp16(pw + c * 16, w + (static_cast<size_t>(n0 / 64 + t) * KG + g) * TILE_BYTES + off * 16);
        }
        if constexpr (T::S > 0) {
            constexpr int PER = T::S / (BN / 64);
            unsigned char* ps = pw + T::W;
            for (int c = tid; c < T::S / 16; c += T::THREADS) {
                const int t = c / (PER / 16), off = c % (PER / 16);
                if (n0 + t * 64 < npad)
                    cp16(ps + c * 16, bs + (static_cast<size_t>(n0 / 64 + t) * KG + g) * PER + off * 16);
            }
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
        if (s < KG) load(s, s);
        commit();
    }
    for (int g = 0; g < KG; ++g) {
        wait<STAGES - 2>();
        __syncthreads();
        if (g + STAGES - 1 < KG) load((g + STAGES - 1) % STAGES, g + STAGES - 1);
        commit();
        const unsigned char* p = stage(g % STAGES);
        const unsigned char* pw = p + T::X;
        const uint8_t* ps = p + T::X + T::W;
        uint4 wq[T::NT];
        uint32_t sv[T::NT][4];
#pragma unroll
        for (int j = 0; j < T::NT; ++j) {
            const int jj = wn * T::NT + j;
            const int col = (jj & 7) * 8 + (lane >> 2);
            if constexpr (MODE == FP4) {
                const uint2 u = reinterpret_cast<const uint2*>(pw)[jj * 32 + lane];
                wq[j] = make_uint4(u.x, u.y, 0u, 0u);
                const uint32_t b4 = *reinterpret_cast<const uint32_t*>(ps + (jj >> 3) * 256 + col * 4);
#pragma unroll
                for (int blk = 0; blk < 4; ++blk)
                    sv[j][blk] = fp8pair(((b4 >> (8 * blk)) & 0xFFu) * 0x101u, 0x7B807B80u);
            } else {
                wq[j] = reinterpret_cast<const uint4*>(pw)[jj * 32 + lane];
                if constexpr (MODE == MXFP8) {
                    const uint32_t e2 = *reinterpret_cast<const uint16_t*>(ps + (jj >> 3) * 128 + col * 2);
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const uint32_t u = (((e2 >> (8 * h)) & 0xFFu) + 120u) << 7;
                        sv[j][h] = u | (u << 16);
                    }
                }
            }
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
                uint32_t b0, b1;
                if constexpr (MODE == FP4) {
                    const uint32_t word = kt < 2 ? wq[j].x : wq[j].y;
                    b0 = mul2(fp4pair(word, (kt & 1) * 8), sv[j][kt]);
                    b1 = mul2(fp4pair(word, (kt & 1) * 8 + 4), sv[j][kt]);
                } else {
                    const uint32_t word = comp(wq[j], kt);
                    const uint32_t unit = MODE == MXFP8 ? sv[j][kt >> 1] : 0x7B807B80u;
                    b0 = fp8pair(word & 0xFFFFu, unit);
                    b1 = fp8pair(word >> 16, unit);
                }
#pragma unroll
                for (int i = 0; i < T::MT; ++i) mma(acc[i][j], a[i], b0, b1);
            }
        }
    }
    wait<0>();
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < T::NT; ++j) {
            const int col = n0 + wn * (BN / WN) + j * 8 + (lane & 3) * 2;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + wm * (BM / WM) + i * 16 + (lane >> 2) + h * 8;
                if (row >= M) continue;
                const float v0 = acc[i][j][2 * h] * scale, v1 = acc[i][j][2 * h + 1] * scale;
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

// The instance the Zig launcher loads by name (PROMPT_TILE 4 in linear.py).
template __global__ void prompt_kernel<FP4, 128, 128, 2, 2, 2, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, int, int, int, int, int, int);
template __global__ void prompt_kernel<FP8, 128, 128, 2, 2, 2, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, int, int, int, int, int, int);
}
