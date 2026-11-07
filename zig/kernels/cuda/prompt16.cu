// Device code of nvfp4/prompt.cu (lines 12-197, comments and ATen dropped), the FP4 and FP8 tile-4 instances.

#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "common/qmm_frag.cuh"
#include "common/fp_pairs.cuh"

namespace tf_prompt16 {

using namespace qmm_frag;

// the lean prompt tile this image's prompt_lean and prompt_many's tf_prompt16_many share
#include "common/prompt_lean.cuh"

using tf_common::fp4pair;



using tf_common::mul2;

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
                    sv[j][blk] = tf_common::fp8pair_unit(((b4 >> (8 * blk)) & 0xFFu) * 0x101u, 0x7B807B80u);
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
                    b0 = tf_common::fp8pair_unit(word & 0xFFFFu, unit);
                    b1 = tf_common::fp8pair_unit(word >> 16, unit);
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

// prompt_kernel's bits on a 128 x 256 tile: same fragments and mma chain over (g, kt); persistent, mbarrier ring
template <int MODE, int BM, int BN, int WM, int WN, int STAGES, bool F32>
__global__ void __launch_bounds__(WM * WN * 32, 1) prompt_lean(
        const __nv_bfloat16* __restrict__ x, const unsigned char* __restrict__ w, const uint8_t* __restrict__ bs,
        float scale, void* __restrict__ out, int M, int N, int K, int npad, int ldx, int group) {
    lean_tile_body<MODE, BM, BN, WM, WN, STAGES, F32>(x, LeanSoloRaw{w, bs, scale, out, N, npad}, M, K, ldx, group);
}

// The instance the Zig launcher loads by name (PROMPT_TILE 4 in linear.py).
template __global__ void prompt_kernel<FP4, 128, 128, 2, 2, 2, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, int, int, int, int, int, int);
template __global__ void prompt_kernel<FP4, 128, 128, 2, 2, 2, true>(
    const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float,
    void* __restrict__, int, int, int, int, int, int);
template __global__ void prompt_kernel<FP8, 128, 128, 2, 2, 2, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, int, int, int, int, int, int);
// The lean tile prompt() serves FP4 on: 128 x 256, eight warps of 128 x 32, three stages (bits equal to the above).
template __global__ void prompt_lean<FP4, 128, 256, 1, 8, 3, false>(
    const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float,
    void* __restrict__, int, int, int, int, int, int);
template __global__ void prompt_lean<FP4, 128, 256, 1, 8, 3, true>(
    const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float,
    void* __restrict__, int, int, int, int, int, int);

// BF16 prompt rows on qmmf's BF16 words with qmmf_kernel's sums (K slices from zero, added in order): the lane's bits.
template <bool F32>
__device__ __forceinline__ void prompt_bf16(const __nv_bfloat16* __restrict__ x, const unsigned char* __restrict__ w,
                                            float scale, void* __restrict__ out, int M, int N, int K, int SK, int npad,
                                            int ldx, int group) {
    constexpr int BM = 128, BN = 128, WM = 2, WN = 4, STAGES = 3;
    using T = Tile<BF16, BM, BN, WM, WN, STAGES>;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    const int KG = K / GS, slice_groups = KG / SK;
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
        constexpr int TILE_BYTES = 64 * GS * 2;
        for (int c = tid; c < T::W / 16; c += T::THREADS) {
            const int t = c / (TILE_BYTES / 16), off = c % (TILE_BYTES / 16);
            if (n0 + t * 64 < npad)
                cp16(pw + c * 16, w + (static_cast<size_t>(n0 / 64 + t) * KG + g) * TILE_BYTES + off * 16);
        }
    };

    float acc[T::MT][T::NT][4];
    float tot[T::MT][T::NT][4];
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < T::NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = tot[i][j][e] = 0.0f;
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KG) load(s, s);
        commit();
    }
    for (int g = 0; g < KG; ++g) {
        if (SK > 1 && g > 0 && g % slice_groups == 0) {
#pragma unroll
            for (int i = 0; i < T::MT; ++i)
#pragma unroll
                for (int j = 0; j < T::NT; ++j)
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        tot[i][j][e] = g == slice_groups ? acc[i][j][e] : tot[i][j][e] + acc[i][j][e];
                        acc[i][j][e] = 0.0f;
                    }
        }
        wait<STAGES - 2>();
        __syncthreads();
        if (g + STAGES - 1 < KG) load((g + STAGES - 1) % STAGES, g + STAGES - 1);
        commit();
        const unsigned char* p = stage(g % STAGES);
        const uint2* pw = reinterpret_cast<const uint2*>(p + T::X);
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
                // n8 tile jj of the block's two 64-column tiles, laid out back to back: (jj, kt, lane) as qmmf reads
                const uint2 v = pw[((wn * T::NT + j) * 4 + kt) * 32 + lane];
#pragma unroll
                for (int i = 0; i < T::MT; ++i) mma(acc[i][j], a[i], v.x, v.y);
            }
        }
    }
    wait<0>();
    if (SK > 1) {
#pragma unroll
        for (int i = 0; i < T::MT; ++i)
#pragma unroll
            for (int j = 0; j < T::NT; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[i][j][e] = tot[i][j][e] + acc[i][j][e];
    }
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

extern "C" __global__ void __launch_bounds__(256) tf_prompt_bf16(const __nv_bfloat16* __restrict__ x, const unsigned char* __restrict__ w,
                                                                 float scale, void* __restrict__ out, int M, int N, int K, int SK,
                                                                 int npad, int ldx, int group) {
    prompt_bf16<false>(x, w, scale, out, M, N, K, SK, npad, ldx, group);
}

extern "C" __global__ void __launch_bounds__(256) tf_prompt_bf16_f32(const __nv_bfloat16* __restrict__ x, const unsigned char* __restrict__ w,
                                                                     float scale, void* __restrict__ out, int M, int N, int K, int SK,
                                                                     int npad, int ldx, int group) {
    prompt_bf16<true>(x, w, scale, out, M, N, K, SK, npad, ldx, group);
}
}

#if defined(TF_NATIVE_FP_DECODE)
// tf-cuda-test native-fp: (native, software) words of all e4m3 pairs and e2m1 bytes; all ones on a plain target
extern "C" __global__ void tf_native_fp_check(uint32_t* __restrict__ out) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 65536u + 256u) return;
#if TF_FP_NATIVE
    if (i < 65536u) {
        out[3 * i] = tf_common::e4m3x2(i);
        out[3 * i + 1] = tf_common::fp8pair_unit(i, 0x7B807B80u);
        out[3 * i + 2] = tf_common::fp8pair(i);
    } else {
        const uint32_t b = i - 65536u;
        out[3 * i] = tf_common::e2m1x2(b);
        out[3 * i + 1] = tf_common::fp4pair((b & 0xFu) | ((b >> 4) << 16), 0);
        out[3 * i + 2] = 0u;
    }
#else
    out[3 * i] = out[3 * i + 1] = out[3 * i + 2] = 0xFFFFFFFFu;
#endif
}
#endif
