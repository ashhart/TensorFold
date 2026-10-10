// Device code of nvfp4/qmmf.cu (lines 14-298, comments and ATen dropped), its FP4, FP8 and FP8G modes plus BF16.

#include <algorithm>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "qmm_frag.cuh"

namespace tf_qmmf {

using namespace qmm_frag;

enum Mode : int { FP4 = 0, FP8 = 1, MXFP8 = 2, FP8G = 3, BF16 = 4 };

constexpr int GS = 64;

__device__ __forceinline__ uint32_t fp4pair(uint32_t w, int s) {
    const uint32_t v = w >> s;
    const uint32_t t = ((v & 0x00070007u) << 6) | ((v & 0x00080008u) << 12);
    uint32_t r;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(t), "r"(0x7E807E80u), "r"(0x80008000u));
    return r;
}

__device__ __forceinline__ uint32_t fp8pair(uint32_t w) {
    const uint32_t x = (w & 0xFFu) | ((w & 0xFF00u) << 8);
    const uint32_t t = ((x & 0x007F007Fu) << 4) | ((x & 0x00800080u) << 8);
    uint32_t r;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(t), "r"(0x7B807B80u), "r"(0x80008000u));
    return r;
}

__device__ __forceinline__ uint32_t comp(const uint4& v, int c) { return c == 0 ? v.x : c == 1 ? v.y : c == 2 ? v.z : v.w; }

__device__ __forceinline__ float e4m3f(uint8_t b) {
    return __half2float(__half(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3)));
}

template <int MODE, int BM, int BN, int WM, int WN, int STAGES>
struct Tile {
    static constexpr int THREADS = WM * WN * 32;
    static constexpr int MT = BM / WM / 16;
    static constexpr int NT = BN / WN / 8;
    static constexpr int ROW = GS * 2;
    static constexpr int CHUNKS = ROW / 16;
    static constexpr int X = BM * ROW;
    static constexpr int W = MODE == FP4 ? BN * GS / 2 : MODE == BF16 ? BN * GS * 2 : BN * GS;
    static constexpr int S = MODE == FP4 || MODE == FP8G ? BN * 4 : MODE == MXFP8 ? BN * 2 : 0;
    static constexpr int STAGE = (X + W + S + 127) / 128 * 128;
    static constexpr int PARTIALS = MT * NT * 4 * THREADS * 4;
    static constexpr int SMEM = STAGES * STAGE > PARTIALS ? STAGES * STAGE : PARTIALS;
};

template <int MODE, int BM, int BN, int WM, int WN, int STAGES, bool F32, bool CLUSTER, bool FUSE = false>
__global__ void __launch_bounds__(WM * WN * 32) qmmf_kernel(
        const __nv_bfloat16* __restrict__ x, const unsigned char* __restrict__ w, const uint8_t* __restrict__ bs,
        float scale, void* __restrict__ out, float* __restrict__ part, int M, int N, int K, int SK, int npad, int ldx,
        int group) {
    using T = Tile<MODE, BM, BN, WM, WN, STAGES>;
    static_assert(BN == 64, "a block reads one 64-column tile of words and scales");
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    const int KG = K / GS, slice_groups = KG / SK, per = FUSE ? KG : slice_groups;
    const int2 at = tile_of(blockIdx.x, M, N, BM, BN, group);
    const int m0 = at.x, n0 = at.y, slice = FUSE ? 0 : blockIdx.z, g0 = slice * per;

    auto stage = [&](int s) { return buf + s * T::STAGE; };
    auto load = [&](int s, int g) {
        unsigned char* p = stage(s);
        for (int c = tid; c < BM * T::CHUNKS; c += T::THREADS) {
            const int r = c / T::CHUNKS, ch = c % T::CHUNKS;
            const int row = min(m0 + r, M - 1);
            cp16z(p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16, x + static_cast<size_t>(row) * ldx + g * GS + ch * 8,
                  m0 + r < M);
        }
        unsigned char* pw = p + T::X;
        constexpr int TILE_BYTES = MODE == FP4 ? 64 * GS / 2 : MODE == BF16 ? 64 * GS * 2 : 64 * GS;
        for (int c = tid; c < T::W / 16; c += T::THREADS) {
            const int t = c / (TILE_BYTES / 16), off = c % (TILE_BYTES / 16);
            cp16(pw + c * 16, w + (static_cast<size_t>(n0 / 64 + t) * KG + g) * TILE_BYTES + off * 16);
        }
        if constexpr (T::S > 0) {
            unsigned char* ps = pw + T::W;
            for (int c = tid; c < T::S / 16; c += T::THREADS)
                cp16(ps + c * 16, bs + (static_cast<size_t>(n0 / 64) * KG + g) * T::S + c * 16);
        }
    };

    float acc[T::MT][T::NT][4];
    float tot[FUSE ? T::MT : 1][FUSE ? T::NT : 1][4];
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
        if constexpr (FUSE) {
            if (it > 0 && it % slice_groups == 0) {
#pragma unroll
                for (int i = 0; i < T::MT; ++i)
#pragma unroll
                    for (int j = 0; j < T::NT; ++j)
#pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            tot[i][j][e] = it == slice_groups ? acc[i][j][e] : tot[i][j][e] + acc[i][j][e];
                            acc[i][j][e] = 0.0f;
                        }
            }
        }
        wait<STAGES - 2>();
        __syncthreads();
        const int next = it + STAGES - 1;
        if (next < per) load(next % STAGES, g0 + next);
        commit();
        const unsigned char* p = stage(it % STAGES);
        const unsigned char* pw = p + T::X;
        const uint8_t* ps = p + T::X + T::W;
        float d[T::MT][T::NT][4];
        uint4 wq[T::NT];
        uint32_t sq[T::NT][2];
#pragma unroll
        for (int j = 0; j < T::NT; ++j) {
            const int jj = wn * T::NT + j;
            const int col = wn * (BN / WN) + j * 8 + (lane & 3) * 2;
            if constexpr (MODE == FP4 || MODE == FP8G) {
                const uint2 v = *reinterpret_cast<const uint2*>(ps + col * 4);
                sq[j][0] = v.x;
                sq[j][1] = v.y;
            } else if constexpr (MODE == MXFP8) {
                const uint32_t v = *reinterpret_cast<const uint32_t*>(ps + col * 2);
                sq[j][0] = v & 0xFFFFu;
                sq[j][1] = v >> 16;
            }
            if constexpr (MODE == FP4) {
                const uint2 u = reinterpret_cast<const uint2*>(pw)[jj * 32 + lane];
                wq[j] = make_uint4(u.x, u.y, 0u, 0u);
            } else if constexpr (MODE != BF16) {
                wq[j] = reinterpret_cast<const uint4*>(pw)[jj * 32 + lane];
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
                    b0 = fp4pair(word, (kt & 1) * 8);
                    b1 = fp4pair(word, (kt & 1) * 8 + 4);
                } else if constexpr (MODE == BF16) {
                    const uint2 v = reinterpret_cast<const uint2*>(pw)[((wn * T::NT + j) * 4 + kt) * 32 + lane];
                    b0 = v.x;
                    b1 = v.y;
                } else {
                    const uint32_t word = comp(wq[j], kt);
                    b0 = fp8pair(word & 0xFFFFu);
                    b1 = fp8pair(word >> 16);
                }
#pragma unroll
                for (int i = 0; i < T::MT; ++i) {
                    if constexpr (MODE == FP8 || MODE == BF16) mma(acc[i][j], a[i], b0, b1);
                    else if (MODE == FP4 || (MODE == MXFP8 && (kt & 1) == 0) || (MODE == FP8G && kt == 0))
                        mma0(d[i][j], a[i], b0, b1);
                    else mma(d[i][j], a[i], b0, b1);
                }
            }
            if constexpr (MODE == FP4 || MODE == MXFP8 || MODE == FP8G) {
                if (MODE == FP4 || (MODE == MXFP8 && (kt & 1)) || (MODE == FP8G && kt == GS / 16 - 1)) {
                    const int blk = MODE == FP4 ? kt : kt / 2;
#pragma unroll
                    for (int j = 0; j < T::NT; ++j) {
                        float s0, s1;
                        if constexpr (MODE == FP8G) {
                            s0 = __uint_as_float(sq[j][0]);
                            s1 = __uint_as_float(sq[j][1]);
                        } else if constexpr (MODE == FP4) {
                            s0 = e4m3f(static_cast<uint8_t>(sq[j][0] >> (8 * blk)));
                            s1 = e4m3f(static_cast<uint8_t>(sq[j][1] >> (8 * blk)));
                        } else {
                            s0 = __int_as_float(static_cast<int>((sq[j][0] >> (8 * blk)) & 0xFFu) << 23);
                            s1 = __int_as_float(static_cast<int>((sq[j][1] >> (8 * blk)) & 0xFFu) << 23);
                        }
#pragma unroll
                        for (int i = 0; i < T::MT; ++i)
#pragma unroll
                            for (int e = 0; e < 4; ++e) acc[i][j][e] = __fmaf_rn(d[i][j][e], (e & 1) ? s1 : s0, acc[i][j][e]);
                    }
                }
            }
        }
    }
    wait<0>();
    __syncthreads();
    if constexpr (FUSE) {
        if (SK > 1) {
#pragma unroll
            for (int i = 0; i < T::MT; ++i)
#pragma unroll
                for (int j = 0; j < T::NT; ++j)
#pragma unroll
                    for (int e = 0; e < 4; ++e) acc[i][j][e] = tot[i][j][e] + acc[i][j][e];
        }
    }
    if constexpr (CLUSTER) {
#if __CUDA_ARCH__ < 900
        __trap();
#else
        auto cluster = cooperative_groups::this_cluster();
        float* mine = reinterpret_cast<float*>(buf);
        if (slice != 0) {
#pragma unroll
            for (int i = 0; i < T::MT; ++i)
#pragma unroll
                for (int j = 0; j < T::NT; ++j)
#pragma unroll
                    for (int e = 0; e < 4; ++e) mine[((i * T::NT + j) * 4 + e) * T::THREADS + tid] = acc[i][j][e];
        }
        cluster.sync();
        if (slice == 0) {
            for (int peer = 1; peer < SK; ++peer) {
                const float* theirs = cluster.map_shared_rank(mine, peer);
#pragma unroll
                for (int i = 0; i < T::MT; ++i)
#pragma unroll
                    for (int j = 0; j < T::NT; ++j)
#pragma unroll
                        for (int e = 0; e < 4; ++e)
                            acc[i][j][e] = acc[i][j][e] + theirs[((i * T::NT + j) * 4 + e) * T::THREADS + tid];
            }
        }
        cluster.sync();
        if (slice != 0) return;
#endif
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
                if (SK > 1 && !CLUSTER && !FUSE) {
                    float* dst = part + (static_cast<size_t>(slice) * M + row) * N + col;
                    if (col < N) dst[0] = acc[i][j][2 * h];
                    if (col + 1 < N) dst[1] = acc[i][j][2 * h + 1];
                    continue;
                }
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

template <bool F32>
__global__ void reduce_kernel(const float* __restrict__ part, void* __restrict__ out, long long total, int SK,
                              float scale) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float acc = part[i];
    for (int s = 1; s < SK; ++s) acc = acc + part[s * total + i];
    acc = acc * scale;
    if (F32) reinterpret_cast<float*>(out)[i] = acc;
    else reinterpret_cast<__nv_bfloat16*>(out)[i] = __float2bfloat16_rn(acc);
}

// The instances the Zig launchers load by name (the Python host instantiates these through its launch switch).
template __global__ void qmmf_kernel<FP8G, 16, 64, 1, 4, 4, false, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8G, 16, 64, 1, 4, 4, false, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8G, 32, 64, 1, 4, 4, false, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8G, 32, 64, 1, 4, 4, false, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8G, 64, 64, 1, 4, 4, false, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8G, 64, 64, 1, 4, 4, false, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8G, 64, 64, 1, 4, 4, false, false, true>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP4, 16, 64, 1, 4, 4, false, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP4, 16, 64, 1, 4, 4, false, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP4, 32, 64, 1, 4, 4, false, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP4, 32, 64, 1, 4, 4, false, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP4, 64, 64, 1, 4, 4, false, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP4, 64, 64, 1, 4, 4, false, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP4, 64, 64, 1, 4, 4, false, false, true>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8, 16, 64, 1, 4, 4, false, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8, 16, 64, 1, 4, 4, false, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8, 32, 64, 1, 4, 4, false, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8, 32, 64, 1, 4, 4, false, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8, 64, 64, 1, 4, 4, false, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8, 64, 64, 1, 4, 4, false, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8, 64, 64, 1, 4, 4, false, false, true>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<BF16, 16, 64, 1, 4, 4, false, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<BF16, 16, 64, 1, 4, 4, false, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<BF16, 32, 64, 1, 4, 4, false, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<BF16, 32, 64, 1, 4, 4, false, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<BF16, 64, 64, 1, 4, 4, false, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<BF16, 64, 64, 1, 4, 4, false, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<BF16, 64, 64, 1, 4, 4, false, false, true>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void reduce_kernel<false>(const float* __restrict__, void* __restrict__, long long, int, float);
}
