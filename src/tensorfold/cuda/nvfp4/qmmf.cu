// Lane matmul for NVFP4 and FP8 weights (W4A16 / W8A16), exact weights in bf16 MMAs: per 16 (NVFP4), 32 (MXFP8) or 64
// inputs (FP8G: fp32 scales) acc = fma(P, block scale, acc), one tensor scale; K slices by shape, no row affects another.

#include <ATen/ATen.h>
#include <algorithm>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "../kernels/qmm_frag.cuh"

namespace {

using namespace qmm_frag;

enum Mode : int { FP4 = 0, FP8 = 1, MXFP8 = 2, FP8G = 3 };

constexpr int GS = 64;                                    // inputs a pipeline stage

// bf16x2 of two e2m1 nibbles (bits [s, s + 4) and [16 + s, 20 + s)): exponent and mantissa into bf16's, times 2^126.
__device__ __forceinline__ uint32_t fp4pair(uint32_t w, int s) {
    const uint32_t v = w >> s;
    const uint32_t t = ((v & 0x00070007u) << 6) | ((v & 0x00080008u) << 12);
    uint32_t r;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(t), "r"(0x7E807E80u), "r"(0x80008000u));
    return r;
}

// bf16x2 of two e4m3 bytes (bits [0, 8) and [8, 16)): exponent and mantissa into bf16's, times 2^120.
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
    static constexpr int W = MODE == FP4 ? BN * GS / 2 : BN * GS;
    static constexpr int S = MODE == FP4 || MODE == FP8G ? BN * 4 : MODE == MXFP8 ? BN * 2 : 0;   // block scales a group
    static constexpr int STAGE = (X + W + S + 127) / 128 * 128;   // on 128-byte lines: shifted stages slow FP8
    static constexpr int PARTIALS = MT * NT * 4 * THREADS * 4;
    static constexpr int SMEM = STAGES * STAGE > PARTIALS ? STAGES * STAGE : PARTIALS;
};

// FUSE (prompt rows): one block runs all SK slices of its tile, each from zero over its own groups, and adds them in
// slice order: the cluster's (or the reduce's) arithmetic without the cluster, the partials or the second pass.
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
        constexpr int TILE_BYTES = MODE == FP4 ? 64 * GS / 2 : 64 * GS;
        for (int c = tid; c < T::W / 16; c += T::THREADS) {
            const int t = c / (TILE_BYTES / 16), off = c % (TILE_BYTES / 16);
            cp16(pw + c * 16, w + (static_cast<size_t>(n0 / 64 + t) * KG + g) * TILE_BYTES + off * 16);
        }
        if constexpr (T::S > 0) {                          // block scales [npad/64][K/64][64][4|2]: one tile's
            unsigned char* ps = pw + T::W;
            for (int c = tid; c < T::S / 16; c += T::THREADS)
                cp16(ps + c * 16, bs + (static_cast<size_t>(n0 / 64) * KG + g) * T::S + c * 16);
        }
    };

    float acc[T::MT][T::NT][4];
    float tot[FUSE ? T::MT : 1][FUSE ? T::NT : 1][4];           // FUSE: the slices' sum so far, in slice order
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
            if (it > 0 && it % slice_groups == 0) {        // a slice ends: add it to the sum, start the next from zero
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
        uint4 wq[T::NT];                                   // a lane's words for the stage, read once (16 or 8 bytes)
        uint32_t sq[T::NT][2];                             // its two columns' block scales for the stage (4 or 2 bytes)
#pragma unroll
        for (int j = 0; j < T::NT; ++j) {
            const int jj = wn * T::NT + j;
            const int col = wn * (BN / WN) + j * 8 + (lane & 3) * 2;
            if constexpr (MODE == FP4 || MODE == FP8G) {   // FP8G: the two columns' fp32 scale bits
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
            } else {
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
                } else {
                    const uint32_t word = comp(wq[j], kt);
                    b0 = fp8pair(word & 0xFFFFu);
                    b1 = fp8pair(word >> 16);
                }
#pragma unroll
                for (int i = 0; i < T::MT; ++i) {
                    if constexpr (MODE == FP8) mma(acc[i][j], a[i], b0, b1);
                    else if (MODE == FP4 || (MODE == MXFP8 && (kt & 1) == 0) || (MODE == FP8G && kt == 0))
                        mma0(d[i][j], a[i], b0, b1);
                    else mma(d[i][j], a[i], b0, b1);
                }
            }
            if constexpr (MODE == FP4 || MODE == MXFP8 || MODE == FP8G) {
                if (MODE == FP4 || (MODE == MXFP8 && (kt & 1)) || (MODE == FP8G && kt == GS / 16 - 1)) {
                    // a block's products, scaled into acc in block order
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
    if constexpr (FUSE) {                                  // the last slice; one slice: acc is the sum already
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
        __trap();                                     // no clusters before sm_90: the host never launches this
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
                if (SK > 1 && !CLUSTER && !FUSE) {        // unscaled slice partials; the reduce scales their sum
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

template <int MODE, int BM, bool F32, bool CLUSTER, bool FUSE = false>
void launch(const at::Tensor& x, const at::Tensor& w, const at::Tensor& bs, double scale, at::Tensor& out,
            const at::Tensor& part, int N, int K, int SK, int npad) {
    constexpr int BN = 64, WM = 1, WN = 4, STAGES = 4;
    using T = Tile<MODE, BM, BN, WM, WN, STAGES>;
    const int M = x.size(0);
    auto kernel = qmmf_kernel<MODE, BM, BN, WM, WN, STAGES, F32, CLUSTER, FUSE>;
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM);
        configured = true;
    }
    cudaLaunchConfig_t config = {};
    const int rows_t = (M + BM - 1) / BM;
    const int group = std::max(1, std::min(rows_t, static_cast<int>((12LL << 20) / (static_cast<long long>(BM) * K * 2))));
    config.gridDim = dim3(rows_t * ((N + BN - 1) / BN), 1, FUSE ? 1 : SK);
    config.blockDim = dim3(T::THREADS);
    config.dynamicSmemBytes = T::SMEM;
    config.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr[1];
    if (CLUSTER) {
        attr[0].id = cudaLaunchAttributeClusterDimension;
        attr[0].val.clusterDim.x = 1;
        attr[0].val.clusterDim.y = 1;
        attr[0].val.clusterDim.z = SK;
        config.attrs = attr;
        config.numAttrs = 1;
    }
    C10_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<const unsigned char*>(w.data_ptr()),
        bs.defined() ? reinterpret_cast<const uint8_t*>(bs.data_ptr()) : nullptr, static_cast<float>(scale),
        out.data_ptr(), part.defined() ? part.data_ptr<float>() : nullptr, M, N, K, SK, npad,
        M == 1 ? K : static_cast<int>(x.stride(0)), group));
}

template <int MODE, bool F32, bool CLUSTER>
void by_rows(int bm, const at::Tensor& x, const at::Tensor& w, const at::Tensor& bs, double scale, at::Tensor& out,
             const at::Tensor& part, int N, int K, int SK, int npad) {
    switch (bm) {
        case 16: launch<MODE, 16, F32, CLUSTER>(x, w, bs, scale, out, part, N, K, SK, npad); break;
        case 32: launch<MODE, 32, F32, CLUSTER>(x, w, bs, scale, out, part, N, K, SK, npad); break;
        case 64: launch<MODE, 64, F32, CLUSTER>(x, w, bs, scale, out, part, N, K, SK, npad); break;
        default: launch<MODE, 64, F32, false, true>(x, w, bs, scale, out, part, N, K, SK, npad); break;   // 0: fused
    }
}

template <int MODE>
void by_output(int bm, bool f32, bool cluster, const at::Tensor& x, const at::Tensor& w, const at::Tensor& bs,
               double scale, at::Tensor& out, const at::Tensor& part, int N, int K, int SK, int npad) {
    if (f32) { if (cluster) by_rows<MODE, true, true>(bm, x, w, bs, scale, out, part, N, K, SK, npad);
               else by_rows<MODE, true, false>(bm, x, w, bs, scale, out, part, N, K, SK, npad); }
    else { if (cluster) by_rows<MODE, false, true>(bm, x, w, bs, scale, out, part, N, K, SK, npad);
           else by_rows<MODE, false, false>(bm, x, w, bs, scale, out, part, N, K, SK, npad); }
}

}  // namespace

void qmmf_cuda(const at::Tensor& x, const at::Tensor& w, const at::Tensor& bs, double scale, at::Tensor& out,
               const at::Tensor& part, int64_t mode, int64_t N, int64_t K, int64_t SK, int64_t npad, int64_t bm,
               bool f32) {
    // slices add in one order via a cluster's shared memory (sm_90 on) or ``part`` and the reduce, or (bm 0, prompt
    // rows) in one block: the same bits
    const bool fused = bm == 0;
    const bool cluster = !fused && SK > 1 && SK <= 8 && !part.defined() &&
                         at::cuda::getCurrentDeviceProperties()->major >= 9;
    const int n = static_cast<int>(N), k = static_cast<int>(K), sk = static_cast<int>(SK), np = static_cast<int>(npad);
    at::Tensor slices = part;                         // sm_89: no clusters, so slices up to 8 meet here too
    if (!fused && SK > 1 && !cluster && !slices.defined())
        slices = at::empty({SK, x.size(0), N}, out.options().dtype(at::kFloat));
    const int b = static_cast<int>(bm);
    if (mode == FP4) by_output<FP4>(b, f32, cluster, x, w, bs, scale, out, slices, n, k, sk, np);
    else if (mode == FP8) by_output<FP8>(b, f32, cluster, x, w, bs, scale, out, slices, n, k, sk, np);
    else if (mode == MXFP8) by_output<MXFP8>(b, f32, cluster, x, w, bs, scale, out, slices, n, k, sk, np);
    else by_output<FP8G>(b, f32, cluster, x, w, bs, scale, out, slices, n, k, sk, np);
    if (!fused && SK > 1 && !cluster) {
        const long long total = static_cast<long long>(x.size(0)) * N;
        const int threads = 256, blocks = static_cast<int>((total + threads - 1) / threads);
        auto stream = at::cuda::getCurrentCUDAStream();
        if (f32) reduce_kernel<true><<<blocks, threads, 0, stream>>>(slices.data_ptr<float>(), out.data_ptr(), total,
                                                                     sk, static_cast<float>(scale));
        else reduce_kernel<false><<<blocks, threads, 0, stream>>>(slices.data_ptr<float>(), out.data_ptr(), total, sk,
                                                                  static_cast<float>(scale));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

namespace {

// NVFP4 weights staged for ``qmm_prefill8w`` (groups of 64): per (64-column tile, group) a block of 8 warps, a lane
// turning its two tiled words (16 codes of one column) into its 16 e4m3 bytes of the same fragment lane (the layouts
// share lanes), scaled by the column's max |value| / 448 over the group, rounded up to bf16.
__global__ void __launch_bounds__(256) stage_fp4_kernel(const uint32_t* __restrict__ words, const uint8_t* __restrict__ bs,
                                                        float global, uint8_t* __restrict__ w8,
                                                        __nv_bfloat16* __restrict__ scales, int KG, int npad) {
    constexpr int OFF[8] = {0, 8, 16, 24, 1, 9, 17, 25};  // nibble slot -> input offset (``qmm.OFFSETS``)
    constexpr int SLOT[8] = {0, 4, 1, 5, 2, 6, 3, 7};     // fragment byte -> nibble slot
    const int t = blockIdx.x, g = blockIdx.y, jj = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int col = t * 64 + jj * 8 + (lane >> 2), tig = lane & 3;
    const size_t at = ((static_cast<size_t>(t) * KG + g) * 8 + jj) * 32 + lane;
    float v[2][8];
    float top = 0.0f;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const uint32_t w = words[at * 2 + h];
#pragma unroll
        for (int p = 0; p < 8; ++p) {
            const int k = 32 * h + 2 * tig + OFF[p];
            const uint32_t code = (w >> (4 * p)) & 0xFu;
            const float mag = (code & 7u) < 2 ? 0.5f * (code & 1u) : static_cast<float>(1u << (((code & 7u) >> 1) - 1)) *
                                                                         (1.0f + 0.5f * (code & 1u));
            const float s = e4m3f(bs[((static_cast<size_t>(t) * KG + g) * 64 + jj * 8 + (lane >> 2)) * 4 + k / 16]);
            v[h][p] = ((code & 8u) ? -mag : mag) * s * global;
            top = fmaxf(top, fabsf(v[h][p]));
        }
    }
    top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, 1));
    top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, 2));
    const __nv_bfloat16 s = __float2bfloat16_ru(top > 0.0f ? top / 448.0f : 1.0f);
    if (tig == 0) scales[static_cast<size_t>(g) * npad + col] = s;
    const float sf = __bfloat162float(s);
    uint32_t out[2][2];
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            uint32_t packed = 0;
#pragma unroll
            for (int i = 0; i < 4; ++i)
                packed |= static_cast<uint32_t>(__nv_cvt_float_to_fp8(v[h][SLOT[r * 4 + i]] / sf, __NV_SATFINITE, __NV_E4M3))
                          << (8 * i);
            out[h][r] = packed;
        }
    uint2* dst = reinterpret_cast<uint2*>(w8 + (static_cast<size_t>(t) * KG + g) * 4096) + (jj * 32 + lane) * 2;
    dst[0] = make_uint2(out[0][0], out[0][1]);
    dst[1] = make_uint2(out[1][0], out[1][1]);
}

}  // namespace

void stage_fp4_cuda(const at::Tensor& words, const at::Tensor& bs, double global, at::Tensor& w8, at::Tensor& scales,
                    int64_t KG, int64_t npad) {
    dim3 grid(static_cast<unsigned>(npad / 64), static_cast<unsigned>(KG));
    stage_fp4_kernel<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint32_t*>(words.data_ptr()), reinterpret_cast<const uint8_t*>(bs.data_ptr()),
        static_cast<float>(global), reinterpret_cast<uint8_t*>(w8.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(scales.data_ptr()), static_cast<int>(KG), static_cast<int>(npad));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
