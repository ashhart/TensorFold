// Lane matmuls in the checkpoint's own math: NVFP4 rows times NVFP4 weights on the block-scaled FP4 mma (A4), and
// FP8 rows times FP8 weights on the e4m3 mma (A8). Rows quantize alone; K slices by shape (never by the row count)
// and slices add in a fixed order, so a row's bits depend only on its own inputs, never on the block it lands in.
// Needs sm_120a / sm_121a for A4.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <type_traits>

#include "../kernels/qmm_frag.cuh"
#include "mma4.cuh"

namespace {

using namespace qmm_frag;
using namespace mma4;

constexpr int STAGES = 4;

// A block: BM x BN on WM x WN warps of (BM / WM) x (BN / WN); SKIP: m16 tiles wholly past M skip their mmas.
template <int BM_, int BN_, int WM_, int WN_, bool SKIP_ = false>
struct Cfg {
    static constexpr int BM = BM_, BN = BN_, WM = WM_, WN = WN_, THREADS = WM_ * WN_ * 32;
    static constexpr bool SKIP = SKIP_;
};

template <int N>
using ic = std::integral_constant<int, N>;

template <int MODE, typename C, bool CLUSTER>
struct Tile {
    static constexpr int MT = C::BM / C::WM / 16;         // m16 tiles a warp
    static constexpr int NT = C::BN / C::WN / 8;          // n8 tiles a warp
    static constexpr int ROW = MODE == A4 ? 32 : 64;      // bytes a row a stage (64 inputs)
    static constexpr int X = C::BM * ROW;
    static constexpr int SX = MODE == A4 ? C::BM * 4 : 0; // row block scales a stage
    static constexpr int W64 = MODE == A4 ? 64 * 32 : 64 * 64;  // one 64-column weight tile a stage
    static constexpr int W = C::BN / 64 * W64;
    static constexpr int SW = MODE == A4 ? C::BN * 4 : 0; // column block scales a stage
    static constexpr int STAGE = (X + SX + W + SW + 127) / 128 * 128;
    static constexpr int PARTIALS = CLUSTER ? C::BM * C::BN * 4 : 0;     // a K slice's sums, parked for the cluster
    static constexpr int SMEM = STAGES * STAGE > PARTIALS ? STAGES * STAGE : PARTIALS;
};

// A4: x codes [M, K/2], x scales [K/64, mpad, 4], w words [npad/64, K/64, 8, 32, 2] (lane (g, t) of n8 tile j:
// column 8j + g, inputs 8t..8t+7 then 32+8t.., low nibble first), w scales [npad/64, K/64, 64, 4].
// A8: x e4m3 [M, K] and w e4m3 [npad/64, K/64, 8, 32, 2, 8], both in the fragment order ``quant8`` writes.
// Every tile runs each output's K steps on the same mma in the same order, so tiles never change bits.
template <int MODE, typename C, bool F32, bool CLUSTER>
__global__ void __launch_bounds__(C::THREADS) lane_kernel(
        const uint8_t* __restrict__ x, const uint8_t* __restrict__ xs, const uint8_t* __restrict__ w,
        const uint8_t* __restrict__ ws, float alpha, void* __restrict__ out, float* __restrict__ part, int M, int N,
        int K, int SK, int mpad, int group) {
    using T = Tile<MODE, C, CLUSTER>;
    constexpr int THREADS = C::THREADS;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    const int r0 = warp / C::WN * (C::BM / C::WM), c0 = warp % C::WN * (C::BN / C::WN);   // the warp's corner
    const int KG = K / 64, per = KG / SK;
    const int2 at = tile_of(blockIdx.x, M, N, C::BM, C::BN, group);
    const int m0 = at.x, n0 = at.y, slice = blockIdx.z, g0 = slice * per;
    constexpr int CH = T::ROW / 16;

    auto stage = [&](int s) { return buf + s * T::STAGE; };
    auto load = [&](int s, int kg) {
        unsigned char* p = stage(s);
        for (int c = tid; c < C::BM * CH; c += THREADS) {
            const int r = c / CH, ch = c % CH;
            const bool in = m0 + r < M;
            cp16z(p + r * T::ROW + chunk<MODE>(r, ch) * 16,
                  x + static_cast<size_t>(in ? m0 + r : 0) * (K / (MODE == A4 ? 2 : 1)) + kg * T::ROW + ch * 16, in);
        }
        if constexpr (MODE == A4) {
            for (int c = tid; c < C::BM / 4; c += THREADS) {
                if constexpr (C::BM <= 64) {          // mpad holds whole 64-row tiles
                    cp16(p + T::X + c * 16, xs + (static_cast<size_t>(kg) * mpad + m0) * 4 + c * 16);
                } else {
                    const bool in = m0 + 4 * c < mpad;
                    cp16z(p + T::X + c * 16, xs + (static_cast<size_t>(kg) * mpad + (in ? m0 + 4 * c : 0)) * 4, in);
                }
            }
        }
        unsigned char* pw = p + T::X + T::SX;
        for (int c = tid; c < T::W / 16; c += THREADS) {
            if constexpr (C::BN == 64) {
                cp16(pw + c * 16, w + (static_cast<size_t>(n0 / 64) * KG + kg) * T::W64 + c * 16);
            } else {                                  // a 64-column tile past N: zeros, never stored
                const int tj = c / (T::W64 / 16), off = c % (T::W64 / 16);
                const bool in = n0 + 64 * tj < N;
                const size_t from = (static_cast<size_t>(n0 / 64 + (in ? tj : 0)) * KG + kg) * T::W64 + off * 16;
                cp16z(pw + c * 16, w + from, in);
            }
        }
        if constexpr (MODE == A4) {
            for (int c = tid; c < T::SW / 16; c += THREADS) {
                const int tj = c / 16, off = c % 16;
                const bool in = C::BN == 64 || n0 + 64 * tj < N;
                const size_t from = (static_cast<size_t>(n0 / 64 + (in ? tj : 0)) * KG + kg) * 256 + off * 16;
                cp16z(pw + T::W + c * 16, ws + from, in);
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
        if (s < per) load(s, g0 + s);
        commit();
    }
    const int q = lane >> 3, rr = lane & 7;
    auto steps = [&](auto live_c) {                           // LIVE: the warp's m16 tiles holding rows below M
        constexpr int LIVE = decltype(live_c)::value;
        for (int it = 0; it < per; ++it) {
            wait<STAGES - 2>();
            __syncthreads();
            const int next = it + STAGES - 1;
            if (next < per) load(next % STAGES, g0 + next);
            commit();
            const unsigned char* p = stage(it % STAGES);
            const unsigned char* pw = p + T::X + T::SX;
            if constexpr (LIVE == 0) {
                continue;
            } else if constexpr (MODE == A4) {
                const uint32_t* sx = reinterpret_cast<const uint32_t*>(p + T::X);
                const uint32_t* sw = reinterpret_cast<const uint32_t*>(pw + T::W);
                uint32_t a[LIVE][4], sa[LIVE];
#pragma unroll
                for (int i = 0; i < LIVE; ++i) {
                    const int r = r0 + i * 16 + rr + (q & 1) * 8;
                    ldmatrix4(a[i], p + r * T::ROW + chunk<MODE>(r, q >> 1) * 16);
                    const uint32_t v = sx[r0 + i * 16 + g + 8 * (t & 1)];   // row g (t 0), g + 8 (t 1): no branch
                    sa[i] = v;                     // lanes 2, 3 repeat rows g, g + 8: never read
                }
#pragma unroll
                for (int j = 0; j < T::NT; ++j) {
                    const int jj = c0 / 8 + j;
                    const uint2 b = reinterpret_cast<const uint2*>(pw)[jj * 32 + lane];
                    const uint32_t sb = sw[jj * 8 + g];      // every lane its column's: lane 0's is read
#pragma unroll
                    for (int i = 0; i < LIVE; ++i) mma_fp4(acc[i][j], a[i], b.x, b.y, sa[i], sb);
                }
            } else {
                uint4 b[T::NT];
#pragma unroll
                for (int j = 0; j < T::NT; ++j) b[j] = reinterpret_cast<const uint4*>(pw)[(c0 / 8 + j) * 32 + lane];
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    uint32_t a[LIVE][4];
#pragma unroll
                    for (int i = 0; i < LIVE; ++i) {
                        const int r = r0 + i * 16 + rr + (q & 1) * 8;
                        ldmatrix4(a[i], p + r * T::ROW + chunk<MODE>(r, 2 * h + (q >> 1)) * 16);
                    }
#pragma unroll
                    for (int j = 0; j < T::NT; ++j)
#pragma unroll
                        for (int i = 0; i < LIVE; ++i)
                            mma_fp8(acc[i][j], a[i], h ? b[j].z : b[j].x, h ? b[j].w : b[j].y);
                }
            }
        }
    };
    if constexpr (!C::SKIP || T::MT == 1) {
        steps(ic<T::MT>());
    } else {
        const int live = min(T::MT, max(0, (M - m0 - r0 + 15) / 16));
        if (live == 0) steps(ic<0>());
        else if (live == 1) steps(ic<1>());
        else if (T::MT > 2 && live == 2) steps(ic<(T::MT > 2 ? 2 : T::MT)>());
        else if (T::MT > 3 && live == 3) steps(ic<(T::MT > 3 ? 3 : T::MT)>());
        else steps(ic<T::MT>());
    }
    wait<0>();
    __syncthreads();
    auto put = [&](int row, int col, float v0, float v1) {        // a scaled pair at (row, col), (row, col + 1)
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
    };
    if constexpr (CLUSTER) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
        __trap();                                 // no clusters before sm_90: lane_cuda reduces through ``part`` there
#else
        // every slice parks its partials; slice s then sums pairs s, s + SK, .. over slices 0, 1, .. in that order
        auto cluster = cooperative_groups::this_cluster();
        float* mine = reinterpret_cast<float*>(buf);
#pragma unroll
        for (int i = 0; i < T::MT; ++i)
#pragma unroll
            for (int j = 0; j < T::NT; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) mine[((i * T::NT + j) * 4 + e) * THREADS + tid] = acc[i][j][e];
        cluster.sync();
        for (int pr = slice; pr < T::MT * T::NT * 2; pr += SK) {     // pair (i * NT + j) * 2 + h: entries 2pr, 2pr + 1
            const int i = pr / (T::NT * 2), j = pr / 2 % T::NT, h = pr % 2;
            const int row = m0 + r0 + i * 16 + g + 8 * h, col = n0 + c0 + j * 8 + t * 2;
            const float* p0 = cluster.map_shared_rank(mine, 0);
            float v0 = p0[2 * pr * THREADS + tid], v1 = p0[(2 * pr + 1) * THREADS + tid];
#pragma unroll 7
            for (int peer = 1; peer < SK; ++peer) {
                const float* theirs = cluster.map_shared_rank(mine, peer);
                v0 = v0 + theirs[2 * pr * THREADS + tid];
                v1 = v1 + theirs[(2 * pr + 1) * THREADS + tid];
            }
            if (row < M) put(row, col, v0 * alpha, v1 * alpha);
        }
        cluster.sync();
#endif
    } else {
#pragma unroll
        for (int i = 0; i < T::MT; ++i)
#pragma unroll
            for (int j = 0; j < T::NT; ++j) {
                const int col = n0 + c0 + j * 8 + t * 2;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int row = m0 + r0 + i * 16 + g + h * 8;
                    if (row >= M) continue;
                    if (SK > 1) {                         // unscaled slice partials; the reduce scales their sum
                        float* dst = part + (static_cast<size_t>(slice) * M + row) * N + col;
                        if (col < N) dst[0] = acc[i][j][2 * h];
                        if (col + 1 < N) dst[1] = acc[i][j][2 * h + 1];
                        continue;
                    }
                    put(row, col, acc[i][j][2 * h] * alpha, acc[i][j][2 * h + 1] * alpha);
                }
            }
    }
}

template <bool F32>
__global__ void reduce_kernel(const float* __restrict__ part, void* __restrict__ out, long long total, int SK,
                              float alpha) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float acc = part[i];
    for (int s = 1; s < SK; ++s) acc = acc + part[s * total + i];
    acc = acc * alpha;
    if (F32) reinterpret_cast<float*>(out)[i] = acc;
    else reinterpret_cast<__nv_bfloat16*>(out)[i] = __float2bfloat16_rn(acc);
}

template <int MODE, typename C, bool F32, bool CLUSTER>
void launch(const at::Tensor& x, const at::Tensor& xs, const at::Tensor& w, const at::Tensor& ws, double alpha,
            at::Tensor& out, const at::Tensor& part, int M, int N, int K, int SK, int mpad) {
    using T = Tile<MODE, C, CLUSTER>;
    auto kernel = lane_kernel<MODE, C, F32, CLUSTER>;
    static bool configured = false;
    if (!configured) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM));
        configured = true;
    }
    const int rows_t = (M + C::BM - 1) / C::BM, row_bytes = MODE == A4 ? K / 2 : K;
    const long long tile_bytes = static_cast<long long>(C::BM) * row_bytes;
    const int group = std::max(1, std::min(rows_t, static_cast<int>((12LL << 20) / tile_bytes)));
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(rows_t * ((N + C::BN - 1) / C::BN), 1, SK);
    config.blockDim = dim3(C::THREADS);
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
    C10_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, reinterpret_cast<const uint8_t*>(x.data_ptr()),
        xs.defined() ? reinterpret_cast<const uint8_t*>(xs.data_ptr()) : nullptr,
        reinterpret_cast<const uint8_t*>(w.data_ptr()),
        ws.defined() ? reinterpret_cast<const uint8_t*>(ws.data_ptr()) : nullptr, static_cast<float>(alpha),
        out.data_ptr(), part.defined() ? part.data_ptr<float>() : nullptr, M, N, K, SK, mpad, group));
}

// ``tile`` (``checkpoint.lane_tile``): 16, 32, 64 rows by 64 columns, else BM * 1000 + BN; part-filled launches skip.
template <int MODE, bool F32, bool CLUSTER>
void by_tile(int tile, const at::Tensor& x, const at::Tensor& xs, const at::Tensor& w, const at::Tensor& ws,
             double alpha, at::Tensor& out, const at::Tensor& part, int M, int N, int K, int SK, int mpad) {
    auto go = [&](auto cfg) {
        launch<MODE, decltype(cfg), F32, CLUSTER>(x, xs, w, ws, alpha, out, part, M, N, K, SK, mpad);
    };
    auto block = [&](auto full, auto skip) { M % decltype(full)::BM ? go(skip) : go(full); };
    switch (tile) {
        case 16: go(Cfg<16, 64, 1, 4>()); break;
        case 32: go(Cfg<32, 64, 1, 4>()); break;
        case 64: block(Cfg<64, 64, 1, 4>(), Cfg<64, 64, 1, 4, true>()); break;
        case 64128: block(Cfg<64, 128, 2, 4>(), Cfg<64, 128, 2, 4, true>()); break;
        case 128064: block(Cfg<128, 64, 2, 2>(), Cfg<128, 64, 2, 2, true>()); break;
        case 128128: block(Cfg<128, 128, 2, 4>(), Cfg<128, 128, 2, 4, true>()); break;
        default: TORCH_CHECK(false, "lane tile ", tile, " is not built");
    }
}

template <int MODE>
void by_output(int tile, bool f32, bool cluster, const at::Tensor& x, const at::Tensor& xs, const at::Tensor& w,
               const at::Tensor& ws, double alpha, at::Tensor& out, const at::Tensor& part, int M, int N, int K,
               int SK, int mpad) {
    if (f32) {
        if (cluster) by_tile<MODE, true, true>(tile, x, xs, w, ws, alpha, out, part, M, N, K, SK, mpad);
        else by_tile<MODE, true, false>(tile, x, xs, w, ws, alpha, out, part, M, N, K, SK, mpad);
    } else {
        if (cluster) by_tile<MODE, false, true>(tile, x, xs, w, ws, alpha, out, part, M, N, K, SK, mpad);
        else by_tile<MODE, false, false>(tile, x, xs, w, ws, alpha, out, part, M, N, K, SK, mpad);
    }
}

}  // namespace

void lane_cuda(int64_t mode, const at::Tensor& x, const at::Tensor& xs, const at::Tensor& w, const at::Tensor& ws,
               double alpha, at::Tensor& out, const at::Tensor& part, int64_t N, int64_t K, int64_t SK, int64_t mpad,
               int64_t tile, bool f32) {
    // slices add in one order via a cluster's shared memory (sm_90 on) or ``part`` and the reduce: the same bits
    const bool cluster = SK > 1 && SK <= 8 && !part.defined() && at::cuda::getCurrentDeviceProperties()->major >= 9;
    const int M = static_cast<int>(out.size(0)), n = static_cast<int>(N), k = static_cast<int>(K);
    const int sk = static_cast<int>(SK), mp = static_cast<int>(mpad), b = static_cast<int>(tile);
    at::Tensor slices = part;                     // sm_89: no clusters, so slices up to 8 meet here too
    if (SK > 1 && !cluster && !slices.defined()) slices = at::empty({SK, M, N}, out.options().dtype(at::kFloat));
    if (mode == A4) by_output<A4>(b, f32, cluster, x, xs, w, ws, alpha, out, slices, M, n, k, sk, mp);
    else by_output<A8>(b, f32, cluster, x, xs, w, ws, alpha, out, slices, M, n, k, sk, mp);
    if (SK > 1 && !cluster) {
        const long long total = static_cast<long long>(M) * N;
        const int threads = 256, blocks = static_cast<int>((total + threads - 1) / threads);
        auto stream = at::cuda::getCurrentCUDAStream();
        if (f32) reduce_kernel<true><<<blocks, threads, 0, stream>>>(slices.data_ptr<float>(), out.data_ptr(), total,
                                                                     sk, static_cast<float>(alpha));
        else reduce_kernel<false><<<blocks, threads, 0, stream>>>(slices.data_ptr<float>(), out.data_ptr(), total, sk,
                                                                  static_cast<float>(alpha));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

namespace {

// NVFP4 checkpoint bytes [N, K/2] (input 2j in byte j's low nibble) -> ``lane_kernel``'s A4 words: lane (g, t) of
// n8 tile j in a (64-column tile, 64-input step) holds column 8j + g's inputs 8t..8t+7 and 32+8t..32+8t+7.
__global__ void pack4_kernel(const uint8_t* __restrict__ src, int N, int K, uint32_t* __restrict__ dst) {
    const int tile = blockIdx.x, step = blockIdx.y, j = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int col = tile * 64 + j * 8 + (lane >> 2), t = lane & 3, KG = K / 64;
    uint32_t b0 = 0u, b1 = 0u;
    if (col < N) {
        const uint8_t* row = src + static_cast<size_t>(col) * (K / 2) + step * 32;
        b0 = *reinterpret_cast<const uint32_t*>(row + 4 * t);
        b1 = *reinterpret_cast<const uint32_t*>(row + 16 + 4 * t);
    }
    uint32_t* out = dst + ((static_cast<size_t>(tile) * KG + step) * 8 + j) * 64 + lane * 2;
    out[0] = b0;
    out[1] = b1;
}

}  // namespace

void pack4_cuda(const at::Tensor& src, int64_t N, int64_t K, at::Tensor& dst) {
    const int npad = static_cast<int>(dst.numel() * 4 / (K / 2));
    const dim3 grid(static_cast<unsigned>(npad / 64), static_cast<unsigned>(K / 64));
    pack4_kernel<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        src.data_ptr<uint8_t>(), static_cast<int>(N), static_cast<int>(K),
        reinterpret_cast<uint32_t*>(dst.data_ptr()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
