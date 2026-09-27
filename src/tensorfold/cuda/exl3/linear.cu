// EXL3 linear layers on CUDA for any codebook and width (decode.cuh, format.py): y = x @ W + bias for 1 to 128
// rows, W = diag(suh) H W_q H diag(svh).
//
// Two kernels:
//   rot_in   xh [M, K] fp16 = fp16(((x * suh) @ H) / sqrt(128)), one warp per (row, 128-input block)
//   linear   program (128-column block nb, K split s) of WK warps; warp w owns a fixed run of the split's k
//            tiles and decodes each one's eight 16x16 tiles straight into mma.m16n8k16 fragments. The trellis
//            words come in through the read-only cache: at 3 bits and up each lane loads its own windows of a
//            tile (a warp's whole search is 8 tiles - 256 B at 2 bits - so the L1 serves it from one line), at
//            1 and 2 bits a lane's windows span two words that the warp instead loads as one 64 or 128 word run
//            (two words a lane) and takes by shuffle. Up to 6 bits the next k step's words are loaded while the
//            current one is decoded, which is where the read bandwidth is won at 2, 5 and 6 bits; at 7 and 8
//            bits a step is 224 or 256 B of words a warp and the same build prefetching there reads 1-2% slower,
//            so it loads per tile. Rows go in passes of 16. The warps' fp32 sums are added in warp order through
//            shared memory, rows 0-7 of the pass and then rows 8-15, so a program holds WK * min(M, 8) rows of
//            128 floats. With one split the program finishes its columns (output
//            Hadamard, * svh, + bias); with SK splits it writes its sum to Z and the last of the column block's
//            programs to arrive (a counter, the only atomic) adds the SK sums in split order and finishes them.
//
// Every output depends only on its own row: rows share an mma but mma keeps rows independent, the k ranges of
// warps and splits depend only on (K, N), and every sum runs in a fixed order.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>

#include "decode.cuh"

using namespace tf_exl3;

namespace {

enum DType : int { F16 = 0, BF16 = 1, F32 = 2 };

__device__ __forceinline__ void load4(const void* p, int dtype, size_t i, float (&v)[4]) {
    if (dtype == F32) {
        const float4 u = *reinterpret_cast<const float4*>(static_cast<const float*>(p) + i);
        v[0] = u.x; v[1] = u.y; v[2] = u.z; v[3] = u.w;
    } else if (dtype == BF16) {
        const uint2 u = *reinterpret_cast<const uint2*>(static_cast<const __nv_bfloat16*>(p) + i);
        const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
        const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
        v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
    } else {
        const uint2 u = *reinterpret_cast<const uint2*>(static_cast<const half*>(p) + i);
        const float2 a = __half22float2(*reinterpret_cast<const half2*>(&u.x));
        const float2 b = __half22float2(*reinterpret_cast<const half2*>(&u.y));
        v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
    }
}

__device__ __forceinline__ void store4(void* p, int dtype, size_t i, const float (&v)[4]) {
    if (dtype == F32) {
        *reinterpret_cast<float4*>(static_cast<float*>(p) + i) = make_float4(v[0], v[1], v[2], v[3]);
    } else if (dtype == BF16) {
        __nv_bfloat162 a = __floats2bfloat162_rn(v[0], v[1]), b = __floats2bfloat162_rn(v[2], v[3]);
        uint2 u;
        u.x = *reinterpret_cast<uint32_t*>(&a);
        u.y = *reinterpret_cast<uint32_t*>(&b);
        *reinterpret_cast<uint2*>(static_cast<__nv_bfloat16*>(p) + i) = u;
    } else {
        half2 a = __floats2half2_rn(v[0], v[1]), b = __floats2half2_rn(v[2], v[3]);
        uint2 u;
        u.x = *reinterpret_cast<uint32_t*>(&a);
        u.y = *reinterpret_cast<uint32_t*>(&b);
        *reinterpret_cast<uint2*>(static_cast<half*>(p) + i) = u;
    }
}

// The finished outputs of one row's 128 columns from their fp32 sums (4 a lane): H / sqrt(128), * svh, + bias.
__device__ __forceinline__ void finish(float (&v)[4], int lane, const half* svh, const half* bias, int col) {
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j] = v[j] * HAD_SCALE * __half2float(__ldg(svh + col + j));
        if (bias) v[j] += __half2float(__ldg(bias + col + j));
    }
}

// The trellis words of one k step (the eight 16x16 tiles of a 128-column block, 8 * TW words in a row in both
// layouts) that a lane needs. At 1 and 2 bits a lane's windows span at most two words and the step is 64 or 128
// words, so the warp loads the step together, TW / 4 words a lane, and each lane takes its two words of a tile
// from their owners by shuffle. At other widths each lane loads its own lane_words of every tile (the same words
// ldg_lane_words reads).
template <int K2>
__host__ __device__ constexpr bool step_shuffled() {
    return K2 == 2 || K2 == 4;
}

template <int K2>
__host__ __device__ constexpr int step_regs() {
    return step_shuffled<K2>() ? tile_words<K2>() / 4 : 8 * lane_words<K2>();
}

template <int K2>
__device__ __forceinline__ void load_step(const uint32_t* step, int lane, uint32_t (&raw)[step_regs<K2>()]) {
    constexpr int TW = tile_words<K2>(), LW = lane_words<K2>();
    if constexpr (step_shuffled<K2>()) {
#pragma unroll
        for (int c = 0; c < step_regs<K2>(); ++c) raw[c] = __ldg(step + c * 32 + lane);
    } else {
        int word, offset;
        lane_start<K2>(lane, word, offset);
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int q = 0; q < LW; ++q) raw[j * LW + q] = __ldg(step + j * TW + (word + q) % TW);
    }
}

// Tile j's lane words from a loaded step; prev: the lane's first window starts in the word before its own.
template <int K2>
__device__ __forceinline__ void step_lane_words(const uint32_t (&raw)[step_regs<K2>()], int j, int lane, bool prev,
                                                uint32_t (&w)[lane_words<K2>()]) {
    constexpr int TW = tile_words<K2>(), LW = lane_words<K2>();
    if constexpr (step_shuffled<K2>()) {
        static_assert(LW == 2, "a lane's windows span two words at 1 and 2 bits");
        constexpr int LPW = 8 / K2;                  // lanes whose windows end in the same word
        const uint32_t r = raw[j * TW / 32];
        const int base = (j * TW) % 32;
        const uint32_t own = __shfl_sync(0xffffffffu, r, base + lane / LPW);
        const uint32_t before = __shfl_sync(0xffffffffu, r, base + (lane / LPW + TW - 1) % TW);
        w[0] = prev ? before : own;
        w[1] = prev ? own : 0u;                      // a window inside one word never reads w[1]
    } else {
#pragma unroll
        for (int q = 0; q < LW; ++q) w[q] = raw[j * LW + q];
    }
}

__global__ void __launch_bounds__(128) rot_in_kernel(const void* __restrict__ x, int x_dtype,
                                                     const half* __restrict__ suh, half* __restrict__ xh, int K) {
    const int blk = blockIdx.x * 4 + (threadIdx.x >> 5), row = blockIdx.y, lane = threadIdx.x & 31;
    if (blk * 128 >= K) return;
    const int k = blk * 128 + 4 * lane;
    float v[4], s[4];
    load4(x, x_dtype, (size_t)row * K + k, v);
    load4(suh, F16, k, s);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= s[j];
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= HAD_SCALE;
    store4(xh, F16, (size_t)row * K + k, v);
}

template <int K2, int CB, int WK>
__global__ void __launch_bounds__(WK * 32) linear_kernel(
    const half* __restrict__ xh, const uint32_t* __restrict__ T, long long stride_k, long long stride_nb,
    const half* __restrict__ svh, const half* __restrict__ bias, void* __restrict__ y, int y_dtype,
    float* __restrict__ Z, int* __restrict__ counters, int M, int K, int N, int SK) {
    constexpr int TW = tile_words<K2>();
    constexpr int LW = lane_words<K2>();
    extern __shared__ __align__(16) float red[];              // WK * RH * 128 floats
    __shared__ int last;
    const int RH = min(M, 8);                                 // rows of red a warp

    const int nb = blockIdx.x, split = blockIdx.y, NB = gridDim.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int per_warp = (K >> 4) / SK / WK;
    const int kt0 = split * (per_warp * WK) + warp * per_warp;
    const uint32_t* tiles = T + nb * stride_nb;
    const int col0 = nb * 128;
    bool prev = false;
    if constexpr (step_shuffled<K2>()) {
        int word, offset;
        lane_start<K2>(lane, word, offset);
        prev = word != lane / (8 / K2);
    }

    for (int m0 = 0, pass = 0; m0 < M; m0 += 16, ++pass) {
        const int R = min(16, M - m0);

        float acc[8][2][4];
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

        // the walk's rows for the mma, clamped so rows past the pass read inside the buffer (their outputs are dropped)
        const int r0 = m0 + (g < R ? g : R - 1), r1 = m0 + (g + 8 < R ? g + 8 : R - 1);
        const half* x0 = xh + (size_t)r0 * K;
        const half* x1 = xh + (size_t)r1 * K;
        const uint32_t* tile = tiles + (size_t)kt0 * stride_k;
        // up to 6 bits the next k step's words are loaded while this one is decoded; 7 and 8 bits load per tile
        constexpr bool PF = K2 <= 12;
        constexpr int SR = step_regs<K2>();
        uint32_t cur[PF ? SR : 1], nxt[PF ? SR : 1];
        if constexpr (PF) load_step<K2>(tile, lane, cur);
#pragma unroll 1
        for (int i = 0; i < per_warp; ++i) {
            const int kt = kt0 + i;
            uint32_t a[4];
            a[0] = __ldg(reinterpret_cast<const uint32_t*>(x0 + kt * 16 + 2 * t));
            a[1] = __ldg(reinterpret_cast<const uint32_t*>(x1 + kt * 16 + 2 * t));
            a[2] = __ldg(reinterpret_cast<const uint32_t*>(x0 + kt * 16 + 2 * t + 8));
            a[3] = __ldg(reinterpret_cast<const uint32_t*>(x1 + kt * 16 + 2 * t + 8));
            if constexpr (PF) {
                if (i + 1 < per_warp) load_step<K2>(tile + (size_t)(i + 1) * stride_k, lane, nxt);
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                uint32_t w[LW];
                if constexpr (PF) step_lane_words<K2>(cur, j, lane, prev, w);
                else ldg_lane_words<K2>(tile + (size_t)i * stride_k + j * TW, lane, w);
                uint32_t b0[2], b1[2];
                decode_lane<K2, CB>(w, lane, b0, b1);
                mma16816(acc[j][0], a, b0);
                mma16816(acc[j][1], a, b1);
            }
            if constexpr (PF) {
#pragma unroll
                for (int q = 0; q < SR; ++q) cur[q] = nxt[q];
            }
        }

        // the warps' sums, added in warp order, rows 0-7 of the pass and then rows 8-15
        for (int rlo = 0; rlo < R; rlo += 8) {
            const int rn = min(R - rlo, 8);
            __syncthreads();                         // red is reused by every half and pass
            if (g < RH) {
#pragma unroll
                for (int i = 0; i < 8; ++i)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int col = i * 16 + h * 8 + 2 * t;
                        *reinterpret_cast<float2*>(red + (warp * RH + g) * 128 + col) =
                            rlo ? make_float2(acc[i][h][2], acc[i][h][3]) : make_float2(acc[i][h][0], acc[i][h][1]);
                    }
            }
            __syncthreads();

            if (SK == 1) {
                for (int r = warp; r < rn; r += WK) {
                    float v[4];
                    const float4 u = *reinterpret_cast<const float4*>(red + r * 128 + 4 * lane);
                    v[0] = u.x; v[1] = u.y; v[2] = u.z; v[3] = u.w;
#pragma unroll
                    for (int w = 1; w < WK; ++w) {
                        const float4 q = *reinterpret_cast<const float4*>(red + (w * RH + r) * 128 + 4 * lane);
                        v[0] += q.x; v[1] += q.y; v[2] += q.z; v[3] += q.w;
                    }
                    finish(v, lane, svh, bias, col0 + 4 * lane);
                    store4(y, y_dtype, (size_t)(m0 + rlo + r) * N + col0 + 4 * lane, v);
                }
            } else {
                for (int idx = threadIdx.x; idx < rn * 32; idx += WK * 32) {
                    const int r = idx >> 5, c = 4 * (idx & 31);
                    float4 s = *reinterpret_cast<const float4*>(red + r * 128 + c);
#pragma unroll
                    for (int w = 1; w < WK; ++w) {
                        const float4 q = *reinterpret_cast<const float4*>(red + (w * RH + r) * 128 + c);
                        s.x += q.x; s.y += q.y; s.z += q.z; s.w += q.w;
                    }
                    *reinterpret_cast<float4*>(Z + ((size_t)split * M + m0 + rlo + r) * N + col0 + c) = s;
                }
            }
        }
        if (SK > 1) {
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0) last = atomicAdd(counters + pass * NB + nb, 1) == SK - 1;
            __syncthreads();
            if (last) {
                __threadfence();
                for (int r = warp; r < R; r += WK) {
                    const size_t at = ((size_t)m0 + r) * N + col0 + 4 * lane;
                    float4 s = __ldcg(reinterpret_cast<const float4*>(Z + at));
                    for (int q = 1; q < SK; ++q) {
                        const float4 u = __ldcg(reinterpret_cast<const float4*>(Z + (size_t)q * M * N + at));
                        s.x += u.x; s.y += u.y; s.z += u.z; s.w += u.w;
                    }
                    float v[4] = {s.x, s.y, s.z, s.w};
                    finish(v, lane, svh, bias, col0 + 4 * lane);
                    store4(y, y_dtype, (size_t)(m0 + r) * N + col0 + 4 * lane, v);
                }
                if (threadIdx.x == 0) counters[pass * NB + nb] = 0;   // every program of the block has arrived
            }
        }
        __syncthreads();                             // red is reused in the next pass
    }
}

// W_q [K, N] fp16 from the trellis words in the stored layout: one warp per tile.
template <int K2, int CB>
__global__ void __launch_bounds__(32) unpack_kernel(const uint32_t* __restrict__ T, half* __restrict__ W, int N) {
    const int kt = blockIdx.y, nt = blockIdx.x, lane = threadIdx.x;
    const int NTILES = N >> 4;
    uint32_t w[lane_words<K2>()];
    ldg_lane_words<K2>(T + ((size_t)kt * NTILES + nt) * tile_words<K2>(), lane, w);
    uint32_t b[2][2];
    decode_lane<K2, CB>(w, lane, b[0], b[1]);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint32_t v = b[j >> 2][(j >> 1) & 1];
        const unsigned short h = (j & 1) ? (unsigned short)(v >> 16) : (unsigned short)(v & 0xffffu);
        W[(size_t)(kt * 16 + value_row(lane, j)) * N + nt * 16 + value_col(lane, j)] = __ushort_as_half(h);
    }
}

int dtype_of(const at::Tensor& t) {
    return t.scalar_type() == at::kFloat ? F32 : t.scalar_type() == at::kBFloat16 ? BF16 : F16;
}

}  // namespace

#define TF_EXL3_WIDTHS(X, CB) X(2, CB) X(4, CB) X(6, CB) X(8, CB) X(10, CB) X(12, CB) X(14, CB) X(16, CB)
#define TF_EXL3_ALL(X) TF_EXL3_WIDTHS(X, 0) TF_EXL3_WIDTHS(X, 1) TF_EXL3_WIDTHS(X, 2) X(3, 2) X(5, 2) X(7, 2)

void exl3_rot_in_cuda(const at::Tensor& x, const at::Tensor& suh, at::Tensor& xh) {
    const int M = (int)x.size(0), K = (int)x.size(1);
    dim3 grid((unsigned)((K / 128 + 3) / 4), (unsigned)M);
    rot_in_kernel<<<grid, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr(), dtype_of(x), reinterpret_cast<const half*>(suh.data_ptr()),
        reinterpret_cast<half*>(xh.data_ptr()), K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exl3_linear_cuda(const at::Tensor& xh, const at::Tensor& T, int64_t stride_k, int64_t stride_nb,
                      const at::Tensor& svh, const c10::optional<at::Tensor>& bias, at::Tensor& y,
                      const c10::optional<at::Tensor>& Z, at::Tensor& counters, int64_t K2, int64_t cb, int64_t SK,
                      int64_t WK) {
    const int M = (int)xh.size(0), K = (int)xh.size(1), N = (int)y.size(1);
    TORCH_CHECK(WK == 2 || WK == 4 || WK == 8, "WK must be 2, 4 or 8");
    TORCH_CHECK((K / 16) % (SK * WK) == 0, "K / 16 must split evenly over SK * WK warps");
    dim3 grid((unsigned)(N / 128), (unsigned)SK);
    auto stream = at::cuda::getCurrentCUDAStream();
    const half* bptr = bias ? reinterpret_cast<const half*>(bias->data_ptr()) : nullptr;
    float* zptr = Z ? Z->data_ptr<float>() : nullptr;
    TORCH_CHECK(SK == 1 || zptr, "Z is needed with more than one split");
#define TF_LAUNCH(K2_, CB_)                                                                                        \
    if (K2 == K2_ && cb == CB_) {                                                                               \
        auto kernel = WK == 2 ? linear_kernel<K2_, CB_, 2>                                                      \
                              : WK == 4 ? linear_kernel<K2_, CB_, 4> : linear_kernel<K2_, CB_, 8>;              \
        const int smem = (int)(WK * std::min(M, 8) * 128 * sizeof(float));                                \
        if (smem > 48 * 1024) cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
        kernel<<<grid, (unsigned)(WK * 32), smem, stream>>>(                                              \
            reinterpret_cast<const half*>(xh.data_ptr()), reinterpret_cast<const uint32_t*>(T.data_ptr()),      \
            stride_k, stride_nb, reinterpret_cast<const half*>(svh.data_ptr()), bptr, y.data_ptr(), dtype_of(y),\
            zptr, counters.data_ptr<int>(), M, K, N, (int)SK);                                                  \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                         \
        return;                                                                                                 \
    }
    TF_EXL3_ALL(TF_LAUNCH)
#undef TF_LAUNCH
    TORCH_CHECK(false, "unsupported EXL3 width/codebook: K2=", K2, " codebook=", cb);
}

void exl3_unpack_cuda(const at::Tensor& T, at::Tensor& W, int64_t K2, int64_t cb) {
    const int K = (int)W.size(0), N = (int)W.size(1);
    dim3 grid((unsigned)(N / 16), (unsigned)(K / 16));
    auto stream = at::cuda::getCurrentCUDAStream();
#define TF_LAUNCH(K2_, CB_)                                                                                         \
    if (K2 == K2_ && cb == CB_) {                                                                                \
        unpack_kernel<K2_, CB_><<<grid, 32, 0, stream>>>(reinterpret_cast<const uint32_t*>(T.data_ptr()),         \
                                                        reinterpret_cast<half*>(W.data_ptr()), N);               \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                          \
        return;                                                                                                  \
    }
    TF_EXL3_ALL(TF_LAUNCH)
#undef TF_LAUNCH
    TORCH_CHECK(false, "unsupported EXL3 width/codebook: K2=", K2, " codebook=", cb);
}
