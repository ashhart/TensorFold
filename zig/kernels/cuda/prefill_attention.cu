// Device code of src/tensorfold/cuda/kernels/prefill_attention.cu (comments, ATen includes dropped), paged keys added.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "common/qmm_frag.cuh"
#include "common/cp_async.cuh"

namespace tf_prefill_attention {

using qmm_frag::commit;
using qmm_frag::cp16z;
using qmm_frag::ldmatrix4;
using qmm_frag::mma;
using qmm_frag::mma0;

constexpr int BN = 64, HALF = 32;             // keys a tile (the fold's unit), keys a staging slot
constexpr float LOG2E = 1.4426950408889634f;
constexpr int SMEM_MAX = 96 * 1024;

using tf_common::cp_wait;

__device__ __forceinline__ void ldmatrix4t(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(qmm_frag::smem(p)));
}

__device__ __forceinline__ float texp(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;\n" : "=f"(y) : "f"(__fmul_rn(x, LOG2E)));
    return y;
}

__device__ __forceinline__ float tdiv(float a, float b) {
    float y;
    asm("div.full.f32 %0, %1, %2;\n" : "=f"(y) : "f"(a), "f"(b));
    return y;
}

__device__ __forceinline__ uint32_t pack(float lo, float hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
}

template <int D>
__device__ __forceinline__ int sw(int r, int c) { return r * (D / 8) + (c ^ (r & 7)); }

__device__ __forceinline__ float rowsum(const float (&c)[8]) {
    float w[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float x = __fadd_rn(c[j], __shfl_xor_sync(0xffffffffu, c[j], 2));
        w[j] = __fadd_rn(x, __shfl_xor_sync(0xffffffffu, x, 1));
    }
    return __fadd_rn(__fadd_rn(__fadd_rn(w[0], w[4]), __fadd_rn(w[2], w[6])),
                     __fadd_rn(__fadd_rn(w[1], w[5]), __fadd_rn(w[3], w[7])));
}

template <int D, int WARPS, int HPC, int NS>
__global__ void __launch_bounds__(32 * WARPS, 1)
pattn_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc,
             const __nv_bfloat16* __restrict__ vc, __nv_bfloat16* __restrict__ out, int p0, int W, int H, int HK,
             int G, float scale, const int* __restrict__ PT, long long PS) {
    constexpr bool causal = true;
    constexpr int skip = 0;
    constexpr bool HPAGED = false, F8 = false, HM = false;  // bf16 keys, row-major
    constexpr int CAP = 0, PTOK = 1;
    const float *ks = nullptr, *vs = nullptr;
    const int HO = H;
    auto first_row = [] { return blockIdx.x * 16 * (WARPS / HPC); };
#include "prefill_attention_rows.cuh"
}

// One segment's rows from row `first` on, keys from p0 on `skip` rows further on: the body for tree_kernel.
template <int D, int WARPS, int HPC, int NS>
__device__ __forceinline__ void
pattn_rows(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc,
           const __nv_bfloat16* __restrict__ vc, __nv_bfloat16* __restrict__ out, int p0, int W, int H, int HK,
           int G, float scale, int first, int skip) {
    constexpr bool causal = true;
    constexpr const int* PT = nullptr;  // contiguous keys
    constexpr long long PS = 0;
    constexpr int PTOK = 1;
    constexpr bool HPAGED = false, F8 = false, HM = false;  // bf16 keys, row-major
    constexpr int CAP = 0;
    const float *ks = nullptr, *vs = nullptr;
    const int HO = H;
    auto first_row = [=] { return first; };
#include "prefill_attention_rows.cuh"
}

// Packed rows: a causal trunk of P rows, then branches at P + cu[b], each over the trunk and its own earlier rows.
template <int D, int WARPS, int HPC, int NS>
__global__ void __launch_bounds__(32 * WARPS, 1)
tree_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc,
            const __nv_bfloat16* __restrict__ vc, __nv_bfloat16* __restrict__ out, const int* __restrict__ cu, int P,
            int n, int H, int HK, int G, float scale) {
    constexpr int ROWS = 16 * (WARPS / HPC);
    int tile = blockIdx.x, b = -1, W = P;                // the trunk's row tiles, then each branch's
    while (tile >= (W + ROWS - 1) / ROWS) {
        tile -= (W + ROWS - 1) / ROWS;
        if (++b == n) return;
        W = cu[b + 1] - cu[b];
    }
    const int64_t at = (int64_t)(b < 0 ? 0 : P + cu[b]) * H * D;
    pattn_rows<D, WARPS, HPC, NS>(q + at, kc, vc, out + at, b < 0 ? 0 : P, W, H, HK, G, scale, tile * ROWS,
                                  b < 0 ? 0 : cu[b]);
}

// One sequence, every row over every key (no mask), a block a head: q, k, v heads of qkv [W, 3 HO D] -> out [W, HO D].
template <int D, int WARPS, int NS>
__global__ void __launch_bounds__(32 * WARPS, 1)
full_kernel(const __nv_bfloat16* __restrict__ qkv, __nv_bfloat16* __restrict__ out, int W, int HO, float scale) {
    constexpr int HPC = 1, G = 1, p0 = 0, skip = 0;
    constexpr bool causal = false;
    constexpr const int* PT = nullptr;  // contiguous keys
    constexpr long long PS = 0;
    constexpr int PTOK = 1;
    constexpr bool HPAGED = false, F8 = false, HM = false;  // bf16 keys, row-major
    constexpr int CAP = 0;
    const float *ks = nullptr, *vs = nullptr;
    const int H = 3 * HO, HK = H;
    const __nv_bfloat16* __restrict__ q = qkv;
    const __nv_bfloat16* __restrict__ kc = qkv + HO * D;
    const __nv_bfloat16* __restrict__ vc = qkv + 2 * HO * D;
    auto first_row = [] { return blockIdx.x * 16 * WARPS; };
#include "prefill_attention_rows.cuh"
}

// A head-major cache (qwen35.cu's: a layer's key heads each CAP rows in a run), the W query rows at positions p0 on.
template <int D, int WARPS, int HPC, int NS>
__global__ void __launch_bounds__(32 * WARPS, 1)
hpattn_whole_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc,
              const __nv_bfloat16* __restrict__ vc, __nv_bfloat16* __restrict__ out, int p0, int W, int H, int HK,
              int G, float scale, int CAP) {
    constexpr bool HPAGED = false, causal = true, F8 = false, HM = true;
    constexpr int skip = 0;
    constexpr const int* PT = nullptr;
    constexpr long long PS = 0;
    constexpr int PTOK = 1;
    const int HO = H;
    const float *ks = nullptr, *vs = nullptr;
    auto first_row = [] { return blockIdx.x * 16 * (WARPS / HPC); };
#include "prefill_attention_rows.cuh"
}

// Whole-window e4m3 rows use an fp32 scale per head and row; staging widens the stored codes.
template <int D, int WARPS, int HPC, int NS>
__global__ void __launch_bounds__(32 * WARPS, 1)
hpattn8_whole_kernel(const __nv_bfloat16* __restrict__ q, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
               const float* __restrict__ ks, const float* __restrict__ vs, __nv_bfloat16* __restrict__ out, int p0,
               int W, int H, int HK, int G, float scale, int CAP) {
    constexpr bool HPAGED = false, causal = true, F8 = true, HM = true;
    constexpr int skip = 0;
    constexpr const int* PT = nullptr;
    constexpr long long PS = 0;
    constexpr int PTOK = 1;
    const int HO = H;
    auto first_row = [] { return blockIdx.x * 16 * (WARPS / HPC); };
#include "prefill_attention_rows.cuh"
}

// A head-major cache (qwen35.cu's: each key head's CAP rows in a run, or PT's pages of PTOK rows a head, PS apart).
template <int D, int WARPS, int HPC, int NS>
__global__ void __launch_bounds__(32 * WARPS, 1)
hpattn_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc,
              const __nv_bfloat16* __restrict__ vc, __nv_bfloat16* __restrict__ out, int p0, int W, int H, int HK,
              int G, float scale, int CAP, const int* __restrict__ PT, long long PS, int PTOK) {
    constexpr bool HPAGED = true, causal = true, F8 = false, HM = true;
    constexpr int skip = 0;
    const int HO = H;
    const float *ks = nullptr, *vs = nullptr;
    auto first_row = [] { return blockIdx.x * 16 * (WARPS / HPC); };
#include "prefill_attention_rows.cuh"
}

// hpattn_kernel over e4m3 rows with an fp32 scale a (key head, row) (qwen35_kv8.cu's layout, PS in bytes).
template <int D, int WARPS, int HPC, int NS>
__global__ void __launch_bounds__(32 * WARPS, 1)
hpattn8_kernel(const __nv_bfloat16* __restrict__ q, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
               const float* __restrict__ ks, const float* __restrict__ vs, __nv_bfloat16* __restrict__ out, int p0,
               int W, int H, int HK, int G, float scale, int CAP, const int* __restrict__ PT, long long PS, int PTOK) {
    constexpr bool HPAGED = true, causal = true, F8 = true, HM = true;
    constexpr int skip = 0;
    const int HO = H;
    auto first_row = [] { return blockIdx.x * 16 * (WARPS / HPC); };
#include "prefill_attention_rows.cuh"
}

// full_kernel's row arithmetic, one barrier a 64-key tile: two buffers of odd-padded rows, the next tile loading.
template <int D, int WARPS>
__global__ void __launch_bounds__(32 * WARPS)
vision_kernel(const __nv_bfloat16* __restrict__ qkv, __nv_bfloat16* __restrict__ out, int W, int HO, float scale) {
    static_assert(D % 16 == 0, "whole 16-dim steps");
    constexpr int C = D / 8, RS = C + 1, K16 = D / 16, NT = BN / 8, THREADS = 32 * WARPS;
    extern __shared__ uint4 smem[];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, qd = lane % 4, mt = lane / 8;
    const int head = blockIdx.y, wrow = (blockIdx.x * WARPS + warp) * 16, last = W - 1, tiles = (W + BN - 1) / BN;
    const int64_t stride = 3LL * HO * D;
    const __nv_bfloat16* __restrict__ q = qkv + head * D;
    const __nv_bfloat16* __restrict__ kc = qkv + (HO + head) * D;
    const __nv_bfloat16* __restrict__ vc = qkv + (2 * HO + head) * D;
    auto stage = [&](int t) {                            // tile t's K rows then V rows into buffer t % 2
        uint4* dst = smem + (t & 1) * 2 * BN * RS;
        for (int i = threadIdx.x; i < 2 * BN * C; i += THREADS) {
            const int kv = i / (BN * C), r = (i / C) % BN, c = i % C, key = t * BN + r;
            const bool in = key < W;
            cp16z(dst + (kv * BN + r) * RS + c, (kv ? vc : kc) + (int64_t)(in ? key : 0) * stride + c * 8, in);
        }
        commit();
    };
    uint32_t qa[K16][4];
    const int r0 = wrow + g, r1 = r0 + 8;
    {
        const uint32_t* q0 = reinterpret_cast<const uint32_t*>(q + (int64_t)min(r0, last) * stride);
        const uint32_t* q1 = reinterpret_cast<const uint32_t*>(q + (int64_t)min(r1, last) * stride);
#pragma unroll
        for (int k = 0; k < K16; ++k) {
            qa[k][0] = r0 < W ? q0[8 * k + qd] : 0u;
            qa[k][1] = r1 < W ? q1[8 * k + qd] : 0u;
            qa[k][2] = r0 < W ? q0[8 * k + qd + 4] : 0u;
            qa[k][3] = r1 < W ? q1[8 * k + qd + 4] : 0u;
        }
    }
    float o[C][4];
#pragma unroll
    for (int i = 0; i < C; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
    stage(0);
    for (int t = 0; t < tiles; ++t) {
        cp_wait<0>();
        __syncthreads();                                 // tile t landed, and every warp is done with tile t - 1
        if (t + 1 < tiles) stage(t + 1);
        const uint4* ks = smem + (t & 1) * 2 * BN * RS;
        const uint4* vs = ks + BN * RS;
        float sc[NT][4];
#pragma unroll
        for (int k = 0; k < K16; ++k) {
#pragma unroll
            for (int n = 0; n < NT; n += 2) {
                uint32_t b[4];
                ldmatrix4(b, ks + (8 * (n + mt / 2) + lane % 8) * RS + 2 * k + mt % 2);
                if (k == 0) {
                    mma0(sc[n], qa[k], b[0], b[1]);
                    mma0(sc[n + 1], qa[k], b[2], b[3]);
                } else {
                    mma(sc[n], qa[k], b[0], b[1]);
                    mma(sc[n + 1], qa[k], b[2], b[3]);
                }
            }
        }
        const int kb = t * BN + 2 * qd;
        float tm0 = -INFINITY, tm1 = -INFINITY;
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            const int c = kb + 8 * n;
            sc[n][0] = c <= last ? __fmul_rn(sc[n][0], scale) : -INFINITY;
            sc[n][1] = c + 1 <= last ? __fmul_rn(sc[n][1], scale) : -INFINITY;
            sc[n][2] = c <= last ? __fmul_rn(sc[n][2], scale) : -INFINITY;
            sc[n][3] = c + 1 <= last ? __fmul_rn(sc[n][3], scale) : -INFINITY;
            tm0 = fmaxf(tm0, fmaxf(sc[n][0], sc[n][1]));
            tm1 = fmaxf(tm1, fmaxf(sc[n][2], sc[n][3]));
        }
        for (int x = 1; x <= 2; x *= 2) {
            tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, x));
            tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, x));
        }
        const bool act0 = tm0 != -INFINITY, act1 = tm1 != -INFINITY;
        const float n0 = act0 ? fmaxf(m0, tm0) : m0, n1 = act1 ? fmaxf(m1, tm1) : m1;
        const float a0 = act0 ? (m0 == -INFINITY ? 0.0f : texp(__fsub_rn(m0, n0))) : 1.0f;
        const float a1 = act1 ? (m1 == -INFINITY ? 0.0f : texp(__fsub_rn(m1, n1))) : 1.0f;
        float c0[NT], c1[NT];
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            const int c = kb + 8 * n;
            sc[n][0] = act0 && c <= last ? texp(__fsub_rn(sc[n][0], n0)) : 0.0f;
            sc[n][1] = act0 && c + 1 <= last ? texp(__fsub_rn(sc[n][1], n0)) : 0.0f;
            sc[n][2] = act1 && c <= last ? texp(__fsub_rn(sc[n][2], n1)) : 0.0f;
            sc[n][3] = act1 && c + 1 <= last ? texp(__fsub_rn(sc[n][3], n1)) : 0.0f;
            c0[n] = __fadd_rn(sc[n][0], sc[n][1]);
            c1[n] = __fadd_rn(sc[n][2], sc[n][3]);
        }
        l0 = __fmaf_rn(l0, a0, rowsum(c0));
        l1 = __fmaf_rn(l1, a1, rowsum(c1));
        m0 = n0;
        m1 = n1;
#pragma unroll
        for (int i = 0; i < C; ++i) {
            o[i][0] = __fmul_rn(o[i][0], a0);
            o[i][1] = __fmul_rn(o[i][1], a0);
            o[i][2] = __fmul_rn(o[i][2], a1);
            o[i][3] = __fmul_rn(o[i][3], a1);
        }
#pragma unroll
        for (int j = 0; j < NT / 2; ++j) {               // 16-key chunks, in key order for every output tile
            const uint32_t pa[4] = {pack(sc[2 * j][0], sc[2 * j][1]), pack(sc[2 * j][2], sc[2 * j][3]),
                                    pack(sc[2 * j + 1][0], sc[2 * j + 1][1]), pack(sc[2 * j + 1][2], sc[2 * j + 1][3])};
#pragma unroll
            for (int i = 0; i < C; i += 2) {
                uint32_t b[4];
                ldmatrix4t(b, vs + (16 * j + 8 * (mt % 2) + lane % 8) * RS + i + mt / 2);
                mma(o[i], pa, b[0], b[1]);
                mma(o[i + 1], pa, b[2], b[3]);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < C; ++i) {
        const int d = 8 * i + 2 * qd;
        if (r0 < W)
            *reinterpret_cast<uint32_t*>(out + ((int64_t)r0 * HO + head) * D + d) =
                pack(tdiv(o[i][0], l0), tdiv(o[i][1], l0));
        if (r1 < W)
            *reinterpret_cast<uint32_t*>(out + ((int64_t)r1 * HO + head) * D + d) =
                pack(tdiv(o[i][2], l1), tdiv(o[i][3], l1));
    }
}

} // namespace tf_prefill_attention

// Head dim 128, eight warps, eight query heads a block, eight staging slots (Nemotron's 16 heads a KV head).
template __global__ void tf_prefill_attention::pattn_kernel<128, 8, 8, 8>(const __nv_bfloat16* __restrict__,
    const __nv_bfloat16* __restrict__, const __nv_bfloat16* __restrict__, __nv_bfloat16* __restrict__, int, int, int,
    int, int, float, const int* __restrict__, long long);
// Head dim 64, eight warps, four query heads a block, eight staging slots (LFM2's 4 heads a KV head, causal or trees).
template __global__ void tf_prefill_attention::tree_kernel<64, 8, 4, 8>(const __nv_bfloat16* __restrict__,
    const __nv_bfloat16* __restrict__, const __nv_bfloat16* __restrict__, __nv_bfloat16* __restrict__, const int* __restrict__, int, int, int, int, int, float);
// Head dim 72, eight warps of one head, eight staging slots (SigLIP2's tower, 72 dims padded to 80 inside).
template __global__ void tf_prefill_attention::full_kernel<72, 8, 8>(const __nv_bfloat16* __restrict__, __nv_bfloat16* __restrict__, int, int,
    float);
// Head dim 80, eight warps, two blocks an SM (RADIO's tower, 16 heads of 80): full_kernel<80>'s bits, 1.35x faster.
template __global__ void tf_prefill_attention::vision_kernel<80, 8>(const __nv_bfloat16* __restrict__, __nv_bfloat16* __restrict__, int, int,
    float);
// Head dim 256, eight warps, two query heads a block, two staging slots (Qwen3.8-27B's prompt chunks, bf16 and fp8).
template __global__ void tf_prefill_attention::hpattn_kernel<256, 8, 2, 2>(const __nv_bfloat16* __restrict__,
    const __nv_bfloat16* __restrict__, const __nv_bfloat16* __restrict__, __nv_bfloat16* __restrict__, int, int, int, int,
    int, float, int, const int* __restrict__, long long, int);
template __global__ void tf_prefill_attention::hpattn8_kernel<256, 8, 2, 2>(const __nv_bfloat16* __restrict__,
    const uint8_t* __restrict__, const uint8_t* __restrict__, const float* __restrict__, const float* __restrict__,
    __nv_bfloat16* __restrict__, int, int, int, int, int, float, int, const int* __restrict__, long long, int);

// Head dim 256, eight warps, two query heads a block, two staging slots (Qwen3.8-27B's prompt chunks, bf16 and fp8).
template __global__ void tf_prefill_attention::hpattn_whole_kernel<256, 8, 2, 2>(const __nv_bfloat16* __restrict__,
    const __nv_bfloat16* __restrict__, const __nv_bfloat16* __restrict__, __nv_bfloat16* __restrict__, int, int, int, int,
    int, float, int);
template __global__ void tf_prefill_attention::hpattn8_whole_kernel<256, 8, 2, 2>(const __nv_bfloat16* __restrict__,
    const uint8_t* __restrict__, const uint8_t* __restrict__, const float* __restrict__, const float* __restrict__,
    __nv_bfloat16* __restrict__, int, int, int, int, int, float, int);
