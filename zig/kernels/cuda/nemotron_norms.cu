// Nemotron's token rows and RMSNorms: one block a row, fixed-order fp32 sums, so a row's bits ignore its window.

#include <cuda_bf16.h>
#include <stdint.h>

#include "glue_math.cuh"

namespace tf_glue {

constexpr int NT = 256;   // threads a row block
constexpr int MAXE = 16;  // elements a thread holds: rows up to NT * MAXE = 4096 wide
constexpr int MAXS = 8;   // MoE slots a row holds: its top-k and the shared halves

// Sum of x^2 over a row: each thread's fma chain in index order, warp butterflies, then warps in order.
__device__ __forceinline__ float row_ss(const float (&x)[MAXE], int D, float* red) {
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < MAXE; ++k)
        if (threadIdx.x + k * NT < D) acc = __fmaf_rn(x[k], x[k], acc);
    acc = warp_sum(acc);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
    __syncthreads();
    float ss = red[0];
    for (int w = 1; w < NT / 32; ++w) ss = __fadd_rn(ss, red[w]);
    return ss;
}

// 1 / sqrt(ss / n + eps) with IEEE division and square root.
__device__ __forceinline__ float inv_rms(float ss, int n, float eps) {
    return __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(ss, static_cast<float>(n)), eps)));
}

// y = bf16((x * inv) * w) into OUT, and each 64-group's sum of the stored values, in order, into XS.
__device__ __forceinline__ void scale_store(const float (&x)[MAXE], int D, float inv, const __nv_bfloat16* __restrict__ W,
                                            __nv_bfloat16* __restrict__ OUT, float* __restrict__ XS, float* ybuf) {
#pragma unroll
    for (int k = 0; k < MAXE; ++k) {
        const int i = threadIdx.x + k * NT;
        if (i >= D) continue;
        const __nv_bfloat16 y = __float2bfloat16_rn(__fmul_rn(__fmul_rn(x[k], inv), bf(W[i])));
        OUT[i] = y;
        ybuf[i] = bf(y);
    }
    __syncthreads();
    for (int g = threadIdx.x; g < D / 64; g += NT) {
        float s = 0.0f;
        for (int j = 0; j < 64; ++j) s = __fadd_rn(s, ybuf[g * 64 + j]);
        XS[g] = s;
    }
}

}  // namespace tf_glue

using namespace tf_glue;

// A bf16 token table's rows (ModelOpt checkpoints keep the embedding unquantized): out = W[ids].
extern "C" __global__ void __launch_bounds__(NT) tf_nemo_embed16(const int* __restrict__ ids, const __nv_bfloat16* __restrict__ W,
                                                                   __nv_bfloat16* __restrict__ out, int D) {
    const int64_t tok = ids[blockIdx.x];
    for (int i = threadIdx.x; i < D; i += NT) out[static_cast<int64_t>(blockIdx.x) * D + i] = W[tok * D + i];
}

// MT threads load and add a row (every load in flight); the first NT sum its squares in row_ss's order.
constexpr int MT = 1024, MTE = (NT * MAXE) / MT;  // elements a load thread holds

// delta = bf16(sum_k<NR fma(wt_k, y_k) + sum_k>=NR y_k) in slot order, h = bf16(x + delta), then add_rmsnorm's tail.
extern "C" __global__ void __launch_bounds__(MT) tf_nemo_add_moe_norm(const __nv_bfloat16* __restrict__ X, const void* __restrict__ Yk,
                                                                        int y_f32, const float* __restrict__ WT, const __nv_bfloat16* __restrict__ W,
                                                                        __nv_bfloat16* __restrict__ HN, __nv_bfloat16* __restrict__ Y,
                                                                        float* __restrict__ XS, float eps, int D, int NR, int NS) {
    __shared__ float red[NT / 32];
    __shared__ float xb[NT * MAXE];  // the row's h, then its stored outputs
    const int64_t row = blockIdx.x;
    float wt[MAXS];
#pragma unroll
    for (int s = 0; s < MAXS; ++s) wt[s] = s < NR ? WT[row * NS + s] : 0.0f;
    float y[MTE][MAXS], xv[MTE];
#pragma unroll
    for (int k = 0; k < MTE; ++k) {
        const int i = threadIdx.x + k * MT;
        xv[k] = i < D ? bf(X[row * D + i]) : 0.0f;
#pragma unroll
        for (int s = 0; s < MAXS; ++s) {
            const int64_t at = (row * NS + s) * D + i;
            y[k][s] = i >= D || s >= NS ? 0.0f : y_f32 ? static_cast<const float*>(Yk)[at] : bf(static_cast<const __nv_bfloat16*>(Yk)[at]);
        }
    }
#pragma unroll
    for (int k = 0; k < MTE; ++k) {
        const int i = threadIdx.x + k * MT;
        if (i >= D) continue;
        float routed = 0.0f, shared = 0.0f;
#pragma unroll
        for (int s = 0; s < MAXS; ++s) {
            if (s >= NS) break;
            if (s < NR) routed = __fmaf_rn(wt[s], y[k][s], routed);
            else shared = __fadd_rn(shared, y[k][s]);
        }
        const float delta = rbf(__fadd_rn(routed, shared));
        const __nv_bfloat16 h = __float2bfloat16_rn(__fadd_rn(xv[k], delta));
        HN[row * D + i] = h;
        xb[i] = bf(h);
    }
    __syncthreads();
    if (threadIdx.x < NT) {  // row_ss's chains, butterflies and warp order
        float acc = 0.0f;
#pragma unroll
        for (int k = 0; k < MAXE; ++k)
            if (threadIdx.x + k * NT < D) acc = __fmaf_rn(xb[threadIdx.x + k * NT], xb[threadIdx.x + k * NT], acc);
        acc = warp_sum(acc);
        if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
    }
    __syncthreads();
    float ss = red[0];
    for (int w = 1; w < NT / 32; ++w) ss = __fadd_rn(ss, red[w]);
    const float inv = inv_rms(ss, D, eps);
#pragma unroll
    for (int k = 0; k < MTE; ++k) {  // each element reads its h and leaves its stored output in place
        const int i = threadIdx.x + k * MT;
        if (i >= D) continue;
        const __nv_bfloat16 o = __float2bfloat16_rn(__fmul_rn(__fmul_rn(xb[i], inv), bf(W[i])));
        Y[row * D + i] = o;
        xb[i] = bf(o);
    }
    __syncthreads();
    for (int g = threadIdx.x; g < D / 64; g += MT) {
        float sum = 0.0f;
        for (int j = 0; j < 64; ++j) sum = __fadd_rn(sum, xb[g * 64 + j]);
        XS[row * (D / 64) + g] = sum;
    }
}

// MTP (row, part) blocks normalize embedding and hidden halves side by side, with group sums.
extern "C" __global__ void __launch_bounds__(NT) tf_nemo_concat_norms(const __nv_bfloat16* __restrict__ E, const __nv_bfloat16* __restrict__ Hd,
                                                                        const __nv_bfloat16* __restrict__ WE, const __nv_bfloat16* __restrict__ WH,
                                                                        __nv_bfloat16* __restrict__ OUT, float* __restrict__ XS, float eps, int D) {
    __shared__ float red[NT / 32];
    __shared__ float ybuf[NT * MAXE];
    const int64_t row = blockIdx.x;
    const int part = blockIdx.y;
    const __nv_bfloat16* src = (part == 0 ? E : Hd) + row * D;
    float x[MAXE];
#pragma unroll
    for (int k = 0; k < MAXE; ++k)
        if (threadIdx.x + k * NT < D) x[k] = bf(src[threadIdx.x + k * NT]);
    const float inv = inv_rms(row_ss(x, D, red), D, eps);
    scale_store(x, D, inv, part == 0 ? WE : WH, OUT + row * 2 * D + part * D, XS + row * (2 * D / 64) + part * (D / 64), ybuf);
}

// Block (row, group) of GS = XD / groups values: n = bf16(v * inv), out = bf16(w * n), and the 64-group sums.
extern "C" __global__ void __launch_bounds__(128) tf_nemo_group_rmsnorm(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ W,
                                                                          __nv_bfloat16* __restrict__ OUT, float* __restrict__ XS, float eps, int XD,
                                                                          int GS) {
    constexpr int T = 128, M = 8;  // groups up to T * M = 1024 wide
    __shared__ float red[T / 32];
    __shared__ float ybuf[T * M];
    const int64_t at = static_cast<int64_t>(blockIdx.x) * XD + blockIdx.y * GS;
    float v[M];
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < M; ++k) {
        const int i = threadIdx.x + k * T;
        if (i >= GS) continue;
        v[k] = bf(X[at + i]);
        acc = __fmaf_rn(v[k], v[k], acc);
    }
    acc = warp_sum(acc);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
    __syncthreads();
    float ss = red[0];
    for (int w = 1; w < T / 32; ++w) ss = __fadd_rn(ss, red[w]);
    const float inv = inv_rms(ss, GS, eps);
#pragma unroll
    for (int k = 0; k < M; ++k) {
        const int i = threadIdx.x + k * T;
        if (i >= GS) continue;
        const float n = rbf(__fmul_rn(v[k], inv));
        const __nv_bfloat16 o = __float2bfloat16_rn(__fmul_rn(bf(W[blockIdx.y * GS + i]), n));
        OUT[at + i] = o;
        ybuf[i] = bf(o);
    }
    __syncthreads();
    for (int g = threadIdx.x; g < GS / 64; g += T) {
        float s = 0.0f;
        for (int j = 0; j < 64; ++j) s = __fadd_rn(s, ybuf[g * 64 + j]);
        XS[at / 64 + g] = s;
    }
}
