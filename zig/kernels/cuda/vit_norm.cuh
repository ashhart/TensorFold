// vit.cu's LayerNorm rows: bf16((x - mean) rstd w + b), two passes, E values a thread; controls drop one of `keep`.
#pragma once

#include <cuda_bf16.h>
#include <stdint.h>

#include "common/glue_math.cuh"

namespace tf_vit_norm {

constexpr int NT = 256;
enum : int { TWO_PASS = 1, ONE_ROUND = 2 };  // the variance about the mean; one rounding, after the bias
constexpr int ALL = 3;

// A block's values summed: warp butterflies, then the warps in order; every thread gets the sum.
__device__ __forceinline__ float block_sum(float v, float* red) {
    v = tf_glue::warp_sum(v);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float s = red[0];
    for (int w = 1; w < NT / 32; ++w) s = __fadd_rn(s, red[w]);
    __syncthreads();
    return s;
}

template <int E>
__device__ __forceinline__ void layer_norm(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ W,
                                           const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ Y, int D,
                                           float eps, int keep) {
    using tf_glue::bf;
    __shared__ float red[NT / 32];
    const int64_t at = static_cast<int64_t>(blockIdx.x) * D;
    float x[E];
    float acc = 0.0f;
    float sq = 0.0f;
#pragma unroll
    for (int k = 0; k < E; ++k) {
        const int i = threadIdx.x + k * NT;
        if (i >= D) continue;
        x[k] = bf(X[at + i]);
        acc = __fadd_rn(acc, x[k]);
        if (!(keep & TWO_PASS)) sq = __fadd_rn(sq, __fmul_rn(x[k], x[k]));
    }
    const float mean = __fdiv_rn(block_sum(acc, red), static_cast<float>(D));
    float var;
    if (keep & TWO_PASS) {
        acc = 0.0f;
#pragma unroll
        for (int k = 0; k < E; ++k) {
            if (threadIdx.x + k * NT >= D) continue;
            x[k] = __fsub_rn(x[k], mean);
            acc = __fadd_rn(acc, __fmul_rn(x[k], x[k]));
        }
        var = __fdiv_rn(block_sum(acc, red), static_cast<float>(D));
    } else {
        var = __fsub_rn(__fdiv_rn(block_sum(sq, red), static_cast<float>(D)), __fmul_rn(mean, mean));
#pragma unroll
        for (int k = 0; k < E; ++k) x[k] = __fsub_rn(x[k], mean);
    }
    const float rstd = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(var, eps)));
#pragma unroll
    for (int k = 0; k < E; ++k) {
        const int i = threadIdx.x + k * NT;
        if (i >= D) continue;
        float v = __fmul_rn(__fmul_rn(x[k], rstd), bf(W[i]));
        if (!(keep & ONE_ROUND)) v = tf_glue::rbf(v);
        Y[at + i] = __float2bfloat16_rn(__fadd_rn(v, bf(B[i])));
    }
}

}  // namespace tf_vit_norm
