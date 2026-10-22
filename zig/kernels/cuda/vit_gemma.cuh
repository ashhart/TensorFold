// Gemma 4 31B's image tower glue at torch's bf16 rounding; `keep` holds the points, a control drops one.
#pragma once

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

#include "glue_math.cuh"

namespace tf_vit_g4 {

using tf_glue::bf;
using tf_glue::rbf;

// qkv_norm_rope: a head's normed row; its rope products. gelu_mul: the gelu. pool_std: the block mean.
enum : int { NORM = 1, ROPE = 2 };
constexpr int QKV_ALL = 3;
enum : int { GELU = 1 };
constexpr int GELU_ALL = 1;
enum : int { MEAN = 1 };
constexpr int POOL_ALL = 1;

__device__ __forceinline__ float pick(int keep, int point, float v) { return (keep & point) ? rbf(v) : v; }

// A warp a (row, head, part) of qkv [P, 3 H D] in place: q, k RMSNorm then 2D RoPE; v weightless RMSNorm. D <= 96.
__device__ __forceinline__ void qkv_norm_rope(__nv_bfloat16* qkv, const __nv_bfloat16* qw, const __nv_bfloat16* kw,
                                              const __nv_bfloat16* cs, const __nv_bfloat16* sn, int P, int H, int D,
                                              float eps, int keep) {
    const int g = blockIdx.x, lane = threadIdx.x;
    const int part = g % 3, head = (g / 3) % H, row = g / (3 * H);
    if (row >= P) return;
    __nv_bfloat16* x = qkv + (static_cast<int64_t>(row) * 3 + part) * H * D + head * D;
    const __nv_bfloat16* w = part == 0 ? qw : kw;
    float v[3];
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const int d = lane + 32 * i;
        v[i] = d < D ? bf(x[d]) : 0.0f;
        s = __fadd_rn(s, __fmul_rn(v[i], v[i]));
    }
    s = tf_glue::warp_sum(s);
    const float r = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(s, static_cast<float>(D)), eps)));
    __shared__ float nrm[96];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const int d = lane + 32 * i;
        if (d < D) nrm[d] = pick(keep, NORM, part == 2 ? __fmul_rn(v[i], r) : __fmul_rn(__fmul_rn(v[i], r), bf(w[d])));
    }
    __syncwarp();
    const int half = D / 2, quarter = half / 2;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const int d = lane + 32 * i;
        if (d >= D) continue;
        float out = nrm[d];
        if (part != 2) {
            const int base = d / half * half, j = d % half;
            const float other = j < quarter ? -nrm[base + j + quarter] : nrm[base + j - quarter];
            const float c = bf(cs[static_cast<int64_t>(row) * D + d]), sv = bf(sn[static_cast<int64_t>(row) * D + d]);
            out = __fadd_rn(pick(keep, ROPE, __fmul_rn(nrm[d], c)), pick(keep, ROPE, __fmul_rn(other, sv)));
        }
        x[d] = __float2bfloat16_rn(out);
    }
}

// torch's gelu (approximate "tanh") in fp32.
__device__ __forceinline__ float gelu_tanh(float x) {
    constexpr float beta = M_SQRT2 * M_2_SQRTPI * 0.5;
    constexpr float kappa = 0.044715f;
    const float cube = __fmul_rn(__fmul_rn(x, x), x);
    const float inner = __fmul_rn(beta, __fadd_rn(x, __fmul_rn(kappa, cube)));
    return __fmul_rn(__fmul_rn(0.5f, x), __fadd_rn(1.0f, tanhf(inner)));
}

// out [rows, n] = bf16(bf16(gelu(gate)) up): torch's F.gelu on a bf16 gate, then the bf16 product.
__device__ __forceinline__ void gelu_mul(const __nv_bfloat16* gate, const __nv_bfloat16* up, __nv_bfloat16* out,
                                         int64_t count, int keep) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    out[i] = __float2bfloat16_rn(__fmul_rn(pick(keep, GELU, gelu_tanh(bf(gate[i]))), bf(up[i])));
}

// out [T, C]: 3 x 3 blocks of x [gh, gw, C], bf16(sum(x / 9)) root, then bf16((h - bias) scale): pool, standardize.
__device__ __forceinline__ void pool_std(const __nv_bfloat16* x, const __nv_bfloat16* bias, const __nv_bfloat16* scale,
                                         __nv_bfloat16* out, int gw, int C, float root, int keep) {
    const int t = blockIdx.x, bw = gw / 3, ty = t / bw, tx = t % bw;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float s = 0.0f;
        for (int dy = 0; dy < 3; ++dy)
            for (int dx = 0; dx < 3; ++dx)
                s = __fadd_rn(s, __fmul_rn(bf(x[(static_cast<int64_t>(3 * ty + dy) * gw + 3 * tx + dx) * C + c]), 1.0f / 9.0f));
        const float h = __fmul_rn(pick(keep, MEAN, s), root);
        out[static_cast<int64_t>(t) * C + c] = __float2bfloat16_rn(__fmul_rn(__fsub_rn(h, bf(bias[c])), bf(scale[c])));
    }
}

}  // namespace tf_vit_g4
