// qk_rows.cu's tf_qk_rows_once and tf_qk_rows_round: head dims to 512, a unit v norm and MLX's rounding points.

#pragma once

#include "qk_rows.cuh"

namespace tf_qk_round {

using namespace tf_qk;

constexpr int WIDE = 512; // head_dim at most

// The options: the normalized row rounded before its weight, the weighted row before RoPE, v by a unit RMS norm.
enum : int { ROW = 1, OUT = 2, VNORM = 4 };
// MLX's bf16 op by op (Metal gemma4/glue.metal's gemma_norm_rope): bf16(w * bf16(x * inv)), then RoPE.
constexpr int MLX = ROW | OUT;

__device__ __forceinline__ float bf(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

// 1 / sqrt(sum / dim + eps) of a head's squares: each warp's xor tree, then the warps in order (tf_qk_rows' order).
__device__ __forceinline__ float inv_rms(float x, int dim, float eps) {
    __shared__ float warps[WIDE / 32];
    float part = __fmul_rn(x, x);
    for (int m = 16; m > 0; m >>= 1) part = __fadd_rn(part, __shfl_xor_sync(0xFFFFFFFFu, part, m));
    if ((threadIdx.x & 31) == 0) warps[threadIdx.x >> 5] = part;
    __syncthreads();
    float s = 0.f;
    for (int w = 0; w < dim / 32; ++w) s = __fadd_rn(s, warps[w]);
    return __fdiv_rn(1.f, __fsqrt_rn(__fadd_rn(__fdiv_rn(s, static_cast<float>(dim)), eps)));
}

// A normed value: w * bf16(x * inv) with ROW, else (x * inv) * w; rounded to bf16 with OUT.
__device__ __forceinline__ float normed(float x, float inv, float w, int pts) {
    const float y = (pts & ROW) ? __fmul_rn(w, bf(__fmul_rn(x, inv))) : __fmul_rn(__fmul_rn(x, inv), w);
    return (pts & OUT) ? bf(y) : y;
}

// tf_qk_rows with head dims to 512 and the options in `pts`: a block a (row, head), a thread an element.
__device__ __forceinline__ void rows(const QkSrc& in, const QkW& nw, const float* cosv, const float* sinv, const float* inv,
                                     int rot, const int* pos, const int* slot, const QkDst& dst, __nv_bfloat16* kc,
                                     __nv_bfloat16* vc, long long len, int h, int hk, float eps, int pts) {
    const int d = threadIdx.x;
    const int dim = blockDim.x;
    const int64_t r = blockIdx.x;
    const int hd = blockIdx.y;
    const __nv_bfloat16* src = head_src(in, r, hd, h, hk);
    const float x = __bfloat162float(src[d]);
    const int p = pos[r];
    const int64_t pl = p < len ? p : p % len;
    const int64_t at = slot != nullptr ? static_cast<int64_t>(slot[r]) * len + pl : pl;
    if (hd >= h + hk) { // v: as is, or by a unit RMS norm
        const int k = hd - h - hk;
        const __nv_bfloat16 out = (pts & VNORM) ? __float2bfloat16_rn(__fmul_rn(x, inv_rms(x, dim, eps))) : src[d];
        if (dst.v != nullptr) dst.v[r * dst.v_ld + k * dst.v_hs + d] = out;
        if (vc != nullptr) vc[(at * hk + k) * dim + d] = out;
        return;
    }
    float y = x;
    float yp = 0.f;
    const int half = rot / 2;
    const int partner = d < half ? d + half : d - half;
    if (nw.q != nullptr) {
        const float rstd = inv_rms(x, dim, eps);
        y = normed(x, rstd, wv(nw, hd >= h, d), pts);
        if (d < rot) yp = normed(__bfloat162float(src[partner]), rstd, wv(nw, hd >= h, partner), pts);
    } else if (d < rot) {
        yp = __bfloat162float(src[partner]);
    }
    if ((cosv != nullptr || inv != nullptr) && d < rot) {
        float c, sn;
        rope_at(cosv, sinv, inv, p, rot, d, c, sn);
        y = __fadd_rn(__fmul_rn(y, c), __fmul_rn(d < half ? -yp : yp, sn));
    }
    const __nv_bfloat16 out = __float2bfloat16_rn(y);
    if (hd < h) {
        dst.q[r * dst.q_ld + hd * dst.q_hs + d] = out;
    } else {
        const int k = hd - h;
        dst.k[r * dst.k_ld + k * dst.k_hs + d] = out;
        if (kc != nullptr) kc[(at * hk + k) * dim + d] = out;
    }
}

}  // namespace tf_qk_round
