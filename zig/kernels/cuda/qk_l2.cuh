// qk_rows.cu's tf_qk_rows_l2 (Dory): RoPE first, then each head's L2 norm (eps on the sum), q times a scale vector.

#pragma once

#include "qk_rows.cuh"

namespace tf_qk_l2 {

using namespace tf_qk;

constexpr int WIDE = 512; // head_dim at most

// The rounding points: the rotated head, and q's normalized head before its scale (k's is its output).
enum : int { ROPE = 1, NORM = 2 };

__device__ __forceinline__ float bf(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

template <int R, int P>
__device__ __forceinline__ float point(float v) {
    if constexpr ((R & P) != 0) return bf(v);
    else return v;
}

// 1 / sqrt(sum + eps) of a head's squares: each warp's xor tree, then the warps in order (tf_qk_rows' order).
__device__ __forceinline__ float inv_l2(float y, int dim, float eps) {
    __shared__ float warps[WIDE / 32];
    float part = __fmul_rn(y, y);
    for (int m = 16; m > 0; m >>= 1) part = __fadd_rn(part, __shfl_xor_sync(0xFFFFFFFFu, part, m));
    if ((threadIdx.x & 31) == 0) warps[threadIdx.x >> 5] = part;
    __syncthreads();
    float s = 0.f;
    for (int w = 0; w < dim / 32; ++w) s = __fadd_rn(s, warps[w]);
    return __fdiv_rn(1.f, __fsqrt_rn(__fadd_rn(s, eps)));
}

// A block a (row, head), a thread an element: q = bf16(qs * point<NORM>(l2(point<ROPE>(rope(x))))), k without qs.
template <int R>
__device__ __forceinline__ void rows(const QkSrc& in, const float* qs, const float* cosv, const float* sinv,
                                     const float* inv, int rot, const int* pos, const int* slot, const QkDst& dst,
                                     __nv_bfloat16* kc, __nv_bfloat16* vc, long long len, int h, int hk, float eps) {
    const int d = threadIdx.x;
    const int dim = blockDim.x;
    const int64_t r = blockIdx.x;
    const int hd = blockIdx.y;
    const __nv_bfloat16* src = head_src(in, r, hd, h, hk);
    const int p = pos[r];
    const int64_t pl = p < len ? p : p % len;
    const int64_t at = slot != nullptr ? static_cast<int64_t>(slot[r]) * len + pl : pl;
    if (hd >= h + hk) { // v: as is
        const int k = hd - h - hk;
        if (dst.v != nullptr) dst.v[r * dst.v_ld + k * dst.v_hs + d] = src[d];
        if (vc != nullptr) vc[(at * hk + k) * dim + d] = src[d];
        return;
    }
    float y = __bfloat162float(src[d]);
    const int half = rot / 2;
    if ((cosv != nullptr || inv != nullptr) && d < rot) {
        const float yp = __bfloat162float(src[d < half ? d + half : d - half]);
        float c, sn;
        rope_at(cosv, sinv, inv, p, rot, d, c, sn);
        y = __fadd_rn(__fmul_rn(y, c), __fmul_rn(d < half ? -yp : yp, sn));
    }
    y = point<R, ROPE>(y);
    const float n = __fmul_rn(y, inv_l2(y, dim, eps));
    if (hd < h) {
        const float out = qs != nullptr ? __fmul_rn(qs[hd * dim + d], point<R, NORM>(n)) : n;
        dst.q[r * dst.q_ld + hd * dst.q_hs + d] = __float2bfloat16_rn(out);
    } else {
        const int k = hd - h;
        const __nv_bfloat16 out = __float2bfloat16_rn(n);
        dst.k[r * dst.k_ld + k * dst.k_hs + d] = out;
        if (kc != nullptr) kc[(at * hk + k) * dim + d] = out;
    }
}

}  // namespace tf_qk_l2
