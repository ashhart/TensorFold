// bf16 residual tails (norms.cu's tf_add_rms_rows_bf16h_tail*): branch norm, add, bf16 scalar, output norm and sums.

#pragma once

#include <cuda_bf16.h>
#include <stdint.h>
#include "common/reduce.cuh"

namespace tf_rms_tail {

constexpr int NT = 256;
constexpr int MOST = 4; // 16-byte loads a thread: rows of width % 64, at most 8,192, 16 B aligned

// The rounding points: the branch's normalized row, the branch's output, the sum before the scalar, the output's row.
enum : int { BRANCH_ROW = 1, BRANCH_OUT = 2, SUM = 4, ROW = 8 };
// One rounding a norm, bf16 rows between the steps.
constexpr int ONCE = BRANCH_OUT | SUM;
// MLX's bf16 op by op (Metal gemma4/glue.metal's tails): bf16(w * bf16(x * inv)) for each norm.
constexpr int MLX = BRANCH_ROW | BRANCH_OUT | SUM | ROW;

__device__ __forceinline__ float bf(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

// v rounded to bf16 where R holds the point P.
template <int R, int P>
__device__ __forceinline__ float point(float v) {
    if constexpr ((R & P) != 0) return bf(v);
    else return v;
}

// A normalized value times its weight: w * bf16(v * inv) where R holds the row's point P, else (v * inv) * w.
template <int R, int P>
__device__ __forceinline__ float normed(float v, float inv, float w) {
    if constexpr ((R & P) != 0) return __fmul_rn(w, bf(__fmul_rn(v, inv)));
    else return __fmul_rn(__fmul_rn(v, inv), w);
}

using tf_common::unpack8;
using tf_common::pack8;
using tf_common::group_sum8;

// 1 / sqrt(sum / width + eps), IEEE division and square root: norms.cu's inv_rms_v tree over the 8 warps.
template <bool ONCE_ONLY>
__device__ __forceinline__ float inv_rms(float part, int width, float eps) {
    __shared__ float warps[NT / 32];
    for (int m = 16; m > 0; m >>= 1) part = __fadd_rn(part, __shfl_xor_sync(0xFFFFFFFFu, part, m));
    if ((threadIdx.x & 31) == 0) warps[threadIdx.x >> 5] = part;
    __syncthreads();
    float s = (threadIdx.x & 31) < NT / 32 ? warps[threadIdx.x & 31] : 0.f;
    for (int m = 4; m > 0; m >>= 1) s = __fadd_rn(s, __shfl_xor_sync(0xFFFFFFFFu, s, m));
    s = __shfl_sync(0xFFFFFFFFu, s, 0);
    const float r = __fdiv_rn(1.f, __fsqrt_rn(__fadd_rn(__fdiv_rn(s, static_cast<float>(width)), eps)));
    if constexpr (!ONCE_ONLY) __syncthreads(); // `warps` is read again by the output's norm
    return r;
}

// b = wb ? bf16(rms(x) wb) : x, h = bf16(sc ? (r + b) sc : r + b) (if r, wb or sc), y = bf16(rms(h) w), xs its sums.
template <int R>
__device__ __forceinline__ void tail(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ r,
                                     const __nv_bfloat16* __restrict__ wb, const __nv_bfloat16* __restrict__ sc,
                                     const __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ h,
                                     __nv_bfloat16* __restrict__ y, float* __restrict__ xs, int width, float eps) {
    const int64_t at = static_cast<int64_t>(blockIdx.x) * width;
    const int chunks = width / 8;
    float v[MOST][8];
    uint4 xp[MOST], rp[MOST], bp[MOST], wp[MOST];
#pragma unroll
    for (int j = 0; j < MOST; ++j) { // every load issued before any is used
        const int c = threadIdx.x + j * NT;
        if (c < chunks) {
            xp[j] = reinterpret_cast<const uint4*>(x + at)[c];
            if (r != nullptr) rp[j] = reinterpret_cast<const uint4*>(r + at)[c];
            if (wb != nullptr) bp[j] = reinterpret_cast<const uint4*>(wb)[c];
            wp[j] = reinterpret_cast<const uint4*>(w)[c];
        } else xp[j] = rp[j] = bp[j] = wp[j] = make_uint4(0, 0, 0, 0); // past the row: zeros that add nothing
    }
#pragma unroll
    for (int j = 0; j < MOST; ++j) unpack8(xp[j], v[j]);
    if (wb != nullptr) {
        float part = 0.f;
#pragma unroll
        for (int j = 0; j < MOST; ++j)
#pragma unroll
            for (int i = 0; i < 8; ++i) part = __fmaf_rn(v[j][i], v[j][i], part); // zeros past the row add nothing
        const float inv = inv_rms<false>(part, width, eps);
#pragma unroll
        for (int j = 0; j < MOST; ++j) {
            if (threadIdx.x + j * NT >= chunks) continue; // past the row v stays 0
            float bw[8];
            unpack8(bp[j], bw);
#pragma unroll
            for (int i = 0; i < 8; ++i) v[j][i] = point<R, BRANCH_OUT>(normed<R, BRANCH_ROW>(v[j][i], inv, bw[i]));
        }
    }
    const float scale = sc != nullptr ? __bfloat162float(*sc) : 1.f;
    if (r != nullptr || wb != nullptr || sc != nullptr) {
#pragma unroll
        for (int j = 0; j < MOST; ++j) {
            const int c = threadIdx.x + j * NT;
            if (c < chunks) {
                if (r != nullptr) {
                    float rv[8];
                    unpack8(rp[j], rv);
#pragma unroll
                    for (int i = 0; i < 8; ++i) v[j][i] = __fadd_rn(rv[i], v[j][i]);
                }
                if (sc != nullptr)
#pragma unroll
                    for (int i = 0; i < 8; ++i) v[j][i] = __fmul_rn(point<R, SUM>(v[j][i]), scale);
                const uint4 hp = pack8(v[j]);
                reinterpret_cast<uint4*>(h + at)[c] = hp;
                unpack8(hp, v[j]); // the output's norm reads h as stored
            }
        }
    }
    float part = 0.f;
#pragma unroll
    for (int j = 0; j < MOST; ++j)
#pragma unroll
        for (int i = 0; i < 8; ++i) part = __fmaf_rn(v[j][i], v[j][i], part);
    const float inv = inv_rms<true>(part, width, eps);
#pragma unroll
    for (int j = 0; j < MOST; ++j) {
        const int c = threadIdx.x + j * NT;
        const bool live = c < chunks;
        float wv[8], o[8];
        unpack8(wp[j], wv);
#pragma unroll
        for (int i = 0; i < 8; ++i) o[i] = normed<R, ROW>(v[j][i], inv, wv[i]);
        const uint4 yp = pack8(o);
        if (live) reinterpret_cast<uint4*>(y + at)[c] = yp;
        if (xs != nullptr && j * NT < chunks) { // whole warps of whole groups: width % 64 == 0
            const float s = group_sum8(yp, live);
            if (live && (c & 7) == 0) xs[static_cast<int64_t>(blockIdx.x) * (width / 64) + c / 8] = s;
        }
    }
}

}  // namespace tf_rms_tail
