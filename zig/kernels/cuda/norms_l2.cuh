// nGPT rows (norms.cu's tf_l2_rows, tf_slerp_rows): L2 norms with eps on the sum, and the normalized lerp residual.

#pragma once

#include <cuda_bf16.h>
#include <stdint.h>

namespace tf_l2 {

constexpr int NT = 256;
constexpr int MOST = 4; // 16-byte loads a thread: rows of width % 64, at most 8,192, 16 B aligned

// The rounding points: the sum x + r before its norm; the slerp's normalized residual a and branch b.
enum : int { SUM = 1, A = 2, B = 4 };

__device__ __forceinline__ float bf(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

// v rounded to bf16 where R holds the point P.
template <int R, int P>
__device__ __forceinline__ float point(float v) {
    if constexpr ((R & P) != 0) return bf(v);
    else return v;
}

__device__ __forceinline__ void unpack8(const uint4& u, float* f) {
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 t = __bfloat1622float2(h[i]);
        f[2 * i] = t.x;
        f[2 * i + 1] = t.y;
    }
}

__device__ __forceinline__ uint4 pack8(const float* f) {
    uint4 u;
    __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i) h[i] = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
    return u;
}

// 1 / sqrt(sum + eps) of the row's squares: a thread's fma chain, the warp's xor tree, the 8 warps' tree (IEEE).
__device__ __forceinline__ float inv_l2(const float (&v)[MOST][8], float eps) {
    __shared__ float warps[NT / 32];
    float part = 0.f;
#pragma unroll
    for (int j = 0; j < MOST; ++j)
#pragma unroll
        for (int i = 0; i < 8; ++i) part = __fmaf_rn(v[j][i], v[j][i], part);
    for (int m = 16; m > 0; m >>= 1) part = __fadd_rn(part, __shfl_xor_sync(0xFFFFFFFFu, part, m));
    if ((threadIdx.x & 31) == 0) warps[threadIdx.x >> 5] = part;
    __syncthreads();
    float s = (threadIdx.x & 31) < NT / 32 ? warps[threadIdx.x & 31] : 0.f;
    for (int m = 4; m > 0; m >>= 1) s = __fadd_rn(s, __shfl_xor_sync(0xFFFFFFFFu, s, m));
    s = __shfl_sync(0xFFFFFFFFu, s, 0);
    __syncthreads(); // `warps` is written again by the row's next norm
    return __fdiv_rn(1.f, __fsqrt_rn(__fadd_rn(s, eps)));
}

// A row's chunks of 8 into v (zeros past the row); with r, v = point<SUM>(x + r).
template <int R>
__device__ __forceinline__ void load(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ r,
                                     int64_t at, int chunks, float (&v)[MOST][8]) {
    uint4 xp[MOST], rp[MOST];
#pragma unroll
    for (int j = 0; j < MOST; ++j) { // every load issued before any is used
        const int c = threadIdx.x + j * NT;
        const bool live = c < chunks;
        xp[j] = live ? reinterpret_cast<const uint4*>(x + at)[c] : make_uint4(0, 0, 0, 0);
        rp[j] = live && r != nullptr ? reinterpret_cast<const uint4*>(r + at)[c] : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int j = 0; j < MOST; ++j) {
        unpack8(xp[j], v[j]);
        if (r != nullptr) {
            float rv[8];
            unpack8(rp[j], rv);
#pragma unroll
            for (int i = 0; i < 8; ++i) v[j][i] = point<R, SUM>(__fadd_rn(v[j][i], rv[i]));
        }
    }
}

__device__ __forceinline__ void store(__nv_bfloat16* __restrict__ y, int64_t at, int chunks, const float (&v)[MOST][8]) {
#pragma unroll
    for (int j = 0; j < MOST; ++j) {
        const int c = threadIdx.x + j * NT;
        if (c < chunks) reinterpret_cast<uint4*>(y + at)[c] = pack8(v[j]);
    }
}

// y = bf16(s * inv_l2(s)), s = x (r null) or point<SUM>(x + r): a block a row of `width`.
template <int R>
__device__ __forceinline__ void l2_rows(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ r,
                                        __nv_bfloat16* __restrict__ y, int width, float eps) {
    const int64_t at = static_cast<int64_t>(blockIdx.x) * width;
    const int chunks = width / 8;
    float v[MOST][8];
    load<R>(x, r, at, chunks, v);
    const float inv = inv_l2(v, eps);
#pragma unroll
    for (int j = 0; j < MOST; ++j)
#pragma unroll
        for (int i = 0; i < 8; ++i) v[j][i] = __fmul_rn(v[j][i], inv);
    store(y, at, chunks, v);
}

// h = bf16(l2(a + lr (b - a))), a = point<A>(l2(res)), b = point<B>(l2(out)), lr = |alpha scale| (fp32 steps).
template <int R>
__device__ __forceinline__ void slerp_rows(const __nv_bfloat16* __restrict__ res, const __nv_bfloat16* __restrict__ out,
                                           const __nv_bfloat16* __restrict__ alpha, float scale,
                                           __nv_bfloat16* __restrict__ h, int width, float eps) {
    const int64_t at = static_cast<int64_t>(blockIdx.x) * width;
    const int chunks = width / 8;
    float a[MOST][8], b[MOST][8], lr[MOST][8];
    load<0>(res, nullptr, at, chunks, a);
    load<0>(out, nullptr, at, chunks, b);
    load<0>(alpha, nullptr, 0, chunks, lr);
    const float ia = inv_l2(a, eps);
    const float ib = inv_l2(b, eps);
#pragma unroll
    for (int j = 0; j < MOST; ++j)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float av = point<R, A>(__fmul_rn(a[j][i], ia));
            const float bv = point<R, B>(__fmul_rn(b[j][i], ib));
            const float l = fabsf(__fmul_rn(lr[j][i], scale));
            a[j][i] = __fadd_rn(av, __fmul_rn(l, __fsub_rn(bv, av))); // past the row: 0 + |0 scale| (0 - 0) = 0
        }
    const float iz = inv_l2(a, eps);
#pragma unroll
    for (int j = 0; j < MOST; ++j)
#pragma unroll
        for (int i = 0; i < 8; ++i) a[j][i] = __fmul_rn(a[j][i], iz);
    store(h, at, chunks, a);
}

}  // namespace tf_l2
