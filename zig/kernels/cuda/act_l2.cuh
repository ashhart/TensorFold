// act_rows.cu's nGPT activations (Dory): a scaled sigmoid gate and a scaled SwiGLU, each row then L2-normed.

#pragma once

#include <cuda_bf16.h>
#include <stdint.h>

namespace tf_act_l2 {

constexpr int NT = 256;
constexpr int MOST = 4; // 16-byte loads a thread: rows of width % 64, at most 8,192, 16 B aligned

// The rounding points: the gated row; SwiGLU's scaled gate, scaled up and their product.
enum : int { GATED = 1, GS = 2, US = 4, PROD = 8 };

__device__ __forceinline__ float bf(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

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

// torch's fp32 sigmoid and silu: 1 / (1 + exp(-x)), x / (1 + exp(-x)), IEEE division.
__device__ __forceinline__ float sigmoid(float x) { return __fdiv_rn(1.f, __fadd_rn(1.f, expf(-x))); }
__device__ __forceinline__ float silu(float x) { return __fdiv_rn(x, __fadd_rn(1.f, expf(-x))); }

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
    return __fdiv_rn(1.f, __fsqrt_rn(__fadd_rn(s, eps)));
}

// One input row's chunks of 8 (zeros past the row), and a scale vector's (fp32, 2 x 16 B a chunk).
__device__ __forceinline__ void rowOf(const __nv_bfloat16* __restrict__ x, int64_t at, int chunks, float (&v)[MOST][8]) {
#pragma unroll
    for (int j = 0; j < MOST; ++j) {
        const int c = threadIdx.x + j * NT;
        unpack8(c < chunks ? reinterpret_cast<const uint4*>(x + at)[c] : make_uint4(0, 0, 0, 0), v[j]);
    }
}

__device__ __forceinline__ void scaleOf(const float* __restrict__ s, int chunks, float (&v)[MOST][8]) {
#pragma unroll
    for (int j = 0; j < MOST; ++j) {
        const int c = threadIdx.x + j * NT;
        const float4 a = c < chunks ? reinterpret_cast<const float4*>(s)[2 * c] : make_float4(0.f, 0.f, 0.f, 0.f);
        const float4 b = c < chunks ? reinterpret_cast<const float4*>(s)[2 * c + 1] : make_float4(0.f, 0.f, 0.f, 0.f);
        v[j][0] = a.x, v[j][1] = a.y, v[j][2] = a.z, v[j][3] = a.w;
        v[j][4] = b.x, v[j][5] = b.y, v[j][6] = b.z, v[j][7] = b.w;
    }
}

__device__ __forceinline__ void normStore(__nv_bfloat16* __restrict__ out, int64_t at, int chunks, float (&v)[MOST][8], float eps) {
    const float inv = inv_l2(v, eps);
#pragma unroll
    for (int j = 0; j < MOST; ++j) {
        const int c = threadIdx.x + j * NT;
        float y[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) y[i] = __fmul_rn(v[j][i], inv);
        if (c < chunks) reinterpret_cast<uint4*>(out + at)[c] = pack8(y);
    }
}

// out = bf16(l2(t)), t = point<GATED>(o * sigmoid(g * sg)) over the row's n values (all heads): a block a row.
template <int R>
__device__ __forceinline__ void gate_rows(const __nv_bfloat16* __restrict__ o, const __nv_bfloat16* __restrict__ g, long long g_ld,
                                          const float* __restrict__ sg, __nv_bfloat16* __restrict__ out, int n, float eps) {
    const int64_t r = blockIdx.x;
    const int chunks = n / 8;
    float ov[MOST][8], gv[MOST][8], sv[MOST][8];
    rowOf(o, r * n, chunks, ov);
    rowOf(g, r * g_ld, chunks, gv);
    scaleOf(sg, chunks, sv);
#pragma unroll
    for (int j = 0; j < MOST; ++j)
#pragma unroll
        for (int i = 0; i < 8; ++i) ov[j][i] = point<R, GATED>(__fmul_rn(ov[j][i], sigmoid(__fmul_rn(gv[j][i], sv[j][i]))));
    normStore(out, r * n, chunks, ov, eps);
}

// out = bf16(l2(m)), m = point<PROD>(silu(gs) us), gs = point<GS>(g sug), us = point<US>(u suu): dory.py's MLP.
template <int R>
__device__ __forceinline__ void swiglu_rows(const __nv_bfloat16* __restrict__ g, long long g_ld, const __nv_bfloat16* __restrict__ u,
                                            long long u_ld, const float* __restrict__ sug, const float* __restrict__ suu,
                                            __nv_bfloat16* __restrict__ out, int n, float eps) {
    const int64_t r = blockIdx.x;
    const int chunks = n / 8;
    float gv[MOST][8], uv[MOST][8], sv[MOST][8];
    rowOf(g, r * g_ld, chunks, gv);
    rowOf(u, r * u_ld, chunks, uv);
    scaleOf(sug, chunks, sv);
#pragma unroll
    for (int j = 0; j < MOST; ++j)
#pragma unroll
        for (int i = 0; i < 8; ++i) gv[j][i] = point<R, GS>(__fmul_rn(gv[j][i], sv[j][i]));
    scaleOf(suu, chunks, sv);
#pragma unroll
    for (int j = 0; j < MOST; ++j)
#pragma unroll
        for (int i = 0; i < 8; ++i) gv[j][i] = point<R, PROD>(__fmul_rn(silu(gv[j][i]), point<R, US>(__fmul_rn(uv[j][i], sv[j][i]))));
    normStore(out, r * n, chunks, gv, eps);
}

}  // namespace tf_act_l2
