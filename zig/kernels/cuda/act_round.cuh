// MLX's bf16-op-by-op activations (act_rows.cu's tf_geglu_rows, tf_swiglu_rows_round); `keep`: the rounding points.

#pragma once

#include <cuda_bf16.h>
#include <stdint.h>
#include "common/reduce.cuh"

namespace tf_act_round {

// mlx_lm's gelu_approx(x) * up as MLX's bf16 ops (Metal gemma4/glue.metal's gemma_geglu): its rounding points.
enum : int { CUBE = 1, CUBE_SCALED = 2, INNER = 4, INNER_SCALED = 8, TANH = 16, ONE_PLUS = 32, GELU = 64 };
constexpr int GEGLU_ALL = 127;
// tf_draft_swiglu's bf16 steps (Metal core/draft_ops.metal, gemma4's DFlash drafter): its rounding points.
enum : int { EXP = 1, DENOM = 2, RECIP = 4, SIGMOID = 8, SILU = 16 };
constexpr int SWIGLU_ALL = 31;
// MLX's logit soft cap cap * tanh(x / cap) as bf16 ops (Metal gemma4/glue.metal's gemma_softcap): its rounding points.
enum : int { CAP_DIV = 1, CAP_TANH = 2 };
constexpr int SOFTCAP_ALL = 3;
// Spark2.5's head gate (the gate cast to bf16 before the product) and exact GELU (the gelu cast before the product).
enum : int { HEAD_GATE = 1, GELU_ERF = 1 };
constexpr int HEAD_GATE_ALL = 1;
constexpr int GELU_ERF_ALL = 1;

__device__ __forceinline__ float bf(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

// v rounded to bf16 where `keep` holds the point p (a constant in the served kernels, so the test folds away).
__device__ __forceinline__ float point(float v, int keep, int p) { return (keep & p) != 0 ? bf(v) : v; }

// 0.5 x (1 + tanh(c0 (x + c1 x^3))) up, each step rounded to bf16 (x^3 exact in fp32 for a bf16 x; c0, c1 as bf16).
__device__ __forceinline__ float geglu(float x, float up, int keep) {
    const float c0 = bf(0.7978845608028654f), c1 = bf(0.044715f);
    const float cube = point(__fmul_rn(__fmul_rn(x, x), x), keep, CUBE);
    const float inner = point(__fadd_rn(x, point(__fmul_rn(c1, cube), keep, CUBE_SCALED)), keep, INNER);
    const float th = point(tanhf(point(__fmul_rn(c0, inner), keep, INNER_SCALED)), keep, TANH);
    const float half_x = bf(__fmul_rn(0.5f, x)); // exact for a bf16 x: no point to drop
    const float gl = point(__fmul_rn(half_x, point(__fadd_rn(1.f, th), keep, ONE_PLUS)), keep, GELU);
    return __fmul_rn(gl, up);
}

// silu(g) up as sigmoid from exp(|g|): 1 / (1 + e) for g < 0, else 1 - that; each step rounded to bf16.
__device__ __forceinline__ float swiglu(float g, float up, int keep) {
    const float e = point(expf(fabsf(g)), keep, EXP);
    const float inv = point(__fdiv_rn(1.f, point(__fadd_rn(1.f, e), keep, DENOM)), keep, RECIP);
    const float sig = g < 0.f ? inv : point(__fsub_rn(1.f, inv), keep, SIGMOID);
    return __fmul_rn(point(__fmul_rn(g, sig), keep, SILU), up);
}

// bf16(bf16(tanh(bf16(x / cap))) * cap): the division and product correctly rounded, tanhf within its 2 ulps.
__device__ __forceinline__ float softcap(float x, float cap, int keep) {
    const float t = point(__fdiv_rn(x, cap), keep, CAP_DIV);
    return __fmul_rn(point(tanhf(t), keep, CAP_TANH), cap);
}

using tf_common::unpack8;
using tf_common::pack8;
using tf_common::group_sum8;

// out[r, i] = bf16(act(gate[r * g_ld + i], up[r * u_ld + i])) for i < n (n % 64 == 0), GEGLU or not; xs its sums.
template <bool GEGLU>
__device__ __forceinline__ void rows(const __nv_bfloat16* __restrict__ gate, long long g_ld,
                                     const __nv_bfloat16* __restrict__ up, long long u_ld, __nv_bfloat16* __restrict__ out,
                                     float* __restrict__ xs, int n, int keep) {
    const int64_t r = blockIdx.x;
    const int c = blockIdx.y * blockDim.x + threadIdx.x; // 8 values
    const bool live = 8 * c < n;
    float g[8], u[8], o[8];
    if (live) {
        unpack8(*reinterpret_cast<const uint4*>(gate + r * g_ld + 8 * c), g);
        unpack8(*reinterpret_cast<const uint4*>(up + r * u_ld + 8 * c), u);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) o[i] = live ? (GEGLU ? geglu(g[i], u[i], keep) : swiglu(g[i], u[i], keep)) : 0.f;
    const uint4 op = pack8(o);
    if (live) *reinterpret_cast<uint4*>(out + r * n + 8 * c) = op;
    if (xs != nullptr) {
        const float s = group_sum8(op, live);
        if (live && (c & 7) == 0) xs[r * (n / 64) + c / 8] = s;
    }
}

// bf16(gelu(x)) * up, gelu PyTorch's exact form in fp32: x / 2 (1 + erf(x / sqrt 2)), erff within its 2 ulps.
__device__ __forceinline__ float gelu_erf(float x, float up, int keep) {
    const float gl = __fmul_rn(__fmul_rn(x, 0.5f), __fadd_rn(1.f, erff(__fmul_rn(x, 0.70710678118654752f))));
    return __fmul_rn(point(gl, keep, GELU_ERF), up);
}

// out[r, i] = bf16(gelu_erf(gate[r * g_ld + i], up[r * u_ld + i])) for i < n (n % 64 == 0); xs (or null) its sums.
__device__ __forceinline__ void gelu_erf_rows(const __nv_bfloat16* __restrict__ gate, long long g_ld,
                                              const __nv_bfloat16* __restrict__ up, long long u_ld,
                                              __nv_bfloat16* __restrict__ out, float* __restrict__ xs, int n, int keep) {
    const int64_t r = blockIdx.x;
    const int c = blockIdx.y * blockDim.x + threadIdx.x; // 8 values
    const bool live = 8 * c < n;
    float g[8], u[8], o[8];
    if (live) {
        unpack8(*reinterpret_cast<const uint4*>(gate + r * g_ld + 8 * c), g);
        unpack8(*reinterpret_cast<const uint4*>(up + r * u_ld + 8 * c), u);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) o[i] = live ? gelu_erf(g[i], u[i], keep) : 0.f;
    const uint4 op = pack8(o);
    if (live) *reinterpret_cast<uint4*>(out + r * n + 8 * c) = op;
    if (xs != nullptr) {
        const float s = group_sum8(op, live);
        if (live && (c & 7) == 0) xs[r * (n / 64) + c / 8] = s;
    }
}

// out[r, h d + e] = bf16(o * bf16(1 / (1 + expf(-g[r * g_ld + h])))), o packed (d % 64 == 0); xs (or null) its sums.
__device__ __forceinline__ void head_gate_rows(const __nv_bfloat16* __restrict__ o, const __nv_bfloat16* __restrict__ g,
                                               long long g_ld, __nv_bfloat16* __restrict__ out, float* __restrict__ xs,
                                               int heads, int d, int keep) {
    const int64_t r = blockIdx.x;
    const int c = blockIdx.y * blockDim.x + threadIdx.x; // 8 values, all of one head
    const int n = heads * d;
    const bool live = 8 * c < n;
    float ov[8], y[8];
    float gate = 0.f;
    if (live) {
        unpack8(*reinterpret_cast<const uint4*>(o + r * n + 8 * c), ov);
        const float x = __bfloat162float(g[r * g_ld + 8 * c / d]);
        gate = point(__fdiv_rn(1.f, __fadd_rn(1.f, expf(-x))), keep, HEAD_GATE);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) y[i] = live ? __fmul_rn(ov[i], gate) : 0.f;
    const uint4 yp = pack8(y);
    if (live) *reinterpret_cast<uint4*>(out + r * n + 8 * c) = yp;
    if (xs != nullptr) {
        const float s = group_sum8(yp, live);
        if (live && (c & 7) == 0) xs[r * (n / 64) + c / 8] = s;
    }
}

// out[r, i] = softcap(x[r, i]) for i < n (n % 8 == 0), rows packed n wide (out may be x); 8 values a thread.
__device__ __forceinline__ void softcap_rows(const __nv_bfloat16* x, __nv_bfloat16* out, int n, float cap, int keep) {
    const int64_t r = blockIdx.x;
    const int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (8 * c >= n) return;
    float v[8];
    unpack8(*reinterpret_cast<const uint4*>(x + r * n + 8 * c), v);
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = softcap(v[i], cap, keep);
    *reinterpret_cast<uint4*>(out + r * n + 8 * c) = pack8(v);
}

}  // namespace tf_act_round
