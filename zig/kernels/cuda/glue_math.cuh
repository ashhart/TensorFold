// The glue kernels' scalar math, op for op as glue_math.zig runs it on the host: IEEE adds, muls, fmas, divs only.
#pragma once

#include <cuda_bf16.h>
#include <stdint.h>

namespace tf_glue {

__device__ __forceinline__ float bf(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float rbf(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

// e^x within one ulp: x = j ln2 + f, a degree-7 Taylor polynomial in f, then 2^j in two exact factors.
__device__ __forceinline__ float tf_exp(float x) {
    if (x != x) return x;
    if (x > 104.0f) return __int_as_float(0x7f800000);
    if (x < -104.0f) return 0.0f;
    const float j = __fsub_rn(__fmaf_rn(x, 0x1.715476p+0f, 0x1.8p+23f), 0x1.8p+23f);
    float f = __fmaf_rn(j, -0x1.62e4p-1f, x);
    f = __fmaf_rn(j, -0x1.7f7d1cp-20f, f);
    float p = 0x1.a01a02p-13f;  // 1/7! .. 1/3!, each the nearest float
    p = __fmaf_rn(p, f, 0x1.6c16c2p-10f);
    p = __fmaf_rn(p, f, 0x1.111112p-7f);
    p = __fmaf_rn(p, f, 0x1.555556p-5f);
    p = __fmaf_rn(p, f, 0x1.555556p-3f);
    p = __fmaf_rn(p, f, 0.5f);
    p = __fmaf_rn(p, f, 1.0f);
    p = __fmaf_rn(p, f, 1.0f);
    const int i = static_cast<int>(j);
    const float a = __int_as_float(i > 0 ? 0x7f000000 : 0x02000000);
    const float b = __int_as_float(i > 0 ? i << 23 : (i + 250) << 23);
    return __fmul_rn(__fmul_rn(p, a), b);
}

// ln(1 + u) for u in [0, 1], within one ulp: 2 atanh(t / (2 + t)) with the quotient carried in two floats.
__device__ __forceinline__ float tf_log1p_unit(float u) {
    const bool hi = u > 0.5f;
    const float t = hi ? __fmul_rn(__fsub_rn(u, 1.0f), 0.5f) : u;  // exact for u in (0.5, 1]: ln(1 + u) = ln2 + ln(1 + t)
    const float s = __fadd_rn(2.0f, t);
    const float err = __fsub_rn(t, __fsub_rn(s, 2.0f));  // 2 + t = s + err exactly
    const float zh = __fdiv_rn(t, s);
    const float zl = __fdiv_rn(__fsub_rn(__fmaf_rn(-zh, s, t), __fmul_rn(zh, err)), s);  // t / (2 + t) = zh + zl to second order
    const float z2 = __fmul_rn(zh, zh);
    float p = 0x1.745d18p-4f;  // 1/11 .. 1/3, each the nearest float
    p = __fmaf_rn(p, z2, 0x1.c71c72p-4f);
    p = __fmaf_rn(p, z2, 0x1.24924ap-3f);
    p = __fmaf_rn(p, z2, 0x1.99999ap-3f);
    p = __fmaf_rn(p, z2, 0x1.555556p-2f);
    const float zz = __fadd_rn(zh, zh);
    const float r = __fadd_rn(zz, __fmaf_rn(__fmul_rn(zz, z2), p, __fadd_rn(zl, zl)));
    if (!hi) return r;
    return __fadd_rn(0x1.62e4p-1f, __fadd_rn(0x1.7f7d1cp-20f, r));
}

__device__ __forceinline__ float tf_sigmoid(float x) { return __fdiv_rn(1.0f, __fadd_rn(1.0f, tf_exp(-x))); }

// SiLU with one rounding fewer than x * sigmoid(x).
__device__ __forceinline__ float tf_silu(float x) { return __fdiv_rn(x, __fadd_rn(1.0f, tf_exp(-x))); }

// The dt softplus without log(1 + e)'s loss for small e.
__device__ __forceinline__ float tf_softplus(float v) {
    return __fadd_rn(fmaxf(v, 0.0f), tf_log1p_unit(tf_exp(-fabsf(v))));
}

// A warp's 32 values summed by xor butterfly (16, 8, 4, 2, 1): every lane ends with the same sum.
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}

}  // namespace tf_glue
