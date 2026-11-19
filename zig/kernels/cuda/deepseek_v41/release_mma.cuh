// DeepSeek V4.1 release weights on sm_121a: warp building blocks over plain pointers and indices (no blockIdx).
#pragma once

#include <cuda_bf16.h>
#include <stdint.h>

#include "common/block_scale_mma.cuh"
#include "common/qmm_frag.cuh"

namespace rel {

// Lane coordinates of an m16n8 tile: g picks rows g and g + 8 (and column g of B), t the K quarter.
struct Lane {
    int g, t;
};

__device__ __forceinline__ Lane lane_of(int lane) { return {lane >> 2, lane & 3}; }

// One 32-wide K group, a ue8m0 scale a row of A (lane 4r, row r + 8 lane 4r + 1) and a column of B (lane 4c).
__device__ __forceinline__ void mma_e4m3(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2], uint32_t sa,
                                         uint32_t sb) {
    tf_common::mma_mx_e4m3(d, a, b, sa, sb);
}

// The same step with B in e2m1, each code in bits [5:2] of its byte (expand_e2m1 gives that from packed nibbles).
__device__ __forceinline__ void mma_e2m1(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2], uint32_t sa,
                                         uint32_t sb) {
    tf_common::mma_mx_e2m1(d, a, b, sa, sb);
}

// One 16-wide K step of BF16 products into FP32.
__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    qmm_frag::mma_v(d, a, b);
}

// Eight packed e2m1 nibbles (low nibble = even k) as eight mma bytes, each code shifted into bits [5:2].
__device__ __forceinline__ void expand_e2m1(uint32_t x, uint32_t (&out)[2]) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const uint32_t v = x >> (16 * h);
        out[h] = ((v & 0xfu) | ((v & 0xf0u) << 4) | ((v & 0xf00u) << 8) | ((v & 0xf000u) << 12)) << 2;
    }
}

// A K step is 32 bytes of a row (32 FP8 codes or 16 BF16 values); lane (g, t) holds bytes 4t.. and 16 + 4t.. of it.
__device__ __forceinline__ uint32_t ld32(const uint8_t* p) { return *reinterpret_cast<const uint32_t*>(p); }

// A rows g and g + 8 of one step at byte kb from global memory; rows at or past `valid` are zero.
__device__ __forceinline__ void frag_rows_global(const uint8_t* a, long long ld, int valid, int kb, Lane l,
                                                 uint32_t (&f)[4]) {
    const uint8_t* lo = a + (long long)l.g * ld + kb + 4 * l.t;
    const uint8_t* hi = lo + 8 * ld;
    f[0] = l.g < valid ? ld32(lo) : 0u;
    f[1] = l.g + 8 < valid ? ld32(hi) : 0u;
    f[2] = l.g < valid ? ld32(lo + 16) : 0u;
    f[3] = l.g + 8 < valid ? ld32(hi + 16) : 0u;
}

// B column g of one step at byte kb from global memory.
__device__ __forceinline__ void frag_cols_global(const uint8_t* w, long long ld, int kb, Lane l, uint32_t (&b)[2]) {
    const uint8_t* p = w + (long long)l.g * ld + kb + 4 * l.t;
    b[0] = ld32(p);
    b[1] = ld32(p + 16);
}

// The A scale this lane supplies for group kg: row g's from t even, row g + 8's from t odd; padded rows 127.
__device__ __forceinline__ uint32_t scale_rows_global(const uint8_t* sa, long long ld, int valid, int kg, Lane l) {
    const int r = l.g + ((l.t & 1) << 3);
    return r < valid ? sa[(long long)r * ld + kg] : 127u;
}

// The mmas of one K step for a warp tile of RM row tiles by NW n8 tiles (the tile's columns start 32-aligned).
template <int RM, int NW, bool FP8>
__device__ __forceinline__ void mma_step(float (&acc)[RM][NW][4], const uint32_t (&a)[RM][4], const uint32_t (&b)[NW][2],
                                         const uint32_t (&sa)[RM], const uint32_t (&sb)[(NW + 3) / 4]) {
#pragma unroll
    for (int i = 0; i < RM; ++i)
#pragma unroll
        for (int j = 0; j < NW; ++j) {
            if constexpr (FP8) {
                mma_e4m3(acc[i][j], a[i], b[j], sa[i], sb[j >> 2]);
            } else {
                mma_bf16(acc[i][j], a[i], b[j]);
            }
        }
}

// K group range [g0, g1) of slice `slice` of s over `groups`: contiguous, whole groups, the same for every row count.
__device__ __forceinline__ int2 slice_groups(int groups, int slice, int s) {
    return make_int2((int)((long long)slice * groups / s), (int)((long long)(slice + 1) * groups / s));
}

// A warp tile's outputs: FP32 partials part[slice][row][col] when s > 1, else y (BF16 or FP32) at column offset y_col.
template <int RM, int NW>
__device__ __forceinline__ void store_acc(const float (&acc)[RM][NW][4], void* y, long long y_ld, long long y_col,
                                          float* part, int m, int n, int s, int slice, bool f32, int row0, int col0,
                                          Lane l) {
#pragma unroll
    for (int i = 0; i < RM; ++i)
#pragma unroll
        for (int j = 0; j < NW; ++j)
#pragma unroll
            for (int c = 0; c < 4; c += 2) {
                const int row = row0 + i * 16 + l.g + 8 * (c >> 1);
                const int col = col0 + j * 8 + l.t * 2;
                if (row >= m || col >= n) continue;
                const float v0 = acc[i][j][c], v1 = acc[i][j][c + 1];
                if (s > 1) {
                    *reinterpret_cast<float2*>(part + ((long long)slice * m + row) * n + col) = make_float2(v0, v1);
                } else if (f32) {
                    float* o = static_cast<float*>(y) + row * y_ld + y_col + col;
                    o[0] = v0;
                    o[1] = v1;
                } else {
                    __nv_bfloat16* o = static_cast<__nv_bfloat16*>(y) + row * y_ld + y_col + col;
                    o[0] = __float2bfloat16_rn(v0);
                    o[1] = __float2bfloat16_rn(v1);
                }
            }
}

// The fixed split tree: element i of s FP32 partials (2, 4 or 8) a stride mn apart, met in adjacent pairs by level.
__device__ __forceinline__ float combine(const float* part, long long mn, long long i, int s) {
    float v[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) v[k] = k < s ? part[k * mn + i] : 0.0f;
    if (s == 8) {
#pragma unroll
        for (int k = 0; k < 4; ++k) v[k] = v[2 * k] + v[2 * k + 1];
    }
    if (s >= 4) {
#pragma unroll
        for (int k = 0; k < 2; ++k) v[k] = v[2 * k] + v[2 * k + 1];
    }
    return v[0] + v[1];
}

// ceil(log2(fp32(max(amax, 1e-4) * fp32(1/448)))) from the product's bits: the round trip's UE8M0 exponent.
__device__ __forceinline__ int fp8_exponent(float amax) {
    const float p = fmaxf(amax, 1e-4f) * (1.0f / 448.0f);
    const uint32_t b = __float_as_uint(p);
    return (int)((b >> 23) & 0xffu) - 127 + ((b & 0x7fffffu) != 0u ? 1 : 0);
}

// |v| / 2^e in [0, 448] as an E4M3 magnitude code, round to nearest even with subnormals.
__device__ __forceinline__ uint32_t e4m3_code(float q) {
    if (q < 0.015625f) return (uint32_t)__float2int_rn(q * 512.0f);
    const uint32_t b = __float_as_uint(q);
    const int e = (int)((b >> 23) & 0xffu) - 127;
    const float m = __uint_as_float((b & 0x7fffffu) | 0x3f800000u);
    return (uint32_t)(((e + 7) << 3) + __float2int_rn((m - 1.0f) * 8.0f));
}

// An E4M3 magnitude code's value.
__device__ __forceinline__ float e4m3_value(uint32_t code) {
    const uint32_t e = code >> 3, m = code & 7u;
    return e == 0 ? (float)m * 0.001953125f : __uint_as_float(((e + 120u) << 23) | (m << 20));
}

__device__ __forceinline__ float pow2(int e) { return __uint_as_float((uint32_t)(e + 127) << 23); }

// One 32-value block's round trip, a value a lane; each output is optional (null).
__device__ __forceinline__ void actq32(float v, int lane, uint8_t* code, uint8_t* scale, __nv_bfloat16* rt) {
    const bool bad = __any_sync(0xffffffffu, !isfinite(v));
    if (bad) {
        if (code) code[lane] = 0x7f;
        if (scale && lane == 0) *scale = 0xff;
        if (rt) rt[lane] = __ushort_as_bfloat16((unsigned short)0x7fc0);
        return;
    }
    float amax = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const int e = fp8_exponent(amax);
    const uint32_t c = e4m3_code(fminf(fabsf(v) * pow2(-e), 448.0f));
    const uint32_t sign = signbit(v) ? 0x80u : 0u;
    if (code) code[lane] = (uint8_t)(c | sign);
    if (scale && lane == 0) *scale = (uint8_t)(e + 127);
    if (rt) {
        const float q = e4m3_value(c) * pow2(e);
        rt[lane] = __float2bfloat16_rn(sign ? -q : q);
    }
}

// Experts' SwiGLU: bf16 gate, up; gate <= 10, |up| <= 10 (NaN passes); silu(g) * u [* route] in FP32, then BF16.
__device__ __forceinline__ float swiglu(float gate_acc, float up_acc, float route, bool routed) {
    float g = __bfloat162float(__float2bfloat16_rn(gate_acc));
    float u = __bfloat162float(__float2bfloat16_rn(up_acc));
    g = g > 10.0f ? 10.0f : g;
    u = u > 10.0f ? 10.0f : (u < -10.0f ? -10.0f : u);
    float v = g / (1.0f + expf(-g)) * u;
    if (routed) v = v * route;
    return __bfloat162float(__float2bfloat16_rn(v));
}

}  // namespace rel
