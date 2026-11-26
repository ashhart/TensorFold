// prefill_attention.cu's staging, MMA and fold helpers, shared with its controls (contract_controls.cu).
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "common/qmm_frag.cuh"
#include "common/cp_async.cuh"

namespace tf_prefill_attention {

using qmm_frag::commit;
using qmm_frag::cp16z;
using qmm_frag::ldmatrix4;
using qmm_frag::mma;
using qmm_frag::mma0;

constexpr int BN = 64, HALF = 32;             // keys a tile (the fold's unit), keys a staging slot
constexpr float LOG2E = 1.4426950408889634f;
constexpr int SMEM_MAX = 96 * 1024;

using tf_common::cp_wait;

using qmm_frag::ldmatrix4t;

__device__ __forceinline__ float texp(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;\n" : "=f"(y) : "f"(__fmul_rn(x, LOG2E)));
    return y;
}

__device__ __forceinline__ float tdiv(float a, float b) {
    float y;
    asm("div.full.f32 %0, %1, %2;\n" : "=f"(y) : "f"(a), "f"(b));
    return y;
}

__device__ __forceinline__ uint32_t pack(float lo, float hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
}

template <int D>
__device__ __forceinline__ int sw(int r, int c) { return r * (D / 8) + (c ^ (r & 7)); }

__device__ __forceinline__ float rowsum(const float (&c)[8]) {
    float w[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float x = __fadd_rn(c[j], __shfl_xor_sync(0xffffffffu, c[j], 2));
        w[j] = __fadd_rn(x, __shfl_xor_sync(0xffffffffu, x, 1));
    }
    return __fadd_rn(__fadd_rn(__fadd_rn(w[0], w[4]), __fadd_rn(w[2], w[6])),
                     __fadd_rn(__fadd_rn(w[1], w[5]), __fadd_rn(w[3], w[7])));
}

// Four e4m3 pairs of a cache row widened by the row's scale into bf16 pairs.
__device__ __forceinline__ uint4 widen8(const uint2& pre, float s) {
    uint4 w;
    const __nv_fp8x2_storage_t* e = reinterpret_cast<const __nv_fp8x2_storage_t*>(&pre);
    uint32_t* o = reinterpret_cast<uint32_t*>(&w);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float2 x = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(e[k], __NV_E4M3)));
        o[k] = pack(__fmul_rn(x.x, s), __fmul_rn(x.y, s));
    }
    return w;
}

// A tile's 64 scores of rows pos0 and pos1 scaled, or masked past the row's position, and each row's maximum.
__device__ __forceinline__ void tile_mask(float (&sc)[8][4], int kb, int pos0, int pos1, float scale, float& tm0,
                                          float& tm1) {
    tm0 = -INFINITY;
    tm1 = -INFINITY;
#pragma unroll
    for (int n = 0; n < 8; ++n) {
        const int c = kb + 8 * n;
        sc[n][0] = c <= pos0 ? __fmul_rn(sc[n][0], scale) : -INFINITY;
        sc[n][1] = c + 1 <= pos0 ? __fmul_rn(sc[n][1], scale) : -INFINITY;
        sc[n][2] = c <= pos1 ? __fmul_rn(sc[n][2], scale) : -INFINITY;
        sc[n][3] = c + 1 <= pos1 ? __fmul_rn(sc[n][3], scale) : -INFINITY;
        tm0 = fmaxf(tm0, fmaxf(sc[n][0], sc[n][1]));
        tm1 = fmaxf(tm1, fmaxf(sc[n][2], sc[n][3]));
    }
    for (int x = 1; x <= 2; x *= 2) {
        tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, x));
        tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, x));
    }
}

// The running maximum after a tile whose maximum is tm (act: the tile holds an unmasked score).
__device__ __forceinline__ float run_max(float m, float tm, bool act) { return act ? fmaxf(m, tm) : m; }

// The factor the sums before a tile take when the running maximum moves from m to n.
__device__ __forceinline__ float run_factor(float m, float n, bool act) {
    return act ? (m == -INFINITY ? 0.0f : texp(__fsub_rn(m, n))) : 1.0f;
}

// A tile's weights exp(score - n) in place of its scores, and their pair sums for rowsum.
__device__ __forceinline__ void tile_weights(float (&sc)[8][4], int kb, int pos0, int pos1, bool act0, bool act1,
                                             float n0, float n1, float (&c0)[8], float (&c1)[8]) {
#pragma unroll
    for (int n = 0; n < 8; ++n) {
        const int c = kb + 8 * n;
        sc[n][0] = act0 && c <= pos0 ? texp(__fsub_rn(sc[n][0], n0)) : 0.0f;
        sc[n][1] = act0 && c + 1 <= pos0 ? texp(__fsub_rn(sc[n][1], n0)) : 0.0f;
        sc[n][2] = act1 && c <= pos1 ? texp(__fsub_rn(sc[n][2], n1)) : 0.0f;
        sc[n][3] = act1 && c + 1 <= pos1 ? texp(__fsub_rn(sc[n][3], n1)) : 0.0f;
        c0[n] = __fadd_rn(sc[n][0], sc[n][1]);
        c1[n] = __fadd_rn(sc[n][2], sc[n][3]);
    }
}

// A tile's weights as the four bf16 fragments the value products read.
__device__ __forceinline__ void tile_pack(uint32_t (&pa)[4][4], const float (&sc)[8][4]) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        pa[j][0] = pack(sc[2 * j][0], sc[2 * j][1]);
        pa[j][1] = pack(sc[2 * j][2], sc[2 * j][3]);
        pa[j][2] = pack(sc[2 * j + 1][0], sc[2 * j + 1][1]);
        pa[j][3] = pack(sc[2 * j + 1][2], sc[2 * j + 1][3]);
    }
}

#ifndef PATTN_VLANE
#define PATTN_VLANE 0  // 1 in contract_controls.cu only: values read from the neighbouring 8 dims
#endif

}  // namespace tf_prefill_attention
