// block_attention.cu's and sparse_attention.cu's fold: 16 mma rows a warp over 32-key tiles, P as bf16 into P x V.
#pragma once
#include "common/softmax_identity.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#include "attn_odd.cuh"
#include "common/qmm_frag.cuh"
#include "common/cp_async.cuh"

namespace attn_fold {

using qmm_frag::commit;
using qmm_frag::cp16z;
using qmm_frag::ldmatrix4;
using qmm_frag::ldmatrix4t;
using qmm_frag::mma;
using qmm_frag::mma0;

constexpr int D = 128, C = D / 8, K16 = D / 16, NB = D / 8, KT = 32;
constexpr float LOG2E = 1.4426950408889634f;

using tf_common::cp_wait;

__device__ __forceinline__ float texp(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;\n" : "=f"(y) : "f"(__fmul_rn(x, LOG2E)));
    return y;
}

__device__ __forceinline__ uint32_t pack(float lo, float hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
}

__device__ __forceinline__ int sw(int r, int c) { return r * C + (c ^ (r & 7)); }

// One warp's fold over a 32-key tile: scores masked by `vis(row, key)`, online softmax, P x V (KEEP: o's inf stays).
struct Fold {
    float o[NB][4];
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

    __device__ __forceinline__ Fold() {
#pragma unroll
        for (int i = 0; i < NB; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    }

    template <bool KEEP = false, typename Vis>
    __device__ __forceinline__ void tile(const uint32_t (&qa)[K16][4], const uint4* ks, const uint4* vs, float scale,
                                         int lane, int k0, Vis vis) {
        const int g8 = lane / 4, qd = lane % 4, mt = lane / 8;
        float sc[4][4];
#pragma unroll
        for (int k = 0; k < K16; ++k) {
#pragma unroll
            for (int n = 0; n < 4; n += 2) {
                uint32_t b[4];
                ldmatrix4(b, ks + sw(8 * (n + mt / 2) + lane % 8, 2 * k + mt % 2));
                if (k == 0) {
                    mma0(sc[n], qa[k], b[0], b[1]);
                    mma0(sc[n + 1], qa[k], b[2], b[3]);
                } else {
                    mma(sc[n], qa[k], b[0], b[1]);
                    mma(sc[n + 1], qa[k], b[2], b[3]);
                }
            }
        }
        float tm0 = -INFINITY, tm1 = -INFINITY;
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            const int c = k0 + 8 * n + 2 * qd;
            sc[n][0] = vis(g8, c) ? __fmul_rn(sc[n][0], scale) : -INFINITY;
            sc[n][1] = vis(g8, c + 1) ? __fmul_rn(sc[n][1], scale) : -INFINITY;
            sc[n][2] = vis(g8 + 8, c) ? __fmul_rn(sc[n][2], scale) : -INFINITY;
            sc[n][3] = vis(g8 + 8, c + 1) ? __fmul_rn(sc[n][3], scale) : -INFINITY;
            tm0 = fmaxf(tm0, fmaxf(sc[n][0], sc[n][1]));
            tm1 = fmaxf(tm1, fmaxf(sc[n][2], sc[n][3]));
        }
        for (int x = 1; x <= 2; x *= 2) {
            tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, x));
            tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, x));
        }
        const float n0 = fmaxf(m0, tm0), n1 = fmaxf(m1, tm1);
        const bool same0 = TF_CUDA_SOFTMAX_IDENTITY_GUARD && isfinite(n0) && n0 == m0, same1 = TF_CUDA_SOFTMAX_IDENTITY_GUARD && isfinite(n1) && n1 == m1;
        const float a0 = same0 ? 1.f : (tm0 == -INFINITY ? 1.f : (m0 == -INFINITY ? 0.f : texp(__fsub_rn(m0, n0))));
        const float a1 = same1 ? 1.f : (tm1 == -INFINITY ? 1.f : (m1 == -INFINITY ? 0.f : texp(__fsub_rn(m1, n1))));
        float c0 = 0.f, c1 = 0.f;
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            sc[n][0] = sc[n][0] != -INFINITY ? texp(__fsub_rn(sc[n][0], n0)) : 0.f;
            sc[n][1] = sc[n][1] != -INFINITY ? texp(__fsub_rn(sc[n][1], n0)) : 0.f;
            sc[n][2] = sc[n][2] != -INFINITY ? texp(__fsub_rn(sc[n][2], n1)) : 0.f;
            sc[n][3] = sc[n][3] != -INFINITY ? texp(__fsub_rn(sc[n][3], n1)) : 0.f;
            c0 = __fadd_rn(c0, __fadd_rn(sc[n][0], sc[n][1]));
            c1 = __fadd_rn(c1, __fadd_rn(sc[n][2], sc[n][3]));
        }
        for (int x = 1; x <= 2; x *= 2) {
            c0 = __fadd_rn(c0, __shfl_xor_sync(0xffffffffu, c0, x));
            c1 = __fadd_rn(c1, __shfl_xor_sync(0xffffffffu, c1, x));
        }
        l0 = (same0 && isfinite(__fadd_rn(l0, c0))) ? __fadd_rn(l0, c0) : __fmaf_rn(l0, a0, c0);
        l1 = (same1 && isfinite(__fadd_rn(l1, c1))) ? __fadd_rn(l1, c1) : __fmaf_rn(l1, a1, c1);
        m0 = n0;
        m1 = n1;
#pragma unroll
        for (int i = 0; i < NB; ++i) {
            if (!same0 || !isfinite(o[i][0])) o[i][0] = KEEP ? attn_odd::keep(o[i][0], a0) : __fmul_rn(o[i][0], a0);
            if (!same0 || !isfinite(o[i][1])) o[i][1] = KEEP ? attn_odd::keep(o[i][1], a0) : __fmul_rn(o[i][1], a0);
            if (!same1 || !isfinite(o[i][2])) o[i][2] = KEEP ? attn_odd::keep(o[i][2], a1) : __fmul_rn(o[i][2], a1);
            if (!same1 || !isfinite(o[i][3])) o[i][3] = KEEP ? attn_odd::keep(o[i][3], a1) : __fmul_rn(o[i][3], a1);
        }
        uint32_t pa[2][4];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            pa[j][0] = pack(sc[2 * j][0], sc[2 * j][1]);
            pa[j][1] = pack(sc[2 * j][2], sc[2 * j][3]);
            pa[j][2] = pack(sc[2 * j + 1][0], sc[2 * j + 1][1]);
            pa[j][3] = pack(sc[2 * j + 1][2], sc[2 * j + 1][3]);
        }
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int i = 0; i < NB; i += 2) {
                uint32_t b[4];
                ldmatrix4t(b, vs + sw(16 * j + 8 * (mt % 2) + lane % 8, i + mt / 2));
                mma(o[i], pa[j], b[0], b[1]);
                mma(o[i + 1], pa[j], b[2], b[3]);
            }
    }
};

}  // namespace attn_fold
