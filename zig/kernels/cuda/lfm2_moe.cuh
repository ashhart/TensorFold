// LFM2.5-24B-A2B's router; each row's bits its own; CTL 1 is the planted control (short conv: liquid_conv.cuh).

#pragma once

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

#include "glue_math.cuh"

namespace tf_lfm2m {

constexpr int NT = 256;    // threads a block
constexpr int E = 64;      // experts
constexpr int TOPK = 4;    // picks a row
constexpr int RB = 8;      // router rows a block: the weights are read once for them
constexpr int DMAX = 2048; // hidden width the router stages

// (value, id) wins over (other value, other id): the larger value, the lower id among equals.
template <int CTL>
__device__ __forceinline__ bool beats(float v, int i, float w, int j) { return v > w || (v == w && (CTL == 1 ? i > j : i < j)); }


// Block of RB rows: fp32 logits (quarters of D in order, (q0 + q1) + (q2 + q3)), top-4 of sigmoid + bias, weights.
template <int CTL>  // CTL 1: ties go to the higher expert id.
__device__ __forceinline__ void lfm2m_route(const __nv_bfloat16* __restrict__ X, const float* __restrict__ W,
                                                                 const float* __restrict__ BIAS, int* __restrict__ PICK,
                                                                 float* __restrict__ WTS, float* __restrict__ LOGITS, int rows,
                                                                 int D, float scale) {
    __shared__ __align__(16) __nv_bfloat16 xs[RB][DMAX];
    __shared__ float lg[RB][E];
    const int row0 = blockIdx.x * RB, t = threadIdx.x, e = t / 4, q = t % 4, n = min(RB, rows - row0);
    for (int i = t; i < n * D; i += NT) xs[i / D][i % D] = X[static_cast<int64_t>(row0 + i / D) * D + i % D];
    __syncthreads();
    float acc[RB];
#pragma unroll
    for (int b = 0; b < RB; ++b) acc[b] = 0.0f;
    const int per = D / 4, k0 = q * per;
    const float* w = W + static_cast<int64_t>(e) * D + k0;
    for (int k = 0; k < per; ++k) {
        const float wk = w[k];
#pragma unroll
        for (int b = 0; b < RB; ++b) acc[b] = __fmaf_rn(tf_glue::bf(xs[b][k0 + k]), wk, acc[b]);
    }
#pragma unroll
    for (int b = 0; b < RB; ++b) {
        float v = __fadd_rn(acc[b], __shfl_xor_sync(0xffffffffu, acc[b], 1));
        v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 2));
        if (q == 0) lg[b][e] = v;
    }
    __syncthreads();
    const int b = t / 32, lane = t % 32;
    if (b >= n) return;
    const int64_t r = row0 + b;
    float gate[2], sel[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const float l = lg[b][lane + 32 * h];
        if (LOGITS != nullptr) LOGITS[r * E + lane + 32 * h] = l;
        gate[h] = tf_glue::tf_sigmoid(l);
        const float v = __fadd_rn(gate[h], BIAS[lane + 32 * h]);
        sel[h] = v != v ? -INFINITY : v;  // a NaN logit is never picked ahead of a number
    }
    float total = 0.0f, mine = 0.0f;
    int pick = 0;
    for (int k = 0; k < TOPK; ++k) {
        float bv = -INFINITY;
        int bi = CTL == 1 ? -1 : 0x7fffffff;
#pragma unroll
        for (int h = 0; h < 2; ++h)
            if (beats<CTL>(sel[h], lane + 32 * h, bv, bi)) bv = sel[h], bi = lane + 32 * h;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
            if (beats<CTL>(ov, oi, bv, bi)) bv = ov, bi = oi;
        }
        float p = 0.0f;
#pragma unroll
        for (int h = 0; h < 2; ++h)
            if (bi == lane + 32 * h) p = gate[h], sel[h] = NAN;  // taken: NaN beats nothing, not even a -inf left
        p = __shfl_sync(0xffffffffu, p, bi % 32);
        total = __fadd_rn(total, p);
        if (lane == k) mine = p, pick = bi;
    }
    if (lane >= TOPK) return;
    PICK[r * TOPK + lane] = pick;
    WTS[r * TOPK + lane] = __fmul_rn(__fdiv_rn(mine, __fadd_rn(total, 1e-6f)), scale);
}

}  // namespace tf_lfm2m
