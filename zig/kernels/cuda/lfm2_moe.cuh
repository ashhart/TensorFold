// LFM2.5-24B-A2B's CUDA glue (short conv, its commit, router); each row's bits its own; CTL 1 is the planted control.

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


// Block (row, channel block): y = bf16(C * bf16(sum of w * bf16(B * x), oldest tap first)); SNAP rows keep state.
template <int CTL>  // CTL 1: bx = B * x not rounded to bf16 before its tap.
__device__ __forceinline__ void lfm2m_conv(const __nv_bfloat16* __restrict__ P, const __nv_bfloat16* __restrict__ Wc,
                                                                const __nv_bfloat16* __restrict__ STATE, const int* __restrict__ SLOT,
                                                                const int* __restrict__ OFF, __nv_bfloat16* __restrict__ Y,
                                                                __nv_bfloat16* __restrict__ SNAP, int SNAP0, int D, long long SS) {
    const int64_t r = blockIdx.x, D3 = 3 * static_cast<int64_t>(D);
    const int c = blockIdx.y * NT + threadIdx.x;
    if (c >= D) return;
    const int o = OFF[r];
    const __nv_bfloat16* st = STATE + SLOT[r] * SS;
    __nv_bfloat16 bx[3];
    float wide[3] = {0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int back = 2 - j;  // rows before this one
        if (o >= back) {
            const int64_t src = r - back;
            const float v = __fmul_rn(tf_glue::bf(P[src * D3 + c]), tf_glue::bf(P[src * D3 + 2 * D + c]));
            bx[j] = __float2bfloat16_rn(v);
            if (CTL == 1) wide[j] = v;
        } else {
            bx[j] = st[static_cast<int64_t>(2 - back + o) * D + c];
            if (CTL == 1) wide[j] = tf_glue::bf(bx[j]);
        }
    }
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < 3; ++j) acc = __fadd_rn(acc, __fmul_rn(tf_glue::bf(Wc[c * 3 + j]), CTL == 1 ? wide[j] : tf_glue::bf(bx[j])));
    Y[r * D + c] = __float2bfloat16_rn(__fmul_rn(tf_glue::bf(P[r * D3 + D + c]), tf_glue::rbf(acc)));
    if (SNAP == nullptr || r < SNAP0) return;
    SNAP[((r - SNAP0) * 2) * D + c] = bx[1];
    SNAP[((r - SNAP0) * 2 + 1) * D + c] = bx[2];
}

// Block (layer, stream): stream k's [2][D] conv state set to its snapshot row ROW[k]; ROW < 0 keeps the state.
template <int CTL>  // CTL 1: every stream's state written to the first stream's slot.
__device__ __forceinline__ void lfm2m_commit(__nv_bfloat16* __restrict__ STATE, const __nv_bfloat16* __restrict__ SNAP,
                                                                  const int* __restrict__ SLOT, const int* __restrict__ ROW, int D,
                                                                  long long state_stride, long long snap_stride, long long SS) {
    const int layer = blockIdx.x, k = blockIdx.y, row = ROW[k];
    if (row < 0) return;
    __nv_bfloat16* dst = STATE + layer * state_stride + SLOT[CTL == 1 ? 0 : k] * SS;
    const __nv_bfloat16* src = SNAP + layer * snap_stride + static_cast<int64_t>(row) * 2 * D;
    for (int i = threadIdx.x; i < 2 * D; i += NT) dst[i] = src[i];
}

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
