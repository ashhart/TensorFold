// A latent MoE's glue: wide router picks, routed pairs summed by share then by pick, the shared add (share order).

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

#include "glue_math.cuh"

using namespace tf_glue;

namespace {

constexpr int MAXC = 32;  // experts a lane holds: up to 1024 (picks a row: one a lane, up to 32)
constexpr int MAXE = 32 * MAXC;  // experts tf_super_topk_few ranks: one a thread
constexpr int MAXK = 32;         // picks a row

// The larger of two (value, index) picks: a higher value, or the lower index on a tie (values are never NaN).
__device__ __forceinline__ bool beats(float v, int i, float bv, int bi) { return v > bv || (v == bv && i < bi); }

}  // namespace

// Picks by sigmoid score plus bias, weighted by the score; pick k's pair ids s * E + expert at (r * TOPK + k) * S + s.
extern "C" __global__ void __launch_bounds__(32)
    tf_super_topk(const float* __restrict__ PART, const float* __restrict__ BIAS, int* __restrict__ IDX,
                  float* __restrict__ WT, int R, float scaling, int E, int SK, int TOPK, int S, int norm) {
    const int64_t r = blockIdx.x;
    const int lane = threadIdx.x;
    float score[MAXC], sel[MAXC];
#pragma unroll
    for (int c = 0; c < MAXC; ++c) {
        const int e = lane + 32 * c;
        score[c] = 0.0f;
        sel[c] = -INFINITY;
        if (e >= E) continue;
        float logit = 0.0f;
        for (int s = 0; s < SK; ++s) logit = __fadd_rn(logit, PART[(s * R + r) * E + e]);
        score[c] = tf_sigmoid(logit);
        const float v = __fadd_rn(score[c], BIAS[e]);
        sel[c] = v != v ? -INFINITY : v;  // a NaN logit is never picked ahead of a number
    }
    float total = 0.0f, mine = 0.0f;
    int mine_id = 0;
    for (int k = 0; k < TOPK; ++k) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
#pragma unroll
        for (int c = 0; c < MAXC; ++c)
            if (lane + 32 * c < E && beats(sel[c], lane + 32 * c, bv, bi)) bv = sel[c], bi = lane + 32 * c;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
            if (beats(ov, oi, bv, bi)) bv = ov, bi = oi;
        }
        float p = 0.0f;
#pragma unroll
        for (int c = 0; c < MAXC; ++c)
            if (bi == lane + 32 * c) p = score[c], sel[c] = -INFINITY;
        p = __shfl_sync(0xffffffffu, p, bi % 32);
        if (lane == k) mine = p, mine_id = bi;  // lane k keeps pick k
        total = __fadd_rn(total, p);
    }
    if (lane >= TOPK) return;
    const float w = norm ? __fmul_rn(__fdiv_rn(mine, __fadd_rn(total, 1e-20f)), scaling) : __fmul_rn(mine, scaling);
    WT[r * TOPK + lane] = w;
    for (int s = 0; s < S; ++s) IDX[(r * TOPK + lane) * S + s] = s * E + mine_id;
}

// tf_super_topk's picks for decode-sized calls, a thread an expert: pick k is the expert k others beat.
extern "C" __global__ void __launch_bounds__(MAXE)
    tf_super_topk_few(const float* __restrict__ PART, const float* __restrict__ BIAS, int* __restrict__ IDX,
                      float* __restrict__ WT, int R, float scaling, int E, int SK, int TOPK, int S, int norm) {
    __shared__ __align__(16) float sel_s[MAXE];  // a block of whole warps: lanes past E hold -inf, never ahead
    __shared__ int pick_s[MAXK];
    __shared__ float prob_s[MAXK];
    const int64_t r = blockIdx.x;
    const int e = threadIdx.x;
    float score = 0.0f, sel = -INFINITY;
    if (e < E) {
        float logit = 0.0f;
        for (int s = 0; s < SK; ++s) logit = __fadd_rn(logit, PART[(s * R + r) * E + e]);
        score = tf_sigmoid(logit);
        const float v = __fadd_rn(score, BIAS[e]);
        sel = v != v ? -INFINITY : v;  // a NaN logit is never picked ahead of a number
    }
    sel_s[e] = sel;
    const int numbers = __syncthreads_count(e < E && sel != -INFINITY);
    if (e < E && sel != -INFINITY) {
        const float4* sel4 = reinterpret_cast<const float4*>(sel_s);
        int rank = 0;  // how many experts beat this one: its pick, if under TOPK
        for (int j = 0; j < blockDim.x && rank < TOPK; j += 16) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const float4 v = sel4[j / 4 + q];
                const int i = j + 4 * q;
                rank += beats(v.x, i, sel, e) + beats(v.y, i + 1, sel, e) + beats(v.z, i + 2, sel, e) +
                        beats(v.w, i + 3, sel, e);
            }
        }
        if (rank < TOPK) pick_s[rank] = e, prob_s[rank] = score;
    }
    if (e == 0)  // past the numbers every pick is expert 0, the lowest id among equal -infs, as tf_super_topk finds
        for (int k = numbers; k < TOPK; ++k) pick_s[k] = 0, prob_s[k] = score;
    __syncthreads();
    if (e >= TOPK) return;
    float total = 0.0f;
    for (int k = 0; k < TOPK; ++k) total = __fadd_rn(total, prob_s[k]);
    const float mine = prob_s[e];
    const float w = norm ? __fmul_rn(__fdiv_rn(mine, __fadd_rn(total, 1e-20f)), scaling) : __fmul_rn(mine, scaling);
    WT[r * TOPK + e] = w;
    for (int s = 0; s < S; ++s) IDX[(r * TOPK + e) * S + s] = s * E + pick_s[e];
}

// Row r's latent sum: each pick's shares added in share order, rounded to bf16, then weighted in pick order.
extern "C" __global__ void __launch_bounds__(256)
    tf_super_combine(const float* __restrict__ P, long long share_stride, long long pair_stride, int S,
                     const float* __restrict__ WT, int K, __nv_bfloat16* __restrict__ OUT, int L) {
    const int64_t r = blockIdx.x;
    const int d = blockIdx.y * 256 + threadIdx.x;
    if (d >= L) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        const float* p = P + (r * K + k) * pair_stride + d;
        float v = p[0];
        for (int s = 1; s < S; ++s) v = __fadd_rn(v, p[s * share_stride]);
        acc = __fmaf_rn(rbf(v), WT[r * K + k], acc);
    }
    OUT[r * L + d] = __float2bfloat16_rn(acc);
}

// The MoE's delta in place: routed rows plus the shared expert's shares, added in share order and rounded first.
extern "C" __global__ void __launch_bounds__(256)
    tf_super_moe_out(__nv_bfloat16* __restrict__ Y, const float* __restrict__ SH, long long share_stride,
                     long long row_stride, int S, int D) {
    const int64_t r = blockIdx.x;
    const int d = blockIdx.y * 256 + threadIdx.x;
    if (d >= D) return;
    const float* p = SH + r * row_stride + d;
    float v = p[0];
    for (int s = 1; s < S; ++s) v = __fadd_rn(v, p[s * share_stride]);
    Y[r * D + d] = __float2bfloat16_rn(__fadd_rn(bf(Y[r * D + d]), rbf(v)));
}

// tf_super_combine over share sums the prompt down kernels rounded already (bf16, a row a pick): 8 columns a thread.
extern "C" __global__ void __launch_bounds__(128)
    tf_super_combine_sums(const __nv_bfloat16* __restrict__ Q, const float* __restrict__ WT, int K,
                          __nv_bfloat16* __restrict__ OUT, int L) {
    const int64_t r = blockIdx.x;
    const int d = (blockIdx.y * 128 + threadIdx.x) * 8;
    if (d >= L) return;
    float acc[8] = {};
    for (int k = 0; k < K; ++k) {
        const uint4 q = *reinterpret_cast<const uint4*>(Q + (r * K + k) * L + d);
        const __nv_bfloat16* v = reinterpret_cast<const __nv_bfloat16*>(&q);
        const float w = WT[r * K + k];
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] = __fmaf_rn(bf(v[i]), w, acc[i]);
    }
    uint4 o;
    __nv_bfloat16* ov = reinterpret_cast<__nv_bfloat16*>(&o);
#pragma unroll
    for (int i = 0; i < 8; ++i) ov[i] = __float2bfloat16_rn(acc[i]);
    *reinterpret_cast<uint4*>(OUT + r * L + d) = o;
}

// tf_super_moe_out over the shared expert's share sums (bf16, rounded by the prompt down kernel): 8 columns a thread.
extern "C" __global__ void __launch_bounds__(256)
    tf_super_moe_out_sums(__nv_bfloat16* __restrict__ Y, const __nv_bfloat16* __restrict__ SH, int D) {
    const int64_t r = blockIdx.x;
    const int d = (blockIdx.y * 256 + threadIdx.x) * 8;
    if (d >= D) return;
    uint4 y = *reinterpret_cast<const uint4*>(Y + r * D + d);
    const uint4 s = *reinterpret_cast<const uint4*>(SH + r * D + d);
    __nv_bfloat16* yv = reinterpret_cast<__nv_bfloat16*>(&y);
    const __nv_bfloat16* sv = reinterpret_cast<const __nv_bfloat16*>(&s);
#pragma unroll
    for (int i = 0; i < 8; ++i) yv[i] = __float2bfloat16_rn(__fadd_rn(bf(yv[i]), bf(sv[i])));
    *reinterpret_cast<uint4*>(Y + r * D + d) = y;
}

// The shared expert's pair ids: pair i runs share i % S (every row's shares, in order).
extern "C" __global__ void __launch_bounds__(256) tf_super_share_ids(int* __restrict__ IDX, int n, int S) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) IDX[i] = i % S;
}
