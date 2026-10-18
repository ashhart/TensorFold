// Softmax top-k routing with a sigmoid-gated shared expert on fp32 logits, as tensorfold/cuda/moe.py routes.

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ float bf(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float rbf(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

constexpr int TILE = 16;  // rows and experts a block covers
constexpr int KC = 256;   // inputs a shared-memory chunk holds

}  // namespace

// OUT[r][e] = fp32 x[r] . w[e] in index order (an fma chain a pair), grid (R / 16, E / 16); skips rows >= P[2].
extern "C" __global__ void __launch_bounds__(256) tf_moe_router(const __nv_bfloat16* __restrict__ X, int x_stride,
                                                                  const __nv_bfloat16* __restrict__ W, float* __restrict__ OUT,
                                                                  int R, int D, int E, const int* __restrict__ P) {
    if (P != nullptr) R = min(R, P[2]);
    if (static_cast<int>(blockIdx.x) * TILE >= R) return;
    __shared__ __nv_bfloat16 xs[TILE][KC + 2];
    __shared__ __nv_bfloat16 ws[TILE][KC + 2];
    const int row0 = blockIdx.x * TILE, e0 = blockIdx.y * TILE;
    const int rr = threadIdx.x / TILE, ee = threadIdx.x % TILE;
    float acc = 0.0f;
    for (int k0 = 0; k0 < D; k0 += KC) {
        __syncthreads();
        for (int c = threadIdx.x; c < TILE * KC; c += 256) {
            const int r = c / KC, k = c % KC;
            const bool kin = k0 + k < D;
            xs[r][k] = kin && row0 + r < R ? X[static_cast<int64_t>(row0 + r) * x_stride + k0 + k] : __float2bfloat16_rn(0.0f);
            ws[r][k] = kin && e0 + r < E ? W[static_cast<int64_t>(e0 + r) * D + k0 + k] : __float2bfloat16_rn(0.0f);
        }
        __syncthreads();
        const int kn = min(KC, D - k0);
        for (int k = 0; k < kn; ++k) acc = __fmaf_rn(bf(xs[rr][k]), bf(ws[ee][k]), acc);
    }
    if (row0 + rr < R && e0 + ee < E) OUT[static_cast<int64_t>(row0 + rr) * E + e0 + ee] = acc;
}

namespace {

constexpr int MAXC = 8;  // experts a lane holds: up to 256 routed experts

// The larger of two (value, index) picks: a higher value, or the lower index on a tie (values are never NaN).
__device__ __forceinline__ bool beats(float v, int i, float bv, int bi) { return v > bv || (v == bv && i < bi); }

}  // namespace

// Row r by a warp: TOPK experts by logit (ties to lower id), bf16(exp(l - top) / sum), shared slot; r < P[2].
extern "C" __global__ void __launch_bounds__(32) tf_moe_topk(const float* __restrict__ L, int* __restrict__ PICK,
                                                              float* __restrict__ WT, int NE, int TOPK,
                                                              const int* __restrict__ P) {
    const int64_t r = blockIdx.x;
    if (P != nullptr && r >= P[2]) return;
    const int lane = threadIdx.x;
    const int NL = NE + 1, SLOTS = TOPK + 1;  // L[r][NE], the shared gate's logit, weighs slot TOPK by bf16(sigmoid(bf16(it)))
    float v[MAXC];
#pragma unroll
    for (int c = 0; c < MAXC; ++c) {
        const int e = lane + 32 * c;
        v[c] = e < NE ? L[r * NL + e] : -INFINITY;
    }
    float top = -INFINITY;
#pragma unroll
    for (int c = 0; c < MAXC; ++c) top = fmaxf(top, v[c]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, off));
    float total = 0.0f;
    float ex[MAXC];
    int id[MAXC];
    for (int k = 0; k < TOPK; ++k) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
#pragma unroll
        for (int c = 0; c < MAXC; ++c)
            if (lane + 32 * c < NE && beats(v[c], lane + 32 * c, bv, bi)) bv = v[c], bi = lane + 32 * c;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
            if (beats(ov, oi, bv, bi)) bv = ov, bi = oi;
        }
        const float e = expf(bv - top);
        total += e;
        if (lane == k % 32) ex[k / 32] = e, id[k / 32] = bi;  // slot k lives in lane k % 32 (TOPK <= 32)
#pragma unroll
        for (int c = 0; c < MAXC; ++c)
            if (lane + 32 * c == bi) v[c] = -INFINITY;
    }
    if (lane < TOPK) {
        PICK[r * SLOTS + lane] = id[0];
        WT[r * SLOTS + lane] = rbf(ex[0] / total);
    }
    if (lane == 0) {
        const float sg = rbf(L[r * NL + NE]);
        PICK[r * SLOTS + TOPK] = NE;
        WT[r * SLOTS + TOPK] = rbf(1.0f / (1.0f + expf(-sg)));
    }
}

// OUT[r][d] = bf16(sum over slots in order of fma(w[r][s], y[r][s][d])), y fp32 or bf16; rows >= P[2] skipped.
extern "C" __global__ void __launch_bounds__(256) tf_moe_combine(const void* __restrict__ Y, int y_f32, const float* __restrict__ WT,
                                                                   __nv_bfloat16* __restrict__ OUT, int S, int D,
                                                                   const int* __restrict__ P) {
    const int64_t r = blockIdx.x;
    if (P != nullptr && r >= P[2]) return;
    for (int d = blockIdx.y * 256 + threadIdx.x; d < D; d += gridDim.y * 256) {
        float acc = 0.0f;
        for (int s = 0; s < S; ++s) {
            const int64_t at = (r * S + s) * D + d;
            const float y = y_f32 ? static_cast<const float*>(Y)[at] : bf(static_cast<const __nv_bfloat16*>(Y)[at]);
            acc = __fmaf_rn(WT[r * S + s], y, acc);
        }
        OUT[r * D + d] = __float2bfloat16_rn(acc);
    }
}
