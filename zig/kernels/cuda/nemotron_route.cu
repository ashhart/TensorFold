// Nemotron's router: fp32 logits in K slices (one fma chain a slice), then each row's top-k and the shared slots.

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

#include "glue_math.cuh"

using namespace tf_glue;

// Block (16-row tile, 16-expert block, K slice s): PART[s][row][e] = x[row] . w[e] over the slice, in index order.
extern "C" __global__ void __launch_bounds__(256) tf_nemo_router(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ W,
                                                                   float* __restrict__ PART, int R, int D, int E, int SK) {
    extern __shared__ uint32_t tile[];  // 16 input rows, then 16 expert rows, each per / 2 words plus one of padding
    const int per = D / SK, words = per / 2, stride = words + 1;
    const int row0 = blockIdx.x * 16, e0 = blockIdx.y * 16, k0 = blockIdx.z * per;
    uint32_t* xs = tile;
    uint32_t* ws = tile + 16 * stride;
    for (int c = threadIdx.x; c < 16 * words; c += 256) {
        const int r = c / words, w = c % words;
        const uint32_t* xr = reinterpret_cast<const uint32_t*>(X + static_cast<int64_t>(row0 + r) * D + k0);
        const uint32_t* wr = reinterpret_cast<const uint32_t*>(W + static_cast<int64_t>(e0 + r) * D + k0);
        xs[r * stride + w] = row0 + r < R ? xr[w] : 0u;
        ws[r * stride + w] = e0 + r < E ? wr[w] : 0u;
    }
    __syncthreads();
    const int rr = threadIdx.x / 16, ee = threadIdx.x % 16;
    const uint32_t* xp = xs + rr * stride;
    const uint32_t* wp = ws + ee * stride;
    float acc = 0.0f;
    for (int w = 0; w < words; ++w) {
        const uint32_t a = xp[w], b = wp[w];
        acc = __fmaf_rn(__uint_as_float(a << 16), __uint_as_float(b << 16), acc);
        acc = __fmaf_rn(__uint_as_float(a & 0xFFFF0000u), __uint_as_float(b & 0xFFFF0000u), acc);
    }
    if (row0 + rr < R && e0 + ee < E) PART[(static_cast<int64_t>(blockIdx.z) * R + row0 + rr) * E + e0 + ee] = acc;
}

namespace {

constexpr int MAXC = 8;  // experts a lane holds: up to 256 experts

// The larger of two (value, index) picks: a higher value, or the lower index on a tie (values are never NaN).
__device__ __forceinline__ bool beats(float v, int i, float bv, int bi) { return v > bv || (v == bv && i < bi); }

}  // namespace

// One warp a row: logits = slices summed in order, scores = sigmoid, top_k picks by score + bias, then the shared slots.
extern "C" __global__ void __launch_bounds__(32) tf_nemo_topk(const float* __restrict__ PART, const float* __restrict__ BIAS,
                                                               int* __restrict__ IDX, float* __restrict__ WT, int R, float scaling, int E,
                                                               int SK, int TOPK, int NS, int norm) {
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
    float total = 0.0f;
    float probs[MAXC];
    int ids[MAXC];
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
        if (k < MAXC) probs[k] = p, ids[k] = bi;
        total = __fadd_rn(total, p);
    }
    if (lane != 0) return;
    for (int k = 0; k < NS; ++k) {
        float w = 1.0f;
        int id = E + (k - TOPK);
        if (k < TOPK) {
            id = ids[k];
            w = norm ? __fmul_rn(__fdiv_rn(probs[k], __fadd_rn(total, 1e-20f)), scaling) : __fmul_rn(probs[k], scaling);
        }
        IDX[r * NS + k] = id;
        WT[r * NS + k] = w;
    }
}
