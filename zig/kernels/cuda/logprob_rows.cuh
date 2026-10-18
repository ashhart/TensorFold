// logprobs (lanes/logprob.zig): log normalizer, the pick's logit, the k best ids and logits (logit desc, id asc), rank.
#pragma once

#include <cuda_bf16.h>
#include <math.h>

namespace tf_logprob {

constexpr unsigned TG = 1024;  // one block a row, as tf_logprob_rows launches it
constexpr unsigned NSG = TG / 32;
constexpr unsigned FULL = 0xFFFFFFFFu;
constexpr unsigned LP_TOP = 20;
constexpr unsigned LP_WORDS = 3 + 2 * LP_TOP;

// (m, s): a running max and the sum of exp(x - m) under it, merged with another part's
__device__ __forceinline__ void lse_merge(float& m, float& s, float m2, float s2) {
    if (m2 == -INFINITY) return;
    if (m == -INFINITY) {
        m = m2;
        s = s2;
    } else if (m2 > m) {
        s = s * expf(m - m2) + s2;
        m = m2;
    } else {
        s += s2 * expf(m2 - m);
    }
}

__device__ __forceinline__ bool better(float v, unsigned i, float bv, unsigned bi) { return v > bv || (v == bv && i < bi); }

// One row: a fixed reduction tree (lane tree, then warp 0 over the warps), so a row's words never depend on timing.
__device__ void logprob_row(const __nv_bfloat16* L, unsigned V, unsigned pick, unsigned k, bool rank, unsigned* out) {
    __shared__ float part_m[NSG], part_s[NSG], part_v[NSG];
    __shared__ unsigned part_i[NSG];
    __shared__ float prev_v;
    __shared__ unsigned prev_i;
    const unsigned t = threadIdx.x, lane = t & 31u, warp = t >> 5;
    float m = -INFINITY, s = 0.0f;
    for (unsigned i = t; i < V; i += TG) {
        const float v = __bfloat162float(L[i]);
        if (v == -INFINITY) continue;
        if (v > m) {
            s = s * expf(m - v) + 1.0f;
            m = v;
        } else {
            s += expf(v - m);
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float m2 = __shfl_down_sync(FULL, m, off), s2 = __shfl_down_sync(FULL, s, off);
        lse_merge(m, s, m2, s2);
    }
    if (lane == 0) {
        part_m[warp] = m;
        part_s[warp] = s;
    }
    if (t == 0) {
        prev_v = INFINITY;
        prev_i = 0;
    }
    __syncthreads();
    if (warp == 0) {
        m = part_m[lane];
        s = part_s[lane];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float m2 = __shfl_down_sync(FULL, m, off), s2 = __shfl_down_sync(FULL, s, off);
            lse_merge(m, s, m2, s2);
        }
        if (lane == 0) {
            out[0] = __float_as_uint(m + logf(s));
            out[1] = __float_as_uint(pick < V ? __bfloat162float(L[pick]) : NAN);
        }
    }
    if (rank) {
        // vLLM's rank: the entries whose logit is at or above the pick's, the pick itself included
        const float pk = pick < V ? __bfloat162float(L[pick]) : INFINITY;
        unsigned above = 0;
        for (unsigned i = t; i < V; i += TG) above += __bfloat162float(L[i]) >= pk ? 1u : 0u;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) above += __shfl_down_sync(FULL, above, off);
        if (lane == 0) part_i[warp] = above;
        __syncthreads();
        if (warp == 0) {
            above = part_i[lane];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) above += __shfl_down_sync(FULL, above, off);
            if (lane == 0) out[2 + 2 * LP_TOP] = above;
        }
        __syncthreads();
    } else if (t == 0) {
        out[2 + 2 * LP_TOP] = 0;
    }
    for (unsigned r = 0; r < k; ++r) {
        // the best entry after the last one taken: a strict total order, so the result is exact
        const float pv = prev_v;
        const unsigned pi = prev_i;
        float bv = -INFINITY;
        unsigned bi = 0xFFFFFFFFu;
        for (unsigned i = t; i < V; i += TG) {
            const float v = __bfloat162float(L[i]);
            if ((v < pv || (v == pv && i > pi)) && better(v, i, bv, bi)) {
                bv = v;
                bi = i;
            }
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float v2 = __shfl_down_sync(FULL, bv, off);
            const unsigned i2 = __shfl_down_sync(FULL, bi, off);
            if (better(v2, i2, bv, bi)) {
                bv = v2;
                bi = i2;
            }
        }
        if (lane == 0) {
            part_v[warp] = bv;
            part_i[warp] = bi;
        }
        __syncthreads();
        if (warp == 0) {
            bv = part_v[lane];
            bi = part_i[lane];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                const float v2 = __shfl_down_sync(FULL, bv, off);
                const unsigned i2 = __shfl_down_sync(FULL, bi, off);
                if (better(v2, i2, bv, bi)) {
                    bv = v2;
                    bi = i2;
                }
            }
            if (lane == 0) {
                out[2 + r] = bi;
                out[2 + LP_TOP + r] = __float_as_uint(bv);
                prev_v = bv;
                prev_i = bi;
            }
        }
        __syncthreads();
    }
}

// A row's k best alone, in two stages: TOPK_SLICE entries a block, then a block over the slices' k best.
constexpr unsigned TOPK_TG = 256;
constexpr unsigned TOPK_SLICE = 4096;

// The best of v[0..n) after (pv, pi), ids from `id` or base + i: logprob_row's order, reduced by a fixed tree.
__device__ void topk_best(const float* v, const unsigned* id, unsigned base, unsigned n, float pv, unsigned pi, float& out_v, unsigned& out_i) {
    __shared__ float part_v[TOPK_TG / 32];
    __shared__ unsigned part_i[TOPK_TG / 32];
    const unsigned t = threadIdx.x, lane = t & 31u, warp = t >> 5;
    float bv = -INFINITY;
    unsigned bi = 0xFFFFFFFFu;
    for (unsigned j = t; j < n; j += TOPK_TG) {
        const float x = v[j];
        const unsigned i = id ? id[j] : base + j;
        if ((x < pv || (x == pv && i > pi)) && better(x, i, bv, bi)) {
            bv = x;
            bi = i;
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float v2 = __shfl_down_sync(FULL, bv, off);
        const unsigned i2 = __shfl_down_sync(FULL, bi, off);
        if (better(v2, i2, bv, bi)) {
            bv = v2;
            bi = i2;
        }
    }
    if (lane == 0) {
        part_v[warp] = bv;
        part_i[warp] = bi;
    }
    __syncthreads();
    if (warp == 0) {
        bv = lane < TOPK_TG / 32 ? part_v[lane] : -INFINITY;
        bi = lane < TOPK_TG / 32 ? part_i[lane] : 0xFFFFFFFFu;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float v2 = __shfl_down_sync(FULL, bv, off);
            const unsigned i2 = __shfl_down_sync(FULL, bi, off);
            if (better(v2, i2, bv, bi)) {
                bv = v2;
                bi = i2;
            }
        }
        if (lane == 0) {
            part_v[0] = bv;
            part_i[0] = bi;
        }
    }
    __syncthreads();
    out_v = part_v[0];
    out_i = part_i[0];
    __syncthreads();
}

// Stage one: block (slice, row) writes its slice's k best to PV/PI[row][slice][0..k).
__device__ void topk_slice(const __nv_bfloat16* L, unsigned V, unsigned k, unsigned slices, float* PV, unsigned* PI) {
    __shared__ float sv[TOPK_SLICE];
    const unsigned row = blockIdx.y, base = blockIdx.x * TOPK_SLICE;
    const unsigned n = V - base < TOPK_SLICE ? V - base : TOPK_SLICE;
    const __nv_bfloat16* x = L + static_cast<size_t>(row) * V + base;
    for (unsigned j = threadIdx.x; j < n; j += TOPK_TG) sv[j] = __bfloat162float(x[j]);
    __syncthreads();
    const size_t at = (static_cast<size_t>(row) * slices + blockIdx.x) * k;
    float pv = INFINITY;
    unsigned pi = 0;
    for (unsigned r = 0; r < k; ++r) {
        topk_best(sv, nullptr, base, n, pv, pi, pv, pi);
        if (threadIdx.x == 0) {
            PV[at + r] = pv;
            PI[at + r] = pi;
        }
    }
}

// Stage two: row blockIdx.x's k best of its slices' candidates, in logprob_row's words (no normalizer, pick or rank).
__device__ void topk_merge(const float* PV, const unsigned* PI, unsigned n, unsigned k, unsigned* out) {
    const float* v = PV + static_cast<size_t>(blockIdx.x) * n;
    const unsigned* id = PI + static_cast<size_t>(blockIdx.x) * n;
    float pv = INFINITY;
    unsigned pi = 0;
    for (unsigned r = 0; r < k; ++r) {
        topk_best(v, id, 0, n, pv, pi, pv, pi);
        if (threadIdx.x == 0) {
            out[2 + r] = pi;
            out[2 + LP_TOP + r] = __float_as_uint(pv);
        }
    }
    if (threadIdx.x == 0) {
        out[0] = 0;
        out[1] = 0;
        out[2 + 2 * LP_TOP] = 0;
    }
}

}  // namespace tf_logprob
