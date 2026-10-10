// logprobs (lanes/logprob.zig): log normalizer, the pick's logit, then the k best ids and logits (logit desc, id asc).
#pragma once

#include <cuda_bf16.h>
#include <math.h>

namespace tf_logprob {

constexpr unsigned TG = 1024;  // one block a row, as tf_logprob_rows launches it
constexpr unsigned NSG = TG / 32;
constexpr unsigned FULL = 0xFFFFFFFFu;
constexpr unsigned LP_TOP = 20;
constexpr unsigned LP_WORDS = 2 + 2 * LP_TOP;

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
__device__ void logprob_row(const __nv_bfloat16* L, unsigned V, unsigned pick, unsigned k, unsigned* out) {
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

}  // namespace tf_logprob
