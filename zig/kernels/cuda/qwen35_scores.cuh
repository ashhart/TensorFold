// FP32 logits to lanes/logprob.zig's disjoint 43-word rows, with a fixed tree and lower-id top ties.
#pragma once
#include <math.h>
#include <stdint.h>
template <bool omit_tail>
__device__ void q35_score_stats_row(const float *__restrict__ L, unsigned V, const unsigned *__restrict__ PICK,
                       unsigned K, unsigned *__restrict__ OUT) {
    constexpr unsigned TOP = 20, WORDS = 43, T = 256;
    __shared__ float val[T], sum[T], prior;
    __shared__ unsigned id[T], nanflag[T], above[T], prior_id;
    const unsigned t = threadIdx.x, r = blockIdx.x;
    L += (size_t)r * V;
    OUT += (size_t)r * WORDS;
    const unsigned count = V - (omit_tail ? 1u : 0u);
    if (t < WORDS) OUT[t] = 0;
    float m = -INFINITY;
    unsigned bad = 0;
    for (unsigned i = t; i < count; i += T) { bad |= isnan(L[i]); m = fmaxf(m, L[i]); }
    val[t] = m; nanflag[t] = bad;
    __syncthreads();
    for (unsigned d = T / 2; d; d >>= 1) {
        if (t < d) { val[t] = fmaxf(val[t], val[t+d]); nanflag[t] |= nanflag[t+d]; }
        __syncthreads();
    }
    m = val[0];
    float s = 0;
    unsigned rank = 0;
    const float pick = PICK[r] < V ? L[PICK[r]] : NAN;
    for (unsigned i = t; i < count; i += T) { s += isfinite(m) ? expf(L[i] - m) : 0; rank += L[i] >= pick; }
    sum[t] = s; above[t] = rank;
    __syncthreads();
    for (unsigned d = T / 2; d; d >>= 1) {
        if (t < d) { sum[t] += sum[t+d]; above[t] += above[t+d]; }
        __syncthreads();
    }
    if (t == 0) {
        OUT[0] = __float_as_uint(nanflag[0] ? NAN : isinf(m) ? m : m + logf(sum[0]));
        OUT[1] = __float_as_uint(pick); OUT[42] = above[0];
        prior = INFINITY; prior_id = 0;
    }
    __syncthreads();
    for (unsigned k = 0; k < K; ++k) {
        float best = -INFINITY; unsigned bi = 0xffffffffu;
        for (unsigned i = t; i < count; i += T) {
            const float v = L[i];
            const bool eligible = k == 0 || v < prior || (v == prior && i > prior_id);
            if (eligible && (v > best || (v == best && i < bi))) { best = v; bi = i; }
        }
        val[t] = best; id[t] = bi;
        __syncthreads();
        for (unsigned d = T / 2; d; d >>= 1) {
            if (t < d && (val[t+d] > val[t] || (val[t+d] == val[t] && id[t+d] < id[t]))) {
                val[t] = val[t+d]; id[t] = id[t+d];
            }
            __syncthreads();
        }
        if (t == 0) { OUT[2+k] = id[0]; OUT[22+k] = __float_as_uint(val[0]); prior = val[0]; prior_id = id[0]; }
        __syncthreads();
    }
}

extern "C" __global__ void __launch_bounds__(256)
    tf_q35_score_stats(const float *L, unsigned V, const unsigned *PICK, unsigned K, unsigned *OUT) {
    q35_score_stats_row<false>(L, V, PICK, K, OUT);
}
#ifdef TF_Q35_SCORE_CONTROL
#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(256)
    tf_ctl_q35_score_tail(const float *L, unsigned V, const unsigned *PICK, unsigned K, unsigned *OUT) {
    q35_score_stats_row<true>(L, V, PICK, K, OUT);
}
// Gate 4: FP16 sums violate the scoring head's FP32 accumulator, without changing its output ABI.
extern "C" __global__ void __launch_bounds__(256)
    tf_ctl_q35_dense_f32_half(const __nv_bfloat16 *X, const __nv_bfloat16 *W, float *Y, int N, int K) {
    if (threadIdx.x != 0) return;
    const size_t row = blockIdx.y, col = blockIdx.x;
    __half sum = __float2half_rn(0.0f);
    for (int i = 0; i < K; ++i)
        sum = __hadd(sum, __hmul(__float2half_rn(__bfloat162float(X[row*K+i])),
                                __float2half_rn(__bfloat162float(W[col*K+i]))));
    Y[row*N+col] = __half2float(sum);
}
#endif
