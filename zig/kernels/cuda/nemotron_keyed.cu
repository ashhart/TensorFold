// Rank MTP candidates by value descending, id ascending; compute the top-K share in fp64.

#include <math.h>
#include <stdint.h>

// One warp a row of C <= 32 candidates: OUT the rank-0 id; PROB, if given, 1 / sum over the top K of e^(v - v0).
extern "C" __global__ void __launch_bounds__(32) tf_nemo_keyed_greedy(const float* __restrict__ VALS, const long long* __restrict__ IDS,
                                                                        int* __restrict__ OUT, float* __restrict__ PROB, int C, int K) {
    constexpr unsigned full = 0xffffffffu;
    const int64_t r = blockIdx.x;
    const int lane = threadIdx.x;
    const bool ok = lane < C;
    const double v = ok ? static_cast<double>(VALS[r * C + lane]) : -INFINITY;
    const long long id = ok ? IDS[r * C + lane] : (1LL << 40);
    int rank = 0;
    for (int j = 0; j < 32; ++j) {
        const double vj = __shfl_sync(full, v, j);
        const long long idj = __shfl_sync(full, id, j);
        if (j < C && (vj > v || (vj == v && idj < id))) ++rank;
    }
    const unsigned first = __ballot_sync(full, ok && rank == 0);
    const long long tok = __shfl_sync(full, id, first ? __ffs(first) - 1 : 0);
    if (lane == 0) OUT[r] = static_cast<int>(tok);
    if (PROB == nullptr) return;
    const bool kept = ok && rank < K;
    double top = kept ? v : -INFINITY;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) top = fmax(top, __shfl_xor_sync(full, top, off));
    double e = kept ? exp(v - top) : 0.0;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) e = e + __shfl_xor_sync(full, e, off);
    if (lane == 0) PROB[r] = static_cast<float>(1.0 / e);
}

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "nemotron_mtp_policy.h"

namespace {
__device__ uint32_t mtp_key(float value) {
    const uint32_t bits = __float_as_uint(value), magnitude = bits & 0x7fffffffu;
    if (magnitude > 0x7f800000u) return UINT32_MAX;
    return bits & 0x80000000u ? ~bits : bits | 0x80000000u;
}
}

// One CTA reproduces topk's radix threshold and column-order compaction, then the original top-20 warp draw.
extern "C" __global__ void __launch_bounds__(256) tf_nemo_mtp_greedy(
    const __nv_bfloat16* L, const long long* IDS, int* OUT, float* PROB, int N) {
    __shared__ uint32_t hist[256], threshold, remaining, above_total;
    __shared__ uint32_t wg[8], we[8], above_start, tied_start;
    __shared__ float values[32];
    __shared__ long long ids[32];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5, C = min(N, 28);
    if (t == 0) { threshold = 0; remaining = C; above_start = 0; tied_start = 0; }
    __syncthreads();
    for (int shift = 24; shift >= 0; shift -= 8) {
        hist[t] = 0;
        __syncthreads();
        const uint32_t mask = shift == 24 ? 0u : (UINT32_MAX << (shift + 8));
        for (int col = t; col < N; col += 256) {
            const uint32_t key = mtp_key(__bfloat162float(L[col]));
            if ((key & mask) == threshold) atomicAdd(&hist[(key >> shift) & 255], 1u);
        }
        __syncthreads();
        if (t == 0) {
            uint32_t before = 0;
            for (int digit = 255; digit >= 0; --digit) {
                if (before + hist[digit] >= remaining) {
                    threshold |= static_cast<uint32_t>(digit) << shift;
                    remaining -= before;
                    break;
                }
                before += hist[digit];
            }
        }
        __syncthreads();
    }
    // There are C-remaining strictly larger keys; threshold ties follow those values in column order.
    if (t == 0) above_total = C - remaining;
    __syncthreads();
    for (int begin = 0; begin < N; begin += 256) {
        const int col = begin + t;
        const float value = col < N ? __bfloat162float(L[col]) : 0.0f;
        const uint32_t key = mtp_key(value);
        const bool above = col < N && key > threshold, tied = col < N && key == threshold;
        const unsigned ga = __ballot_sync(0xffffffffu, above), eq = __ballot_sync(0xffffffffu, tied);
        if (lane == 0) { wg[warp] = __popc(ga); we[warp] = __popc(eq); }
        __syncthreads();
        const unsigned prior = lane == 0 ? 0u : (1u << lane) - 1u;
        uint32_t ai = above_start + __popc(ga & prior), ei = tied_start + __popc(eq & prior);
        uint32_t ac = 0, ec = 0;
        for (int w = 0; w < 8; ++w) {
            if (w < warp) { ai += wg[w]; ei += we[w]; }
            ac += wg[w]; ec += we[w];
        }
        if (above) { values[ai] = value; ids[ai] = IDS ? IDS[col] : col; }
        if (tied && above_total + ei < static_cast<unsigned>(C)) {
            values[above_total + ei] = value; ids[above_total + ei] = IDS ? IDS[col] : col;
        }
        __syncthreads();
        if (t == 0) { above_start += ac; tied_start += ec; }
        __syncthreads();
    }
    if (t >= 32) return;
    const bool ok = lane < C;
    const double v = ok ? static_cast<double>(values[lane]) : -INFINITY;
    const long long id = ok ? ids[lane] : (1LL << 40);
    int rank = 0;
    for (int j = 0; j < 32; ++j) {
        const double vj = __shfl_sync(0xffffffffu, v, j);
        const long long idj = __shfl_sync(0xffffffffu, id, j);
        if (j < C && (vj > v || (vj == v && idj < id))) ++rank;
    }
    const unsigned first = __ballot_sync(0xffffffffu, ok && rank == 0);
    const long long tok = __shfl_sync(0xffffffffu, id, first ? __ffs(first) - 1 : 0);
    if (lane == 0) OUT[0] = static_cast<int>(tok);
    const bool kept = ok && rank < 20;
    double top = kept ? v : -INFINITY;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) top = fmax(top, __shfl_xor_sync(0xffffffffu, top, off));
    double e = kept ? exp(v - top) : 0.0;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) e = e + __shfl_xor_sync(0xffffffffu, e, off);
    if (lane == 0 && PROB) PROB[0] = static_cast<float>(1.0 / e);
}

// No target state is read: initialize loop metadata from the head's already-uploaded begin row.
extern "C" __global__ void __launch_bounds__(1) tf_nemo_mtp_start(TfMtpState* s, const uint32_t* meta) {
    s->run = 1.0; s->j = 1; s->n = 0; s->levels = 0;
    s->meta[0] = meta[1] - 1; s->meta[1] = meta[1];
}

// Save even the rejected final level, as the host loop does; only n controls the target verify width.
extern "C" __global__ void __launch_bounds__(1) tf_nemo_mtp_control(
    TfMtpState* s, uint32_t* ids, float* probs, cudaGraphConditionalHandle handle) {
    const uint32_t j = s->j;
    ids[j] = s->ids[j - 1] = s->token;
    probs[j - 1] = s->probs[j - 1] = s->probability;
    s->levels = j;
    const bool more = tf_mtp_advance(s->policy, s->run, j, s->n, s->probability);
    if (more) { ++s->j; ++s->meta[0]; ++s->meta[1]; }
    cudaGraphSetConditional(handle, more ? 1u : 0u);
}

extern "C" __global__ void __launch_bounds__(1) tf_ctl_nemo_mtp_feed(TfMtpState* s, const float* probabilities) {
    s->token = s->j * 17; s->probability = probabilities[s->j - 1];
}
