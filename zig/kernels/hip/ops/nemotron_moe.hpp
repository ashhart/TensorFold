#pragma once

#include "common.hpp"

// Nemotron-H's MoE pick, activation and sum: sigmoid scores ranked with a correction bias, relu^2 of an ungated up.

namespace {

constexpr int kNemoMaxExperts = 1024;
constexpr int kNemoMaxTop = 32;

}  // namespace

// A row's top-k by sigmoid(logit) + bias (the lower id on a tie, a NaN never ahead of a number), weighted by its scores
// over their sum times `scaling` (unnormalized when !norm); with `items`, one row's plan: an item a slot.
extern "C" __global__ void __launch_bounds__(32)
    tf_nemo_select(const float* __restrict__ logits, const float* __restrict__ bias, int* __restrict__ pick,
                   float* __restrict__ wts, int* __restrict__ items, int* __restrict__ members, int capacity,
                   int experts, int top_k, float scaling, int norm) {
    __shared__ float sel[kNemoMaxExperts];
    __shared__ float score[kNemoMaxExperts];
    __shared__ float picked[kNemoMaxTop];
    const int lane = threadIdx.x;
    const int row = blockIdx.x;
    const float* l = logits + static_cast<long long>(row) * experts;
    for (int i = lane; i < experts; i += 32) {
        const float s = sigmoid(l[i]);
        const float v = s + bias[i];
        score[i] = s;
        sel[i] = v != v ? -INFINITY : v;
    }
    __syncthreads();
    for (int k = 0; k < top_k; ++k) {
        float best = -INFINITY;
        int idx = 0x7fffffff;
        for (int i = lane; i < experts; i += 32) {
            const float v = sel[i];
            if (v > best || (v == best && i < idx)) {
                best = v;
                idx = i;
            }
        }
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            const float other = __shfl_xor(best, mask, 32);
            const int id = __shfl_xor(idx, mask, 32);
            if (other > best || (other == best && id < idx)) {
                best = other;
                idx = id;
            }
        }
        if (lane == 0) {
            pick[row * top_k + k] = idx;
            picked[k] = score[idx];
            sel[idx] = -INFINITY;
        }
        __syncthreads();
    }
    if (lane == 0) {
        float total = 0.f;
        for (int k = 0; k < top_k; ++k) total += picked[k];
        for (int k = 0; k < top_k; ++k) {
            wts[row * top_k + k] = norm ? (picked[k] / (total + 1e-20f)) * scaling : picked[k] * scaling;
        }
    }
    if (items != nullptr) {
        __syncthreads();
        for (int k = lane; k < capacity; k += 32) {
            const bool real = k < top_k;
            items[3 * k] = real ? pick[k] : 0;
            items[3 * k + 1] = real ? k : 0;
            items[3 * k + 2] = real ? 1 : 0;
            if (real) members[k] = k;
        }
    }
}

constexpr int kNemoRankThreads = 512;

// tf_nemo_select's picks and weights by rank: expert i is the k-th when k experts come before it (a higher value, or an
// equal one at a lower id), counted by `per` lanes over every per-th expert and summed across them.
extern "C" __global__ void __launch_bounds__(kNemoRankThreads)
    tf_nemo_select_rank(const float* __restrict__ logits, const float* __restrict__ bias, int* __restrict__ pick,
                        float* __restrict__ wts, int* __restrict__ items, int* __restrict__ members, int capacity,
                        int experts, int top_k, float scaling, int norm) {
    __shared__ float sel[kNemoMaxExperts];
    __shared__ float score[kNemoMaxExperts];
    __shared__ int top[kNemoMaxTop];
    const int row = blockIdx.x;
    const float* l = logits + static_cast<long long>(row) * experts;
    for (int i = threadIdx.x; i < experts; i += kNemoRankThreads) {
        const float s = sigmoid(l[i]);
        const float v = s + bias[i];
        score[i] = s;
        sel[i] = v != v ? -INFINITY : v;
    }
    __syncthreads();
    int per = 1;
    while (per < 32 && per * 2 * experts <= kNemoRankThreads) per *= 2;
    const int part = threadIdx.x % per;
    const int groups = kNemoRankThreads / per;
    // every lane of a wave reaches the shuffles: an expert past the last counts nothing
    for (int base = 0; base < experts; base += groups) {
        const int i = base + threadIdx.x / per;
        int rank = 0;
        if (i < experts) {
            const float vi = sel[i];
            for (int j = part; j < experts; j += per) {
                const float vj = sel[j];
                rank += (vj > vi) || (vj == vi && j < i);
            }
        }
        for (int m = per / 2; m > 0; m >>= 1) rank += __shfl_xor(rank, m, 32);
        if (i < experts && part == 0 && rank < top_k) top[rank] = i;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.f;
        for (int k = 0; k < top_k; ++k) {
            pick[row * top_k + k] = top[k];
            total += score[top[k]];
        }
        for (int k = 0; k < top_k; ++k) {
            const float p = score[top[k]];
            wts[row * top_k + k] = norm ? (p / (total + 1e-20f)) * scaling : p * scaling;
        }
    }
    if (items != nullptr) {
        __syncthreads();
        for (int k = threadIdx.x; k < capacity; k += kNemoRankThreads) {
            const bool real = k < top_k;
            items[3 * k] = real ? pick[k] : 0;
            items[3 * k + 1] = real ? k : 0;
            items[3 * k + 2] = real ? 1 : 0;
            if (real) members[k] = k;
        }
    }
}

// out = relu(round(up))^2 rounded to `kind` (1 fp16, 2 bf16): the expert's up rounded as the activation, then squared.
extern "C" __global__ void tf_nemo_relu2(const float* __restrict__ up, void* __restrict__ out, int kind, long long n) {
    const long long i = gid();
    if (i >= n) return;
    const float r = rounded(up[i], kind);
    const float u = r != r ? r : fmaxf(r, 0.f);  // a NaN stays NaN, as torch's relu keeps it
    store(out, kind, i, u * u);
}

// A rank's local id of each pick: pick - first for its `own` experts from `first`, else `own` (an item of no work).
extern "C" __global__ void tf_nemo_localize(const int* __restrict__ pick, int* __restrict__ out, int n, int first,
                                            int own) {
    const long long i = gid();
    if (i >= n) return;
    const int local = pick[i] - first;
    out[i] = local >= 0 && local < own ? local : own;
}

// out = sum over slots in order of y * wts, plus the shared expert's row when given, rounded once to `kind` (0 fp32).
extern "C" __global__ void tf_nemo_combine(const float* __restrict__ y, const float* __restrict__ wts,
                                           const float* __restrict__ shared, void* __restrict__ out, int kind,
                                           int slots, int d) {
    const int row = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= d) return;
    float acc = 0.f;
    for (int s = 0; s < slots; ++s) acc += y[(static_cast<long long>(row) * slots + s) * d + c] * wts[row * slots + s];
    if (shared != nullptr) acc += shared[static_cast<long long>(row) * d + c];
    store(out, kind, static_cast<long long>(row) * d + c, acc);
}
