#pragma once

// A decode step's expert picks; decode.hip includes it after its router.

namespace {

constexpr int kSelectThreads = 512;
constexpr int kSelectMax = 1024;

__device__ __forceinline__ float round_bf16_f(float v) { return round_kind(v, 2); }

// act.hip's moe_select with picks by rank: expert i is in the top k when fewer than k beat it; `items` writes the plan.
__device__ void select_body(const float* logits, int* pick, float* wts, int* items, int* members, int capacity,
                            int experts, int top_k) {
    __shared__ float v[kSelectMax];
    __shared__ float top_v[32];
    __shared__ int top_i[32];
    const int row = blockIdx.x;
    const int slots = top_k + 1;
    const float* l = logits + static_cast<long long>(row) * (experts + 1);
    for (int i = threadIdx.x; i < experts; i += kSelectThreads) v[i] = l[i];
    __syncthreads();
    for (int i = threadIdx.x; i < experts; i += kSelectThreads) {
        const float vi = v[i];
        int rank = 0;
#pragma unroll 16
        for (int j = 0; j < experts; ++j) {
            const float vj = v[j];
            rank += (vj > vi) || (vj == vi && j < i);
        }
        if (rank < top_k) {
            top_v[rank] = vi;
            top_i[rank] = i;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const float top = top_v[0];
        float ex[32];
        float total = 0.f;
        for (int k = 0; k < top_k; ++k) {
            ex[k] = expf(top_v[k] - top);
            total += ex[k];
        }
        for (int k = 0; k < top_k; ++k) {
            pick[row * slots + k] = top_i[k];
            wts[row * slots + k] = round_bf16_f(ex[k] / total);
        }
        const float gate = round_bf16_f(l[experts]);
        pick[row * slots + top_k] = experts;
        wts[row * slots + top_k] = round_bf16_f(1.f / (1.f + expf(-gate)));
    }
    if (items != nullptr) {
        __syncthreads();
        for (int k = threadIdx.x; k < capacity; k += kSelectThreads) {
            const bool real = k < slots;
            items[3 * k] = real ? pick[k] : 0;
            items[3 * k + 1] = real ? k : 0;
            items[3 * k + 2] = real ? 1 : 0;
            if (real) members[k] = k;
        }
    }
}

}  // namespace
