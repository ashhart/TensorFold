#pragma once

#include "common.hpp"
#include "draw.hpp"

// Tensor-parallel forward ops: residual add of an fp32 sum, expert-share remap and masks; MTP token probability.

namespace {

__device__ __forceinline__ long long tp_gid() { return static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; }

constexpr int kProbThreads = 256;

}  // namespace

// out = round(float(x) + y): the residual of a rank-summed fp32 share.
extern "C" __global__ void tf_add_wide(const void* x, const float* y, void* out, int kind, long long n) {
    long long i = tp_gid();
    if (i >= n) return;
    if (kind == 1) {
        float v = __half2float(static_cast<const __half*>(x)[i]) + y[i];
        static_cast<__half*>(out)[i] = __float2half_rn(v);
    } else {
        float v = bf16_to_float(static_cast<const uint16_t*>(x)[i]) + y[i];
        static_cast<uint16_t*>(out)[i] = float_to_bf16(v);
    }
}

// out[i] = the rank's local id of expert pick[i], or `skip` (an id past the rank's experts) when another rank holds it.
extern "C" __global__ void tf_moe_localize(const int* pick, const int* remap, int* out, int n, int skip) {
    int i = static_cast<int>(tp_gid());
    if (i >= n) return;
    int local = remap[pick[i]];
    out[i] = local >= 0 ? local : skip;
}

// An item of the skip expert does no work.
extern "C" __global__ void tf_moe_foreign_items(int* items, int count, int skip) {
    int i = static_cast<int>(tp_gid());
    if (i < count && items[i * 3] == skip) items[i * 3 + 2] = 0;
}

// A pair another rank's expert answers has zero rows (selected, not multiplied: the buffer holds anything).
extern "C" __global__ void tf_moe_zero_foreign(float* y, const int* picks, int skip, long long pairs, int d) {
    long long i = tp_gid();
    if (i >= pairs * d) return;
    if (picks[i / d] == skip) y[i] = 0.f;
}

// out[r] = softmax(row r)[id[r]], or of the row's largest value when `id` is null; one block a row, a NaN in gives NaN.
extern "C" __global__ void tf_token_prob(const uint16_t* logits, int kind, int n, const int* id, float* out) {
    const uint16_t* row = logits + static_cast<long long>(blockIdx.x) * n;
    __shared__ float shared[kProbThreads];
    float top = -INFINITY;
    bool nan = false;
    for (int i = threadIdx.x; i < n; i += kProbThreads) {
        float v = widen(row[i], kind);
        nan |= v != v;
        top = fmaxf(top, v);
    }
    shared[threadIdx.x] = nan ? NAN : top;
    __syncthreads();
    for (int stride = kProbThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            float a = shared[threadIdx.x], b = shared[threadIdx.x + stride];
            shared[threadIdx.x] = (a != a || b != b) ? NAN : fmaxf(a, b);
        }
        __syncthreads();
    }
    const float peak = shared[0];
    __syncthreads();
    float total = 0.f;
    for (int i = threadIdx.x; i < n; i += kProbThreads) total += expf(widen(row[i], kind) - peak);
    shared[threadIdx.x] = total;
    __syncthreads();
    for (int stride = kProbThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const float chosen = id ? expf(widen(row[id[blockIdx.x]], kind) - peak) : 1.f;
        out[blockIdx.x] = chosen / shared[0];
    }
}
