// Living Weights' gated attention backward.

#define TF_ATTN_BACK(name) tf_attn_##name
#include "attn_back.cuh"


// ---- shared by the learners' products and nudges ------------------------------------------------------------------

// C [M, N] (rows ldc apart) (+)= the split products' partials [splits, M, N], summed in split order.
extern "C" __global__ void tf_sum_parts(const float* parts, int splits, int M, int N, float* C, int ldc, int accumulate) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)M * N) return;
    float s = 0;
    for (int z = 0; z < splits; z++) s += parts[(size_t)z * M * N + i];
    float* out = C + (size_t)(i / N) * ldc + i % N;
    *out = accumulate ? *out + s : s;
}

// y[i] += x[i] for n floats: a nudge onto its row of a layer's output.
extern "C" __global__ void tf_axpy(float* y, const float* x, int n) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) y[i] += x[i];
}
