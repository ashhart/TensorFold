// ERNIE 5.5's router: two routers' logits as two sigmoid scores summed, the top k of score + bias, weights scaled.

#include <math.h>

#include "common/glue_math.cuh"
#include "common/topk_rows.cuh"

namespace ernie55r {

constexpr int E = 512, T = 64, K = 32;

enum Fault { none, one_sigmoid, biased_weights };

// Row blockIdx.x: logits [2 E] ls apart (the first router's row, then the second's); picks and weights k a row.
template <Fault F>
__device__ __forceinline__ void route(const float* __restrict__ logits, long long ls, const float* __restrict__ bias,
                                      const float* __restrict__ scale, int* __restrict__ ids, float* __restrict__ wts,
                                      int k) {
    __shared__ __align__(16) float sel[E];
    __shared__ float score[E];
    __shared__ int pick[K];
    const long long row = blockIdx.x;
    const float* l = logits + row * ls;
    for (int e = threadIdx.x; e < E; e += T) {
        // sigmoid(l0) + sigmoid(l1) (the control: one sigmoid of l0 + l1), then + bias; a NaN ranks as -inf
        const float s = F == one_sigmoid ? tf_glue::tf_sigmoid(__fadd_rn(l[e], l[E + e]))
                                         : __fadd_rn(tf_glue::tf_sigmoid(l[e]), tf_glue::tf_sigmoid(l[E + e]));
        const float v = __fadd_rn(s, bias[e]);
        score[e] = s;
        sel[e] = v == v ? v : -INFINITY;
    }
    __syncthreads();
    tf_topk::select_row<T>(sel, E, k, pick);  // the k best, ties to the lower id, in ascending ids
    __syncthreads();
    if (threadIdx.x != 0) return;
    // the picks' scores (the control: score + bias) summed in id order, over max(sum, 1e-20), times the expert's scale
    float total = 0.0f;
    for (int j = 0; j < k; ++j) total = __fadd_rn(total, F == biased_weights ? sel[pick[j]] : score[pick[j]]);
    const float d = fmaxf(total, 1e-20f);
    for (int j = 0; j < k; ++j) {
        const int e = pick[j];
        ids[row * k + j] = e;
        wts[row * k + j] = __fmul_rn(__fdiv_rn(F == biased_weights ? sel[e] : score[e], d), scale[e]);
    }
}

}  // namespace ernie55r

extern "C" __global__ void __launch_bounds__(64)
    tf_ernie55_route(const float* __restrict__ logits, long long ls, const float* __restrict__ bias,
                     const float* __restrict__ scale, int* __restrict__ ids, float* __restrict__ wts, int k) {
    ernie55r::route<ernie55r::none>(logits, ls, bias, scale, ids, wts, k);
}

// The contract's planted controls: one sigmoid of the two logits summed, and weights from score + bias.
extern "C" __global__ void __launch_bounds__(64)
    tf_ctl_ernie55_route_one_sigmoid(const float* __restrict__ logits, long long ls, const float* __restrict__ bias,
                                     const float* __restrict__ scale, int* __restrict__ ids, float* __restrict__ wts,
                                     int k) {
    ernie55r::route<ernie55r::one_sigmoid>(logits, ls, bias, scale, ids, wts, k);
}

extern "C" __global__ void __launch_bounds__(64)
    tf_ctl_ernie55_route_biased_weights(const float* __restrict__ logits, long long ls,
                                        const float* __restrict__ bias, const float* __restrict__ scale,
                                        int* __restrict__ ids, float* __restrict__ wts, int k) {
    ernie55r::route<ernie55r::biased_weights>(logits, ls, bias, scale, ids, wts, k);
}
