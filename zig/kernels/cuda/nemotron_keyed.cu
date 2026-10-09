// The MTP head's greedy draw over a row's candidates: rank 0 by (value desc, id asc) and its share of the top K in fp64.

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
