// Flash Next's MTP drafts (cuda_mtp.zig): a score row's best value, its first id and the row's log-sum-exp.

#include <cuda_bf16.h>
#include <math_constants.h>
#include <stdint.h>

namespace fn_mtp {

static __device__ __forceinline__ void better(float& v, int& i, float ov, int oi) {
    if (ov > v || (ov == v && oi < i)) {
        v = ov;
        i = oi;
    }
}

}  // namespace fn_mtp

// Program (row), 1024 threads: out [rows, 4] fp32 holds the best of a row's first V values, its log-sum-exp, its id, 0.
extern "C" __global__ void __launch_bounds__(1024) fn_row_best(const __nv_bfloat16* __restrict__ L, int V, long long ld,
                                                               float* __restrict__ out) {
    using namespace fn_mtp;
    const int r = blockIdx.x, t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const __nv_bfloat16* row = L + static_cast<long long>(r) * ld;
    float v = -CUDART_INF_F;
    int at = 0x7fffffff;
    for (int i = t; i < V; i += blockDim.x) {
        const float x = __bfloat162float(row[i]);
        if (x > v) {
            v = x;
            at = i;
        }
    }
    for (int o = 16; o; o >>= 1) better(v, at, __shfl_xor_sync(0xffffffffu, v, o), __shfl_xor_sync(0xffffffffu, at, o));
    __shared__ float sv[32];
    __shared__ int si[32];
    __shared__ float top;
    __shared__ int top_at;
    if (lane == 0) {
        sv[warp] = v;
        si[warp] = at;
    }
    __syncthreads();
    if (warp == 0) {
        v = lane < (int)(blockDim.x >> 5) ? sv[lane] : -CUDART_INF_F;
        at = lane < (int)(blockDim.x >> 5) ? si[lane] : 0x7fffffff;
        for (int o = 16; o; o >>= 1) better(v, at, __shfl_xor_sync(0xffffffffu, v, o), __shfl_xor_sync(0xffffffffu, at, o));
        if (lane == 0) {
            top = v;
            top_at = at;
        }
    }
    __syncthreads();
    const float m = top;
    float sum = 0.0f;
    for (int i = t; i < V; i += blockDim.x) sum += expf(__bfloat162float(row[i]) - m);
    for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    __shared__ float ss[32];
    if (lane == 0) ss[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < (int)(blockDim.x >> 5) ? ss[lane] : 0.0f;
        for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
        if (lane == 0) {
            out[r * 4 + 0] = m;
            out[r * 4 + 1] = m + logf(sum);
            out[r * 4 + 2] = __int_as_float(top_at);
            out[r * 4 + 3] = 0.0f;
        }
    }
}
