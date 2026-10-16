// The kernel contract's planted controls on CUDA: each breaks one gate on purpose, and that gate must go red.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>

__device__ __forceinline__ float tf_ctl_x(const __nv_bfloat16* x, int64_t i) { return __bfloat162float(x[i]); }

// Gate 1: a row's maximum by fmaxf, which drops a NaN input, so a row holding NaN reads as its largest number.
extern "C" __global__ void tf_ctl_fmax_max(const __nv_bfloat16* X, int K, float* Y) {
    const int row = blockIdx.x, lane = threadIdx.x;
    float m = -INFINITY;
    for (int i = lane; i < K; i += 32) m = fmaxf(m, tf_ctl_x(X, static_cast<int64_t>(row) * K + i));
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (lane == 0) Y[row] = m;
}

// Gate 2: row sums by an atomicAdd an input from threads all over the grid, so the order varies run to run.
extern "C" __global__ void tf_ctl_atomic_sum(const __nv_bfloat16* X, int K, float* Y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
    if (i < K) atomicAdd(Y + row, tf_ctl_x(X, static_cast<int64_t>(row) * K + i));
}

// A warp's sum of every 32nd input of a row, in lane order then a butterfly.
__device__ __forceinline__ float tf_ctl_rowsum(const __nv_bfloat16* X, int K, int row, int end, int lane) {
    float s = 0.0f;
    for (int i = lane; i < end; i += 32) s += tf_ctl_x(X, static_cast<int64_t>(row) * K + i);
    return s;
}

__device__ __forceinline__ float tf_ctl_warp(float s) {
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    return s;
}

// Gate 2 too: a row sum that writes the sum back into its own input, which its spec says it never does.
extern "C" __global__ void tf_ctl_scribble_sum(__nv_bfloat16* X, int K, float* Y) {
    const int row = blockIdx.x, lane = threadIdx.x;
    const float s = tf_ctl_warp(tf_ctl_rowsum(X, K, row, K, lane));
    __syncwarp();
    if (lane == 0) {
        Y[row] = s;
        X[static_cast<int64_t>(row) * K] = __float2bfloat16(s);
    }
}

// Gate 3: a row sum whose partial tail loses its last input: right whenever K is a whole number of 32s.
extern "C" __global__ void tf_ctl_tail_sum(const __nv_bfloat16* X, int K, float* Y) {
    const int row = blockIdx.x, lane = threadIdx.x, whole = K / 32 * 32;
    float s = tf_ctl_rowsum(X, K, row, whole, lane);
    if (lane < K - whole - 1) s += tf_ctl_x(X, static_cast<int64_t>(row) * K + whole + lane);
    s = tf_ctl_warp(s);
    if (lane == 0) Y[row] = s;
}

// Gate 4: a dot product summed in half, while its spec declares fp32 sums.
extern "C" __global__ void tf_ctl_half_dot(const __nv_bfloat16* X, int K, float* Y, const __nv_bfloat16* W) {
    const int row = blockIdx.x, lane = threadIdx.x;
    __half s = __float2half(0.0f);
    for (int i = lane; i < K; i += 32) s = __hadd(s, __float2half(tf_ctl_x(X, static_cast<int64_t>(row) * K + i) * tf_ctl_x(W, i)));
    for (int o = 16; o > 0; o >>= 1) s = __hadd(s, __shfl_xor_sync(0xffffffffu, s, o));
    if (lane == 0) Y[row] = __half2float(s);
}

// Gate 5: bounded to 256 threads a block, which the registry says its launcher gives 512.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_bounded(const float* X, float* Y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    Y[i] = X[i] * 2.0f;
}

// Gate 5 too: 160 live fp32 values a thread and no bounds, launched at 1,024 threads: more registers than a block has.
extern "C" __global__ void tf_ctl_heavy(const float* X, float* Y) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    float v[160];
#pragma unroll
    for (int j = 0; j < 160; ++j) v[j] = X[t * 160 + j];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
#pragma unroll
        for (int j = 0; j < 160; ++j) v[j] = fmaf(v[j], v[(j + 1) % 160], v[(j + 37) % 160]);
    }
    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < 160; ++j) s += v[j] * v[159 - j];
    Y[t] = s;
}
