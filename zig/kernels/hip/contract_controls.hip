// The kernel contract's planted controls on HIP: each breaks one gate on purpose, and that gate must go red.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

__device__ __forceinline__ float tf_ctl_x(const uint16_t* x, int64_t i) {
    return __uint_as_float(static_cast<uint32_t>(x[i]) << 16);
}

// Gate 1: a row's maximum by fmaxf, which drops a NaN input, so a row holding NaN reads as its largest number.
extern "C" __global__ void tf_ctl_fmax_max(const uint16_t* X, int K, float* Y) {
    const int row = blockIdx.x, lane = threadIdx.x;
    float m = -INFINITY;
    for (int i = lane; i < K; i += 32) m = fmaxf(m, tf_ctl_x(X, static_cast<int64_t>(row) * K + i));
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor(m, o, 32));
    if (lane == 0) Y[row] = m;
}

// Gate 2: row sums by an atomicAdd an input from threads all over the grid, so the order varies run to run.
extern "C" __global__ void tf_ctl_atomic_sum(const uint16_t* X, int K, float* Y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
    if (i < K) atomicAdd(Y + row, tf_ctl_x(X, static_cast<int64_t>(row) * K + i));
}

// A wave's sum of every 32nd input of a row up to `end`, in lane order.
__device__ __forceinline__ float tf_ctl_rowsum(const uint16_t* X, int K, int row, int end, int lane) {
    float s = 0.0f;
    for (int i = lane; i < end; i += 32) s += tf_ctl_x(X, static_cast<int64_t>(row) * K + i);
    return s;
}

__device__ __forceinline__ float tf_ctl_wave(float s) {
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor(s, o, 32);
    return s;
}

// Gate 2 too: a row sum that writes the sum back into its own input, which its spec says it never does.
extern "C" __global__ void tf_ctl_scribble_sum(uint16_t* X, int K, float* Y) {
    const int row = blockIdx.x, lane = threadIdx.x;
    const float s = tf_ctl_wave(tf_ctl_rowsum(X, K, row, K, lane));
    if (lane == 0) {
        Y[row] = s;
        X[static_cast<int64_t>(row) * K] = static_cast<uint16_t>(__float_as_uint(s) >> 16);
    }
}

// Gate 3: a row sum whose partial tail loses its last input: right whenever K is a whole number of 32s.
extern "C" __global__ void tf_ctl_tail_sum(const uint16_t* X, int K, float* Y) {
    const int row = blockIdx.x, lane = threadIdx.x, whole = K / 32 * 32;
    float s = tf_ctl_rowsum(X, K, row, whole, lane);
    if (lane < K - whole - 1) s += tf_ctl_x(X, static_cast<int64_t>(row) * K + whole + lane);
    s = tf_ctl_wave(s);
    if (lane == 0) Y[row] = s;
}

// Gate 4: a dot product summed in half, while its spec declares fp32 sums.
extern "C" __global__ void tf_ctl_half_dot(const uint16_t* X, int K, float* Y, const uint16_t* W) {
    const int row = blockIdx.x, lane = threadIdx.x;
    __half s = __float2half(0.0f);
    for (int i = lane; i < K; i += 32)
        s = __hadd(s, __float2half(tf_ctl_x(X, static_cast<int64_t>(row) * K + i) * tf_ctl_x(W, i)));
    for (int o = 16; o > 0; o >>= 1) s = __hadd(s, __float2half(__shfl_xor(__half2float(s), o, 32)));
    if (lane == 0) Y[row] = __half2float(s);
}

// Gate 5: bounded to 256 threads a workgroup, which the registry says its launcher gives 512.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_bounded(const float* X, float* Y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    Y[i] = X[i] * 2.0f;
}

// Gate 5 too: 40 KiB of static LDS, which the registry says its launcher adds 32 KiB of dynamic LDS to.
extern "C" __global__ void tf_ctl_big_lds(const float* X, float* Y) {
    __shared__ float fixed[10240];
    extern __shared__ float more[];
    const int t = threadIdx.x;
    for (int i = t; i < 10240; i += blockDim.x) fixed[i] = X[i];
    for (int i = t; i < 8192; i += blockDim.x) more[i] = X[i] * 3.0f;
    __syncthreads();
    Y[t] = fixed[(t * 37) % 10240] + more[(t * 11) % 8192];
}
