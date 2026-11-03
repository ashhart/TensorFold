// The kernel contract's planted controls on HIP: each breaks one gate on purpose, and that gate must go red.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

#include "quant/dense_tiles.hpp"

__device__ __forceinline__ float tf_ctl_x(const uint16_t* x, int64_t i) {
    return __uint_as_float(static_cast<uint32_t>(x[i]) << 16);
}

#include "columns_controls.hpp"

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

// Not a control: tf_embed_dense's fp32 gather as it was before its table kind, the baseline its fp32 rows must equal.
__device__ __forceinline__ uint16_t tf_base_bf16(float f) {
    if (f != f) return 0x7fc0;
    uint32_t u = __float_as_uint(f);
    return static_cast<uint16_t>((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}

extern "C" __global__ void tf_base_embed_dense(const float* table, const int* ids, int n, int k, void* out, int out_kind) {
    long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<long long>(n) * k) return;
    int r = static_cast<int>(i / k), c = static_cast<int>(i % k);
    float v = table[static_cast<long long>(ids[r]) * k + c];
    if (out_kind == 0) static_cast<float*>(out)[i] = v;
    else if (out_kind == 1) static_cast<__half*>(out)[i] = __float2half_rn(v);
    else static_cast<uint16_t*>(out)[i] = tf_base_bf16(v);
}

// Gate 4: the WMMA dense tile adding each 16-deep step in BF16 (v_wmma_bf16), while its spec declares fp32 sums.
struct TfCtlBf16Step {
#if TF_DEVICE_WMMA_GFX12
    __device__ static tf::rocm::f32x8 mma(tf::rocm::bf16x8 a, tf::rocm::bf16x8 b, tf::rocm::f32x8 c) {
        tf::rocm::bf16x8 s;
        for (int e = 0; e < 8; ++e) s[e] = static_cast<__bf16>(c[e]);
        s = __builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32_gfx12(a, b, s);
        tf::rocm::f32x8 d;
        for (int e = 0; e < 8; ++e) d[e] = static_cast<float>(s[e]);
        return d;
    }
#endif
};

extern "C" __global__ void __launch_bounds__(tf::rocm::W12Small::Threads) tf_ctl_wmma12_bf16_sum(tf::rocm::Affine a) {
    using namespace tf::rocm;
    wmma12_gemm_tile<W12Small, IdentityAct<DotBF16>, GemmOut<DotBF16>, TfCtlBf16Step>(a);
}

__device__ __forceinline__ uint16_t tf_ctl_bf16(float f) {
    const uint32_t u = __float_as_uint(f);
    return f != f ? 0x7fc0 : static_cast<uint16_t>((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}

__device__ __forceinline__ float tf_ctl_r16(float f) { return __uint_as_float(static_cast<uint32_t>(tf_ctl_bf16(f)) << 16); }

// Gate 4: tf_js_layer_norm with both sums rounded to bf16 at every add, while its spec declares fp32 sums.
extern "C" __global__ void tf_ctl_js_norm_bf16(const uint16_t* x, const uint16_t* w, const uint16_t* b, uint16_t* y,
                                               int dim, float eps) {
    if (threadIdx.x != 0) return;
    const int64_t at = static_cast<int64_t>(blockIdx.x) * dim;
    float s = 0.0f;
    for (int i = 0; i < dim; ++i) s = tf_ctl_r16(s + tf_ctl_x(x, at + i));
    const float mean = s / static_cast<float>(dim);
    float q = 0.0f;
    for (int i = 0; i < dim; ++i) q = tf_ctl_r16(q + (tf_ctl_x(x, at + i) - mean) * (tf_ctl_x(x, at + i) - mean));
    const float inv = 1.0f / sqrtf(q / static_cast<float>(dim) + eps);
    for (int i = 0; i < dim; ++i) y[at + i] = tf_ctl_bf16((tf_ctl_x(x, at + i) - mean) * inv * tf_ctl_x(w, i) + tf_ctl_x(b, i));
}

// Gate 1: tf_js_softmax_rows that skips non-finite scores as masked, so a row holding NaN or infinity stays finite.
extern "C" __global__ void tf_ctl_js_softmax_safe(float* s, int cols) {
    if (threadIdx.x != 0) return;
    float* row = s + static_cast<int64_t>(blockIdx.x) * cols;
    float m = -INFINITY;
    for (int i = 0; i < cols; ++i) if (isfinite(row[i])) m = fmaxf(m, row[i]);
    float t = 0.0f;
    for (int i = 0; i < cols; ++i) t += isfinite(row[i]) ? expf(row[i] - m) : 0.0f;
    for (int i = 0; i < cols; ++i) row[i] = isfinite(row[i]) ? expf(row[i] - m) / t : 0.0f;
}

// Gate 3: tf_js_combine without its three intermediate roundings, so its logits miss the head's BF16 points.
extern "C" __global__ void tf_ctl_js_combine_unrounded(const float* prior, const float* cosine, const uint16_t* res, int n,
                                                       float joint_scale, float gate, uint16_t* logits) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    logits[j] = tf_ctl_bf16(prior[j] + gate * (joint_scale * cosine[j] + tf_ctl_x(res, j)));
}
