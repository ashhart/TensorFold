// The 27B's own small kernels: batched state copies, the drafter's taps and its fp32 RMSNorm.
#include <cuda_bf16.h>
#include <stdint.h>

struct TfCopy {
    unsigned long long dst, src, bytes;  // 16-byte aligned pointers, bytes a multiple of 16
};

// One block an item, its threads striding through the item's 16-byte words.
extern "C" __global__ void __launch_bounds__(256) tf_copy_batch(const TfCopy* __restrict__ items) {
    const TfCopy it = items[blockIdx.x];
    const uint4* src = reinterpret_cast<const uint4*>(it.src);
    uint4* dst = reinterpret_cast<uint4*>(it.dst);
    const unsigned long long n = it.bytes / 16;
    for (unsigned long long i = threadIdx.x + static_cast<unsigned long long>(blockIdx.y) * blockDim.x; i < n;
         i += static_cast<unsigned long long>(blockDim.x) * gridDim.y)
        dst[i] = src[i];
}

// forward.tree_forward's taps: (x.float() + pending.float()).to(bfloat16), row r into columns [col, col + D) of OUT.
extern "C" __global__ void __launch_bounds__(256) tf_tap(const __nv_bfloat16* __restrict__ x,
                                                         const __nv_bfloat16* __restrict__ pending,
                                                         __nv_bfloat16* __restrict__ out, int D, int out_stride, int col) {
    const long long row = blockIdx.x;
    for (int i = threadIdx.x + blockIdx.y * blockDim.x; i < D; i += blockDim.x * gridDim.y)
        out[row * out_stride + col + i] =
            __float2bfloat16_rn(__bfloat162float(x[row * D + i]) + __bfloat162float(pending[row * D + i]));
}

// The drafter's RMSNorm in fp32: y = bf16(x * rsqrt(mean(x^2) + eps) * w), a 256-thread block a row.
extern "C" __global__ void __launch_bounds__(256) tf_rms_norm(const __nv_bfloat16* __restrict__ x, int x_stride,
                                                              const __nv_bfloat16* __restrict__ w,
                                                              __nv_bfloat16* __restrict__ y, int D, float eps) {
    __shared__ float part[8];
    const long long row = blockIdx.x;
    const __nv_bfloat16* xr = x + row * x_stride;
    float s = 0.0f;
    for (int i = threadIdx.x; i < D; i += 256) {
        const float v = __bfloat162float(xr[i]);
        s += v * v;
    }
    for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xFFFFFFFFu, s, off);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = s;
    __syncthreads();
    float total = 0.0f;
    for (int k = 0; k < 8; ++k) total += part[k];
    const float inv = rsqrtf(total / D + eps);
    for (int i = threadIdx.x; i < D; i += 256)
        y[row * D + i] = __float2bfloat16_rn(__bfloat162float(xr[i]) * inv * __bfloat162float(w[i]));
}

// The drafter's inverse frequencies as dflash2 makes them on the GPU: 1 / theta ** (2i / dim), fp32 powf then a divide.
extern "C" __global__ void tf_inv_freq(float* __restrict__ inv, float theta, int half, int dim) {
    const int i = threadIdx.x;
    if (i < half) inv[i] = 1.0f / powf(theta, static_cast<float>(i) * 2.0f / static_cast<float>(dim));
}

// dflash2._rotary and add_taps_streams' tables: cos and sin of float(position) * inv_freq, fp32, a block a row.
extern "C" __global__ void tf_rotary(const int* __restrict__ pos, const float* __restrict__ inv,
                                     float* __restrict__ cos_out, float* __restrict__ sin_out, int half) {
    const int row = blockIdx.x;
    const float p = static_cast<float>(pos[row]);
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        const float phase = p * inv[i];
        cos_out[row * half + i] = cosf(phase);
        sin_out[row * half + i] = sinf(phase);
    }
}
