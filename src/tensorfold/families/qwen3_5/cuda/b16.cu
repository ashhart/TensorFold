// A plain weight linear for the Qwen3.8 engines: y = x @ W + bias with W fp16 or bf16 as stored.
//
// Row invariance (the verify path's contract) holds by construction here, not by tuning: one warp owns one
// output element. Lane l sums k = 8l .. 8l+7, then 8l+256 .. 8l+263 and so on, in fp32 and in that order, and
// the 32 lane sums are added by a fixed xor butterfly (every lane ends with the same value), so an element's
// bits are a function of its own row and column alone. No cuBLAS, no cross-row reduction, nothing shape
// dependent. This covers the handful of tensors an EXL3 checkpoint leaves unquantized (in_proj_a/b); the
// trellis weights go through src/tensorfold/cuda/exl3.
//
// The first version gave each output one thread walking all of K alone: 96 launches over 0.5 MB weights took
// 15.8 ms of a 27B decode step. A warp reading 16 bytes a lane streams the same weights in ~0.3 ms.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

template <typename T>
__device__ __forceinline__ float f2f(T v);
template <>
__device__ __forceinline__ float f2f<half>(half v) { return __half2float(v); }
template <>
__device__ __forceinline__ float f2f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T>
__device__ __forceinline__ T f_from(float v);
template <>
__device__ __forceinline__ half f_from<half>(float v) { return __float2half_rn(v); }
template <>
__device__ __forceinline__ __nv_bfloat16 f_from<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }

// Eight elements (16 bytes) as fp32.
template <typename T>
__device__ __forceinline__ void ld8(const T* __restrict__ p, float (&v)[8]) {
    const uint4 u = __ldg(reinterpret_cast<const uint4*>(p));
    const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if constexpr (std::is_same_v<T, half>) {
            const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&w[i]));
            v[2 * i] = f.x; v[2 * i + 1] = f.y;
        } else {
            const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w[i]));
            v[2 * i] = f.x; v[2 * i + 1] = f.y;
        }
    }
}

constexpr int WARPS = 4;

// Grid (ceil(N / WARPS), M); warp w of a block owns output column blockIdx.x * WARPS + w of row blockIdx.y.
// K % 8 == 0 and 16-byte aligned rows (checked by the host) for the vector loads; the tail past the last
// full 256-element stride is summed per lane in the same fixed order.
template <typename T>
__global__ void __launch_bounds__(WARPS * 32) b16_kernel(const T* __restrict__ x, const T* __restrict__ w,
                                                         const T* __restrict__ bias, T* __restrict__ y, int K, int N) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.y;
    const int col = blockIdx.x * WARPS + (threadIdx.x >> 5);
    if (col >= N) return;
    const T* xr = x + (size_t)row * K;
    const T* wr = w + (size_t)col * K;
    float acc = 0.f;
    for (int k = 8 * lane; k + 8 <= K; k += 256) {
        float a[8], b[8];
        ld8(xr + k, a);
        ld8(wr + k, b);
#pragma unroll
        for (int i = 0; i < 8; ++i) acc = fmaf(a[i], b[i], acc);
    }
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
    if (lane == 0) {
        if (bias != nullptr) acc += f2f(__ldg(bias + col));
        y[(size_t)row * N + col] = f_from<T>(acc);
    }
}

}  // namespace

at::Tensor b16_linear(const at::Tensor& x, const at::Tensor& w, const at::Tensor& bias) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "x must be contiguous CUDA");
    TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.dim() == 2, "w must be a contiguous 2-d CUDA tensor");
    TORCH_CHECK(x.scalar_type() == w.scalar_type(), "x and w must share a dtype (cast x first)");
    const bool is_half = x.scalar_type() == at::kHalf;
    TORCH_CHECK(is_half || x.scalar_type() == at::kBFloat16, "only fp16 and bf16 weights are supported");
    const int M = (int)x.size(0), K = (int)x.size(1), N = (int)w.size(0);
    TORCH_CHECK((int)w.size(1) == K, "w's K must be x's K");
    TORCH_CHECK(K % 8 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
                    reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
                "K must be a multiple of 8 and x, w 16-byte aligned");
    at::cuda::CUDAGuard guard(x.device());
    auto y = at::empty({M, N}, x.options());
    const void* bp = nullptr;
    if (bias.defined() && bias.numel()) {
        TORCH_CHECK(bias.is_cuda() && bias.is_contiguous() && bias.numel() == N, "bias must be [N]");
        bp = bias.data_ptr();
    }
    const dim3 block(WARPS * 32), grid((unsigned)((N + WARPS - 1) / WARPS), (unsigned)M);
    auto stream = at::cuda::getCurrentCUDAStream();
    if (is_half) {
        b16_kernel<half><<<grid, block, 0, stream>>>(reinterpret_cast<const half*>(x.data_ptr()),
                                                     reinterpret_cast<const half*>(w.data_ptr()),
                                                     reinterpret_cast<const half*>(bp),
                                                     reinterpret_cast<half*>(y.data_ptr()), K, N);
    } else {
        b16_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(bp),
            reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), K, N);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
