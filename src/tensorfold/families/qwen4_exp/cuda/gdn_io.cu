// Flash Next's DeltaNet front and back ends, with gdn.cu's chain-kernel operations and lane order, so the same bits.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int DK = 128, DV = 128, TAPS = 4;

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__device__ __forceinline__ float warp_sum(float x) {
    for (int o = 16; o; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    return x;
}

__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ float softplusf_(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

// conv + SiLU of channel c for row r: taps from the row's window, [conv state (3 rows) | window rows]
__device__ __forceinline__ float conv_act(const __nv_bfloat16* P, const __nv_bfloat16* cs, const __nv_bfloat16* cw,
                                          const int* win, int r, int c, int C, int PW) {
    float acc = 0.0f;
#pragma unroll
    for (int tap = 0; tap < TAPS; ++tap) {
        const int src = win[r * TAPS + tap];
        const float x = src < TAPS - 1 ? __bfloat162float(cs[src * C + c])
                                       : __bfloat162float(P[(size_t)(src - (TAPS - 1)) * PW + c]);
        acc = acc + __bfloat162float(cw[c * TAPS + tap]) * x;
    }
    return bf(acc / (1.0f + expf(-acc)));
}

// Block (row, head): heads < NK L2-norm q (times DK^-0.5) and k as the chain kernel's warps 0 and 1; others v, g, beta.
template <int NK, int NV>
__global__ void __launch_bounds__(128) front_kernel(
        const __nv_bfloat16* __restrict__ P, const long long* __restrict__ conv_ptrs, const int* __restrict__ sid,
        const int* __restrict__ win, const __nv_bfloat16* __restrict__ cw, const float* __restrict__ a_log,
        const float* __restrict__ dt_bias, float* __restrict__ q, float* __restrict__ k,
        __nv_bfloat16* __restrict__ v, float* __restrict__ g, float* __restrict__ beta) {
    constexpr int C = 2 * NK * DK + NV * DV, PW = C + NV * DV + 2 * NV;
    const int r = blockIdx.x, head = blockIdx.y, t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const auto* cs = reinterpret_cast<const __nv_bfloat16*>(conv_ptrs[sid[r]]);
    __shared__ float xs[2][DK];
    if (head < NK) {
        xs[0][t] = conv_act(P, cs, cw, win, r, head * DK + t, C, PW);
        xs[1][t] = conv_act(P, cs, cw, win, r, NK * DK + head * DK + t, C, PW);
        __syncthreads();
        if (warp < 2) {
            float v4[4], ss = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) { v4[i] = xs[warp][lane * 4 + i]; ss = ss + v4[i] * v4[i]; }
            ss = warp_sum(ss);
            float inv = 1.0f / sqrtf(ss + 1e-6f);
            if (warp == 0) inv = inv * (1.0f / sqrtf((float)DK));
            float* out = (warp == 0 ? q : k) + ((size_t)r * NK + head) * DK + lane * 4;
#pragma unroll
            for (int i = 0; i < 4; ++i) out[i] = v4[i] * inv;
        }
        return;
    }
    const int hv = head - NK;
    v[((size_t)r * NV + hv) * DV + t] = __float2bfloat16_rn(conv_act(P, cs, cw, win, r, 2 * NK * DK + hv * DV + t,
                                                                     C, PW));
    if (t == 0) {
        const float b = __bfloat162float(P[(size_t)r * PW + C + NV * DV + hv]);
        const float a = __bfloat162float(P[(size_t)r * PW + C + NV * DV + NV + hv]);
        g[r * NV + hv] = expf(-expf(a_log[hv]) * softplusf_(a + dt_bias[hv]));
        beta[r * NV + hv] = bf(sigmoidf_(b));
    }
}

// Block (row, value head): gated RMSNorm on warp 0's sum as the chain kernel, then bf16 out and 32-channel group sums.
template <int NK, int NV>
__global__ void __launch_bounds__(128) back_kernel(
        const __nv_bfloat16* __restrict__ y, const __nv_bfloat16* __restrict__ P,
        const __nv_bfloat16* __restrict__ norm_w, float eps, __nv_bfloat16* __restrict__ out,
        float* __restrict__ xs) {
    constexpr int C = 2 * NK * DK + NV * DV, PW = C + NV * DV + 2 * NV;
    const int r = blockIdx.x, hv = blockIdx.y, t = threadIdx.x, warp = t >> 5, lane = t & 31;
    __shared__ float ys[DV];
    __shared__ float rinv;
    ys[t] = __bfloat162float(y[((size_t)r * NV + hv) * DV + t]);
    __syncthreads();
    if (warp == 0) {
        float ss = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) { const float yy = ys[lane * 4 + i]; ss = ss + yy * yy; }
        ss = warp_sum(ss);
        if (lane == 0) rinv = 1.0f / sqrtf(ss / (float)DV + eps);
    }
    __syncthreads();
    const float yn = bf(bf(ys[t] * rinv) * __bfloat162float(norm_w[t]));
    const float z = __bfloat162float(P[(size_t)r * PW + C + hv * DV + t]);
    const float o = bf(yn * sigmoidf_(z));
    out[(size_t)r * NV * DV + hv * DV + t] = __float2bfloat16_rn(o);
    const float gs = warp_sum(o);
    if (lane == 0) xs[(size_t)r * (NV * DV / 32) + hv * (DV / 32) + warp] = gs;
}

}  // namespace

void gdn_front_cuda(const at::Tensor& P, const at::Tensor& conv_ptrs, const at::Tensor& sid, const at::Tensor& win,
                    const at::Tensor& cw, const at::Tensor& a_log, const at::Tensor& dt_bias, at::Tensor& q,
                    at::Tensor& k, at::Tensor& v, at::Tensor& g, at::Tensor& beta) {
    auto stream = at::cuda::getCurrentCUDAStream();
    const int rows = (int)win.size(0), nv = (int)a_log.numel();
    auto launch = [&](auto kernel, int nk) {
        kernel<<<dim3(rows, nk + nv), 128, 0, stream>>>(
            (const __nv_bfloat16*)P.data_ptr(), (const long long*)conv_ptrs.data_ptr(), sid.data_ptr<int>(),
            win.data_ptr<int>(), (const __nv_bfloat16*)cw.data_ptr(), a_log.data_ptr<float>(),
            dt_bias.data_ptr<float>(), q.data_ptr<float>(), k.data_ptr<float>(), (__nv_bfloat16*)v.data_ptr(),
            g.data_ptr<float>(), beta.data_ptr<float>());
    };
    if (nv == 48) launch(front_kernel<16, 48>, 16);
    else if (nv == 24) launch(front_kernel<8, 24>, 8);
    else TORCH_CHECK(false, "gdn front: 48 or 24 value heads");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gdn_back_cuda(const at::Tensor& y, const at::Tensor& P, const at::Tensor& norm_w, double eps, at::Tensor& out,
                   at::Tensor& xs) {
    auto stream = at::cuda::getCurrentCUDAStream();
    const int rows = (int)y.size(0), nv = (int)y.size(1);
    auto launch = [&](auto kernel) {
        kernel<<<dim3(rows, nv), 128, 0, stream>>>(
            (const __nv_bfloat16*)y.data_ptr(), (const __nv_bfloat16*)P.data_ptr(),
            (const __nv_bfloat16*)norm_w.data_ptr(), (float)eps, (__nv_bfloat16*)out.data_ptr(), xs.data_ptr<float>());
    };
    if (nv == 48) launch(back_kernel<16, 48>);
    else if (nv == 24) launch(back_kernel<8, 24>);
    else TORCH_CHECK(false, "gdn back: 48 or 24 value heads");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
