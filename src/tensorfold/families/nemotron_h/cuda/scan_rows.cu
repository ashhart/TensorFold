// Mamba-2 prompt scan: a step's bits never depend on where its chunk starts; not the verify kernel's arithmetic.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int DS = 128;      // states a row
constexpr int TPR = 4;       // threads a value row
constexpr int ROWS = 32;     // value rows a block
constexpr int STEPS = 16;    // chunk rows staged at a time

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__global__ void __launch_bounds__(TPR * ROWS) scan_kernel(
        const __nv_bfloat16* __restrict__ proj, const __nv_bfloat16* __restrict__ xc, float* __restrict__ state,
        const float* __restrict__ a, const float* __restrict__ dsk, const float* __restrict__ dtb,
        __nv_bfloat16* __restrict__ y, int W, int proj_w, int xd, int cd, int dt_off, int dh, int per_group,
        int groups, float lo, float hi) {
    __shared__ float4 bs[STEPS][DS / 4], cs[STEPS][DS / 4];
    __shared__ float xs[STEPS][ROWS], gzs[STEPS][ROWS], dts[STEPS], das[STEPS];
    const int head = blockIdx.x, g = head / per_group;
    const int local = threadIdx.x / TPR, q = threadIdx.x % TPR, row = blockIdx.y * ROWS + local;
    const float ah = a[head], dh_skip = dsk[head], bias = dtb[head];
    constexpr int NJ = DS / TPR / 4;
    float s[4 * NJ];
    float* s0 = state + (static_cast<size_t>(head) * dh + row) * DS;
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
        const float4 t = *reinterpret_cast<const float4*>(s0 + 4 * (TPR * j + q));
        s[4 * j] = t.x; s[4 * j + 1] = t.y; s[4 * j + 2] = t.z; s[4 * j + 3] = t.w;
    }
    const int b_col = xd + g * DS, c_col = xd + groups * DS + g * DS;
    for (int t0 = 0; t0 < W; t0 += STEPS) {
        const int n = min(STEPS, W - t0);
        __syncthreads();
        for (int i = threadIdx.x; i < n * (DS / 4); i += TPR * ROWS) {
            const int st = i / (DS / 4), c = i % (DS / 4);
            const __nv_bfloat16* src = xc + static_cast<size_t>(t0 + st) * cd;
            const uint2 wb = *reinterpret_cast<const uint2*>(src + b_col + 4 * c);
            const uint2 wc = *reinterpret_cast<const uint2*>(src + c_col + 4 * c);
            const __nv_bfloat162* pb = reinterpret_cast<const __nv_bfloat162*>(&wb);
            const __nv_bfloat162* pc = reinterpret_cast<const __nv_bfloat162*>(&wc);
            bs[st][c] = make_float4(__low2float(pb[0]), __high2float(pb[0]), __low2float(pb[1]), __high2float(pb[1]));
            cs[st][c] = make_float4(__low2float(pc[0]), __high2float(pc[0]), __low2float(pc[1]), __high2float(pc[1]));
        }
        for (int i = threadIdx.x; i < n * ROWS; i += TPR * ROWS) {
            const int st = i / ROWS, r = blockIdx.y * ROWS + i % ROWS;
            const size_t t = static_cast<size_t>(t0 + st);
            xs[st][i % ROWS] = __bfloat162float(xc[t * cd + head * dh + r]);
            const float z = __bfloat162float(proj[t * proj_w + head * dh + r]);
            gzs[st][i % ROWS] = bf(z / (1.0f + expf(-z)));
        }
        for (int i = threadIdx.x; i < n; i += TPR * ROWS) {
            const float v = __bfloat162float(proj[static_cast<size_t>(t0 + i) * proj_w + dt_off + head]) + bias;
            const float dt = fminf(fmaxf(fmaxf(v, 0.0f) + logf(1.0f + expf(-fabsf(v))), lo), hi);
            dts[i] = dt;
            das[i] = expf(ah * dt);
        }
        __syncthreads();
        for (int tt = 0; tt < n; ++tt) {
            const float x = xs[tt][local], da = das[tt], xdt = x * dts[tt];
            float m[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int j = 0; j < NJ; ++j) {
                const float4 b = bs[tt][TPR * j + q], c = cs[tt][TPR * j + q];
                s[4 * j] = __fmaf_rn(xdt, b.x, __fmul_rn(s[4 * j], da));
                s[4 * j + 1] = __fmaf_rn(xdt, b.y, __fmul_rn(s[4 * j + 1], da));
                s[4 * j + 2] = __fmaf_rn(xdt, b.z, __fmul_rn(s[4 * j + 2], da));
                s[4 * j + 3] = __fmaf_rn(xdt, b.w, __fmul_rn(s[4 * j + 3], da));
                m[0] = __fmaf_rn(s[4 * j], c.x, m[0]);
                m[1] = __fmaf_rn(s[4 * j + 1], c.y, m[1]);
                m[2] = __fmaf_rn(s[4 * j + 2], c.z, m[2]);
                m[3] = __fmaf_rn(s[4 * j + 3], c.w, m[3]);
            }
            float out = (m[0] + m[1]) + (m[2] + m[3]);
            out = out + __shfl_xor_sync(0xffffffffu, out, 1);
            out = out + __shfl_xor_sync(0xffffffffu, out, 2);
            if (q == 0)
                y[static_cast<size_t>(t0 + tt) * xd + head * dh + row] =
                    __float2bfloat16_rn(gzs[tt][local] * bf(__fmaf_rn(x, dh_skip, out)));
        }
    }
#pragma unroll
    for (int j = 0; j < NJ; ++j)
        *reinterpret_cast<float4*>(s0 + 4 * (TPR * j + q)) = make_float4(s[4 * j], s[4 * j + 1], s[4 * j + 2], s[4 * j + 3]);
}

} // namespace

// A chunk of W rows: y (W, heads * dh) bf16; ``state`` (heads, dh, 128) fp32 in place, then the last row's state.
void scan_rows_cuda(const at::Tensor& proj, const at::Tensor& xc, at::Tensor& state, const at::Tensor& a,
                    const at::Tensor& dsk, const at::Tensor& dtb, at::Tensor& y, int dt_off, int groups, double lo,
                    double hi) {
    const int W = xc.size(0), heads = state.size(0), dh = state.size(1);
    const dim3 grid(heads, dh / ROWS);
    scan_kernel<<<grid, TPR * ROWS, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(proj.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(xc.data_ptr()),
        state.data_ptr<float>(), a.data_ptr<float>(), dsk.data_ptr<float>(), dtb.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), W, proj.size(1), heads * dh, xc.size(1), dt_off, dh,
        heads / groups, groups, static_cast<float>(lo), static_cast<float>(hi));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
