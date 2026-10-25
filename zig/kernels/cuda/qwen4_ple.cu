#include <cuda_bf16.h>
#include "glue_math.cuh"
using namespace tf_glue;
__device__ __forceinline__ float bsig(float x) { const float e = tf_exp(-fabsf(x)); return rbf(__fdiv_rn(x < 0 ? e : 1.0f, __fadd_rn(1.0f, e))); }
__device__ __forceinline__ float bsilu(float x) { const float e = tf_exp(-fabsf(x)); return rbf(__fdiv_rn(x < 0 ? __fmul_rn(x, e) : x, __fadd_rn(1.0f, e))); }
__device__ __forceinline__ float block_sum(float v, float* scratch) {
    v = warp_sum(v);
    const int warps = blockDim.x >> 5;
    if ((threadIdx.x & 31) == 0) scratch[threadIdx.x >> 5] = v;
    __syncthreads();
    float total = 0.0f;
    for (int w = 0; w < warps; ++w) total = __fadd_rn(total, scratch[w]);
    __syncthreads();
    return total;
}
extern "C" __global__ void __launch_bounds__(512) qwen4_ple_gate(const __nv_bfloat16* __restrict__ KEYS, const __nv_bfloat16* __restrict__ VALS,
                                                              const __nv_bfloat16* __restrict__ H, const float* __restrict__ NK,
                                                              const float* __restrict__ NQ, __nv_bfloat16* __restrict__ GATED,
                                                              float* __restrict__ PSS, float eps, int D, int S) {
    __shared__ float scratch[32];
    const int r = blockIdx.x;
    const float fd = static_cast<float>(D);
    for (int s = 0; s < S; ++s) {
        const long long row = static_cast<long long>(r) * S * D + static_cast<long long>(s) * D;
        float kss = 0.0f, qss = 0.0f;
        for (int d = threadIdx.x; d < D; d += blockDim.x) {
            const float kv = bf(KEYS[static_cast<long long>(r) * (S + 1) * D + s * D + d]), hv = bf(H[row + d]);
            kss = __fmaf_rn(kv, kv, kss);
            qss = __fmaf_rn(hv, hv, qss);
        }
        kss = block_sum(kss, scratch);
        qss = block_sum(qss, scratch);
        const float krinv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(kss, fd), eps)));
        const float qrinv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(qss, fd), eps)));
        float dot = 0.0f;
        for (int d = threadIdx.x; d < D; d += blockDim.x) {
            const float kn = rbf(__fmul_rn(__fmul_rn(bf(KEYS[static_cast<long long>(r) * (S + 1) * D + s * D + d]), krinv), NK[s * D + d]));
            const float qn = rbf(__fmul_rn(__fmul_rn(bf(H[row + d]), qrinv), NQ[s * D + d]));
            dot = __fadd_rn(dot, rbf(__fmul_rn(kn, qn)));
        }
        dot = block_sum(dot, scratch);
        float gate = rbf(__fdiv_rn(rbf(dot), __fsqrt_rn(fd)));
        const float mag = __fsqrt_rn(fmaxf(fabsf(gate), 1e-6f));
        gate = rbf(gate > 0.0f ? mag : (gate < 0.0f ? -mag : 0.0f));
        const float sg = bsig(gate);
        float gss = 0.0f;
        for (int d = threadIdx.x; d < D; d += blockDim.x) {
            const float gv = rbf(__fmul_rn(sg, bf(VALS[static_cast<long long>(r) * (S + 1) * D + d])));
            GATED[row + d] = __float2bfloat16_rn(gv);
            gss = __fmaf_rn(gv, gv, gss);
        }
        gss = block_sum(gss, scratch);
        if (threadIdx.x == 0) PSS[static_cast<long long>(r) * S + s] = gss;
    }
}
extern "C" __global__ void __launch_bounds__(512) qwen4_ple_conv(const __nv_bfloat16* __restrict__ GATED, const float* __restrict__ PSS,
                                                              const float* __restrict__ NC, const __nv_bfloat16* __restrict__ TAIL,
                                                              const float* __restrict__ CW, const __nv_bfloat16* H,
                                                              __nv_bfloat16* HOUT, __nv_bfloat16* __restrict__ NROW, float eps,
                                                              int D, int S, int TAPS, int DIL) {
    const int r = blockIdx.x;
    const int w = S * D, nt = (TAPS - 1) * DIL;
    const int e = blockIdx.y * blockDim.x + threadIdx.x;
    if (e >= w) return;
    const int s = e / D;
    const float fd = static_cast<float>(D);
    float acc = 0.0f;
    for (int t = 0; t < TAPS; ++t) {
        const int at = r + t * DIL;
        float xv;
        if (at < nt) {
            xv = bf(TAIL[static_cast<long long>(at) * w + e]);
        } else {
            const int rr = at - nt;
            const float rinv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(PSS[rr * S + s], fd), eps)));
            xv = rbf(__fmul_rn(__fmul_rn(bf(GATED[static_cast<long long>(rr) * w + e]), rinv), NC[e]));
        }
        acc = __fmaf_rn(CW[static_cast<long long>(e) * TAPS + t], xv, acc);
    }
    const float act = bsilu(rbf(acc));
    const float g = bf(GATED[static_cast<long long>(r) * w + e]);
    const float ple = rbf(__fadd_rn(g, act));
    const long long at = static_cast<long long>(r) * w + e;
    HOUT[at] = __float2bfloat16_rn(__fadd_rn(bf(H[at]), ple));
    const float rinv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(PSS[r * S + s], fd), eps)));
    NROW[at] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(g, rinv), NC[e]));
}
