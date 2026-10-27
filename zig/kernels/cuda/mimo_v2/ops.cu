// MiMo-V2's row ops (norms, SwiGLU, router, top-k): fixed orders, no row-count tiles, a row's bits its own. Glue flags.

#include <cstdint>
#include <cuda_fp16.h>
#include <math_constants.h>

namespace {

constexpr int kMaxPer = 16;  // values a thread holds: rows up to 16 * blockDim.x wide

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Sum over the block (blockDim.x a multiple of 32, at most 1024): warp trees, then the warp sums' tree.
__device__ float block_sum(float v) {
    __shared__ float part[32];
    __shared__ float total;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) part[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float s = lane < nw ? part[lane] : 0.f;
        s = warp_sum(s);
        if (lane == 0) total = s;
    }
    __syncthreads();
    return total;
}

}  // namespace

// out = half(v * rsqrt(mean(v^2) + eps) * w), v = x, or res += x (fp32, in place) and v = res. A block a row.
extern "C" __global__ void mimo_rms(const float* __restrict__ x, float* __restrict__ res, const float* __restrict__ w,
                                    __half* __restrict__ out, int D, float eps) {
    const int64_t r = blockIdx.x;
    float v[kMaxPer];
    float ss = 0.f;
    int n = 0;
    for (int d = threadIdx.x; d < D; d += blockDim.x, ++n) {
        float t = x[r * D + d];
        if (res) {
            t = res[r * D + d] + t;
            res[r * D + d] = t;
        }
        v[n] = t;
        ss += t * t;
    }
    const float rs = rsqrtf(block_sum(ss) / (float)D + eps);
    n = 0;
    for (int d = threadIdx.x; d < D; d += blockDim.x, ++n) out[r * D + d] = __float2half_rn(v[n] * rs * w[d]);
}

// out = half(silu(g) * u), fp32 math; grid (rows, N / blockDim.x).
extern "C" __global__ void mimo_silu_mul(const __half* __restrict__ g, const __half* __restrict__ u, __half* __restrict__ out,
                                         int N) {
    const int64_t r = blockIdx.x;
    const int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (c >= N) return;
    const float gf = __half2float(g[r * N + c]), uf = __half2float(u[r * N + c]);
    out[r * N + c] = __float2half_rn(gf / (1.f + expf(-gf)) * uf);
}

// out [rows, D] fp32 = sum over slots k of w[row, k] * y[row, k, :] (fp32, slot order, fused): the prompt experts' sum.
extern "C" __global__ void mimo_wsum(const float* __restrict__ y, const float* __restrict__ w, float* __restrict__ out, int K, int D) {
    const int64_t r = blockIdx.x;
    const int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (d >= D) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc = fmaf(w[r * K + k], y[(r * K + k) * D + d], acc);
    out[r * D + d] = acc;
}

// logits [rows, E] fp32 = x . W[e] (fp16): a warp an expert, lane pairs 2 lane + 64 j in order, then the warp's tree.
extern "C" __global__ void mimo_router(const __half* __restrict__ x, const __half* __restrict__ W,
                                       float* __restrict__ out, int D, int E) {
    extern __shared__ float xs[];
    const int64_t r = blockIdx.y;
    for (int d = threadIdx.x; d < D; d += blockDim.x) xs[d] = __half2float(x[r * D + d]);
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int e = blockIdx.x * (blockDim.x >> 5) + warp;
    if (e >= E) return;
    const __half2* wr = (const __half2*)(W + (int64_t)e * D);
    float acc = 0.f;
    for (int j = lane; j < D / 2; j += 32) {
        const float2 wf = __half22float2(__ldg(wr + j));
        acc = __fmaf_rn(xs[2 * j], wf.x, acc);
        acc = __fmaf_rn(xs[2 * j + 1], wf.y, acc);
    }
    acc = warp_sum(acc);
    if (lane == 0) out[r * E + e] = acc;
}

// Top K of sigmoid(logits) + bias (ties: lower id) in pick order, w = s / (sum + 1e-20) * scale; a block (>= E) a row.
extern "C" __global__ void mimo_topk(const float* __restrict__ logits, const float* __restrict__ bias, int E, int K,
                                     float scale, int32_t* __restrict__ pick, float* __restrict__ wts) {
    __shared__ float bv[1024];
    __shared__ int bi[1024];
    __shared__ int sel[32];
    __shared__ float sc[1024];
    const int64_t r = blockIdx.x;
    const int t = threadIdx.x;
    float c = -CUDART_INF_F;
    if (t < E) {
        const float s = 1.f / (1.f + expf(-logits[r * E + t]));
        sc[t] = s;
        c = s + bias[t];
    }
    for (int k = 0; k < K; ++k) {
        bv[t] = c;
        bi[t] = t;
        __syncthreads();
        for (int o = blockDim.x >> 1; o > 0; o >>= 1) {
            if (t < o) {
                const float v = bv[t + o];
                const int i = bi[t + o];
                if (v > bv[t] || (v == bv[t] && i < bi[t])) {
                    bv[t] = v;
                    bi[t] = i;
                }
            }
            __syncthreads();
        }
        if (t == 0) sel[k] = bi[0];
        __syncthreads();
        if (t == sel[k]) c = -CUDART_INF_F;
        __syncthreads();
    }
    if (t == 0) {
        float sum = 0.f;
        for (int k = 0; k < K; ++k) sum += sc[sel[k]];
        const float den = sum + 1e-20f;
        for (int k = 0; k < K; ++k) {
            pick[r * K + k] = sel[k];
            wts[r * K + k] = sc[sel[k]] / den * scale;
        }
    }
}
