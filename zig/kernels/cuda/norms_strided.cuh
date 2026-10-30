// Nemotron's strided rows (thread t holds t, t + T, ...): fma chains, xor butterflies, warps and 64-sums in order.

#pragma once

#include <cuda_bf16.h>
#include <stdint.h>

namespace tf_rms_strided {

__device__ __forceinline__ float bf(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float rbf(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, m));
    return v;
}

// The block's sum: each warp's butterfly into red, then red's T / 32 entries in order (every thread reads it).
template <int T>
__device__ __forceinline__ float block_sum(float acc, float* red) {
    acc = warp_sum(acc);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
    __syncthreads();
    float s = red[0];
    for (int w = 1; w < T / 32; ++w) s = __fadd_rn(s, red[w]);
    return s;
}

// 1 / sqrt(ss / n + eps) with IEEE division and square root.
__device__ __forceinline__ float inv_rms(float ss, int n, float eps) {
    return __fdiv_rn(1.f, __fsqrt_rn(__fadd_rn(__fdiv_rn(ss, static_cast<float>(n)), eps)));
}

// Each 64-value group of buf summed in order into xs, by the first T threads (buf written before a barrier).
template <int T>
__device__ __forceinline__ void group_sums(const float* buf, int width, float* __restrict__ xs) {
    for (int g = threadIdx.x; g < width / 64; g += T) {
        float s = 0.f;
        for (int j = 0; j < 64; ++j) s = __fadd_rn(s, buf[g * 64 + j]);
        xs[g] = s;
    }
}

// y = bf16((x inv) w), or MLX's bf16(w bf16(x inv)) with ROUND, of one row of `width` <= T M; xs y's 64-sums.
template <int T, int M, bool ROUND>
__device__ __forceinline__ void row(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
                                    __nv_bfloat16* __restrict__ y, float* __restrict__ xs, int width, float eps) {
    __shared__ float red[T / 32];
    __shared__ float ybuf[T * M];
    float v[M];
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < M; ++k) {
        const int i = threadIdx.x + k * T;
        if (i >= width) continue;
        v[k] = bf(x[i]);
        acc = __fmaf_rn(v[k], v[k], acc);
    }
    const float inv = inv_rms(block_sum<T>(acc, red), width, eps);
#pragma unroll
    for (int k = 0; k < M; ++k) {
        const int i = threadIdx.x + k * T;
        if (i >= width) continue;
        const float o = ROUND ? __fmul_rn(bf(w[i]), rbf(__fmul_rn(v[k], inv))) : __fmul_rn(__fmul_rn(v[k], inv), bf(w[i]));
        const __nv_bfloat16 ob = __float2bfloat16_rn(o);
        y[i] = ob;
        ybuf[i] = bf(ob);
    }
    if (xs == nullptr) return;
    __syncthreads();
    group_sums<T>(ybuf, width, xs);
}

// Part p of row r at x + r x_ld + p x_part (w + p w_part; y and xs at r y_ld + p y_part): row's norm, a block each.
template <int T, int M, bool ROUND>
__device__ __forceinline__ void parts(const __nv_bfloat16* x, long long x_ld, long long x_part, const __nv_bfloat16* w,
                                      long long w_part, __nv_bfloat16* y, long long y_ld, long long y_part, float* xs,
                                      int width, float eps) {
    const int64_t r = blockIdx.x, p = blockIdx.y, yo = r * y_ld + p * y_part;
    row<T, M, ROUND>(x + r * x_ld + p * x_part, w + p * w_part, y + yo, xs != nullptr ? xs + yo / 64 : nullptr, width, eps);
}

constexpr int LT = 1024;  // moe_row's threads: each loads and adds LE elements, every load in flight
constexpr int NT = 256;   // the threads whose chains take the squares, MOST / NT elements each
constexpr int MOST = 4096;
constexpr int LE = MOST / LT;
constexpr int SLOTS = 8;  // expert slots a row holds

// h = bf16(x + bf16(routed slots' fma chain + the others' sum)) (KEEP: unrounded), then the strided norm of h.
template <bool KEEP>
__device__ __forceinline__ void moe_row(const __nv_bfloat16* __restrict__ x, const void* __restrict__ ys, int y_f32,
                                        const float* __restrict__ wt, const __nv_bfloat16* __restrict__ w,
                                        __nv_bfloat16* __restrict__ h, __nv_bfloat16* __restrict__ y,
                                        float* __restrict__ xs, float eps, int width, int routed, int slots) {
    __shared__ float red[NT / 32];
    __shared__ float xb[MOST];  // the row's h, then its stored outputs
    const int64_t r = blockIdx.x;
    float wk[SLOTS];
#pragma unroll
    for (int s = 0; s < SLOTS; ++s) wk[s] = s < routed ? wt[r * slots + s] : 0.f;
    float yv[LE][SLOTS], xv[LE];
#pragma unroll
    for (int k = 0; k < LE; ++k) {
        const int i = threadIdx.x + k * LT;
        xv[k] = i < width ? bf(x[r * width + i]) : 0.f;
#pragma unroll
        for (int s = 0; s < SLOTS; ++s) {
            const int64_t at = (r * slots + s) * width + i;
            yv[k][s] = i >= width || s >= slots ? 0.f
                       : y_f32                  ? static_cast<const float*>(ys)[at]
                                                : bf(static_cast<const __nv_bfloat16*>(ys)[at]);
        }
    }
#pragma unroll
    for (int k = 0; k < LE; ++k) {
        const int i = threadIdx.x + k * LT;
        if (i >= width) continue;
        float rs = 0.f, ss = 0.f;
#pragma unroll
        for (int s = 0; s < SLOTS; ++s) {
            if (s >= slots) break;
            if (s < routed) rs = __fmaf_rn(wk[s], yv[k][s], rs);
            else ss = __fadd_rn(ss, yv[k][s]);
        }
        const float delta = KEEP ? __fadd_rn(rs, ss) : rbf(__fadd_rn(rs, ss));
        const __nv_bfloat16 hv = __float2bfloat16_rn(__fadd_rn(xv[k], delta));
        h[r * width + i] = hv;
        xb[i] = bf(hv);
    }
    __syncthreads();
    if (threadIdx.x < NT) {
        float acc = 0.f;
#pragma unroll
        for (int k = 0; k < MOST / NT; ++k)
            if (threadIdx.x + k * NT < width) acc = __fmaf_rn(xb[threadIdx.x + k * NT], xb[threadIdx.x + k * NT], acc);
        acc = warp_sum(acc);
        if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
    }
    __syncthreads();
    float sum = red[0];
    for (int q = 1; q < NT / 32; ++q) sum = __fadd_rn(sum, red[q]);
    const float inv = inv_rms(sum, width, eps);
#pragma unroll
    for (int k = 0; k < LE; ++k) {  // each element reads its h and leaves its stored output in place
        const int i = threadIdx.x + k * LT;
        if (i >= width) continue;
        const __nv_bfloat16 o = __float2bfloat16_rn(__fmul_rn(__fmul_rn(xb[i], inv), bf(w[i])));
        y[r * width + i] = o;
        xb[i] = bf(o);
    }
    __syncthreads();
    group_sums<LT>(xb, width, xs + r * (width / 64));
}

}  // namespace tf_rms_strided
