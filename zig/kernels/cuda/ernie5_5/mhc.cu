// ERNIE 5.5's mHC: the pre side (a row's mixes, Sinkhorn, the collapse and its RMSNorm) and the post side.

#include <cuda_bf16.h>

#include "common/reduce.cuh"

namespace ernie55m {

using tf_common::warp_sum;

constexpr int D = 4096, S = 4, T = 256, M = 24;
constexpr unsigned all = 0xffffffffu;

enum Fault { none, plain_rms, column_softmax };

// 1 / (1 + e^-v), each step rounded once.
__device__ __forceinline__ float sigmoid(float v) { return __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-v))); }

// The 4x4 sums, lane & 15 = 4i + j: a row's over j (xor 2, then 1), a column's over i (xor 8, then 4).
__device__ __forceinline__ float row4(float v) {
    v = __fadd_rn(v, __shfl_xor_sync(all, v, 2));
    return __fadd_rn(v, __shfl_xor_sync(all, v, 1));
}
__device__ __forceinline__ float col4(float v) {
    v = __fadd_rn(v, __shfl_xor_sync(all, v, 8));
    return __fadd_rn(v, __shfl_xor_sync(all, v, 4));
}

// Row blockIdx.x: thread t holds k = c 2048 + 8t + j (c, j < 8; k = 4096 s + h) of x and of each fn row.
template <Fault F>
__device__ __forceinline__ void pre(const __nv_bfloat16* __restrict__ x, long long xs, const float* __restrict__ fn,
                                    const float* __restrict__ scale, const float* __restrict__ base,
                                    const __nv_bfloat16* __restrict__ w, float* __restrict__ post,
                                    float* __restrict__ comb, __nv_bfloat16* __restrict__ out, long long os,
                                    float rms_eps, float hc_eps, float norm_eps, int iters) {
    __shared__ float part[M + 1][T / 32];
    __shared__ float mix[M + 1];
    __shared__ float pmix[S];
    __shared__ float sq[T / 32];
    const long long row = blockIdx.x;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const __nv_bfloat16* xr = x + row * xs;
    // each mix, then the squares (m = 24): one fma chain over c, then j; x's 16 bytes read again for each
    for (int m = 0; m <= M; ++m) {
        const float* f = fn + static_cast<long long>(m < M ? m : 0) * (S * D) + t * 8;
        float acc = 0.0f;
#pragma unroll 1
        for (int c = 0; c < 8; ++c) {
            const uint4 xv = __ldg(reinterpret_cast<const uint4*>(xr + c * 2048 + t * 8));
            const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&xv);
            if (m < M) {
                const float4 lo = __ldg(reinterpret_cast<const float4*>(f + c * 2048));
                const float4 hi = __ldg(reinterpret_cast<const float4*>(f + c * 2048 + 4));
                const float fv[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
#pragma unroll
                for (int j = 0; j < 8; ++j) acc = __fmaf_rn(__bfloat162float(h[j]), fv[j], acc);
            } else {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float v = __bfloat162float(h[j]);
                    acc = __fmaf_rn(v, v, acc);
                }
            }
        }
        acc = warp_sum(acc);
        if (lane == 0) part[m][warp] = acc;
    }
    __syncthreads();
    if (t <= M) {
        float v = part[t][0];
        for (int i = 1; i < T / 32; ++i) v = __fadd_rn(v, part[t][i]);
        mix[t] = v;
    }
    __syncthreads();
    if (warp == 0) {
        const float mean = __fmul_rn(mix[M], 1.0f / (S * D));  // a power of two: exact
        // ERNIE's mix RMS, 1 / (sqrt(mean) + eps); the control takes the plain 1 / sqrt(mean + eps)
        const float r = F == plain_rms ? __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(mean, rms_eps)))
                                       : __fdiv_rn(1.0f, __fadd_rn(__fsqrt_rn(mean), rms_eps));
        const int idx = lane & 15, k = lane & 3;
        if (lane < S) {
            pmix[lane] = __fadd_rn(sigmoid(__fmaf_rn(__fmul_rn(mix[k], r), scale[0], base[k])), hc_eps);
            const float sp = sigmoid(__fmaf_rn(__fmul_rn(mix[S + k], r), scale[1], base[S + k]));
            post[row * S + lane] = __fadd_rn(sp, sp);
        }
        const float cl = __fmaf_rn(__fmul_rn(mix[2 * S + idx], r), scale[2], base[2 * S + idx]);
        // softmax over j (a row); the control takes it over i (a column)
        float cmax;
        if (F == column_softmax) {
            cmax = fmaxf(cl, __shfl_xor_sync(all, cl, 8));
            cmax = fmaxf(cmax, __shfl_xor_sync(all, cmax, 4));
        } else {
            cmax = fmaxf(cl, __shfl_xor_sync(all, cl, 2));
            cmax = fmaxf(cmax, __shfl_xor_sync(all, cmax, 1));
        }
        const float e = expf(__fsub_rn(cl, cmax));
        float cm = __fadd_rn(__fdiv_rn(e, F == column_softmax ? col4(e) : row4(e)), hc_eps);
        cm = __fdiv_rn(cm, __fadd_rn(col4(cm), hc_eps));
        for (int it = 1; it < iters; ++it) {
            cm = __fdiv_rn(cm, __fadd_rn(row4(cm), hc_eps));
            cm = __fdiv_rn(cm, __fadd_rn(col4(cm), hc_eps));
        }
        if (lane < 16) comb[row * 16 + lane] = cm;
    }
    __syncthreads();
    // thread t's outputs h = 2048 half + 8t + j: sum_s pre_s x_s from stream 0, squares of the unrounded sums
    uint4 xu[8];
#pragma unroll
    for (int c = 0; c < 8; ++c) xu[c] = __ldg(reinterpret_cast<const uint4*>(xr + c * 2048 + t * 8));
    float o[2][8];
    float ss = 0.0f;
#pragma unroll
    for (int half = 0; half < 2; ++half)
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            float v = 0.0f;
#pragma unroll
            for (int s = 0; s < S; ++s) {
                const float xv = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&xu[2 * s + half])[j]);
                v = s == 0 ? __fmul_rn(pmix[0], xv) : __fmaf_rn(pmix[s], xv, v);
            }
            o[half][j] = v;
            ss = __fmaf_rn(v, v, ss);
        }
    ss = warp_sum(ss);
    if (lane == 0) sq[warp] = ss;
    __syncthreads();
    float tot = sq[0];
    for (int i = 1; i < T / 32; ++i) tot = __fadd_rn(tot, sq[i]);
    const float r2 = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fmul_rn(tot, 1.0f / D), norm_eps)));
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        uint4 ou;
        __nv_bfloat16* oh = reinterpret_cast<__nv_bfloat16*>(&ou);
        const uint4 wu = __ldg(reinterpret_cast<const uint4*>(w + half * 2048 + t * 8));
        const __nv_bfloat16* wh = reinterpret_cast<const __nv_bfloat16*>(&wu);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float b = __bfloat162float(__float2bfloat16_rn(o[half][j]));
            oh[j] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(b, r2), __bfloat162float(wh[j])));
        }
        *reinterpret_cast<uint4*>(out + row * os + half * 2048 + t * 8) = ou;
    }
}

enum PostFault { post_none, comb_transposed, no_post };

// Row blockIdx.x, h = 2048 half + 8t + j: stream o = bf16(post[o] x + sum_i comb[i][o] res[i]), the product first.
template <PostFault F>
__device__ __forceinline__ void post_row(const __nv_bfloat16* __restrict__ x, long long xs,
                                         const __nv_bfloat16* __restrict__ res, long long rs,
                                         const float* __restrict__ post, const float* __restrict__ comb,
                                         __nv_bfloat16* __restrict__ out, long long os) {
    const long long row = blockIdx.x;
    const int t = threadIdx.x;
    float p[S], c[S][S];
#pragma unroll
    for (int i = 0; i < S; ++i) {
        p[i] = F == no_post ? 1.0f : post[row * S + i];
#pragma unroll
        for (int o = 0; o < S; ++o) c[i][o] = F == comb_transposed ? comb[row * 16 + o * S + i] : comb[row * 16 + i * S + o];
    }
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const long long h0 = half * 2048 + t * 8;
        const uint4 xu = __ldg(reinterpret_cast<const uint4*>(x + row * xs + h0));
        uint4 ru[S];
#pragma unroll
        for (int i = 0; i < S; ++i) ru[i] = __ldg(reinterpret_cast<const uint4*>(res + row * rs + i * D + h0));
        const __nv_bfloat16* xh = reinterpret_cast<const __nv_bfloat16*>(&xu);
#pragma unroll
        for (int o = 0; o < S; ++o) {
            uint4 ou;
            __nv_bfloat16* oh = reinterpret_cast<__nv_bfloat16*>(&ou);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                float v = __fmul_rn(p[o], __bfloat162float(xh[j]));
#pragma unroll
                for (int i = 0; i < S; ++i)
                    v = __fmaf_rn(c[i][o], __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&ru[i])[j]), v);
                oh[j] = __float2bfloat16_rn(v);
            }
            *reinterpret_cast<uint4*>(out + row * os + o * D + h0) = ou;
        }
    }
}

}  // namespace ernie55m

// Rows of 4 bf16 streams xs apart, fn [24][16384] and the norm's w: post [4], comb [16] (fp32), bf16 rows os apart.
extern "C" __global__ void __launch_bounds__(256)
    tf_ernie55_mhc_pre(const __nv_bfloat16* __restrict__ x, long long xs, const float* __restrict__ fn,
                       const float* __restrict__ scale, const float* __restrict__ base,
                       const __nv_bfloat16* __restrict__ w, float* __restrict__ post, float* __restrict__ comb,
                       __nv_bfloat16* __restrict__ out, long long os, float rms_eps, float hc_eps, float norm_eps,
                       int iters) {
    ernie55m::pre<ernie55m::none>(x, xs, fn, scale, base, w, post, comb, out, os, rms_eps, hc_eps, norm_eps, iters);
}

// The contract's planted control: the plain mix RMS, 1 / sqrt(mean + eps), in place of 1 / (sqrt(mean) + eps).
extern "C" __global__ void __launch_bounds__(256)
    tf_ctl_ernie55_mhc_pre_plain_rms(const __nv_bfloat16* __restrict__ x, long long xs, const float* __restrict__ fn,
                                     const float* __restrict__ scale, const float* __restrict__ base,
                                     const __nv_bfloat16* __restrict__ w, float* __restrict__ post,
                                     float* __restrict__ comb, __nv_bfloat16* __restrict__ out, long long os,
                                     float rms_eps, float hc_eps, float norm_eps, int iters) {
    ernie55m::pre<ernie55m::plain_rms>(x, xs, fn, scale, base, w, post, comb, out, os, rms_eps, hc_eps, norm_eps,
                                       iters);
}

// The contract's planted control: comb's softmax over a column in place of a row.
extern "C" __global__ void __launch_bounds__(256)
    tf_ctl_ernie55_mhc_pre_column_softmax(const __nv_bfloat16* __restrict__ x, long long xs,
                                          const float* __restrict__ fn, const float* __restrict__ scale,
                                          const float* __restrict__ base, const __nv_bfloat16* __restrict__ w,
                                          float* __restrict__ post, float* __restrict__ comb,
                                          __nv_bfloat16* __restrict__ out, long long os, float rms_eps, float hc_eps,
                                          float norm_eps, int iters) {
    ernie55m::pre<ernie55m::column_softmax>(x, xs, fn, scale, base, w, post, comb, out, os, rms_eps, hc_eps,
                                            norm_eps, iters);
}

// A row's layer output x (bf16, rows xs apart) into its 4 streams res (rows rs apart): post [4], comb [16] fp32.
extern "C" __global__ void __launch_bounds__(256)
    tf_ernie55_mhc_post(const __nv_bfloat16* __restrict__ x, long long xs, const __nv_bfloat16* __restrict__ res,
                        long long rs, const float* __restrict__ post, const float* __restrict__ comb,
                        __nv_bfloat16* __restrict__ out, long long os) {
    ernie55m::post_row<ernie55m::post_none>(x, xs, res, rs, post, comb, out, os);
}

// The contract's planted control: comb read transposed, comb[o][i] in place of comb[i][o].
extern "C" __global__ void __launch_bounds__(256)
    tf_ctl_ernie55_mhc_post_comb_transposed(const __nv_bfloat16* __restrict__ x, long long xs,
                                            const __nv_bfloat16* __restrict__ res, long long rs,
                                            const float* __restrict__ post, const float* __restrict__ comb,
                                            __nv_bfloat16* __restrict__ out, long long os) {
    ernie55m::post_row<ernie55m::comb_transposed>(x, xs, res, rs, post, comb, out, os);
}

// The contract's planted control: the layer output added as it is, without post.
extern "C" __global__ void __launch_bounds__(256)
    tf_ctl_ernie55_mhc_post_no_post(const __nv_bfloat16* __restrict__ x, long long xs,
                                    const __nv_bfloat16* __restrict__ res, long long rs,
                                    const float* __restrict__ post, const float* __restrict__ comb,
                                    __nv_bfloat16* __restrict__ out, long long os) {
    ernie55m::post_row<ernie55m::no_post>(x, xs, res, rs, post, comb, out, os);
}
