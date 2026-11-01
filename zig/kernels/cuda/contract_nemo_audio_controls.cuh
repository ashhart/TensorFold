// nemo_audio.cu's planted controls (Nemotron 3.5 Transcribe's audio encoder); only the contract controls include this.
#pragma once
#include <cuda_bf16.h>

// tf_na_qk_rope with a one-pass variance, E[x^2] - mean^2: rows far from zero lose their digits, so gate 3 goes red.
__device__ __forceinline__ float tf_ctl_na_bf(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float tf_ctl_na_rbf(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ __forceinline__ float tf_ctl_na_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

extern "C" __global__ void __launch_bounds__(32) tf_ctl_na_qk_onepass(__nv_bfloat16* qkv, const __nv_bfloat16* qw, const __nv_bfloat16* qb,
                                                                          const __nv_bfloat16* kw, const __nv_bfloat16* kb, const __nv_bfloat16* cs,
                                                                          const __nv_bfloat16* sn, int H, int D, int R, float eps) {
    __shared__ float x[128];
    const int which = blockIdx.x & 1;
    const int h = (blockIdx.x >> 1) % H;
    const int t = (blockIdx.x >> 1) / H;
    __nv_bfloat16* row = qkv + ((static_cast<int64_t>(t) * 3 + which) * H + h) * D;
    const int lane = threadIdx.x;
    for (int i = lane; i < D; i += 32) x[i] = tf_ctl_na_bf(row[i]);
    __syncwarp();
    const __nv_bfloat16* w = which ? kw : qw;
    const __nv_bfloat16* b = which ? kb : qb;
    float acc = 0.0f, sq = 0.0f;
    for (int i = lane; i < D; i += 32) {
        acc = __fadd_rn(acc, x[i]);
        sq = __fadd_rn(sq, __fmul_rn(x[i], x[i]));
    }
    const float mean = __fdiv_rn(tf_ctl_na_sum(acc), static_cast<float>(D));
    const float var = __fsub_rn(__fdiv_rn(tf_ctl_na_sum(sq), static_cast<float>(D)), __fmul_rn(mean, mean));
    const float rstd = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(fmaxf(var, 0.0f), eps)));
    float y[4];
    for (int i = lane, k = 0; i < D; i += 32, ++k) y[k] = tf_ctl_na_rbf(__fadd_rn(__fmul_rn(__fmul_rn(__fsub_rn(x[i], mean), rstd), tf_ctl_na_bf(w[i])), tf_ctl_na_bf(b[i])));
    __syncwarp();
    for (int i = lane, k = 0; i < D; i += 32, ++k) x[i] = y[k];
    __syncwarp();
    const int half = R / 2;
    for (int i = lane; i < D; i += 32) {
        float v = x[i];
        if (i < R) {
            const float c = tf_ctl_na_bf(cs[static_cast<int64_t>(t) * R + i]);
            const float s = tf_ctl_na_bf(sn[static_cast<int64_t>(t) * R + i]);
            const float r = i < half ? -x[i + half] : x[i - half];
            v = __fadd_rn(tf_ctl_na_rbf(__fmul_rn(v, c)), tf_ctl_na_rbf(__fmul_rn(r, s)));
        }
        row[i] = __float2bfloat16_rn(v);
    }
}

// Gate 1: nemo_audio's relu as fmaxf(x, 0), which turns a NaN into 0 where tf_na_relu keeps it.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_na_relu_fmax(__nv_bfloat16* x, int64_t count) {
    const int64_t at = static_cast<int64_t>(blockIdx.x) * 256 + threadIdx.x;
    if (at < count) x[at] = __float2bfloat16_rn(fmaxf(__bfloat162float(x[at]), 0.0f));
}
