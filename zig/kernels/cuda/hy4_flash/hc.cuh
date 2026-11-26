// Hy4-flash hyper-connection stream mix and branch merge helpers (`hc.cu` and planted controls).
#pragma once

#include "../common/reduce.cuh"
#include <cuda_bf16.h>

namespace hy4_hc {
constexpr int kStreams = 4;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;

__device__ __forceinline__ float bf(unsigned short bits) { return __uint_as_float((unsigned)bits << 16); }

__device__ __forceinline__ unsigned short round_bf(float v) { return __bfloat16_as_ushort(__float2bfloat16_rn(v)); }

__device__ __forceinline__ float round_bf_f(float v) { return bf(round_bf(v)); }

// Row sums over the block via xor halving and warp fold; BF16=true plants reduced-precision sums.
template <int N, bool BF16>
__device__ __forceinline__ void reduce_streams(float (&v)[N], float (*shared)[kWarps]) {
    for (int j = 0; j < N; j++) {
        if constexpr (!BF16) {
            v[j] = tf_common::warp_sum(v[j]);
        } else {
            for (int o = 16; o > 0; o >>= 1) v[j] = round_bf_f(__fadd_rn(v[j], __shfl_xor_sync(0xffffffffu, v[j], o)));
        }
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0)
        for (int j = 0; j < N; j++) shared[j][warp] = v[j];
    __syncthreads();
    for (int j = 0; j < N; j++) {
        float s = shared[j][0];
        for (int w = 1; w < kWarps; w++) {
            s = s + shared[j][w];
            if constexpr (BF16) s = round_bf_f(s);
        }
        v[j] = s;
    }
}

// Stream weights and sum of one row matching `tf_hy4_ihc_collapse`: MIXES 8 for pre, 4 for head; CONTROL 1..4 faults.
template <int MIXES, int CONTROL>
__device__ __forceinline__ void hc_mix(const unsigned short* __restrict__ x, const float* __restrict__ fn,
                                       const float* __restrict__ base, const float* __restrict__ scale, float eps,
                                       float hc_eps, float magnitude, unsigned short* __restrict__ out,
                                       float* __restrict__ post, int hidden, int repeat) {
    constexpr bool kBadSums = CONTROL == 4;
    __shared__ float partial[1 + MIXES][kWarps];
    __shared__ float weight[kStreams];
    const int width = kStreams * hidden;
    const unsigned short* row = x + (unsigned long long)blockIdx.x * width;
    float acc[1 + MIXES];
    for (int j = 0; j <= MIXES; j++) acc[j] = 0.0f;
    for (int k = threadIdx.x; k < width; k += kThreads) {
        const float v = bf(row[k]);
        const float sq = __fmaf_rn(v, v, acc[0]);
        acc[0] = kBadSums ? round_bf_f(sq) : sq;
        for (int j = 0; j < MIXES; j++) {
            const float d = __fmaf_rn(v, fn[(unsigned long long)j * width + k], acc[1 + j]);
            acc[1 + j] = kBadSums ? round_bf_f(d) : d;
        }
    }
    reduce_streams<1 + MIXES, kBadSums>(acc, partial);
    if (threadIdx.x == 0) {
        const float r = rsqrtf(acc[0] / (float)width + eps);
        for (int i = 0; i < kStreams; i++) {
            const float mix = acc[1 + i] * r;
            const float gate = 1.0f / (1.0f + expf(-(mix * scale[0] + base[i])));
            weight[i] = gate + hc_eps;
        }
        if (MIXES == 2 * kStreams)
            for (int i = 0; i < kStreams; i++) {
                const float mix = acc[1 + kStreams + i] * r;
                const float gate = 1.0f / (1.0f + expf(-(mix * scale[1] + base[kStreams + i])));
                post[(unsigned long long)blockIdx.x * kStreams + i] = magnitude * gate + hc_eps;
            }
    }
    __syncthreads();
    unsigned short* o = out + (unsigned long long)blockIdx.x * hidden;
    for (int h = threadIdx.x; h < hidden; h += kThreads) {
        if (CONTROL == 3 && h == hidden - 1) continue;
        float s = 0.0f;
        for (int i = 0; i < kStreams; i++) s += weight[i] * bf(row[i * hidden + h]);
        unsigned short bits = round_bf(s);
        if (CONTROL == 1 && isnan(bf(bits))) bits = 0;
        if (CONTROL == 2) bits = round_bf(__fadd_rn(bf(bits), (float)repeat));
        o[h] = bits;
    }
}

// Branch output back into the streams matching `tf_hy4_ihc_add`: bf16(residual_i + post_i * f).
template <int CONTROL>
__device__ __forceinline__ void hc_merge(const unsigned short* __restrict__ residual, const unsigned short* __restrict__ f,
                                         const float* __restrict__ post, unsigned short* __restrict__ out, int rows,
                                         int hidden, int repeat) {
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long count = (unsigned long long)rows * kStreams * hidden;
    if (i >= count || (CONTROL == 3 && i == count - 1)) return;
    const unsigned long long row = i / ((unsigned long long)kStreams * hidden), h = i % hidden;
    const int s = (int)((i / hidden) % kStreams);
    unsigned short bits = round_bf(bf(residual[i]) + post[row * kStreams + s] * bf(f[row * hidden + h]));
    if (CONTROL == 1 && isnan(bf(bits))) bits = 0;
    if (CONTROL == 2) bits = round_bf(__fadd_rn(bf(bits), (float)repeat));
    out[i] = bits;
}
}  // namespace hy4_hc
