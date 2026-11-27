// Hy4-flash FP8 Engram templates: 100-byte row decode + per-head RMSNorm, rank-0 bias, and 4-stream gate.
#pragma once
#include "../common/glue_math.cuh"
#include "../common/reduce.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace hy4_engram_fp8_impl {
constexpr int kStreams = 4;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;

__device__ __forceinline__ float bf(unsigned short bits) { return __uint_as_float((unsigned)bits << 16); }

__device__ __forceinline__ unsigned short round_bf(float v) { return __bfloat16_as_ushort(__float2bfloat16_rn(v)); }

__device__ __forceinline__ float round_bf_f(float v) { return bf(round_bf(v)); }

__device__ __forceinline__ float round_f16_f(float v) { return __half2float(__float2half_rn(v)); }

// Exact IEEE float8_e4m3fn -> float32 on every SM (0x7f / 0xff -> quiet NaN).
__device__ __forceinline__ float e4m3(unsigned char b) {
    const unsigned s = (unsigned)(b & 0x80u) << 24;
    const unsigned e = (b >> 3) & 0x0fu;
    const unsigned m = b & 0x07u;
    if (e == 15 && m == 7) return __uint_as_float(s | 0x7fc00000u);
    if (e == 0) return m == 0 ? __uint_as_float(s) : __fmul_rn((float)(b & 0x80u ? -(int)m : (int)m), 0x1.0p-9f);
    return __uint_as_float(s | ((e + 120u) << 23) | (m << 20));
}

__device__ __forceinline__ float read_scale(const unsigned char* p) {
    const unsigned u = (unsigned)p[0] | ((unsigned)p[1] << 8) | ((unsigned)p[2] << 16) | ((unsigned)p[3] << 24);
    return __uint_as_float(u);
}

__device__ __forceinline__ float sigmoid(float x) {
    const float e = tf_glue::tf_exp(-fabsf(x));
    return x < 0 ? __fdiv_rn(e, __fadd_rn(1.0f, e)) : __fdiv_rn(1.0f, __fadd_rn(1.0f, e));
}

template <int N, bool BF16>
__device__ __forceinline__ void reduce_channels(float (&v)[N], float (*shared)[kWarps]) {
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
            s = __fadd_rn(s, shared[j][w]);
            if constexpr (BF16) s = round_bf_f(s);
        }
        v[j] = s;
    }
    __syncthreads();
}

// One warp per (row, head) pair: decodes `dim` e4m3 codes * fp32 scale and applies per-head RMSNorm into bf16.
template <int CONTROL>
__device__ __forceinline__ void engram_embed(const unsigned char* __restrict__ raw,
                                             const unsigned short* __restrict__ head_norms, float eps,
                                             unsigned short* __restrict__ out, int pairs, int heads, int dim,
                                             int repeat) {
    const int pair = (int)blockIdx.x * kWarps + (threadIdx.x >> 5);
    if (pair >= pairs) return;
    const int lane = threadIdx.x & 31;
    const int head = pair % heads;
    const unsigned char* rec = raw + (unsigned long long)pair * (dim + 4);
    const float scale = read_scale(rec + dim);
    float sq = 0.0f;
    for (int d = lane; d < dim; d += 32) {
        const float v = __fmul_rn(e4m3(rec[d]), scale);
        const float s = __fadd_rn(sq, __fmul_rn(v, v));
        sq = (CONTROL == 4) ? round_f16_f(s) : s;
    }
    for (int o = 16; o > 0; o >>= 1) {
        const float s = __fadd_rn(sq, __shfl_xor_sync(0xffffffffu, sq, o));
        sq = (CONTROL == 4) ? round_f16_f(s) : s;
    }
    const float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(sq, (float)dim), eps)));
    unsigned short* dst = out + (unsigned long long)pair * dim;
    const unsigned short* nw = head_norms + (unsigned long long)head * dim;
    for (int d = lane; d < dim; d += 32) {
        if (CONTROL == 3 && pair == pairs - 1 && d == dim - 1) continue;
        const float v = __fmul_rn(e4m3(rec[d]), scale);
        unsigned short bits = round_bf(__fmul_rn(__fmul_rn(v, inv), bf(nw[d])));
        if (CONTROL == 1 && isnan(bf(bits))) bits = 0;
        if (CONTROL == 2) bits = round_bf(__fadd_rn(bf(bits), (float)repeat));
        dst[d] = bits;
    }
}

// Rank-0 projection biases added once into fp32 partial sums [rows, cols] before the all-reduce seam.
template <int CONTROL>
__device__ __forceinline__ void engram_bias(const float* __restrict__ x, const unsigned short* __restrict__ bias,
                                            float* __restrict__ out, int rows, int cols, int repeat) {
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long count = (unsigned long long)rows * cols;
    if (i >= count || (CONTROL == 3 && i == count - 1)) return;
    float v = __fadd_rn(x[i], bf(bias[i % cols]));
    if (CONTROL == 1 && isnan(v)) v = 0.0f;
    if (CONTROL == 2) v = __fadd_rn(v, (float)repeat);
    out[i] = v;
}

// Post-allreduce unified RMSNorm, 4-stream signed-sqrt sigmoid gate, scales, mask and residual add.
template <int CONTROL>
__device__ __forceinline__ void engram_gate(const unsigned short* __restrict__ streams_in, const float* __restrict__ proj,
                                            const unsigned short* __restrict__ norm_keys,
                                            const unsigned short* __restrict__ norm_queries,
                                            const unsigned short* __restrict__ pre_gamma,
                                            const unsigned short* __restrict__ stream_scales,
                                            const unsigned char* __restrict__ mask, float eps,
                                            unsigned short* engram_out, unsigned short* streams_out, int hidden,
                                            int repeat) {
    constexpr bool kBadSums = CONTROL == 4;
    __shared__ float shared[9][kWarps];
    __shared__ float inv_s[9];
    __shared__ float gate_s[kStreams];
    const unsigned long long r = blockIdx.x;
    const bool active = (mask == nullptr) || (mask[r] != 0);
    const float* pv = proj + r * (1 + kStreams) * hidden;
    const unsigned short* sin = streams_in + r * kStreams * hidden;

    float acc[9];
    for (int j = 0; j < 9; j++) acc[j] = 0.0f;
    for (int h = threadIdx.x; h < hidden; h += kThreads) {
        const float vv = pv[h];
        const float s0 = __fadd_rn(acc[0], __fmul_rn(vv, vv));
        acc[0] = kBadSums ? round_bf_f(s0) : s0;
        for (int s = 0; s < kStreams; s++) {
            const float kv = pv[(1 + s) * hidden + h];
            const float sk = __fadd_rn(acc[1 + s], __fmul_rn(kv, kv));
            acc[1 + s] = kBadSums ? round_bf_f(sk) : sk;
            const float qv = bf(sin[s * hidden + h]);
            const float sq = __fadd_rn(acc[5 + s], __fmul_rn(qv, qv));
            acc[5 + s] = kBadSums ? round_bf_f(sq) : sq;
        }
    }
    reduce_channels<9, kBadSums>(acc, shared);
    if (threadIdx.x < 9) {
        inv_s[threadIdx.x] = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(acc[threadIdx.x], (float)hidden), eps)));
    }
    __syncthreads();

    float dot[kStreams];
    for (int s = 0; s < kStreams; s++) dot[s] = 0.0f;
    for (int h = threadIdx.x; h < hidden; h += kThreads) {
        for (int s = 0; s < kStreams; s++) {
            const float kh = __fmul_rn(__fmul_rn(pv[(1 + s) * hidden + h], inv_s[1 + s]), bf(norm_keys[s * hidden + h]));
            const float qh = __fmul_rn(__fmul_rn(bf(sin[s * hidden + h]), inv_s[5 + s]), bf(norm_queries[s * hidden + h]));
            const float d = __fadd_rn(dot[s], __fmul_rn(kh, qh));
            dot[s] = kBadSums ? round_bf_f(d) : d;
        }
    }
    reduce_channels<kStreams, kBadSums>(dot, shared);
    if (threadIdx.x < kStreams) {
        const int s = threadIdx.x;
        const float g = __fmul_rn(dot[s], __fdiv_rn(1.0f, __fsqrt_rn((float)hidden)));
        const float mag = __fsqrt_rn(fmaxf(fabsf(g), 1.0e-6f));
        const float ssqrt = isnan(g) ? g : (g > 0.0f ? mag : (g < 0.0f ? -mag : 0.0f));
        gate_s[s] = sigmoid(ssqrt);
    }
    __syncthreads();

    const float inv_v = inv_s[0];
    for (int h = threadIdx.x; h < hidden; h += kThreads) {
        const float main_h = __fmul_rn(bf(pre_gamma[h]), __fmul_rn(pv[h], inv_v));
        for (int s = 0; s < kStreams; s++) {
            if (CONTROL == 3 && s == kStreams - 1 && h == hidden - 1) continue;
            const unsigned long long idx = (r * kStreams + s) * hidden + h;
            const unsigned short st_in_bits = streams_in[idx];
            unsigned short eo_bits = 0;
            unsigned short st_bits = st_in_bits;
            if (active) {
                const float eo_f = __fadd_rn(__fmul_rn(gate_s[s], main_h),
                                             __fmul_rn(bf(stream_scales[s * hidden + h]), main_h));
                eo_bits = round_bf(eo_f);
                st_bits = round_bf(__fadd_rn(bf(st_in_bits), bf(eo_bits)));
            }
            if (CONTROL == 1) {
                if (isnan(bf(eo_bits))) eo_bits = 0;
                if (isnan(bf(st_bits))) st_bits = 0;
            }
            if (CONTROL == 2) {
                eo_bits = round_bf(__fadd_rn(bf(eo_bits), (float)repeat));
                st_bits = round_bf(__fadd_rn(bf(st_bits), (float)repeat));
            }
            if (engram_out) engram_out[idx] = eo_bits;
            if (streams_out) streams_out[idx] = st_bits;
        }
    }
}
}  // namespace hy4_engram_fp8_impl
