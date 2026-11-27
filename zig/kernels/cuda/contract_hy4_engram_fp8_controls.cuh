// Planted controls and GPU serial FP32 oracle kernels for Hy4-flash FP8 Engram (contract_controls only).
#include "hy4_flash/engram_fp8.cuh"

namespace hy4_engram_fp8_ref {
using namespace hy4_engram_fp8_impl;

__device__ __forceinline__ float serial_warp_sum(const float (&lanes)[32]) {
    float v[32];
    for (int i = 0; i < 32; i++) v[i] = lanes[i];
    for (int o = 16; o > 0; o >>= 1) {
        float next[32];
        for (int l = 0; l < 32; l++) next[l] = __fadd_rn(v[l], v[l ^ o]);
        for (int l = 0; l < 32; l++) v[l] = next[l];
    }
    return v[0];
}

__device__ __forceinline__ float serial_block_reduce(const float (&thr)[kThreads]) {
    float wsum[kWarps];
    for (int w = 0; w < kWarps; w++) {
        float lanes[32];
        for (int l = 0; l < 32; l++) lanes[l] = thr[w * 32 + l];
        wsum[w] = serial_warp_sum(lanes);
    }
    float s = wsum[0];
    for (int w = 1; w < kWarps; w++) s = __fadd_rn(s, wsum[w]);
    return s;
}

// Old local butterfly reduction (pre-tf_common::warp_sum) for direct byte-equivalence verification.
template <int N>
__device__ __forceinline__ void reduce_channels_old_local(float (&v)[N], float (*shared)[kWarps]) {
    for (int j = 0; j < N; j++) {
        for (int o = 16; o > 0; o >>= 1) v[j] = __fadd_rn(v[j], __shfl_xor_sync(0xffffffffu, v[j], o));
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0)
        for (int j = 0; j < N; j++) shared[j][warp] = v[j];
    __syncthreads();
    for (int j = 0; j < N; j++) {
        float s = shared[j][0];
        for (int w = 1; w < kWarps; w++) s = __fadd_rn(s, shared[j][w]);
        v[j] = s;
    }
    __syncthreads();
}

// Wrong-order reduction control: reverses warp merge order so FP64 bounds pass while exact FP32 bytes differ.
template <int N>
__device__ __forceinline__ void reduce_channels_rev_order(float (&v)[N], float (*shared)[kWarps]) {
    for (int j = 0; j < N; j++) v[j] = tf_common::warp_sum(v[j]);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0)
        for (int j = 0; j < N; j++) shared[j][warp] = v[j];
    __syncthreads();
    for (int j = 0; j < N; j++) {
        float s = shared[j][kWarps - 1];
        for (int w = kWarps - 2; w >= 0; w--) s = __fadd_rn(s, shared[j][w]);
        v[j] = s;
    }
    __syncthreads();
}

template <bool OLD_LOCAL>
__device__ __forceinline__ void engram_gate_variant(const unsigned short* __restrict__ streams_in,
                                                    const float* __restrict__ proj,
                                                    const unsigned short* __restrict__ norm_keys,
                                                    const unsigned short* __restrict__ norm_queries,
                                                    const unsigned short* __restrict__ pre_gamma,
                                                    const unsigned short* __restrict__ stream_scales,
                                                    const unsigned char* __restrict__ mask, float eps,
                                                    unsigned short* engram_out, unsigned short* streams_out, int hidden) {
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
        acc[0] = __fadd_rn(acc[0], __fmul_rn(vv, vv));
        for (int s = 0; s < kStreams; s++) {
            const float kv = pv[(1 + s) * hidden + h];
            acc[1 + s] = __fadd_rn(acc[1 + s], __fmul_rn(kv, kv));
            const float qv = bf(sin[s * hidden + h]);
            acc[5 + s] = __fadd_rn(acc[5 + s], __fmul_rn(qv, qv));
        }
    }
    if constexpr (OLD_LOCAL) {
        reduce_channels_old_local<9>(acc, shared);
    } else {
        reduce_channels_rev_order<9>(acc, shared);
    }
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
            dot[s] = __fadd_rn(dot[s], __fmul_rn(kh, qh));
        }
    }
    if constexpr (OLD_LOCAL) {
        reduce_channels_old_local<kStreams>(dot, shared);
    } else {
        reduce_channels_rev_order<kStreams>(dot, shared);
    }
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
            if (engram_out) engram_out[idx] = eo_bits;
            if (streams_out) streams_out[idx] = st_bits;
        }
    }
}
}  // namespace hy4_engram_fp8_ref

#define HY4_ENGRAM_FP8_EMBED_CTL(NAME, CONTROL)                                                                        \
    extern "C" __global__ void __launch_bounds__(hy4_engram_fp8_impl::kThreads)                                        \
        NAME(const unsigned char* raw, const unsigned short* head_norms, float eps, unsigned short* out, int pairs,    \
             int heads, int dim, int repeat) {                                                                         \
        hy4_engram_fp8_impl::engram_embed<CONTROL>(raw, head_norms, eps, out, pairs, heads, dim, repeat);              \
    }
HY4_ENGRAM_FP8_EMBED_CTL(tf_ctl_hy4_engram_fp8_embed_bad_nan, 1)
HY4_ENGRAM_FP8_EMBED_CTL(tf_ctl_hy4_engram_fp8_embed_bad_repeat, 2)
HY4_ENGRAM_FP8_EMBED_CTL(tf_ctl_hy4_engram_fp8_embed_bad_tail, 3)
HY4_ENGRAM_FP8_EMBED_CTL(tf_ctl_hy4_engram_fp8_embed_bad_sums, 4)

#define HY4_ENGRAM_FP8_BIAS_CTL(NAME, CONTROL)                                                                         \
    extern "C" __global__ void __launch_bounds__(hy4_engram_fp8_impl::kThreads)                                        \
        NAME(const float* x, const unsigned short* bias, float* out, int rows, int cols, int repeat) {                 \
        hy4_engram_fp8_impl::engram_bias<CONTROL>(x, bias, out, rows, cols, repeat);                                   \
    }
HY4_ENGRAM_FP8_BIAS_CTL(tf_ctl_hy4_engram_fp8_bias_bad_nan, 1)
HY4_ENGRAM_FP8_BIAS_CTL(tf_ctl_hy4_engram_fp8_bias_bad_repeat, 2)
HY4_ENGRAM_FP8_BIAS_CTL(tf_ctl_hy4_engram_fp8_bias_bad_tail, 3)

#define HY4_ENGRAM_FP8_GATE_CTL(NAME, CONTROL)                                                                         \
    extern "C" __global__ void __launch_bounds__(hy4_engram_fp8_impl::kThreads)                                        \
        NAME(const unsigned short* streams_in, const float* proj, const unsigned short* norm_keys,                     \
             const unsigned short* norm_queries, const unsigned short* pre_gamma,                                      \
             const unsigned short* stream_scales, const unsigned char* mask, float eps, unsigned short* engram_out,    \
             unsigned short* streams_out, int hidden, int repeat) {                                                    \
        hy4_engram_fp8_impl::engram_gate<CONTROL>(streams_in, proj, norm_keys, norm_queries, pre_gamma, stream_scales, \
                                                  mask, eps, engram_out, streams_out, hidden, repeat);                 \
    }
HY4_ENGRAM_FP8_GATE_CTL(tf_ctl_hy4_engram_fp8_gate_bad_nan, 1)
HY4_ENGRAM_FP8_GATE_CTL(tf_ctl_hy4_engram_fp8_gate_bad_repeat, 2)
HY4_ENGRAM_FP8_GATE_CTL(tf_ctl_hy4_engram_fp8_gate_bad_tail, 3)
HY4_ENGRAM_FP8_GATE_CTL(tf_ctl_hy4_engram_fp8_gate_bad_sums, 4)

extern "C" __global__ void __launch_bounds__(hy4_engram_fp8_impl::kThreads)
    tf_ctl_hy4_engram_fp8_gate_bad_order(const unsigned short* streams_in, const float* proj,
                                         const unsigned short* norm_keys, const unsigned short* norm_queries,
                                         const unsigned short* pre_gamma, const unsigned short* stream_scales,
                                         const unsigned char* mask, float eps, unsigned short* engram_out,
                                         unsigned short* streams_out, int hidden, int repeat) {
    (void)repeat;
    hy4_engram_fp8_ref::engram_gate_variant<false>(streams_in, proj, norm_keys, norm_queries, pre_gamma, stream_scales,
                                                   mask, eps, engram_out, streams_out, hidden);
}

extern "C" __global__ void __launch_bounds__(hy4_engram_fp8_impl::kThreads)
    tf_ref_hy4_engram_fp8_gate_old_reduce(const unsigned short* streams_in, const float* proj,
                                          const unsigned short* norm_keys, const unsigned short* norm_queries,
                                          const unsigned short* pre_gamma, const unsigned short* stream_scales,
                                          const unsigned char* mask, float eps, unsigned short* engram_out,
                                          unsigned short* streams_out, int hidden, int repeat) {
    (void)repeat;
    hy4_engram_fp8_ref::engram_gate_variant<true>(streams_in, proj, norm_keys, norm_queries, pre_gamma, stream_scales,
                                                  mask, eps, engram_out, streams_out, hidden);
}

// Single-thread serial GPU FP32 reference for embed (1 thread per (row, head) pair).
extern "C" __global__ void __launch_bounds__(1)
    tf_ref_hy4_engram_fp8_embed(const unsigned char* raw, const unsigned short* head_norms, float eps,
                                unsigned short* out, int pairs, int heads, int dim, int repeat) {
    (void)repeat;
    using namespace hy4_engram_fp8_impl;
    const int pair = (int)blockIdx.x;
    if (pair >= pairs) return;
    const int head = pair % heads;
    const unsigned char* rec = raw + (unsigned long long)pair * (dim + 4);
    const float scale = read_scale(rec + dim);
    float lanes[32];
    for (int lane = 0; lane < 32; lane++) {
        float sq = 0.0f;
        for (int d = lane; d < dim; d += 32) {
            const float v = __fmul_rn(e4m3(rec[d]), scale);
            sq = __fadd_rn(sq, __fmul_rn(v, v));
        }
        lanes[lane] = sq;
    }
    const float sq = hy4_engram_fp8_ref::serial_warp_sum(lanes);
    const float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(sq, (float)dim), eps)));
    unsigned short* dst = out + (unsigned long long)pair * dim;
    const unsigned short* nw = head_norms + (unsigned long long)head * dim;
    for (int d = 0; d < dim; d++) {
        const float v = __fmul_rn(e4m3(rec[d]), scale);
        dst[d] = round_bf(__fmul_rn(__fmul_rn(v, inv), bf(nw[d])));
    }
}

// Single-thread serial GPU FP32 reference for bias (exact device NaN payloads).
extern "C" __global__ void __launch_bounds__(1)
    tf_ref_hy4_engram_fp8_bias(const float* x, const unsigned short* bias, float* out, int rows, int cols, int repeat) {
    (void)repeat;
    using namespace hy4_engram_fp8_impl;
    const unsigned long long r = blockIdx.x;
    if (r >= (unsigned long long)rows) return;
    for (int c = 0; c < cols; c++) {
        const unsigned long long i = r * (unsigned long long)cols + (unsigned long long)c;
        out[i] = __fadd_rn(x[i], bf(bias[c]));
    }
}

// Single-thread serial GPU FP32 reference for gate (1 thread per row, exact device intrinsics & tf_exp).
extern "C" __global__ void __launch_bounds__(1)
    tf_ref_hy4_engram_fp8_gate(const unsigned short* streams_in, const float* proj, const unsigned short* norm_keys,
                               const unsigned short* norm_queries, const unsigned short* pre_gamma,
                               const unsigned short* stream_scales, const unsigned char* mask, float eps,
                               unsigned short* engram_out, unsigned short* streams_out, int hidden, int repeat) {
    (void)repeat;
    using namespace hy4_engram_fp8_impl;
    const unsigned long long r = blockIdx.x;
    const bool active = (mask == nullptr) || (mask[r] != 0);
    const float* pv = proj + r * (1 + kStreams) * hidden;
    const unsigned short* sin = streams_in + r * kStreams * hidden;

    float inv_s[9];
    for (int j = 0; j < 9; j++) {
        float thr[kThreads];
        for (int t = 0; t < kThreads; t++) {
            float acc = 0.0f;
            for (int h = t; h < hidden; h += kThreads) {
                float v = 0.0f;
                if (j == 0) {
                    v = pv[h];
                } else if (j < 5) {
                    v = pv[j * hidden + h];
                } else {
                    v = bf(sin[(j - 5) * hidden + h]);
                }
                acc = __fadd_rn(acc, __fmul_rn(v, v));
            }
            thr[t] = acc;
        }
        const float sum = hy4_engram_fp8_ref::serial_block_reduce(thr);
        inv_s[j] = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(sum, (float)hidden), eps)));
    }

    float gate_s[kStreams];
    for (int s = 0; s < kStreams; s++) {
        float thr[kThreads];
        for (int t = 0; t < kThreads; t++) {
            float d = 0.0f;
            for (int h = t; h < hidden; h += kThreads) {
                const float kh = __fmul_rn(__fmul_rn(pv[(1 + s) * hidden + h], inv_s[1 + s]), bf(norm_keys[s * hidden + h]));
                const float qh = __fmul_rn(__fmul_rn(bf(sin[s * hidden + h]), inv_s[5 + s]), bf(norm_queries[s * hidden + h]));
                d = __fadd_rn(d, __fmul_rn(kh, qh));
            }
            thr[t] = d;
        }
        const float dot = hy4_engram_fp8_ref::serial_block_reduce(thr);
        const float g = __fmul_rn(dot, __fdiv_rn(1.0f, __fsqrt_rn((float)hidden)));
        const float mag = __fsqrt_rn(fmaxf(fabsf(g), 1.0e-6f));
        const float ssqrt = isnan(g) ? g : (g > 0.0f ? mag : (g < 0.0f ? -mag : 0.0f));
        gate_s[s] = sigmoid(ssqrt);
    }

    const float inv_v = inv_s[0];
    for (int h = 0; h < hidden; h++) {
        const float main_h = __fmul_rn(bf(pre_gamma[h]), __fmul_rn(pv[h], inv_v));
        for (int s = 0; s < kStreams; s++) {
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
            if (engram_out) engram_out[idx] = eo_bits;
            if (streams_out) streams_out[idx] = st_bits;
        }
    }
}

#undef HY4_ENGRAM_FP8_EMBED_CTL
#undef HY4_ENGRAM_FP8_BIAS_CTL
#undef HY4_ENGRAM_FP8_GATE_CTL
