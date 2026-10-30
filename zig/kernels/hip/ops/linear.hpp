#pragma once

#include "common.hpp"

// The linear attention's prefill: conv, gate and gated norm.

namespace {

// Conv prefill: window [state | x], taps summed in order, then silu; row block 0 writes new_state, never `state`.
constexpr int kConvRows = 16;   // rows a conv_prefill thread slides over
constexpr int kConvTaps = 8;    // most taps it holds

extern "C" __global__ void tf_conv_prefill(const void* x, int kind, const float* weight, const float* state, float* out,
                                    float* new_state, int len, int channels, int kernel) {
    // a thread one channel over kConvRows rows, the window sliding in registers (each input read once)
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const int r0 = blockIdx.y * kConvRows;
    const int kept = kernel - 1;
    auto window = [&](int t) -> float {
        if (t < kept) return state ? state[static_cast<long long>(t) * channels + c] : 0.0f;
        return load(x, kind, static_cast<long long>(t - kept) * channels + c);
    };
    if (blockIdx.y == 0)
        for (int t = 0; t < kept; ++t) new_state[static_cast<long long>(t) * channels + c] = window(len + t);
    float w[kConvTaps], win[kConvTaps];
    for (int tap = 0; tap < kernel; ++tap) w[tap] = weight[static_cast<long long>(c) * kernel + tap];
    for (int tap = 0; tap + 1 < kernel; ++tap) win[tap] = window(r0 + tap);
    const int r1 = r0 + kConvRows < len ? r0 + kConvRows : len;
    for (int r = r0; r < r1; ++r) {
        win[kernel - 1] = window(r + kernel - 1);
        float acc = 0.0f;
        for (int tap = 0; tap < kernel; ++tap) {
            float p = win[tap] * w[tap];
            acc = acc + p;
        }
        out[static_cast<long long>(r) * channels + c] = silu(acc);
        for (int tap = 0; tap + 1 < kernel; ++tap) win[tap] = win[tap + 1];
    }
}

// A DeltaNet layer's prompt prework in one launch, the chain's values: conv, q/k head norms, v, new state, gate, beta.
constexpr int kPrepSeg = 16;

__device__ __forceinline__ float prep_window(const void* x, int kind, const float* state, int kept, int channels, int c,
                                             int t) {
    if (t < kept) return state ? state[static_cast<long long>(t) * channels + c] : 0.0f;
    return load(x, kind, static_cast<long long>(t - kept) * channels + c);
}

extern "C" __global__ void __launch_bounds__(256)
tf_gdn_prompt_prep(const void* x, int kind, const float* weight, const float* state, float* new_state, float* q_out,
                   float* k_out, float* v_out, const float* qw, const float* kw_w, float eps, int len, int kwidth,
                   int vwidth, int kernel, const void* a, const void* b, const float* a_log, const float* dt_bias,
                   float* gate, float* beta, int heads) {
    const int channels = 2 * kwidth + vwidth, kept = kernel - 1;
    const int segs = (len + kPrepSeg - 1) / kPrepSeg;
    const int qk_heads = 2 * kwidth / 128;
    const int nA = (qk_heads * segs + 7) / 8, nB = ((vwidth + 255) / 256) * segs;
    const int nC = (len * heads + 255) / 256;
    int blk = blockIdx.x;
    if (blk < nA) {
        // q / k: a wave a head a segment
        const int unit = blk * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
        if (unit >= qk_heads * segs) return;
        const int hd = unit % qk_heads, sg = unit / qk_heads;
        const bool is_q = hd < qk_heads / 2;
        const int r0 = sg * kPrepSeg, r1 = min(len, r0 + kPrepSeg);
        float w[4][kConvTaps], win[4][kConvTaps];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int c = hd * 128 + lane + 32 * j;
            for (int tap = 0; tap < kernel; ++tap) w[j][tap] = weight[static_cast<long long>(c) * kernel + tap];
            for (int tap = 0; tap + 1 < kernel; ++tap) win[j][tap] = prep_window(x, kind, state, kept, channels, c, r0 + tap);
        }
        const float* nw = is_q ? qw : kw_w;
        float* out = is_q ? q_out : k_out;
        const int head = is_q ? hd : hd - qk_heads / 2;
        for (int r = r0; r < r1; ++r) {
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = hd * 128 + lane + 32 * j;
                win[j][kernel - 1] = prep_window(x, kind, state, kept, channels, c, r + kernel - 1);
                float acc = 0.0f;
                for (int tap = 0; tap < kernel; ++tap) {
                    float p = win[j][tap] * w[j][tap];
                    acc = acc + p;
                }
                v[j] = silu(acc);
                for (int tap = 0; tap + 1 < kernel; ++tap) win[j][tap] = win[j][tap + 1];
            }
            float sum = 0.f;
#pragma unroll
            for (int j = 0; j < 4; ++j) sum += v[j] * v[j];
#pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) sum += __shfl_xor(sum, mask, 32);
            const float inv = rsqrtf(__shfl(sum, 0, 32) / 128.f + eps);
            float* row = out + (static_cast<long long>(r) * (kwidth / 128) + head) * 128;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int i = lane + 32 * j;
                float value = v[j] * inv;
                value *= nw[i];
                row[i] = value;
            }
        }
        return;
    }
    blk -= nA;
    if (blk < nB) {
        // v: a thread a channel a segment
        const int per = (vwidth + 255) / 256;
        const int vc = (blk % per) * 256 + threadIdx.x, sg = blk / per;
        if (vc >= vwidth) return;
        const int c = 2 * kwidth + vc;
        const int r0 = sg * kPrepSeg, r1 = min(len, r0 + kPrepSeg);
        float w[kConvTaps], win[kConvTaps];
        for (int tap = 0; tap < kernel; ++tap) w[tap] = weight[static_cast<long long>(c) * kernel + tap];
        for (int tap = 0; tap + 1 < kernel; ++tap) win[tap] = prep_window(x, kind, state, kept, channels, c, r0 + tap);
        for (int r = r0; r < r1; ++r) {
            win[kernel - 1] = prep_window(x, kind, state, kept, channels, c, r + kernel - 1);
            float acc = 0.0f;
            for (int tap = 0; tap < kernel; ++tap) {
                float p = win[tap] * w[tap];
                acc = acc + p;
            }
            v_out[static_cast<long long>(r) * vwidth + vc] = silu(acc);
            for (int tap = 0; tap + 1 < kernel; ++tap) win[tap] = win[tap + 1];
        }
        return;
    }
    blk -= nB;
    if (blk < nC) {
        // tf_gdn_gate_prefill's gate and beta
        const long long i = static_cast<long long>(blk) * 256 + threadIdx.x;
        if (i >= static_cast<long long>(len) * heads) return;
        const int h = static_cast<int>(i % heads);
        beta[i] = sigmoid(load(b, kind, i));
        float xa = load(a, kind, i) + dt_bias[h];
        float xb = xa * 1.0f;
        float sp = xb > 20.0f ? xa : log1pf(expf(xb)) / 1.0f;
        float e = -expf(a_log[h]);
        gate[i] = expf(e * sp);
        return;
    }
    blk -= nC;
    // the conv's new state: the last kept inputs of every channel
    const long long i = static_cast<long long>(blk) * 256 + threadIdx.x;
    if (i >= static_cast<long long>(kept) * channels) return;
    const int t = static_cast<int>(i / channels), c = static_cast<int>(i % channels);
    new_state[i] = prep_window(x, kind, state, kept, channels, c, len + t);
}

// _gate_beta: beta = sigmoid(b), gate = exp(-exp(a_log) * softplus(a + dt_bias)) with torch's softplus.
extern "C" __global__ void tf_gdn_gate_prefill(const void* a, const void* b, int kind, const float* a_log,
                                        const float* dt_bias, float* gate, float* beta, int count, int heads) {
    long long i = gid();
    if (i >= static_cast<long long>(count) * heads) return;
    int h = static_cast<int>(i % heads);
    beta[i] = sigmoid(load(b, kind, i));
    float x = load(a, kind, i) + dt_bias[h];
    float xb = x * 1.0f;
    float sp = xb > 20.0f ? x : log1pf(expf(xb)) / 1.0f;
    float e = -expf(a_log[h]);
    gate[i] = expf(e * sp);
}

// out = y * silu(z) widened, rounded to kind: the linear-attention output before its projection.
extern "C" __global__ void tf_gnorm_silu(const float* y, const void* z, void* out, int kind, long long n,
                                         int sigmoid_gate) {
    long long i = gid();
    if (i >= n) return;
    const float zv = load(z, kind, i);
    float s = rounded(sigmoid_gate ? sigmoid(zv) : silu(zv), kind);
    store(out, kind, i, y[i] * s);
}

}  // namespace
