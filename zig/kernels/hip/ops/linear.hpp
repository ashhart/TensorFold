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
extern "C" __global__ void tf_gnorm_silu(const float* y, const void* z, void* out, int kind, long long n) {
    long long i = gid();
    if (i >= n) return;
    float s = rounded(silu(load(z, kind, i)), kind);
    store(out, kind, i, y[i] * s);
}

}  // namespace

