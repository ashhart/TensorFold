#pragma once

#include "../ops/common.hpp"

// Flash Next's residual streams: the hyper-connection norm, gates and mix, and the n-gram (PLE) gate and conv.

namespace {

constexpr int kFnThreads = 256;
constexpr int kFnWaves = kFnThreads / 32;
// The widest stream a thread walks in registers: dims / kFnThreads values each.
constexpr int kFnPer = 16;
// The hyper-connection norm's block: a stream's dims over 1024 threads, four values each at most.
constexpr int kNormThreads = 1024;
constexpr int kNormPer = 4;

// A block's sum of a value a thread (`scratch`: a value a wave): each wave by a fixed xor tree, then waves in order.
__device__ float fn_block_sum(float v, float* scratch) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) v += __shfl_xor(v, mask, 32);
    const int wave = threadIdx.x >> 5, waves = (blockDim.x + 31) >> 5;
    __syncthreads();
    if ((threadIdx.x & 31) == 0) scratch[wave] = v;
    __syncthreads();
    float total = 0.f;
    for (int w = 0; w < waves; ++w) total += scratch[w];
    return total;
}

// Block (s, r): h_s += round(branch * inject[s]) (tp: `world` fp32 shares, summed in order), then x = h rinv scale.
extern "C" __global__ void __launch_bounds__(kNormThreads)
tf_fn_hc_norm(const void* h, const void* branch, const void* inject, void* h_out, void* x_out, const float* scale,
              int kind, int streams, int dims, float eps, int world, long long stride) {
    __shared__ float scratch[kNormThreads / 32];
    const int s = blockIdx.x, r = blockIdx.y;
    const long long base = (static_cast<long long>(r) * streams + s) * dims;
    float v[kNormPer];
    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < kNormPer; ++j) {
        const int d = threadIdx.x + j * kNormThreads;
        v[j] = 0.f;
        if (d >= dims) continue;
        float x = load(h, kind, base + d);
        if (branch != nullptr) {
            const long long at = static_cast<long long>(r) * dims + d;
            float b;
            if (world > 0) {
                b = 0.f;
                for (int w = 0; w < world; ++w) b += static_cast<const float*>(branch)[w * stride + at];
                b = rounded(b, kind);
            } else {
                b = load(branch, kind, at);
            }
            const float add = rounded(b * load(inject, kind, static_cast<long long>(r) * streams + s), kind);
            x = rounded(x + add, kind);
        }
        v[j] = x;
        sum += x * x;
    }
    sum = fn_block_sum(sum, scratch);
    const float rinv = rsqrtf(sum / static_cast<float>(dims) + eps);
#pragma unroll
    for (int j = 0; j < kNormPer; ++j) {
        const int d = threadIdx.x + j * kNormThreads;
        if (d >= dims) continue;
        if (h_out != nullptr) store(h_out, kind, base + d, v[j]);
        store(x_out, kind, base + d, (v[j] * rinv) * scale[s * dims + d]);
    }
}

// From z [rows, low (+ streams with inject)] / streams: silu for the up projection, 2 sigmoid for the inject gates.
extern "C" __global__ void tf_fn_hc_gates(const float* z, void* act, void* inject, int kind, int rows, int low,
                                          int streams) {
    const long long i = gid();
    const int width = low + (inject != nullptr ? streams : 0);
    if (i >= static_cast<long long>(rows) * width) return;
    const int r = static_cast<int>(i / width), c = static_cast<int>(i % width);
    const float v = rounded(rounded(z[i], kind) / static_cast<float>(streams), kind);
    if (c < low) store(act, kind, static_cast<long long>(r) * low + c, silu(v));
    else store(inject, kind, static_cast<long long>(r) * streams + c - low, 2.f * rounded(sigmoid(v), kind));
}

// Row r of x normed over all its `width` values (fp32 sums: each thread's strided values, then the block's sum).
extern "C" __global__ void __launch_bounds__(kFnThreads)
tf_fn_rms_wide(const void* x, const float* scale, void* y, int kind, int width, float eps) {
    __shared__ float scratch[kFnWaves];
    const long long base = static_cast<long long>(blockIdx.x) * width;
    float sum = 0.f;
    for (int i = threadIdx.x; i < width; i += kFnThreads) {
        const float v = load(x, kind, base + i);
        sum += v * v;
    }
    const float rinv = rsqrtf(fn_block_sum(sum, scratch) / static_cast<float>(width) + eps);
    for (int i = threadIdx.x; i < width; i += kFnThreads) {
        store(y, kind, base + i, (load(x, kind, base + i) * rinv) * scale[i]);
    }
}

// mixed[r, d] = mean over the streams (in order) of round(sigmoid(round(g)) * x): g the up projection's fp32 sums.
extern "C" __global__ void tf_fn_hc_mix(const float* g, const void* x, void* mixed, int kind, int rows, int streams,
                                        int dims) {
    const long long i = gid();
    if (i >= static_cast<long long>(rows) * dims) return;
    const long long r = i / dims;
    const int d = static_cast<int>(i % dims);
    float total = 0.f;
    for (int s = 0; s < streams; ++s) {
        const long long at = (r * streams + s) * dims + d;
        const float gate = rounded(sigmoid(rounded(g[at], kind)), kind);
        total += rounded(gate * load(x, kind, at), kind);
    }
    store(mixed, kind, i, total / static_cast<float>(streams));
}

// Block (s, r): gated = sigmoid(signed sqrt(k . q / sqrt(dims))) v, k and q normed; normed is the conv input.
extern "C" __global__ void __launch_bounds__(kFnThreads)
tf_fn_ple_gate(const void* kv, const void* h, const float* key_scale, const float* query_scale, const float* conv_scale,
               void* gated, void* normed, int kind, int streams, int dims, float eps) {
    __shared__ float scratch[kFnWaves];
    const int s = blockIdx.x, r = blockIdx.y;
    const long long wide = static_cast<long long>(streams) * dims;
    const long long krow = r * (wide + dims), hrow = r * wide;
    float k[kFnPer], q[kFnPer], v[kFnPer];
    float sk = 0.f, sq = 0.f;
#pragma unroll
    for (int j = 0; j < kFnPer; ++j) {
        const int d = threadIdx.x + j * kFnThreads;
        k[j] = q[j] = v[j] = 0.f;
        if (d >= dims) continue;
        k[j] = load(kv, kind, krow + static_cast<long long>(s) * dims + d);
        q[j] = load(h, kind, hrow + static_cast<long long>(s) * dims + d);
        v[j] = load(kv, kind, krow + wide + d);
        sk += k[j] * k[j];
        sq += q[j] * q[j];
    }
    const float rk = rsqrtf(fn_block_sum(sk, scratch) / static_cast<float>(dims) + eps);
    const float rq = rsqrtf(fn_block_sum(sq, scratch) / static_cast<float>(dims) + eps);
    float dot = 0.f;
#pragma unroll
    for (int j = 0; j < kFnPer; ++j) {
        const int d = threadIdx.x + j * kFnThreads;
        if (d >= dims) continue;
        const float kn = rounded((k[j] * rk) * key_scale[s * dims + d], kind);
        const float qn = rounded((q[j] * rq) * query_scale[s * dims + d], kind);
        dot += rounded(kn * qn, kind);
    }
    dot = fn_block_sum(dot, scratch);
    const float g1 = rounded(rounded(dot, kind) / rounded(sqrtf(static_cast<float>(dims)), kind), kind);
    const float root = rounded(sqrtf(fmaxf(fabsf(g1), 1e-6f)), kind);
    const float g2 = rounded(g1 < 0.f ? -root : (g1 > 0.f ? root : 0.f), kind);
    const float gate = rounded(sigmoid(g2), kind);
    float sc = 0.f;
#pragma unroll
    for (int j = 0; j < kFnPer; ++j) {
        v[j] = rounded(gate * v[j], kind);
        sc += v[j] * v[j];
    }
    const float rc = rsqrtf(fn_block_sum(sc, scratch) / static_cast<float>(dims) + eps);
#pragma unroll
    for (int j = 0; j < kFnPer; ++j) {
        const int d = threadIdx.x + j * kFnThreads;
        if (d >= dims) continue;
        const long long at = hrow + static_cast<long long>(s) * dims + d;
        store(gated, kind, at, v[j]);
        store(normed, kind, at, (v[j] * rc) * conv_scale[s * dims + d]);
    }
}

// Channel c of row r: out = h + (gated + silu(conv)), the dilated causal conv over cin, which holds the tail first.
extern "C" __global__ void tf_fn_ple_conv(const void* cin, const float* weight, const void* gated, const void* h,
                                          void* out, int kind, int rows, int width, int taps, int dilation) {
    const long long i = gid();
    if (i >= static_cast<long long>(rows) * width) return;
    const long long r = i / width;
    const int c = static_cast<int>(i % width);
    float y = 0.f;
    for (int j = 0; j < taps; ++j) y = fmaf(weight[c * taps + j], load(cin, kind, (r + dilation * j) * width + c), y);
    const float yb = rounded(y, kind);
    const float act = rounded(yb * rounded(sigmoid(yb), kind), kind);
    store(out, kind, i, load(h, kind, i) + rounded(load(gated, kind, i) + act, kind));
}

}  // namespace
