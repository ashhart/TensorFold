// Gated DeltaNet inputs for every CUDA family (float64 reference: core/gdn_prep_ref.zig): conv, norms, g, beta.

// A warp a head (q heads, k heads, then v heads), lane l holding channels J l .. J l + J - 1; a row's bits are its own.

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

namespace tf_gdn_prep {

constexpr int MAXT = 8; // conv taps at most

struct Io {
    const __nv_bfloat16* qkv; // [rows, qkv_ld]: q (kh dk), k (kh dk), v (vh dk) channels
    long long qkv_ld;
    const __nv_bfloat16* state; // [streams * keep, c]: each stream's kept conv rows
    const __nv_bfloat16* conv;  // [c, keep + 1] taps a channel
    const int* win;             // [rows, keep + 1]: < keep a kept row, else qkv row win - keep
    const int* sid;             // [rows] stream (null: stream 0)
    const __nv_bfloat16* a;     // [rows, a_ld] the v heads' dt inputs
    long long a_ld;
    const __nv_bfloat16* b;     // [rows, b_ld] the v heads' beta inputs
    long long b_ld;
    const float* a_log;         // [vh]
    const float* dt_bias;       // [vh]
    __nv_bfloat16* q;           // [rows, kh, dk]
    __nv_bfloat16* k;           // [rows, kh, dk]
    __nv_bfloat16* v;           // [rows, vh, dk]
    float* g;                   // [rows, vh]
    float* beta;                // [rows, vh]
};

// Exactly rounded division on purpose: the same bits on every GPU, where div.full and ex2.approx are not.
__device__ __forceinline__ float sigmoid(float x) { return __fdiv_rn(1.f, __fadd_rn(1.f, expf(-x))); }

template <int J>
__device__ __forceinline__ void prep(const Io& io, int c, int kh, int vh, int keep, float eps) {
    constexpr int DK = 32 * J;
    const int lane = threadIdx.x & 31;
    const int64_t r = blockIdx.x;
    const int hd = blockIdx.y * (blockDim.x / 32) + threadIdx.x / 32;
    if (hd >= 2 * kh + vh) return;
    const int ch = hd * DK + J * lane;
    const int64_t srow = (io.sid != nullptr ? static_cast<int64_t>(io.sid[r]) : 0) * keep;
    int src[MAXT];
#pragma unroll
    for (int t = 0; t < MAXT; ++t) src[t] = t <= keep ? io.win[r * (keep + 1) + t] : 0;
    __nv_bfloat16 xv[MAXT][J], kv[MAXT][J];
#pragma unroll
    for (int t = 0; t < MAXT; ++t) { // every tap's loads issued before any is used
        if (t > keep) break;
        const __nv_bfloat16* x = src[t] < keep ? io.state + (srow + src[t]) * c : io.qkv + (src[t] - keep) * io.qkv_ld;
#pragma unroll
        for (int i = 0; i < J; ++i) xv[t][i] = x[ch + i], kv[t][i] = io.conv[(ch + i) * (keep + 1) + t];
    }
    float acc[J];
#pragma unroll
    for (int i = 0; i < J; ++i) acc[i] = 0.f;
#pragma unroll
    for (int t = 0; t < MAXT; ++t) {
        if (t > keep) break;
#pragma unroll
        for (int i = 0; i < J; ++i) acc[i] = __fmaf_rn(__bfloat162float(xv[t][i]), __bfloat162float(kv[t][i]), acc[i]);
    }
    float cv[J];
#pragma unroll
    for (int i = 0; i < J; ++i) cv[i] = __bfloat162float(__float2bfloat16_rn(__fmul_rn(acc[i], sigmoid(acc[i]))));
    if (hd < 2 * kh) { // q: l2norm / sqrt(dk), k: l2norm (rms with eps, scaled)
        float part = 0.f;
#pragma unroll
        for (int i = 0; i < J; ++i) part = __fmaf_rn(cv[i], cv[i], part);
        for (int m = 16; m > 0; m >>= 1) part = __fadd_rn(part, __shfl_xor_sync(0xFFFFFFFFu, part, m));
        const float inv = __fdiv_rn(1.f, __fsqrt_rn(__fadd_rn(__fdiv_rn(part, static_cast<float>(DK)), eps)));
        const bool is_q = hd < kh;
        const float scale = is_q ? 1.f / DK : __fdiv_rn(1.f, __fsqrt_rn(static_cast<float>(DK)));
        __nv_bfloat16* out = (is_q ? io.q : io.k) + (r * kh + (is_q ? hd : hd - kh)) * DK + J * lane;
#pragma unroll
        for (int i = 0; i < J; ++i) out[i] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(cv[i], inv), scale));
        return;
    }
    const int hv = hd - 2 * kh;
#pragma unroll
    for (int i = 0; i < J; ++i) io.v[(r * vh + hv) * DK + J * lane + i] = __float2bfloat16_rn(cv[i]);
    if (lane == 0) {
        const float x = __fadd_rn(__bfloat162float(io.a[r * io.a_ld + hv]), io.dt_bias[hv]);
        const float sp = x > 20.f ? x : log1pf(expf(x));
        io.g[r * vh + hv] = expf(-__fmul_rn(expf(io.a_log[hv]), sp));
        io.beta[r * vh + hv] = sigmoid(__bfloat162float(io.b[r * io.b_ld + hv]));
    }
}

}  // namespace tf_gdn_prep

using namespace tf_gdn_prep;

extern "C" __global__ void __launch_bounds__(128) tf_gdn_prep_64(const Io io, int c, int kh, int vh, int keep, float eps) {
    prep<2>(io, c, kh, vh, keep, eps);
}

extern "C" __global__ void __launch_bounds__(128) tf_gdn_prep_128(const Io io, int c, int kh, int vh, int keep, float eps) {
    prep<4>(io, c, kh, vh, keep, eps);
}

extern "C" __global__ void __launch_bounds__(128) tf_gdn_prep_256(const Io io, int c, int kh, int vh, int keep, float eps) {
    prep<8>(io, c, kh, vh, keep, eps);
}
