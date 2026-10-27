// DFlash2 drafter glue on CUDA (cuda_draft.zig): a block of rows over the target's taps, draft_ops.metal's rounding.

#include <cuda_bf16.h>
#include <stdint.h>
#include "common/reduce.cuh"

namespace tf_q35d {

constexpr int HD = 128;  // the drafter's head width (q, k, v)

__device__ __forceinline__ float bf(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float rbf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

using tf_common::warp_sum;

}  // namespace tf_q35d

using namespace tf_q35d;

// Tap rows (start + i) of a stream's ring of R rows, `width` bf16 each, into OUT rows 0..n. Grid (n), 256 threads.
extern "C" __global__ void tf_q35d_gather_taps(const __nv_bfloat16* __restrict__ RING, int R, long long start, int width,
                                                 __nv_bfloat16* __restrict__ OUT) {
    const long long row = start + blockIdx.x;
    const __nv_bfloat16* src = RING + (size_t)(row % R) * width;
    __nv_bfloat16* dst = OUT + (size_t)blockIdx.x * width;
    for (int i = threadIdx.x; i < width; i += blockDim.x) dst[i] = src[i];
}

// Rows of a window to gather, by value (a tree's kept path).
struct TapRows {
    int n;
    int rows[64];
};

// Tap rows (start + rows.rows[i]) of a stream's ring into OUT rows 0..n: a kept path's rows in order. Grid (n).
extern "C" __global__ void tf_q35d_gather_rows(const __nv_bfloat16* __restrict__ RING, int R, long long start, TapRows rows, int width,
                                                 __nv_bfloat16* __restrict__ OUT) {
    const long long row = start + rows.rows[blockIdx.x];
    const __nv_bfloat16* src = RING + (size_t)(row % R) * width;
    __nv_bfloat16* dst = OUT + (size_t)blockIdx.x * width;
    for (int i = threadIdx.x; i < width; i += blockDim.x) dst[i] = src[i];
}

// y[t, c] = sum over lags l of (BASE[l, c] + DYN[t, l, c / group]) * x[t - l, c] in bf16 steps; x[-1] = 0; RES last.
extern "C" __global__ void tf_q35d_dyn_conv(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ DYN,
                                              const __nv_bfloat16* __restrict__ BASE, const __nv_bfloat16* __restrict__ RES,
                                              __nv_bfloat16* __restrict__ OUT, int D, int group, int branch) {
    const int t = blockIdx.x;
    const int groups = D / group;
    for (int c = threadIdx.x; c < D; c += blockDim.x) {
        const int g = c / group;
        float total = 0.0f;
#pragma unroll
        for (int l = 0; l < 2; l++) {
            if (t - l < 0) continue;
            const float x = bf(X[(size_t)(t - l) * D + c]);
            total = rbf(total + rbf(bf(BASE[(branch * 2 + l) * D + c]) * x));
            total = rbf(total + rbf(bf(DYN[(size_t)t * (4 * groups) + (branch * 2 + l) * groups + g]) * x));
        }
        if (RES != nullptr) total = rbf(total + bf(RES[(size_t)t * D + c]));
        OUT[(size_t)t * D + c] = __float2bfloat16_rn(total);
    }
}

// Per-head RMSNorm then rotate-half RoPE at position pos0 + row; OUT packed (rows, heads, HD), grid (rows * heads).
extern "C" __global__ void __launch_bounds__(32) tf_q35d_norm_rope(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ W,
                                                                     __nv_bfloat16* __restrict__ OUT, int heads, int stride, long long pos0,
                                                                     float eps, float theta) {
    const int head = blockIdx.x % heads;
    const size_t m = blockIdx.x / heads;
    const int t = threadIdx.x;
    const __nv_bfloat16* x = X + m * stride + head * HD;
    float v[4], acc = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        v[i] = bf(x[4 * t + i]);
        acc += v[i] * v[i];
    }
    acc = warp_sum(acc);
    const float inv = rsqrtf(acc / (float)HD + eps);
    float n[4];
#pragma unroll
    for (int i = 0; i < 4; i++) n[i] = rbf(rbf(v[i] * inv) * bf(W[4 * t + i]));
    // the pair of lane element d is d + HD / 2: this lane's partner values come from the lane HD / 8 = 16 away
    const float pos = (float)(pos0 + (long long)m);
    __nv_bfloat16* out = OUT + (m * heads + head) * HD;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int d = 4 * t + i;
        const float partner = __shfl_xor_sync(0xffffffffu, n[i], 16);
        const int j = d < HD / 2 ? d : d - HD / 2;
        const float angle = pos * powf(theta, -(float)j / (float)(HD / 2));
        float sn, cs;
        sincosf(angle, &sn, &cs);
        out[d] = __float2bfloat16_rn(d < HD / 2 ? n[i] * cs - partner * sn : n[i] * cs + partner * sn);
    }
}

// Rows of packed (kv_heads, HD) keys and values at positions pos0 + i into a layer's rings (slot pos % W).
extern "C" __global__ void tf_q35d_ring_put(const __nv_bfloat16* __restrict__ K, const __nv_bfloat16* __restrict__ V,
                                              __nv_bfloat16* __restrict__ RK, __nv_bfloat16* __restrict__ RV, int width, long long pos0, int W) {
    const int i = blockIdx.x;
    const size_t slot = (size_t)((pos0 + i) % W);
    for (int c = threadIdx.x; c < width; c += blockDim.x) {
        RK[slot * width + c] = K[(size_t)i * width + c];
        RV[slot * width + c] = V[(size_t)i * width + c];
    }
}

// Block attention: a row sees ring positions [max(begin, p - window + 1), pos0) and every block row; a warp a row.
extern "C" __global__ void __launch_bounds__(32) tf_q35d_attn(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ KB,
                                                                const __nv_bfloat16* __restrict__ VB, const __nv_bfloat16* __restrict__ RK,
                                                                const __nv_bfloat16* __restrict__ RV, int rows, int heads, int kv_heads, int W,
                                                                long long begin, long long pos0, int window, __nv_bfloat16* __restrict__ OUT) {
    constexpr int DPL = HD / 32;
    const int lane = threadIdx.x;
    const int m = blockIdx.x, h = blockIdx.y;
    const int kvh = h / (heads / kv_heads);
    const int width = kv_heads * HD;
    const float scale = rsqrtf((float)HD);
    float q[DPL], o[DPL];
#pragma unroll
    for (int i = 0; i < DPL; i++) {
        q[i] = bf(Q[((size_t)m * heads + h) * HD + lane * DPL + i]);
        o[i] = 0.0f;
    }
    float mx = -INFINITY, l = 0.0f;
    const long long p = pos0 + m;
    long long first = p - window + 1;
    if (first < begin) first = begin;
    for (long long pos = first; pos < pos0; pos++) {
        const __nv_bfloat16* k = RK + (size_t)(pos % W) * width + kvh * HD + lane * DPL;
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < DPL; i++) dot = fmaf(q[i], bf(k[i]), dot);
        const float s = warp_sum(dot) * scale;
        const float mn = fmaxf(mx, s);
        const float a = __expf(mx - mn), b = __expf(s - mn);
        l = l * a + b;
        const __nv_bfloat16* v = RV + (size_t)(pos % W) * width + kvh * HD + lane * DPL;
#pragma unroll
        for (int i = 0; i < DPL; i++) o[i] = fmaf(b, bf(v[i]), o[i] * a);
        mx = mn;
    }
    for (int r = 0; r < rows; r++) {
        const __nv_bfloat16* k = KB + ((size_t)r * kv_heads + kvh) * HD + lane * DPL;
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < DPL; i++) dot = fmaf(q[i], bf(k[i]), dot);
        const float s = warp_sum(dot) * scale;
        const float mn = fmaxf(mx, s);
        const float a = __expf(mx - mn), b = __expf(s - mn);
        l = l * a + b;
        const __nv_bfloat16* v = VB + ((size_t)r * kv_heads + kvh) * HD + lane * DPL;
#pragma unroll
        for (int i = 0; i < DPL; i++) o[i] = fmaf(b, bf(v[i]), o[i] * a);
        mx = mn;
    }
#pragma unroll
    for (int i = 0; i < DPL; i++) OUT[((size_t)m * heads + h) * HD + lane * DPL + i] = __float2bfloat16_rn(o[i] / l);
}

#include "qwen35_draft_frame.cuh"
