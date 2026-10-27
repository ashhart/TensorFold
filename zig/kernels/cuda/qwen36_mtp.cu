// The Qwen3.6 MTP drafter's glue on CUDA (cuda_mtp.zig): its attention at the target's head width over a ring.

#include <cuda_bf16.h>
#include <stdint.h>
#include "common/reduce.cuh"

namespace tf_q36m {

constexpr int HD = 256;  // the head width (q, k, v)
constexpr int ROT = 64;  // rotary dims: pairs (i, i + 32), the target's partial rotary

__device__ __forceinline__ float bf(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float rbf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

using tf_common::warp_sum;

}  // namespace tf_q36m

using namespace tf_q36m;

// A warp a (row, head): RMS norm, rope on [0, ROT) as pairs (i, i + 32) at pos0 + row * step; OUT (rows, heads, HD)
extern "C" __global__ void __launch_bounds__(32) tf_q36m_norm_rope(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ W,
                                                                     __nv_bfloat16* __restrict__ OUT, int heads, int stride, int in_head,
                                                                     long long pos0, int step, float eps, float theta) {
    constexpr int DPL = HD / 32;
    const int head = blockIdx.x % heads;
    const size_t m = blockIdx.x / heads;
    const int t = threadIdx.x;
    const __nv_bfloat16* x = X + m * stride + (size_t)head * in_head;  // in_head 512 for q beside its gate, 256 for k
    float v[DPL], acc = 0.0f;
#pragma unroll
    for (int i = 0; i < DPL; i++) {
        v[i] = bf(x[DPL * t + i]);
        acc = fmaf(v[i], v[i], acc);
    }
    acc = warp_sum(acc);
    const float inv = rsqrtf(acc / (float)HD + eps);
    float n[DPL];
#pragma unroll
    for (int i = 0; i < DPL; i++) n[i] = rbf(bf(W[DPL * t + i]) * (v[i] * inv));
    // a rotary pair's partner sits 32 elements away: DPL = 8 a lane, so 4 lanes away at the same i
    const float pos = (float)(pos0 + (long long)m * step);
    __nv_bfloat16* out = OUT + (m * heads + head) * HD;
#pragma unroll
    for (int i = 0; i < DPL; i++) {
        const int d = DPL * t + i;
        const float partner = __shfl_xor_sync(0xffffffffu, n[i], 4);
        float r = n[i];
        if (d < ROT) {
            const int j = d < ROT / 2 ? d : d - ROT / 2;
            const float angle = pos * powf(theta, -(float)j / (float)(ROT / 2));
            float sn, cs;
            sincosf(angle, &sn, &cs);
            r = d < ROT / 2 ? n[i] * cs - partner * sn : n[i] * cs + partner * sn;
        }
        out[d] = __float2bfloat16_rn(r);
    }
}

// A warp a (row, head): ring slots [begin, pos0), then block rows 0..m; OUT = softmax-weighted values x sigmoid(gate).
extern "C" __global__ void __launch_bounds__(32) tf_q36m_attn(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ G, int gstride,
                                                                const __nv_bfloat16* __restrict__ KB, const __nv_bfloat16* __restrict__ VB,
                                                                const __nv_bfloat16* __restrict__ RK, const __nv_bfloat16* __restrict__ RV, int heads,
                                                                int kv_heads, int W, long long begin, long long pos0, __nv_bfloat16* __restrict__ OUT) {
    constexpr int DPL = HD / 32;
    const int lane = threadIdx.x;
    const int m = blockIdx.x, h = blockIdx.y;
    const int kvh = h / (heads / kv_heads);
    const int width = kv_heads * HD;
    const float scale = 1.0f / 16.0f;
    float q[DPL], o[DPL];
#pragma unroll
    for (int i = 0; i < DPL; i++) {
        q[i] = bf(Q[((size_t)m * heads + h) * HD + lane * DPL + i]);
        o[i] = 0.0f;
    }
    float mx = -INFINITY, l = 0.0f;
    for (long long pos = begin; pos < pos0; pos++) {
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
    for (int r = 0; r <= m; r++) {
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
    const __nv_bfloat16* g = G + (size_t)m * gstride + (size_t)h * (2 * HD) + HD + lane * DPL;  // the gate half
#pragma unroll
    for (int i = 0; i < DPL; i++) {
        const float gate = 1.0f / (1.0f + __expf(-bf(g[i])));
        OUT[((size_t)m * heads + h) * HD + lane * DPL + i] = __float2bfloat16_rn(rbf(o[i] / l) * gate);
    }
}

// A packed row's ring: its keys and values, its slots [begin, end) of a ring of w, and its segment's first row.
struct RingRow {
    const __nv_bfloat16* rk;
    const __nv_bfloat16* rv;
    long long begin;
    long long end;
    int seg;
    int w;
};

// tf_q36m_attn over many streams' rows, a warp a (row, head): row m's ring slots T[m], then the block rows T[m].seg..m.
extern "C" __global__ void __launch_bounds__(32) tf_q36m_attn_rows(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ G, int gstride,
                                                                     const __nv_bfloat16* __restrict__ KB, const __nv_bfloat16* __restrict__ VB,
                                                                     const RingRow* __restrict__ T, int heads, int kv_heads,
                                                                     __nv_bfloat16* __restrict__ OUT) {
    constexpr int DPL = HD / 32;
    const int lane = threadIdx.x;
    const int m = blockIdx.x, h = blockIdx.y;
    const RingRow t = T[m];
    const int kvh = h / (heads / kv_heads);
    const int width = kv_heads * HD;
    const float scale = 1.0f / 16.0f;
    float q[DPL], o[DPL];
#pragma unroll
    for (int i = 0; i < DPL; i++) {
        q[i] = bf(Q[((size_t)m * heads + h) * HD + lane * DPL + i]);
        o[i] = 0.0f;
    }
    float mx = -INFINITY, l = 0.0f;
    for (long long pos = t.begin; pos < t.end; pos++) {
        const __nv_bfloat16* k = t.rk + (size_t)(pos % t.w) * width + kvh * HD + lane * DPL;
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < DPL; i++) dot = fmaf(q[i], bf(k[i]), dot);
        const float s = warp_sum(dot) * scale;
        const float mn = fmaxf(mx, s);
        const float a = __expf(mx - mn), b = __expf(s - mn);
        l = l * a + b;
        const __nv_bfloat16* v = t.rv + (size_t)(pos % t.w) * width + kvh * HD + lane * DPL;
#pragma unroll
        for (int i = 0; i < DPL; i++) o[i] = fmaf(b, bf(v[i]), o[i] * a);
        mx = mn;
    }
    for (int r = t.seg; r <= m; r++) {
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
    const __nv_bfloat16* g = G + (size_t)m * gstride + (size_t)h * (2 * HD) + HD + lane * DPL;  // the gate half
#pragma unroll
    for (int i = 0; i < DPL; i++) {
        const float gate = 1.0f / (1.0f + __expf(-bf(g[i])));
        OUT[((size_t)m * heads + h) * HD + lane * DPL + i] = __float2bfloat16_rn(rbf(o[i] / l) * gate);
    }
}

// Tree rows at one depth, a warp a (row, head): ring [begin, end), the ancestor lanes ANC[m] names, then its own key.
extern "C" __global__ void __launch_bounds__(32) tf_q36m_attn_tree(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ G, int gstride,
                                                                     const __nv_bfloat16* __restrict__ KB, const __nv_bfloat16* __restrict__ VB,
                                                                     const __nv_bfloat16* __restrict__ LK, const __nv_bfloat16* __restrict__ LV,
                                                                     const int* __restrict__ ANC, int anc_stride, const __nv_bfloat16* __restrict__ RK,
                                                                     const __nv_bfloat16* __restrict__ RV, int heads, int kv_heads, int W, long long begin,
                                                                     long long end, __nv_bfloat16* __restrict__ OUT) {
    constexpr int DPL = HD / 32;
    const int lane = threadIdx.x;
    const int m = blockIdx.x, h = blockIdx.y;
    const int kvh = h / (heads / kv_heads);
    const int width = kv_heads * HD;
    const float scale = 1.0f / 16.0f;
    float q[DPL], o[DPL];
#pragma unroll
    for (int i = 0; i < DPL; i++) {
        q[i] = bf(Q[((size_t)m * heads + h) * HD + lane * DPL + i]);
        o[i] = 0.0f;
    }
    float mx = -INFINITY, l = 0.0f;
    for (long long pos = begin; pos < end; pos++) {
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
    const int* anc = ANC + (size_t)m * anc_stride;
    const int n = anc[0];
    for (int j = 0; j <= n; j++) {
        const bool self = j == n;
        const __nv_bfloat16* k = (self ? KB + (size_t)m * width : LK + (size_t)anc[1 + j] * width) + kvh * HD + lane * DPL;
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < DPL; i++) dot = fmaf(q[i], bf(k[i]), dot);
        const float s = warp_sum(dot) * scale;
        const float mn = fmaxf(mx, s);
        const float a = __expf(mx - mn), b = __expf(s - mn);
        l = l * a + b;
        const __nv_bfloat16* v = (self ? VB + (size_t)m * width : LV + (size_t)anc[1 + j] * width) + kvh * HD + lane * DPL;
#pragma unroll
        for (int i = 0; i < DPL; i++) o[i] = fmaf(b, bf(v[i]), o[i] * a);
        mx = mn;
    }
    const __nv_bfloat16* g = G + (size_t)m * gstride + (size_t)h * (2 * HD) + HD + lane * DPL;
#pragma unroll
    for (int i = 0; i < DPL; i++) {
        const float gate = 1.0f / (1.0f + __expf(-bf(g[i])));
        OUT[((size_t)m * heads + h) * HD + lane * DPL + i] = __float2bfloat16_rn(rbf(o[i] / l) * gate);
    }
}
