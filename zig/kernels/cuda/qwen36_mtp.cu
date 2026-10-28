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

constexpr int DPL = HD / 32;  // a lane's elements of a head row
constexpr int AHEAD = 8;      // ring slots loaded a group, the next group's loads in flight while one folds

// A key's score: the lane's products summed in order, then across the warp, scaled by 1/sqrt(HD).
__device__ __forceinline__ float score(const float (&q)[DPL], const __nv_bfloat16* k) {
    float dot = 0.0f;
#pragma unroll
    for (int i = 0; i < DPL; i++) dot = fmaf(q[i], bf(k[i]), dot);
    return warp_sum(dot) * (1.0f / 16.0f);
}

// One key folded into the running softmax (mx, l) and the lane's weighted values o.
__device__ __forceinline__ void fold(float s, const __nv_bfloat16* v, float& mx, float& l, float (&o)[DPL]) {
    const float mn = fmaxf(mx, s);
    const float a = __expf(mx - mn), b = __expf(s - mn);
    l = l * a + b;
#pragma unroll
    for (int i = 0; i < DPL; i++) o[i] = fmaf(b, bf(v[i]), o[i] * a);
    mx = mn;
}

// Ring slots [begin, end) of a ring of w folded in order: the same terms one at a time, with loads run ahead.
__device__ __forceinline__ void fold_ring(const __nv_bfloat16* rk, const __nv_bfloat16* rv, int w, int width, size_t off, long long begin,
                                          long long end, const float (&q)[DPL], float& mx, float& l, float (&o)[DPL]) {
    long long pos = begin;
    int slot = (int)(begin % w);
    uint4 kn[AHEAD], vn[AHEAD];
    const auto load = [&](uint4 (&k)[AHEAD], uint4 (&v)[AHEAD]) {
#pragma unroll
        for (int u = 0; u < AHEAD; u++) {
            k[u] = __ldg(reinterpret_cast<const uint4*>(rk + (size_t)slot * width + off));  // read-only path, as a __restrict__ ring's
            v[u] = __ldg(reinterpret_cast<const uint4*>(rv + (size_t)slot * width + off));
            slot = slot + 1 == w ? 0 : slot + 1;
        }
    };
    if (end - pos >= AHEAD) load(kn, vn);
    for (; end - pos >= AHEAD; pos += AHEAD) {
        uint4 kr[AHEAD], vr[AHEAD];
#pragma unroll
        for (int u = 0; u < AHEAD; u++) {
            kr[u] = kn[u];
            vr[u] = vn[u];
        }
        if (end - pos >= 2 * AHEAD) load(kn, vn);  // differences: no position near the type's end overflows
        float s[AHEAD];
#pragma unroll
        for (int u = 0; u < AHEAD; u++) s[u] = score(q, reinterpret_cast<const __nv_bfloat16*>(&kr[u]));
#pragma unroll
        for (int u = 0; u < AHEAD; u++) fold(s[u], reinterpret_cast<const __nv_bfloat16*>(&vr[u]), mx, l, o);
    }
    for (; pos < end; pos++) {
        fold(score(q, rk + (size_t)slot * width + off), rv + (size_t)slot * width + off, mx, l, o);
        slot = slot + 1 == w ? 0 : slot + 1;
    }
}

}  // namespace tf_q36m

using namespace tf_q36m;

// A warp a (row, head): RMS norm, rope on [0, ROT) as pairs (i, i + 32) at pos0 + row * step; OUT (rows, heads, HD)
extern "C" __global__ void __launch_bounds__(32) tf_q36m_norm_rope(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ W,
                                                                     __nv_bfloat16* __restrict__ OUT, int heads, int stride, int in_head,
                                                                     long long pos0, int step, float eps, float theta) {
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
    const int lane = threadIdx.x;
    const int m = blockIdx.x, h = blockIdx.y;
    const int kvh = h / (heads / kv_heads);
    const int width = kv_heads * HD;
    float q[DPL], o[DPL];
#pragma unroll
    for (int i = 0; i < DPL; i++) {
        q[i] = bf(Q[((size_t)m * heads + h) * HD + lane * DPL + i]);
        o[i] = 0.0f;
    }
    float mx = -INFINITY, l = 0.0f;
    fold_ring(RK, RV, W, width, (size_t)kvh * HD + lane * DPL, begin, pos0, q, mx, l, o);
    for (int r = 0; r <= m; r++) {
        const size_t at = ((size_t)r * kv_heads + kvh) * HD + lane * DPL;
        fold(score(q, KB + at), VB + at, mx, l, o);
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
    const int lane = threadIdx.x;
    const int m = blockIdx.x, h = blockIdx.y;
    const RingRow t = T[m];
    const int kvh = h / (heads / kv_heads);
    const int width = kv_heads * HD;
    float q[DPL], o[DPL];
#pragma unroll
    for (int i = 0; i < DPL; i++) {
        q[i] = bf(Q[((size_t)m * heads + h) * HD + lane * DPL + i]);
        o[i] = 0.0f;
    }
    float mx = -INFINITY, l = 0.0f;
    fold_ring(t.rk, t.rv, t.w, width, (size_t)kvh * HD + lane * DPL, t.begin, t.end, q, mx, l, o);
    for (int r = t.seg; r <= m; r++) {
        const size_t at = ((size_t)r * kv_heads + kvh) * HD + lane * DPL;
        fold(score(q, KB + at), VB + at, mx, l, o);
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
    const int lane = threadIdx.x;
    const int m = blockIdx.x, h = blockIdx.y;
    const int kvh = h / (heads / kv_heads);
    const int width = kv_heads * HD;
    float q[DPL], o[DPL];
#pragma unroll
    for (int i = 0; i < DPL; i++) {
        q[i] = bf(Q[((size_t)m * heads + h) * HD + lane * DPL + i]);
        o[i] = 0.0f;
    }
    float mx = -INFINITY, l = 0.0f;
    const size_t off = (size_t)kvh * HD + lane * DPL;
    fold_ring(RK, RV, W, width, off, begin, end, q, mx, l, o);
    const int* anc = ANC + (size_t)m * anc_stride;
    const int n = anc[0];
    for (int j = 0; j <= n; j++) {
        const size_t at = (j == n ? (size_t)m : (size_t)anc[1 + j]) * width + off;
        fold(score(q, (j == n ? KB : LK) + at), (j == n ? VB : LV) + at, mx, l, o);
    }
    const __nv_bfloat16* g = G + (size_t)m * gstride + (size_t)h * (2 * HD) + HD + lane * DPL;
#pragma unroll
    for (int i = 0; i < DPL; i++) {
        const float gate = 1.0f / (1.0f + __expf(-bf(g[i])));
        OUT[((size_t)m * heads + h) * HD + lane * DPL + i] = __float2bfloat16_rn(rbf(o[i] / l) * gate);
    }
}
