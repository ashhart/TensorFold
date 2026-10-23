// Ours: --kv-cache fp8 for Qwen3.5: key and value rows as e4m3, an fp32 scale (amax / 448) each, as qwen27 keeps them.

#include <cuda_fp8.h>

#include "qwen35.cuh"

using namespace tf_q35;

// A layer's block in Seg.kv: kvh * cap rows of HD bytes for keys then values, then kvh * cap fp32 scales for each.
__device__ __forceinline__ unsigned char* layer_at(const Seg& sg, int layer, int kvh) {
    return reinterpret_cast<unsigned char*>(sg.kv) + (size_t)layer * 2 * kvh * sg.cap * (HD + 4);
}

#include "kv_fp8.cuh"

// A key and value as e4m3 rows and their scales in their cache slot; every thread of the block calls it together.
struct StoreFp8 {
    int layer, kvh;
    float* red;
    __device__ __forceinline__ void operator()(const Seg& sg, int h, int slot, int d, __nv_bfloat16 k, __nv_bfloat16 v) const {
        unsigned char* keys = layer_at(sg, layer, kvh);
        unsigned char* values = keys + (size_t)kvh * sg.cap * HD;
        float* kscale = reinterpret_cast<float*>(values + (size_t)kvh * sg.cap * HD);
        float* vscale = kscale + (size_t)kvh * sg.cap;
        const size_t row = (size_t)h * sg.cap + slot;
        tf_kv_fp8::quant_row<HD>(bf(k), d, keys + row * HD, kscale + row, red);
        tf_kv_fp8::quant_row<HD>(bf(v), d, values + row * HD, vscale + row, red);
    }
};

// tf_q35_rope's rows and t, h, w positions (rope_rows) with the keys and values written as e4m3 rows and their scales.
extern "C" __global__ void __launch_bounds__(256) tf_q35_rope8(Geo geo, __nv_bfloat16* __restrict__ Qn, const __nv_bfloat16* __restrict__ Kn,
                                                                 const __nv_bfloat16* __restrict__ Vr, const Seg* __restrict__ segs,
                                                                 const int* __restrict__ rowseg, const Ancestor* __restrict__ rowanc, int layer,
                                                                 const int* __restrict__ P, const int* __restrict__ M) {
    __shared__ float red[HD / 32];
    rope_rows(geo, Qn, Kn, Vr, segs, rowseg, rowanc, P, M, StoreFp8{layer, geo.kvh, red});
}

// Eight e4m3 values as floats.
__device__ __forceinline__ void widen8(uint2 w, float (&f)[8]) {
    const __nv_fp8x2_storage_t* p = reinterpret_cast<const __nv_fp8x2_storage_t*>(&w);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const float2 v = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(p[i], __NV_E4M3)));
        f[2 * i] = v.x;
        f[2 * i + 1] = v.y;
    }
}

// tf_q35_attn_partial over e4m3 rows: a key's dot taken on its codes, then scaled; a value's weight carries its scale.
extern "C" __global__ void tf_q35_attn_partial8(Geo geo, const __nv_bfloat16* __restrict__ Q, const Seg* __restrict__ segs,
                                                  const int* __restrict__ rowseg, const Ancestor* __restrict__ rowanc, int layer,
                                                  const int* __restrict__ P, float scale,
                                                  float* __restrict__ PM, float* __restrict__ PL, float* __restrict__ PO) {
    constexpr int DPL = HD / 32;
    const int QH = geo.qh, KVH = geo.kvh;
    const int lane = threadIdx.x & 31, g = threadIdx.x >> 5;
    const int c = blockIdx.z, h = blockIdx.y;
    if (c >= MAXCH || past_rows(P, blockIdx.x)) return;
    const size_t m = blockIdx.x;
    const int qh = h * (QH / KVH) + g;
    const Seg sg = segs[rowseg[m]];
    const Ancestor anc = rowanc[m];
    const int keys = sg.len + rows_seen(anc), ck = chunk_keys(keys);
    const int k0 = c * ck, k1 = min(k0 + ck, keys);
    const size_t slot = (m * QH + qh) * MAXCH + c;
    if (k0 >= k1) {
        if (lane == 0) {
            PM[slot] = -INFINITY;
            PL[slot] = 0.0f;
        }
        return;
    }
    const unsigned char* K = layer_at(sg, layer, KVH);
    const unsigned char* V = K + (size_t)KVH * sg.cap * HD;
    const float* KS = reinterpret_cast<const float*>(V + (size_t)KVH * sg.cap * HD);
    const float* VS = KS + (size_t)KVH * sg.cap;
    float q[DPL], o[DPL];
#pragma unroll
    for (int i = 0; i < DPL; i++) {
        q[i] = bf(Q[(m * QH + qh) * HD + lane * DPL + i]);
        o[i] = 0.0f;
    }
    float mx = -INFINITY, l = 0.0f;
    for (int t = k0; t < k1; t++) {
        const int pos = t < sg.len ? t : sg.len + seen_row(anc, t - sg.len);
        const size_t row = (size_t)h * sg.cap + pos;
        float kf[DPL], vf[DPL];
        widen8(*reinterpret_cast<const uint2*>(K + row * HD + lane * DPL), kf);
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < DPL; i++) dot = fmaf(q[i], kf[i], dot);
        const float s = warp_sum(dot) * KS[row] * scale;
        const float mn = fmaxf(mx, s);
        const float a = __expf(mx - mn), b = __expf(s - mn);
        l = l * a + b;
        widen8(*reinterpret_cast<const uint2*>(V + row * HD + lane * DPL), vf);
        const float bv = b * VS[row];
#pragma unroll
        for (int i = 0; i < DPL; i++) o[i] = fmaf(bv, vf[i], o[i] * a);
        mx = mn;
    }
    if (lane == 0) {
        PM[slot] = mx;
        PL[slot] = l;
    }
#pragma unroll
    for (int i = 0; i < DPL; i++) PO[slot * HD + lane * DPL + i] = o[i];
}

// tf_q35k_compact over e4m3 rows: the path's keys, values and their scales move down to the slots after the cache.
extern "C" __global__ void __launch_bounds__(256) tf_q35k_compact8(Geo geo, Seg sg, Path path) {
    const int KVH = geo.kvh;
    const int layer = blockIdx.x, h = blockIdx.y;
    unsigned char* keys = layer_at(sg, layer, KVH);
    unsigned char* values = keys + (size_t)KVH * sg.cap * HD;
    float* kscale = reinterpret_cast<float*>(values + (size_t)KVH * sg.cap * HD);
    float* vscale = kscale + (size_t)KVH * sg.cap;
    const size_t base = (size_t)h * sg.cap;
    const int d = threadIdx.x;
    for (int j = 0; j < path.n; j++) {
        const int from = sg.len + path.rows[j], to = sg.len + j;
        if (from != to) {
            keys[(base + to) * HD + d] = keys[(base + from) * HD + d];
            values[(base + to) * HD + d] = values[(base + from) * HD + d];
            if (d == 0) {
                kscale[base + to] = kscale[base + from];
                vscale[base + to] = vscale[base + from];
            }
        }
        __syncthreads();
    }
}
