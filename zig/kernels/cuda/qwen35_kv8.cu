// Ours: --kv-cache fp8 for Qwen3.5: key and value rows as e4m3, an fp32 scale (amax / 448) each, as qwen27 keeps them.

#include <cuda_fp8.h>

#include "qwen35.cuh"
#include "qwen35_page_run.h"

using namespace tf_q35;

// A key's and value's codes and scales.
struct Row8 {
    unsigned char *k, *v;
    float *ks, *vs;
};

// Codes and scales at (layer, h, pos): a layer's block of n = cap rows a head, or of ptok rows in pos's page.
__device__ __forceinline__ Row8 row8(const Seg& sg, int layer, int kvh, int h, int pos) {
    unsigned char* base = reinterpret_cast<unsigned char*>(sg.kv);
    int n = sg.cap, p = pos;
    if (sg.pt != 0) {
        base += (size_t)reinterpret_cast<const int*>(sg.pt)[pos / sg.ptok] * sg.ps;
        n = sg.ptok;
        p = pos % sg.ptok;
    }
    unsigned char* keys = base + (size_t)layer * 2 * kvh * n * (HD + 4);
    unsigned char* values = keys + (size_t)kvh * n * HD;
    float* kscale = reinterpret_cast<float*>(values + (size_t)kvh * n * HD);
    const size_t row = (size_t)h * n + p;
    return {keys + row * HD, values + row * HD, kscale + row, kscale + (size_t)kvh * n + row};
}

#include "kv_fp8.cuh"

// A key and value as e4m3 rows and their scales in their cache slot; every thread of the block calls it together.
struct StoreFp8 {
    int layer, kvh;
    float* red;
    __device__ __forceinline__ void operator()(const Seg& sg, int h, int slot, int d, __nv_bfloat16 k, __nv_bfloat16 v) const {
        const Row8 r = row8(sg, layer, kvh, h, slot);
        tf_kv_fp8::quant_row<HD>(bf(k), d, r.k, r.ks, red);
        tf_kv_fp8::quant_row<HD>(bf(v), d, r.v, r.vs, red);
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


#include "qwen35_attn_tc.cuh"

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

// Decode attention over e4m3 rows on tensor cores: a short stream's rows a block each, a long stream's in groups.


// A layer's block in Seg.kv: kvh * cap rows of HD bytes for keys then values, then kvh * cap fp32 scales for each.
__device__ __forceinline__ unsigned char* layer_at(const Seg& sg, int layer, int kvh) {
    return reinterpret_cast<unsigned char*>(sg.kv) + (size_t)layer * 2 * kvh * sg.cap * (HD + 4);
}

// Contiguous attention retains the pre-paging FP8 body.
extern "C" __global__ void tf_q35_attn_partial8_contiguous(Geo geo, const __nv_bfloat16* __restrict__ Q, const Seg* __restrict__ segs,
                                                  const int* __restrict__ rowseg, const Ancestor* __restrict__ rowanc, int layer,
                                                  const int* __restrict__ P, float scale,
                                                  float* __restrict__ PM, float* __restrict__ PL, float* __restrict__ PO) {
    constexpr int DPL = HD / 32;
    const int QH = geo.qh, KVH = geo.kvh;
    const int lane = threadIdx.x & 31, g = threadIdx.x >> 5;
    const int c = blockIdx.x, h = blockIdx.y;
    if (c >= MAXCH || past_rows(P, blockIdx.z)) return;
    const size_t m = blockIdx.z;
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

// tf_q35_attn_partial over e4m3 rows: a key's dot taken on its codes, then scaled; a value's weight carries its scale.
extern "C" __global__ void tf_q35_attn_partial8_legacy(Geo geo, const __nv_bfloat16* __restrict__ Q, const Seg* __restrict__ segs,
                                                  const int* __restrict__ rowseg, const Ancestor* __restrict__ rowanc, int layer,
                                                  const int* __restrict__ P, float scale,
                                                  float* __restrict__ PM, float* __restrict__ PL, float* __restrict__ PO) {
    constexpr int DPL = HD / 32;
    const int QH = geo.qh, KVH = geo.kvh;
    const int lane = threadIdx.x & 31, g = threadIdx.x >> 5;
    const int c = blockIdx.x, h = blockIdx.y;
    if (c >= MAXCH || past_rows(P, blockIdx.z)) return;
    const size_t m = blockIdx.z;
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
    float q[DPL], o[DPL];
#pragma unroll
    for (int i = 0; i < DPL; i++) {
        q[i] = bf(Q[(m * QH + qh) * HD + lane * DPL + i]);
        o[i] = 0.0f;
    }
    float mx = -INFINITY, l = 0.0f;
    // the keys in order; a row's codes and scales by `at` (one branch here, not one a key)
    const auto scan = [&](auto at) {
        for (int t = k0; t < k1; t++) {
            const int pos = t < sg.len ? t : sg.len + seen_row(anc, t - sg.len);
            const Row8 r = at(pos);
            float kf[DPL], vf[DPL];
            widen8(*reinterpret_cast<const uint2*>(r.k + lane * DPL), kf);
            float dot = 0.0f;
#pragma unroll
            for (int i = 0; i < DPL; i++) dot = fmaf(q[i], kf[i], dot);
            const float s = warp_sum(dot) * *r.ks * scale;
            const float mn = fmaxf(mx, s);
            const float a = __expf(mx - mn), b = __expf(s - mn);
            l = l * a + b;
            widen8(*reinterpret_cast<const uint2*>(r.v + lane * DPL), vf);
            const float bv = b * *r.vs;
#pragma unroll
            for (int i = 0; i < DPL; i++) o[i] = fmaf(bv, vf[i], o[i] * a);
            mx = mn;
        }
    };
    if (sg.pt == 0) {
        const Row8 r0 = row8(sg, layer, KVH, h, 0); // row p is row 0's plus p rows
        scan([&](int pos) { return Row8{r0.k + (size_t)pos * HD, r0.v + (size_t)pos * HD, r0.ks + pos, r0.vs + pos}; });
    } else {
        scan([&](int pos) { return row8(sg, layer, KVH, h, pos); });
    }
    if (lane == 0) {
        PM[slot] = mx;
        PL[slot] = l;
    }
#pragma unroll
    for (int i = 0; i < DPL; i++) PO[slot * HD + lane * DPL + i] = o[i];
}

extern "C" __global__ void tf_q35_attn_partial8_paged(Geo geo, const __nv_bfloat16* __restrict__ Q, const Seg* __restrict__ segs,
                                                  const int* __restrict__ rowseg, const Ancestor* __restrict__ rowanc, int layer,
                                                  const int* __restrict__ P, float scale,
                                                  float* __restrict__ PM, float* __restrict__ PL, float* __restrict__ PO) {
    constexpr int DPL = HD / 32;
    const int QH = geo.qh, KVH = geo.kvh;
    const int lane = threadIdx.x & 31, g = threadIdx.x >> 5;
    const int c = blockIdx.x, h = blockIdx.y;
    if (c >= MAXCH || past_rows(P, blockIdx.z)) return;
    const size_t m = blockIdx.z;
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
    float q[DPL], o[DPL];
#pragma unroll
    for (int i = 0; i < DPL; i++) {
        q[i] = bf(Q[(m * QH + qh) * HD + lane * DPL + i]);
        o[i] = 0.0f;
    }
    float mx = -INFINITY, l = 0.0f;
    // the keys in order; a row's codes and scales by `at` (one branch here, not one a key)
    const auto scan = [&](auto at) {
        for (int t = k0; t < k1; t++) {
            const int pos = t < sg.len ? t : sg.len + seen_row(anc, t - sg.len);
            const Row8 r = at(pos);
            float kf[DPL], vf[DPL];
            widen8(*reinterpret_cast<const uint2*>(r.k + lane * DPL), kf);
            float dot = 0.0f;
#pragma unroll
            for (int i = 0; i < DPL; i++) dot = fmaf(q[i], kf[i], dot);
            const float s = warp_sum(dot) * *r.ks * scale;
            const float mn = fmaxf(mx, s);
            const float a = __expf(mx - mn), b = __expf(s - mn);
            l = l * a + b;
            widen8(*reinterpret_cast<const uint2*>(r.v + lane * DPL), vf);
            const float bv = b * *r.vs;
#pragma unroll
            for (int i = 0; i < DPL; i++) o[i] = fmaf(bv, vf[i], o[i] * a);
            mx = mn;
        }
    };
    if (sg.pt == 0) {
        const Row8 r0 = row8(sg, layer, KVH, h, 0); // row p is row 0's plus p rows
        scan([&](int pos) { return Row8{r0.k + (size_t)pos * HD, r0.v + (size_t)pos * HD, r0.ks + pos, r0.vs + pos}; });
    } else {
        TfQ35PageRun page{reinterpret_cast<const int*>(sg.pt), sg.ptok, -1, sg.ps, 0};
        scan([&](int pos) {
            unsigned char* base = reinterpret_cast<unsigned char*>(sg.kv) + tf_q35_page_base(&page, pos);
            unsigned char* keys = base + (size_t)layer * 2 * KVH * sg.ptok * (HD + 4);
            unsigned char* values = keys + (size_t)KVH * sg.ptok * HD;
            float* scales = reinterpret_cast<float*>(values + (size_t)KVH * sg.ptok * HD);
            const size_t row = (size_t)h * sg.ptok + pos % sg.ptok;
            return Row8{keys + row * HD, values + row * HD, scales + row, scales + (size_t)KVH * sg.ptok + row};
        });
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
    const int d = threadIdx.x;
    for (int j = 0; j < path.n; j++) {
        const int from = sg.len + path.rows[j], to = sg.len + j;
        if (from != to) {
            const Row8 a = row8(sg, layer, KVH, h, from), b = row8(sg, layer, KVH, h, to);
            b.k[d] = a.k[d];
            b.v[d] = a.v[d];
            if (d == 0) {
                *b.ks = *a.ks;
                *b.vs = *a.vs;
            }
        }
        __syncthreads();
    }
}


extern "C" __global__ void __launch_bounds__(128) tf_q35_attn_partial8(Geo geo, const __nv_bfloat16* __restrict__ Q, const Seg* __restrict__ segs,
                                                                         const int* __restrict__ rowseg, const Ancestor* __restrict__ rowanc, int layer,
                                                                         const int* __restrict__ P, float scale,
                                                                         float* __restrict__ PM, float* __restrict__ PL, float* __restrict__ PO) {
    __shared__ __align__(16) uint4 kt[tf_q35_tc::TK * tf_q35_tc::CH], vt[tf_q35_tc::TK * tf_q35_tc::CH];
    __shared__ float ksc[tf_q35_tc::TK], vsc[tf_q35_tc::TK];
    tf_q35_tc::attn_one<tf_q35_tc::Fp8Tiles<128>>({kt, vt, ksc, vsc}, geo, Q, segs, rowseg, rowanc, layer, P, scale, PM, PL, PO);
}

extern "C" __global__ void __launch_bounds__(256) tf_q35_attn_rows8(Geo geo, const __nv_bfloat16* __restrict__ Q, const Seg* __restrict__ segs,
                                                                      const int* __restrict__ rowseg, const Ancestor* __restrict__ rowanc, int layer,
                                                                      const int* __restrict__ P, float scale, int total,
                                                                      float* __restrict__ PM, float* __restrict__ PL, float* __restrict__ PO) {
    __shared__ __align__(16) uint4 kt[tf_q35_tc::TK * tf_q35_tc::CH], vt[tf_q35_tc::TK * tf_q35_tc::CH];
    __shared__ float ksc[tf_q35_tc::TK], vsc[tf_q35_tc::TK];
    tf_q35_tc::attn_rows<4, tf_q35_tc::Fp8Tiles<256>>({kt, vt, ksc, vsc}, geo, Q, segs, rowseg, rowanc, layer, P, scale, total, PM, PL, PO);
}

extern "C" __global__ void __launch_bounds__(128) tf_q35_attn_partial8_whole(Geo geo, const __nv_bfloat16* __restrict__ Q, const Seg* __restrict__ segs,
                                                                         const int* __restrict__ rowseg, const Ancestor* __restrict__ rowanc, int layer,
                                                                         const int* __restrict__ P, float scale,
                                                                         float* __restrict__ PM, float* __restrict__ PL, float* __restrict__ PO) {
    __shared__ __align__(16) uint4 kt[tf_q35_tc::TK * tf_q35_tc::CH], vt[tf_q35_tc::TK * tf_q35_tc::CH];
    __shared__ float ksc[tf_q35_tc::TK], vsc[tf_q35_tc::TK];
    tf_q35_tc::attn_one<tf_q35_tc::WholeFp8Tiles<128>>({kt, vt, ksc, vsc}, geo, Q, segs, rowseg, rowanc, layer, P, scale, PM, PL, PO);
}

extern "C" __global__ void __launch_bounds__(256) tf_q35_attn_rows8_whole(Geo geo, const __nv_bfloat16* __restrict__ Q, const Seg* __restrict__ segs,
                                                                      const int* __restrict__ rowseg, const Ancestor* __restrict__ rowanc, int layer,
                                                                      const int* __restrict__ P, float scale, int total,
                                                                      float* __restrict__ PM, float* __restrict__ PL, float* __restrict__ PO) {
    __shared__ __align__(16) uint4 kt[tf_q35_tc::TK * tf_q35_tc::CH], vt[tf_q35_tc::TK * tf_q35_tc::CH];
    __shared__ float ksc[tf_q35_tc::TK], vsc[tf_q35_tc::TK];
    tf_q35_tc::attn_rows<4, tf_q35_tc::WholeFp8Tiles<256>>({kt, vt, ksc, vsc}, geo, Q, segs, rowseg, rowanc, layer, P, scale, total, PM, PL, PO);
}
