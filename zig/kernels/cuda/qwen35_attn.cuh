// Contiguous attention retains the pre-paging body; the general entry also serves mixed cache layouts.
#pragma once
#include "qwen35_page_run.h"

extern "C" __global__ void tf_q35_attn_partial_contiguous(Geo geo, const __nv_bfloat16* __restrict__ Q, const Seg* __restrict__ segs,
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
    // chunks cut at logical positions (a window key's: the stream's length plus its ancestors), as a step's are
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
    const __nv_bfloat16* K = reinterpret_cast<const __nv_bfloat16*>(sg.kv) + (size_t)layer * 2 * KVH * sg.cap * HD;
    const __nv_bfloat16* V = K + (size_t)KVH * sg.cap * HD;
    float q[DPL], o[DPL];
#pragma unroll
    for (int i = 0; i < DPL; i++) {
        q[i] = bf(Q[(m * QH + qh) * HD + lane * DPL + i]);
        o[i] = 0.0f;
    }
    float mx = -INFINITY, l = 0.0f;
    for (int t = k0; t < k1; t++) {
        const int pos = t < sg.len ? t : sg.len + seen_row(anc, t - sg.len);
        const uint4 kw = *reinterpret_cast<const uint4*>(K + ((size_t)h * sg.cap + pos) * HD + lane * DPL);
        const __nv_bfloat16* kb = reinterpret_cast<const __nv_bfloat16*>(&kw);
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < DPL; i++) dot = fmaf(q[i], bf(kb[i]), dot);
        const float s = warp_sum(dot) * scale;
        const float mn = fmaxf(mx, s);
        const float a = __expf(mx - mn), b = __expf(s - mn);
        l = l * a + b;
        const uint4 vw = *reinterpret_cast<const uint4*>(V + ((size_t)h * sg.cap + pos) * HD + lane * DPL);
        const __nv_bfloat16* vb = reinterpret_cast<const __nv_bfloat16*>(&vw);
#pragma unroll
        for (int i = 0; i < DPL; i++) o[i] = fmaf(b, bf(vb[i]), o[i] * a);
        mx = mn;
    }
    if (lane == 0) {
        PM[slot] = mx;
        PL[slot] = l;
    }
#pragma unroll
    for (int i = 0; i < DPL; i++) PO[slot * HD + lane * DPL + i] = o[i];
}

extern "C" __global__ void tf_q35_attn_partial(Geo geo, const __nv_bfloat16* __restrict__ Q, const Seg* __restrict__ segs,
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
    // chunks cut at logical positions (a window key's: the stream's length plus its ancestors), as a step's are
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
    const __nv_bfloat16* KV = reinterpret_cast<const __nv_bfloat16*>(sg.kv);
    float q[DPL], o[DPL];
#pragma unroll
    for (int i = 0; i < DPL; i++) {
        q[i] = bf(Q[(m * QH + qh) * HD + lane * DPL + i]);
        o[i] = 0.0f;
    }
    float mx = -INFINITY, l = 0.0f;
    // the keys in order; a key's and value's rows by `row` (one branch here, not one a key)
    const auto scan = [&](auto row) {
        for (int t = k0; t < k1; t++) {
            const int pos = t < sg.len ? t : sg.len + seen_row(anc, t - sg.len);
            const uint4 kw = *reinterpret_cast<const uint4*>(row(0, pos) + lane * DPL);
            const __nv_bfloat16* kb = reinterpret_cast<const __nv_bfloat16*>(&kw);
            float dot = 0.0f;
#pragma unroll
            for (int i = 0; i < DPL; i++) dot = fmaf(q[i], bf(kb[i]), dot);
            const float s = warp_sum(dot) * scale;
            const float mn = fmaxf(mx, s);
            const float a = __expf(mx - mn), b = __expf(s - mn);
            l = l * a + b;
            const uint4 vw = *reinterpret_cast<const uint4*>(row(1, pos) + lane * DPL);
            const __nv_bfloat16* vb = reinterpret_cast<const __nv_bfloat16*>(&vw);
#pragma unroll
            for (int i = 0; i < DPL; i++) o[i] = fmaf(b, bf(vb[i]), o[i] * a);
            mx = mn;
        }
    };
    if (sg.pt == 0) {
        const __nv_bfloat16* K = KV + kv_row(sg, KVH, layer, 0, h, 0);
        const __nv_bfloat16* V = KV + kv_row(sg, KVH, layer, 1, h, 0);
        scan([&](int s, int pos) { return (s == 0 ? K : V) + (size_t)pos * HD; });
    } else {
        scan([&](int s, int pos) { return KV + kv_row(sg, KVH, layer, s, h, pos); });
    }
    if (lane == 0) {
        PM[slot] = mx;
        PL[slot] = l;
    }
#pragma unroll
    for (int i = 0; i < DPL; i++) PO[slot * HD + lane * DPL + i] = o[i];
}

// Mixed streams branch once a block; each visited page ID is reused by every key and value in that run.
extern "C" __global__ void tf_q35_attn_partial_paged(Geo geo, const __nv_bfloat16* __restrict__ Q, const Seg* __restrict__ segs,
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
    // chunks cut at logical positions (a window key's: the stream's length plus its ancestors), as a step's are
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
    const __nv_bfloat16* KV = reinterpret_cast<const __nv_bfloat16*>(sg.kv);
    float q[DPL], o[DPL];
#pragma unroll
    for (int i = 0; i < DPL; i++) {
        q[i] = bf(Q[(m * QH + qh) * HD + lane * DPL + i]);
        o[i] = 0.0f;
    }
    float mx = -INFINITY, l = 0.0f;
    // the keys in order; a key's and value's rows by `row` (one branch here, not one a key)
    const auto scan = [&](auto row) {
        for (int t = k0; t < k1; t++) {
            const int pos = t < sg.len ? t : sg.len + seen_row(anc, t - sg.len);
            const uint4 kw = *reinterpret_cast<const uint4*>(row(0, pos) + lane * DPL);
            const __nv_bfloat16* kb = reinterpret_cast<const __nv_bfloat16*>(&kw);
            float dot = 0.0f;
#pragma unroll
            for (int i = 0; i < DPL; i++) dot = fmaf(q[i], bf(kb[i]), dot);
            const float s = warp_sum(dot) * scale;
            const float mn = fmaxf(mx, s);
            const float a = __expf(mx - mn), b = __expf(s - mn);
            l = l * a + b;
            const uint4 vw = *reinterpret_cast<const uint4*>(row(1, pos) + lane * DPL);
            const __nv_bfloat16* vb = reinterpret_cast<const __nv_bfloat16*>(&vw);
#pragma unroll
            for (int i = 0; i < DPL; i++) o[i] = fmaf(b, bf(vb[i]), o[i] * a);
            mx = mn;
        }
    };
    if (sg.pt == 0) {
        const __nv_bfloat16* K = KV + kv_row(sg, KVH, layer, 0, h, 0);
        const __nv_bfloat16* V = KV + kv_row(sg, KVH, layer, 1, h, 0);
        scan([&](int s, int pos) { return (s == 0 ? K : V) + (size_t)pos * HD; });
    } else {
        TfQ35PageRun page{reinterpret_cast<const int*>(sg.pt), sg.ptok, -1, sg.ps, 0};
        scan([&](int s, int pos) {
            const size_t blk = ((size_t)layer * 2 + s) * KVH + h;
            return KV + tf_q35_page_base(&page, pos) + (blk * sg.ptok + pos % sg.ptok) * HD;
        });
    }
    if (lane == 0) {
        PM[slot] = mx;
        PL[slot] = l;
    }
#pragma unroll
    for (int i = 0; i < DPL; i++) PO[slot * HD + lane * DPL + i] = o[i];
}
