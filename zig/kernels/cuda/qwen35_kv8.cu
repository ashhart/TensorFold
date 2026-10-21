// Ours: --kv-cache fp8 for Qwen3.5: key and value rows as e4m3, an fp32 scale (amax / 448) each, as qwen27 keeps them.

#include <cuda_fp8.h>

#include "qwen35.cuh"

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

// The block's 256 threads' row as e4m3 into q[d], its scale (1 for a row of zeros) to *s; tf_kv_quant's bits.
__device__ __forceinline__ void quant_row(float x, int d, unsigned char* q, float* s, float* red) {
    float amax = fabsf(x);
#pragma unroll
    for (int o = 16; o > 0; o /= 2) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    if ((d & 31) == 0) red[d >> 5] = amax;
    __syncthreads();
    amax = red[0];
#pragma unroll
    for (int w = 1; w < HD / 32; w++) amax = fmaxf(amax, red[w]);
    __syncthreads();
    const float inv = amax > 0.0f ? __fdiv_rn(448.0f, amax) : 1.0f;
    q[d] = __nv_cvt_float_to_fp8(__fmul_rn(x, inv), __NV_SATFINITE, __NV_E4M3);
    if (d == 0) *s = amax > 0.0f ? __fdiv_rn(amax, 448.0f) : 1.0f;
}

// A key and value as e4m3 rows and their scales in their cache slot; every thread of the block calls it together.
struct StoreFp8 {
    int layer, kvh;
    float* red;
    __device__ __forceinline__ void operator()(const Seg& sg, int h, int slot, int d, __nv_bfloat16 k, __nv_bfloat16 v) const {
        const Row8 r = row8(sg, layer, kvh, h, slot);
        quant_row(bf(k), d, r.k, r.ks, red);
        quant_row(bf(v), d, r.v, r.vs, red);
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
