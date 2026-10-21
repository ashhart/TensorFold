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


#include "qwen35_attn_tc.cuh"

// Decode attention over e4m3 rows on tensor cores: a short stream's rows a block each, a long stream's in groups.
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
