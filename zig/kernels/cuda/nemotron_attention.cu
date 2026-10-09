// Merge absolute 512-key attention chunks in order, so a row's bits ignore its window.

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

#include "glue_math.cuh"

using namespace tf_glue;

namespace {

constexpr int HD = 128;     // head dim the chunk kernels take
constexpr int GMAX = 16;    // query heads a KV head
constexpr int TILE = 64;    // keys a tile: the online softmax's unit
constexpr int KW = HD / 2 + 2;  // a key row's words in shared memory, padded so 64-bit reads of 16 rows miss no bank

__device__ __forceinline__ float lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float hi(uint32_t w) { return __uint_as_float(w & 0xFFFF0000u); }

// Max over the 16 lanes of a half warp (xor 8, 4, 2, 1).
__device__ __forceinline__ float half_max(float v) {
#pragma unroll
    for (int off = 8; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}

// Sum over the 16 lanes of a half warp (xor 8, 4, 2, 1): each lane ends with the same sum.
__device__ __forceinline__ float half_sum(float v) {
#pragma unroll
    for (int off = 8; off > 0; off >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}

}  // namespace

// Row r's keys and values written at its position: K and V follow the query columns of each qkv row.
extern "C" __global__ void tf_nemo_kv_write(const __nv_bfloat16* __restrict__ QKV, __nv_bfloat16* __restrict__ KC,
                                             __nv_bfloat16* __restrict__ VC, const int* __restrict__ META, int NQKV, int QD, int KVD) {
    const int64_t r = blockIdx.x, pos = META[0];
    for (int o = threadIdx.x; o < KVD; o += blockDim.x) {
        KC[(pos + r) * KVD + o] = QKV[r * NQKV + QD + o];
        VC[(pos + r) * KVD + o] = QKV[r * NQKV + QD + KVD + o];
    }
}

// Block (row, KV head, z): the row's G query heads against each z-th chunk's keys below its limit, in 64-key tiles.
extern "C" __global__ void __launch_bounds__(256) tf_nemo_attn_chunk(const __nv_bfloat16* __restrict__ QKV, const __nv_bfloat16* __restrict__ KC,
                                                                       const __nv_bfloat16* __restrict__ VC, const int* __restrict__ META,
                                                                       float* __restrict__ PO, float* __restrict__ PM, float* __restrict__ PL,
                                                                       int NQKV, int H, int HK, int G, int CH, int NCH, float scale) {
    __shared__ __align__(16) float qs[GMAX][HD];
    __shared__ __align__(16) uint32_t kw[TILE][KW];
    __shared__ __align__(16) uint32_t vw[TILE][HD / 2];
    __shared__ float ps[GMAX][TILE];
    const int r = blockIdx.x, hk = blockIdx.y, t = threadIdx.x;
    const int limit = META[0] + r + 1;
    if (static_cast<int>(blockIdx.z) * CH >= limit) return;
    const int g = t / 16, jj = t % 16;     // softmax and values: head g, lanes jj of its 16
    const int key = t % TILE, quad = t / TILE;  // scores: one key, heads 4 quad .. 4 quad + 3
    for (int i = t; i < GMAX * HD; i += 256) {
        const int gg = i / HD, d = i % HD;
        qs[gg][d] = gg < G ? bf(QKV[static_cast<int64_t>(r) * NQKV + (hk * G + gg) * HD + d]) : 0.0f;
    }
    for (int c = blockIdx.z; c * CH < limit; c += gridDim.z) {  // this block's chunks: every gridDim.z-th
        float m = -INFINITY, den = 0.0f, o[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) o[e] = 0.0f;
        for (int key0 = c * CH; key0 < (c + 1) * CH && key0 < limit; key0 += TILE) {
            __syncthreads();
            for (int i = t; i < TILE * (HD / 2); i += 256) {
                const int j = i / (HD / 2), w = i % (HD / 2);
                const int64_t at = (static_cast<int64_t>(key0 + j) * HK + hk) * HD;
                const bool in = key0 + j < limit;
                kw[j][w] = in ? reinterpret_cast<const uint32_t*>(KC + at)[w] : 0u;
                vw[j][w] = in ? reinterpret_cast<const uint32_t*>(VC + at)[w] : 0u;
            }
            __syncthreads();
            float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 4
            for (int w = 0; w < HD / 2; w += 2) {
                const uint2 kk = *reinterpret_cast<const uint2*>(&kw[key][w]);
                const float k0 = lo(kk.x), k1 = hi(kk.x), k2 = lo(kk.y), k3 = hi(kk.y);
#pragma unroll
                for (int h = 0; h < 4; ++h) {
                    const float4 q = *reinterpret_cast<const float4*>(&qs[4 * quad + h][2 * w]);
                    acc[h] = __fmaf_rn(q.x, k0, acc[h]);
                    acc[h] = __fmaf_rn(q.y, k1, acc[h]);
                    acc[h] = __fmaf_rn(q.z, k2, acc[h]);
                    acc[h] = __fmaf_rn(q.w, k3, acc[h]);
                }
            }
#pragma unroll
            for (int h = 0; h < 4; ++h) ps[4 * quad + h][key] = key0 + key < limit ? __fmul_rn(acc[h], scale) : -INFINITY;
            __syncthreads();
            float sc[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) sc[u] = ps[g][jj + 16 * u];
            const float tile_m = half_max(fmaxf(fmaxf(sc[0], sc[1]), fmaxf(sc[2], sc[3])));
            const bool active = tile_m != -INFINITY;
            const float next = active ? fmaxf(m, tile_m) : m;
            const float alpha = active ? (m == -INFINITY ? 0.0f : tf_exp(__fsub_rn(m, next))) : 1.0f;
            float local = 0.0f;
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const float p = active && key0 + jj + 16 * u < limit ? tf_exp(__fsub_rn(sc[u], next)) : 0.0f;
                ps[g][jj + 16 * u] = p;
                local = __fadd_rn(local, p);
            }
            den = __fmaf_rn(den, alpha, half_sum(local));
            m = next;
            __syncthreads();
            float pv[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) pv[e] = 0.0f;
            for (int j = 0; j < TILE; ++j) {
                const float p = ps[g][j];
                const uint4 v = *reinterpret_cast<const uint4*>(&vw[j][4 * jj]);
                const uint32_t vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    pv[2 * e] = __fmaf_rn(p, lo(vv[e]), pv[2 * e]);
                    pv[2 * e + 1] = __fmaf_rn(p, hi(vv[e]), pv[2 * e + 1]);
                }
            }
#pragma unroll
            for (int e = 0; e < 8; ++e) o[e] = __fmaf_rn(o[e], alpha, pv[e]);
        }
        if (g >= G) continue;
        const int64_t base = (static_cast<int64_t>(r) * NCH + c) * H + hk * G + g;
        float4* out = reinterpret_cast<float4*>(PO + base * HD + 8 * jj);
        out[0] = make_float4(o[0], o[1], o[2], o[3]);
        out[1] = make_float4(o[4], o[5], o[6], o[7]);
        if (jj == 0) {
            PM[base] = m;
            PL[base] = den;
        }
    }
}

// Block (row, KV head): the row's chunk partials merged in chunk order; out = bf16(o / den) and its 64-group sums.
extern "C" __global__ void __launch_bounds__(256) tf_nemo_attn_merge(const float* __restrict__ PO, const float* __restrict__ PM,
                                                                       const float* __restrict__ PL, const int* __restrict__ META,
                                                                       __nv_bfloat16* __restrict__ OUT, float* __restrict__ XS, int H, int HK,
                                                                       int G, int CH, int NCH) {
    __shared__ float ys[GMAX][HD];
    const int r = blockIdx.x, hk = blockIdx.y, t = threadIdx.x, g = t / 16, jj = t % 16;
    const int limit = META[0] + r + 1, nch = (limit + CH - 1) / CH, head = hk * G + g;
    float m = -INFINITY, den = 0.0f, o[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) o[e] = 0.0f;
    if (g < G) {
        for (int c = 0; c < nch; ++c) {
            const int64_t base = (static_cast<int64_t>(r) * NCH + c) * H + head;
            const float cm = PM[base], cl = PL[base];
            const bool active = cl > 0.0f;
            const float next = active ? fmaxf(m, cm) : m;
            const float a = active ? (m == -INFINITY ? 0.0f : tf_exp(__fsub_rn(m, next))) : 1.0f;
            const float b = active ? tf_exp(__fsub_rn(cm, next)) : 0.0f;
            const float4* po = reinterpret_cast<const float4*>(PO + base * HD + 8 * jj);
            const float4 p0 = po[0], p1 = po[1];
            const float co[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
#pragma unroll
            for (int e = 0; e < 8; ++e) o[e] = __fmaf_rn(o[e], a, __fmul_rn(co[e], b));
            den = __fmaf_rn(den, a, __fmul_rn(cl, b));
            m = next;
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const __nv_bfloat16 y = __float2bfloat16_rn(__fdiv_rn(o[e], den));
            OUT[(static_cast<int64_t>(r) * H + head) * HD + 8 * jj + e] = y;
            ys[g][8 * jj + e] = bf(y);
        }
    }
    __syncthreads();
    if (t >= GMAX * (HD / 64)) return;
    const int gg = t / (HD / 64), gi = t % (HD / 64);
    if (gg >= G) return;
    float s = 0.0f;
    for (int j = 0; j < 64; ++j) s = __fadd_rn(s, ys[gg][gi * 64 + j]);
    XS[static_cast<int64_t>(r) * (H * HD / 64) + (hk * G + gg) * (HD / 64) + gi] = s;
}
