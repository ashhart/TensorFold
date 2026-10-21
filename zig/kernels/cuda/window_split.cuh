// Sliding-window decode split by key tile (float64 references: core/attention_ref.zig), and its merge.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#include "attn_odd.cuh"

// Decode: a block a (row, key head); keys walk 64-key tiles by absolute position, so a row's bits are its own.

namespace tf_window {

constexpr int NT = 256;
constexpr int BN = 64;     // keys a tile
constexpr int GMAX = 16;   // query heads a key head
constexpr int KT = BN + 2;  // kt's row stride: a bank further each dim
constexpr int DMAX = 128;  // head dim (static shared memory: q, a K and a V tile, scores within 48 KiB)

}  // namespace tf_window

using namespace tf_window;

// tf_window_part's NaN or infinite outputs again: the finite values' sum, then each odd value at its float64 weight.
__device__ __noinline__ void window_part_odd(const __nv_bfloat16* q, const __nv_bfloat16* kc, const float* sc,
                                             const __nv_bfloat16* vs, double* m64, float* po, int64_t r, int kh,
                                             int g, int h, int hk, int d, int t, int lo, int p, int64_t base,
                                             long long ring, float scale) {
    const auto krow = [&](int key) { return kc + ((base + key % ring) * hk + kh) * d; };
    const auto qrow = [&](int hd) { return q + (r * h + kh * g + hd) * d; };
    { // each head's float64 max over the row's window, attention_ref's scores: a warp a head, lanes over keys
        const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
        for (int hd = warp; hd < g; hd += NT / 32) {
            double mx = -INFINITY;
            for (int key = lo + lane; key <= p; key += 32) mx = fmax(mx, attn_odd::score64(qrow(hd), krow(key), d, scale));
            for (int k = 16; k > 0; k >>= 1) mx = fmax(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, k));
            if (lane == 0) m64[hd] = mx;
        }
    }
    __syncthreads();
    for (int o = threadIdx.x; o < g * d; o += NT) {
        if (isfinite(po[o])) continue;
        const int hd = o / d, e = o % d;
        float a = 0.f;
        for (int jj = 0; jj < BN; ++jj) {
            const float v = __bfloat162float(vs[jj * d + e]);
            if (isfinite(v)) a = __fmaf_rn(sc[hd * BN + jj], v, a);
        }
        for (int jj = 0; jj < BN; ++jj) {
            const float v = __bfloat162float(vs[jj * d + e]);
            const int key = t * BN + jj;
            if (isfinite(v) || key < lo || key > p) continue;
            const bool w = __dsub_rn(attn_odd::score64(qrow(hd), krow(key), d, scale), m64[hd]) >= attn_odd::least();
            a = __fadd_rn(a, w ? v : __fmul_rn(0.f, v)); // NaN score: no weight
        }
        po[o] = a;
    }
}

// Decode split by key tile (block: row, key head, tile); tf_window_merge folds the partials in tile order.
extern "C" __global__ void __launch_bounds__(NT) tf_window_part(const __nv_bfloat16* __restrict__ q,
                                                                const __nv_bfloat16* __restrict__ kc,
                                                                const __nv_bfloat16* __restrict__ vc,
                                                                const int* __restrict__ pos,
                                                                const int* __restrict__ slot, float* __restrict__ po,
                                                                float* __restrict__ pm, float* __restrict__ pl, int h,
                                                                int hk, int d, int win, long long ring, float scale) {
    __shared__ float qs[GMAX * DMAX];
    __shared__ __nv_bfloat16 kt[DMAX * KT]; // K transposed, [d][key], padded: a warp reads 32 keys of one dim
    __shared__ __nv_bfloat16 vs[BN * DMAX];
    __shared__ float sc[GMAX * BN];
    __shared__ float mx[GMAX], ls[GMAX];
    const int64_t r = blockIdx.x;
    const int kh = blockIdx.y, tiles = gridDim.z, j = blockIdx.z;
    const int g = h / hk;
    const int p = pos[r];
    const int64_t base = (slot != nullptr ? static_cast<int64_t>(slot[r]) : 0) * ring;
    const int lo = win > 0 ? max(p - win + 1, 0) : 0;
    const int t = lo / BN + j;
    const int64_t part = (r * hk + kh) * tiles + j; // this block's partial, g heads of d values
    if (t > p / BN) { // past the row's last tile: an empty partial
        for (int i = threadIdx.x; i < g; i += NT) pm[part * g + i] = -INFINITY, pl[part * g + i] = 0.f;
        return;
    }
    for (int i = threadIdx.x; i < g * d; i += NT) qs[i] = __bfloat162float(q[(r * h + kh * g + i / d) * d + i % d]);
    for (int i = threadIdx.x; i < BN * d; i += NT) {
        const int key = t * BN + i / d;
        const bool in = key >= lo && key <= p;
        const int64_t at = ((base + (in ? key : lo) % ring) * hk + kh) * d + i % d;
        kt[(i % d) * KT + i / d] = in ? kc[at] : __float2bfloat16_rn(0.0f); // outside the window: zeros
        vs[i] = in ? vc[at] : __float2bfloat16_rn(0.0f);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < g * BN; i += NT) { // score (head i / BN, key i % BN)
        const int hd = i / BN, jj = i % BN, key = t * BN + jj;
        float s = 0.f;
#pragma unroll 8
        for (int e = 0; e < d; ++e) s = __fmaf_rn(qs[hd * d + e], __bfloat162float(kt[e * KT + jj]), s);
        sc[i] = (key >= lo && key <= p) ? __fmul_rn(s, scale) : -INFINITY;
    }
    __syncthreads();
    { // a warp a head: the tile's max, the weights exp(s - max) and their sum, lanes in a fixed tree
        const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
        for (int hd = warp; hd < g; hd += NT / 32) {
            float a = sc[hd * BN + lane], b = sc[hd * BN + 32 + lane];
            float m = fmaxf(a, b);
            for (int k = 16; k > 0; k >>= 1) m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, k));
            a = a == -INFINITY ? 0.f : expf(__fsub_rn(a, m));
            b = b == -INFINITY ? 0.f : expf(__fsub_rn(b, m));
            sc[hd * BN + lane] = a;
            sc[hd * BN + 32 + lane] = b;
            float l = __fadd_rn(a, b);
            for (int k = 16; k > 0; k >>= 1) l = __fadd_rn(l, __shfl_xor_sync(0xFFFFFFFFu, l, k));
            if (lane == 0) mx[hd] = m, ls[hd] = l;
        }
    }
    __syncthreads();
    bool odd = false;
    for (int o = threadIdx.x; o < g * d; o += NT) {
        const int hd = o / d, e = o % d;
        float a = 0.f;
        for (int jj = 0; jj < BN; ++jj) a = __fmaf_rn(sc[hd * BN + jj], __bfloat162float(vs[jj * d + e]), a);
        po[part * g * d + o] = a;
        odd |= !isfinite(a);
    }
    if (threadIdx.x < g) pm[part * g + threadIdx.x] = mx[threadIdx.x], pl[part * g + threadIdx.x] = ls[threadIdx.x];
    if (__syncthreads_or(odd)) { // a NaN or infinite output: those again (finite outputs keep their bits)
        __shared__ double m64[GMAX];
        window_part_odd(q, kc, sc, vs, m64, po + part * g * d, r, kh, g, h, hk, d, t, lo, p, base, ring, scale);
    }
}

// out [W, H, D] bf16: a row's partials folded in tile order, o = sum_j exp(m_j - M) o_j / sum_j exp(m_j - M) l_j.
extern "C" __global__ void __launch_bounds__(NT) tf_window_merge(const float* __restrict__ po,
                                                                 const float* __restrict__ pm,
                                                                 const float* __restrict__ pl,
                                                                 __nv_bfloat16* __restrict__ out, int h, int hk, int d,
                                                                 int tiles) {
    const int64_t r = blockIdx.x;
    const int kh = blockIdx.y, g = h / hk;
    const int64_t part0 = (r * hk + kh) * tiles;
    for (int o = threadIdx.x; o < g * d; o += NT) {
        const int hd = o / d;
        float m = -INFINITY;
        for (int j = 0; j < tiles; ++j) m = fmaxf(m, pm[(part0 + j) * g + hd]);
        float acc = 0.f, l = 0.f;
        for (int j = 0; j < tiles; ++j) {
            const float mj = pm[(part0 + j) * g + hd];
            if (mj == -INFINITY) continue;
            const float a = expf(__fsub_rn(mj, m));
            acc = __fmaf_rn(a, po[(part0 + j) * g * d + o], acc);
            l = __fmaf_rn(a, pl[(part0 + j) * g + hd], l);
        }
        if (acc != acc) { // NaN: a replayed infinity (weighed in float64) may have met a factor of 0; again, kept
            acc = 0.f;
            for (int j = 0; j < tiles; ++j) {
                const float mj = pm[(part0 + j) * g + hd];
                if (mj != -INFINITY) acc = attn_odd::fold(po[(part0 + j) * g * d + o], expf(__fsub_rn(mj, m)), acc);
            }
        }
        out[(r * h + kh * g + hd) * d + o % d] = __float2bfloat16_rn(__fdiv_rn(acc, l));
    }
}
