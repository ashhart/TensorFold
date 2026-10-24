// window_attention at head dims 256 and 512: decode by 256-key chunks and a merge, prompt tiles (attention_range.zig).
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#include "qmm_frag.cuh"
#include "window_fallback.h"

namespace tf_wrows {

constexpr int NT = 256;      // threads a block
constexpr int NW = NT / 32;  // warps a block
constexpr int CK = 256;      // keys a decode chunk: chunk c holds keys [c * CK, c * CK + CK)
constexpr int GMAX = 16;     // query heads a key head
constexpr int SP = GMAX + 1; // the decode scores' key stride (a bank further each key)
constexpr int BN = 32;       // keys a prompt tile
constexpr unsigned FULL = 0xffffffffu;

// Planted faults, compiled only into contract_controls.cu's controls; every served kernel has NONE.
enum Fault : int {
    NONE = 0,
    EDGE = 1,    // the window's first key one too early
    LAST = 2,    // a prompt row's last key: its own position, the given one ignored
    RESCALE = 3, // the merge adds the chunks' partials without exp(m_c - M)
    TRUNC = 4,   // the merge's output rounded toward zero
};

using qmm_frag::commit;
using qmm_frag::cp16z;
using qmm_frag::ldmatrix4;
using qmm_frag::mma;
using qmm_frag::mma0;

__device__ __forceinline__ float lo16(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float hi16(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

// A float64 value to bf16 in one rounding, ties to even: toward zero to fp32 with a sticky last bit, then nearest.
__device__ __forceinline__ __nv_bfloat16 d2bf(double x) {
    if (x != x) return __float2bfloat16_rn(__int_as_float(0x7fc00000));
    float f = __double2float_rz(x);
    if ((double)f != x && isfinite(f)) f = __int_as_float(__float_as_int(f) | 1);
    return __float2bfloat16_rn(f);
}

// q . k in float64, lanes over the head's dims, then a xor tree; every lane gets the sum.
template <int D>
__device__ __forceinline__ double dot64(const __nv_bfloat16* q, const __nv_bfloat16* k, int lane) {
    double s = 0.0;
#pragma unroll 4
    for (int e = lane; e < D; e += 32) s += (double)__bfloat162float(q[e]) * (double)__bfloat162float(k[e]);
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(FULL, s, o);
    return s;
}

// A head over keys [lo, hi] in float64 for columns c0 + threadIdx.x + NT i < c1 (kv(key, 0): K row, 1: V); a block.
template <int D, typename KV>
__device__ __noinline__ void slow_head(const __nv_bfloat16* qh, KV kv, int lo, int hi, double scale,
                                       __nv_bfloat16* out, int c0, int c1) {
    __shared__ double red[NW];
    __shared__ double pw[NW];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    double m = -INFINITY;
    for (int j = lo + warp; j <= hi; j += NW) m = fmax(m, dot64<D>(qh, kv(j, 0), lane) * scale);
    __syncthreads();
    if (lane == 0) red[warp] = m;
    __syncthreads();
    m = -INFINITY;
    for (int w = 0; w < NW; ++w) m = fmax(m, red[w]);
    constexpr int CPT = D / NT;
    double acc[CPT], l = 0.0;
#pragma unroll
    for (int i = 0; i < CPT; ++i) acc[i] = 0.0;
    for (int jb = lo; jb <= hi; jb += NW) {
        const int j = jb + warp;
        if (j <= hi) {
            const double s = dot64<D>(qh, kv(j, 0), lane) * scale;
            if (lane == 0) pw[warp] = exp(s - m);
        }
        __syncthreads();
        const int nb = min(NW, hi - jb + 1);
        for (int w = 0; w < nb; ++w) {
            const double p = pw[w];
            l += p;
            const __nv_bfloat16* v = kv(jb + w, 1);
#pragma unroll
            for (int i = 0; i < CPT; ++i) {
                const int c = c0 + threadIdx.x + NT * i;
                if (c < c1) acc[i] += p * (double)__bfloat162float(v[c]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < CPT; ++i) {
        const int c = c0 + threadIdx.x + NT * i;
        if (c < c1) out[c] = d2bf(acc[i] / l);
    }
}

// Decode partials, a block a (row, key head, chunk): the chunk's max, weights and o; empty chunks write m = -inf.
template <int D, int FAULT>
__device__ __forceinline__ void part(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc,
                                     const __nv_bfloat16* __restrict__ vc, const int* __restrict__ pos,
                                     const int* __restrict__ slot, float* __restrict__ po, float* __restrict__ pm,
                                     float* __restrict__ pl, int h, int hk, int win, long long ring, float scale) {
    constexpr int DPL = D / 32;  // a lane's dims of q and k, and its columns of v
    constexpr int HW = 32 / DPL; // query heads a warp takes: 32 q values, then 32 accumulators, a lane
    constexpr int U = 4;         // keys a warp loads at once
    __shared__ float sp[CK * SP];
    __shared__ float mx[GMAX], ls[GMAX];
    __shared__ float ox[7 * 4 * 256]; // key runs 1.. partial o, (runs - 1) G D floats at most (G HW: runs 8)
    const int64_t r = blockIdx.x;
    const int kh = blockIdx.y, c = blockIdx.z, chunks = gridDim.z, G = h / hk;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int p = pos[r];
    const int64_t base = (slot != nullptr ? static_cast<int64_t>(slot[r]) : 0) * ring;
    const int lo = win > 0 ? max(p - win + (FAULT == EDGE ? 0 : 1), 0) : 0;
    const int k0 = max((lo / CK + c) * CK, lo), k1 = min((lo / CK + c) * CK + CK - 1, p);
    const int64_t part = (r * hk + kh) * chunks + c; // this block's partial: G heads of D values
    if (k0 > k1) {
        if (threadIdx.x < G) pm[part * G + threadIdx.x] = -INFINITY, pl[part * G + threadIdx.x] = 0.0f;
        return;
    }
    const int nk = k1 - k0 + 1;
    const int64_t r0 = k0 % ring;
    // warp (hg, kg): heads hg HW.., keys [j0, j1), the kg-th of WG runs in key order
    const int HG = (G + HW - 1) / HW, WG = max(1, NW / HG);
    const int hg = warp % HG, kg = warp / HG, run = (nk + WG - 1) / WG;
    const bool busy = warp < HG * WG;
    const int j0 = min(kg * run, nk), j1 = min(j0 + run, nk);
    const auto rowp = [&](const __nv_bfloat16* cache, int jj) {
        int64_t row = r0 + jj;
        if (row >= ring) row -= ring;
        return cache + ((base + row) * hk + kh) * D + lane * DPL;
    };
    const auto load = [&](const __nv_bfloat16* src, float (&x)[DPL]) {
#pragma unroll
        for (int i = 0; i < DPL; i += 8) {
            const uint4 u = *reinterpret_cast<const uint4*>(src + i);
            const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
            for (int t = 0; t < 4; ++t) x[i + 2 * t] = lo16(w[t]), x[i + 2 * t + 1] = hi16(w[t]);
        }
    };
    if (busy) { // scores: a lane's DPL dims in an fma chain, then a xor tree over the lanes
        float qv[HW][DPL];
#pragma unroll
        for (int hh = 0; hh < HW; ++hh) {
            const int hd = hg * HW + hh;
            load(q + (r * h + kh * G + min(hd, G - 1)) * D + lane * DPL, qv[hh]);
        }
        for (int jj = j0; jj < j1; jj += U) {
            float kf[U][DPL];
#pragma unroll
            for (int u = 0; u < U; ++u)
                if (jj + u < j1) load(rowp(kc, jj + u), kf[u]);
#pragma unroll
            for (int u = 0; u < U; ++u) {
                if (jj + u >= j1) break;
#pragma unroll
                for (int hh = 0; hh < HW; ++hh) {
                    if (hg * HW + hh >= G) break; // the same for the whole warp
                    float s = 0.0f;
#pragma unroll
                    for (int i = 0; i < DPL; ++i) s = __fmaf_rn(qv[hh][i], kf[u][i], s);
#pragma unroll
                    for (int o = 16; o > 0; o >>= 1) s = __fadd_rn(s, __shfl_xor_sync(FULL, s, o));
                    if (lane == 0) sp[(jj + u) * SP + hg * HW + hh] = __fmul_rn(s, scale);
                }
            }
        }
    }
    __syncthreads();
    for (int hd = warp; hd < G; hd += NW) { // a warp a head: max, then weights and their sum (lanes, then a xor tree)
        float m = -INFINITY;
        for (int jj = lane; jj < nk; jj += 32) m = fmaxf(m, sp[jj * SP + hd]);
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(FULL, m, o));
        float l = 0.0f;
        for (int jj = lane; jj < nk; jj += 32) {
            const float s = sp[jj * SP + hd];
            const float e = s == -INFINITY ? 0.0f : expf(__fsub_rn(s, m));
            sp[jj * SP + hd] = e;
            l = __fadd_rn(l, e);
        }
        for (int o = 16; o > 0; o >>= 1) l = __fadd_rn(l, __shfl_xor_sync(FULL, l, o));
        if (lane == 0) mx[hd] = m, ls[hd] = l;
    }
    __syncthreads();
    // o: a lane's DPL columns of heads hg HW.., an fma chain over its run's keys in order; then the runs in order
    float acc[HW][DPL];
#pragma unroll
    for (int hh = 0; hh < HW; ++hh)
#pragma unroll
        for (int i = 0; i < DPL; ++i) acc[hh][i] = 0.0f;
    if (busy) {
        for (int jj = j0; jj < j1; jj += U) {
            float vf[U][DPL];
#pragma unroll
            for (int u = 0; u < U; ++u)
                if (jj + u < j1) load(rowp(vc, jj + u), vf[u]);
#pragma unroll
            for (int u = 0; u < U; ++u) {
                if (jj + u >= j1) break;
#pragma unroll
                for (int hh = 0; hh < HW; ++hh) {
                    if (hg * HW + hh >= G) break;
                    const float w = sp[(jj + u) * SP + hg * HW + hh];
#pragma unroll
                    for (int i = 0; i < DPL; ++i) acc[hh][i] = __fmaf_rn(w, vf[u][i], acc[hh][i]);
                }
            }
        }
        if (kg > 0)
#pragma unroll
            for (int hh = 0; hh < HW; ++hh) {
                const int hd = hg * HW + hh;
                if (hd >= G) break;
#pragma unroll
                for (int i = 0; i < DPL; ++i) ox[((kg - 1) * G + hd) * D + lane * DPL + i] = acc[hh][i];
            }
    }
    __syncthreads();
    if (busy && kg == 0) {
#pragma unroll
        for (int hh = 0; hh < HW; ++hh) {
            const int hd = hg * HW + hh;
            if (hd >= G) break;
#pragma unroll
            for (int i = 0; i < DPL; ++i) {
                float a = acc[hh][i];
                for (int k = 1; k < WG; ++k) a = __fadd_rn(a, ox[((k - 1) * G + hd) * D + lane * DPL + i]);
                po[(part * G + hd) * D + lane * DPL + i] = a;
            }
        }
    }
    if (threadIdx.x < G) pm[part * G + threadIdx.x] = mx[threadIdx.x], pl[part * G + threadIdx.x] = ls[threadIdx.x];
}

// out = sum_c a_c o_c / sum_c a_c l_c, a_c = expf(m_c - M), chunks in order; NaN or inf: the head again in float64.
template <int D, int FAULT>
__device__ __forceinline__ void merge(const float* __restrict__ po, const float* __restrict__ pm,
                                      const float* __restrict__ pl, __nv_bfloat16* __restrict__ out, int h, int hk,
                                      int chunks, const __nv_bfloat16* __restrict__ q,
                                      const __nv_bfloat16* __restrict__ kc, const __nv_bfloat16* __restrict__ vc,
                                      const int* __restrict__ pos, const int* __restrict__ slot, int win,
                                      long long ring, float scale) {
    const int64_t r = blockIdx.x;
    const int kh = blockIdx.y, G = h / hk;
    const int o = blockIdx.z * NT + threadIdx.x; // NT divides D: the block's outputs are one head's
    const int hd = o / D, e = o % D;
    const int64_t part0 = (r * hk + kh) * chunks;
    float m = -INFINITY;
    for (int c = 0; c < chunks; ++c) m = fmaxf(m, pm[(part0 + c) * G + hd]);
    float acc = 0.0f, l = 0.0f;
    for (int c = 0; c < chunks; ++c) {
        const float mc = pm[(part0 + c) * G + hd];
        if (mc == -INFINITY) continue;
        const float a = FAULT == RESCALE ? 1.0f : expf(__fsub_rn(mc, m));
        acc = __fmaf_rn(a, po[((part0 + c) * G + hd) * D + e], acc);
        l = __fmaf_rn(a, pl[(part0 + c) * G + hd], l);
    }
    const float x = __fdiv_rn(acc, l);
    __nv_bfloat16* oh = out + (r * h + kh * G + hd) * D;
    oh[e] = FAULT == TRUNC ? __float2bfloat16_rz(x) : __float2bfloat16_rn(x);
    if (__syncthreads_or(!isfinite(x))) {
        const int p = pos[r];
        const int64_t base = (slot != nullptr ? static_cast<int64_t>(slot[r]) : 0) * ring;
        const auto kv = [&](int key, int which) {
            return (which == 0 ? kc : vc) + ((base + key % ring) * hk + kh) * D;
        };
        const int c0 = blockIdx.z * NT % D;
        slow_head<D>(q + (r * h + kh * G + hd) * D, kv, win > 0 ? max(p - win + 1, 0) : 0, p, (double)scale, oh, c0,
                     c0 + NT);
    }
}

namespace prompt {

__device__ __forceinline__ void ldmatrix4t(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(qmm_frag::smem(p)));
}

template <int N>
__device__ __forceinline__ void wait_group() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ uint32_t pack(float lo, float hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
}

// The bf16 pair's fp32 values.
__device__ __forceinline__ float2 unpack(uint32_t w) { return make_float2(lo16(w), hi16(w)); }

template <int D>
__device__ __forceinline__ int sw(int r, int c) { return r * (D / 8) + (c ^ (r & 7)); }

template <int D>
struct Cfg {
    static constexpr int SL = 128;                       // dims a warp's slice: its share of q . k, its columns of o
    static constexpr int CS = D / SL;                    // warps sharing 16 rows
    static constexpr int K16 = SL / 16, NB = SL / 8;     // a slice's mma k-steps and n8 output tiles
    static constexpr int NG = NW / CS;                   // 16-row groups a block
    static constexpr int C = D / 8;                      // 16-byte chunks a key row
    static constexpr int NS = D == 256 ? 4 : 2;          // staging slots of BN keys (K or V)
    static constexpr int SLOT = BN * C;                  // uint4 a slot
    static constexpr int XS = NW * 16 * 32;              // floats: the warps' partial scores, [warp][element][lane]
    static constexpr int SMEM = NS * SLOT * 16 + XS * 4; // dynamic shared bytes
};

// Prompt rows p0..: row i sees keys [p0 + i - win + 1, hi[i]]; CS warps a 16-row band, slices summed; P32: P halves.
template <int D, bool P32, int FAULT>
__device__ __forceinline__ bool rows(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc,
                                     const __nv_bfloat16* __restrict__ vc, __nv_bfloat16* __restrict__ out,
                                     const int* __restrict__ hi, int p0, int W, int H, int HK, int G, float scale,
                                     int win, int ring) {
    using K = Cfg<D>;
    constexpr int CS = K::CS, NG = K::NG, C = K::C, NS = K::NS, SL = K::SL, K16 = K::K16, NB = K::NB;
    extern __shared__ uint4 wsmem[];
    uint4* slots = wsmem;
    float* xs = reinterpret_cast<float*>(wsmem + NS * K::SLOT);
    __shared__ int kmax;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, qd = lane % 4, mt = lane / 8;
    const int grp = warp / CS, cs = warp % CS;
    const int HB = G % NG == 0 ? NG : 1, RB = NG / HB; // heads and 16-row bands a block
    const int hk = blockIdx.y / (G / HB), hg = blockIdx.y % (G / HB);
    const int head = hk * G + hg * HB + grp % HB;
    const int row0 = blockIdx.x * 16 * RB, wrow = row0 + (grp / HB) * 16;
    const int r0 = wrow + g, r1 = r0 + 8;
    const int pos0 = p0 + r0, pos1 = p0 + r1;
    const auto first = [&](int p) { return win > 0 ? max(p - win + (FAULT == EDGE ? 0 : 1), 0) : 0; };
    const auto last = [&](int r) { return hi != nullptr && FAULT != LAST && r < W ? hi[r] : p0 + r; };
    const int lo0 = first(pos0), lo1 = first(pos1), hi0 = last(r0), hi1 = last(r1);
    if (threadIdx.x == 0) kmax = 0;
    __syncthreads();
    if (threadIdx.x < 16 * RB && row0 + (int)threadIdx.x < W) atomicMax(&kmax, last(row0 + threadIdx.x));
    __syncthreads();
    const int kfirst = first(p0 + row0), klast = kmax, limit = p0 + W;
    const int t0 = kfirst / BN, tiles = klast / BN + 1 - t0, items = 2 * tiles;
    auto stage = [&](int item) { // item 2t: tile t's keys, 2t + 1: its values
        const __nv_bfloat16* src = item % 2 == 0 ? kc : vc;
        const int key0 = (t0 + item / 2) * BN;
        uint4* dst = slots + (item % NS) * K::SLOT;
        for (int i = threadIdx.x; i < BN * C; i += NT) {
            const int key = key0 + i / C;
            const bool in = key >= kfirst && key <= klast && key < limit; // outside every row's keys: zeros
            const int row = in ? (ring > 0 ? key % ring : key) : 0;
            cp16z(dst + sw<D>(i / C, i % C), src + ((int64_t)row * HK + hk) * D + (i % C) * 8, in);
        }
    };
    uint32_t qa[K16][4]; // this warp's SL dims of rows r0, r1
    {
        const uint32_t* q0 = reinterpret_cast<const uint32_t*>(q + ((int64_t)min(r0, W - 1) * H + head) * D + cs * SL);
        const uint32_t* q1 = reinterpret_cast<const uint32_t*>(q + ((int64_t)min(r1, W - 1) * H + head) * D + cs * SL);
#pragma unroll
        for (int k = 0; k < K16; ++k) {
            qa[k][0] = r0 < W ? q0[8 * k + qd] : 0u;
            qa[k][1] = r1 < W ? q1[8 * k + qd] : 0u;
            qa[k][2] = r0 < W ? q0[8 * k + qd + 4] : 0u;
            qa[k][3] = r1 < W ? q1[8 * k + qd + 4] : 0u;
        }
    }
#pragma unroll
    for (int i = 0; i < NS; ++i) {
        if (i < items) stage(i);
        commit();
    }
    auto take = [&](int it) -> const uint4* { // item it's slot, once every thread's copy has landed
        wait_group<NS - 1>();
        __syncthreads();
        return slots + (it % NS) * K::SLOT;
    };
    auto done = [&](int it) { // the slot is free: stage the item NS ahead into it
        __syncthreads();
        if (it + NS < items) stage(it + NS);
        commit();
    };
    float o[NB][4];
#pragma unroll
    for (int i = 0; i < NB; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
    for (int t = 0; t < tiles; ++t) {
        float s[4][4];
        {
            const uint4* sl = take(2 * t);
#pragma unroll
            for (int k = 0; k < K16; ++k) {
#pragma unroll
                for (int n = 0; n < 4; n += 2) {
                    uint32_t b[4];
                    ldmatrix4(b, sl + sw<D>(8 * (n + mt / 2) + lane % 8, cs * (SL / 8) + 2 * k + mt % 2));
                    if (k == 0) {
                        mma0(s[n], qa[k], b[0], b[1]);
                        mma0(s[n + 1], qa[k], b[2], b[3]);
                    } else {
                        mma(s[n], qa[k], b[0], b[1]);
                        mma(s[n + 1], qa[k], b[2], b[3]);
                    }
                }
            }
#pragma unroll
            for (int n = 0; n < 4; ++n)
#pragma unroll
                for (int j = 0; j < 4; ++j) xs[(warp * 16 + 4 * n + j) * 32 + lane] = s[n][j];
            done(2 * t);
        }
        // the group's scores: its slices' partials in slice order
#pragma unroll
        for (int n = 0; n < 4; ++n)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float a = xs[(grp * CS * 16 + 4 * n + j) * 32 + lane];
#pragma unroll
                for (int c = 1; c < CS; ++c) a = __fadd_rn(a, xs[((grp * CS + c) * 16 + 4 * n + j) * 32 + lane]);
                s[n][j] = a;
            }
        const int kb = (t0 + t) * BN + 2 * qd;
        float tm0 = -INFINITY, tm1 = -INFINITY;
        bool in[4][4];
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            const int c = kb + 8 * n;
            in[n][0] = c >= lo0 && c <= hi0;
            in[n][1] = c + 1 >= lo0 && c + 1 <= hi0;
            in[n][2] = c >= lo1 && c <= hi1;
            in[n][3] = c + 1 >= lo1 && c + 1 <= hi1;
#pragma unroll
            for (int j = 0; j < 4; ++j) s[n][j] = in[n][j] ? __fmul_rn(s[n][j], scale) : -INFINITY;
            tm0 = fmaxf(tm0, fmaxf(s[n][0], s[n][1]));
            tm1 = fmaxf(tm1, fmaxf(s[n][2], s[n][3]));
        }
        for (int x = 1; x <= 2; x *= 2) {
            tm0 = fmaxf(tm0, __shfl_xor_sync(FULL, tm0, x));
            tm1 = fmaxf(tm1, __shfl_xor_sync(FULL, tm1, x));
        }
        const bool act0 = tm0 != -INFINITY, act1 = tm1 != -INFINITY;
        const float n0 = act0 ? fmaxf(m0, tm0) : m0, n1 = act1 ? fmaxf(m1, tm1) : m1;
        const float a0 = act0 ? (m0 == -INFINITY ? 0.0f : __expf(__fsub_rn(m0, n0))) : 1.0f;
        const float a1 = act1 ? (m1 == -INFINITY ? 0.0f : __expf(__fsub_rn(m1, n1))) : 1.0f;
        float w0[4], w1[4];
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            s[n][0] = in[n][0] && act0 ? __expf(__fsub_rn(s[n][0], n0)) : 0.0f;
            s[n][1] = in[n][1] && act0 ? __expf(__fsub_rn(s[n][1], n0)) : 0.0f;
            s[n][2] = in[n][2] && act1 ? __expf(__fsub_rn(s[n][2], n1)) : 0.0f;
            s[n][3] = in[n][3] && act1 ? __expf(__fsub_rn(s[n][3], n1)) : 0.0f;
            const float x0 = __fadd_rn(s[n][0], s[n][1]), x1 = __fadd_rn(s[n][2], s[n][3]);
            const float y0 = __fadd_rn(x0, __shfl_xor_sync(FULL, x0, 2)), y1 = __fadd_rn(x1, __shfl_xor_sync(FULL, x1, 2));
            w0[n] = __fadd_rn(y0, __shfl_xor_sync(FULL, y0, 1));
            w1[n] = __fadd_rn(y1, __shfl_xor_sync(FULL, y1, 1));
        }
        l0 = __fmaf_rn(l0, a0, __fadd_rn(__fadd_rn(w0[0], w0[2]), __fadd_rn(w0[1], w0[3])));
        l1 = __fmaf_rn(l1, a1, __fadd_rn(__fadd_rn(w1[0], w1[2]), __fadd_rn(w1[1], w1[3])));
        m0 = n0;
        m1 = n1;
        uint32_t ph[2][4], pr[2][4]; // P's bf16 fragments, and (P32) the rest P - bf16(P)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            ph[j][0] = pack(s[2 * j][0], s[2 * j][1]);
            ph[j][1] = pack(s[2 * j][2], s[2 * j][3]);
            ph[j][2] = pack(s[2 * j + 1][0], s[2 * j + 1][1]);
            ph[j][3] = pack(s[2 * j + 1][2], s[2 * j + 1][3]);
            if constexpr (P32) {
                const float2 h0 = unpack(ph[j][0]), h1 = unpack(ph[j][1]), h2 = unpack(ph[j][2]), h3 = unpack(ph[j][3]);
                pr[j][0] = pack(__fsub_rn(s[2 * j][0], h0.x), __fsub_rn(s[2 * j][1], h0.y));
                pr[j][1] = pack(__fsub_rn(s[2 * j][2], h1.x), __fsub_rn(s[2 * j][3], h1.y));
                pr[j][2] = pack(__fsub_rn(s[2 * j + 1][0], h2.x), __fsub_rn(s[2 * j + 1][1], h2.y));
                pr[j][3] = pack(__fsub_rn(s[2 * j + 1][2], h3.x), __fsub_rn(s[2 * j + 1][3], h3.y));
            }
        }
        {
            const uint4* sl = take(2 * t + 1);
#pragma unroll
            for (int hf = 0; hf < NB / 8; ++hf) { // 64 columns at a time: the tile's P x V, then folded into o
                float tv[8][4];
#pragma unroll
                for (int j = 0; j < 2; ++j) {
#pragma unroll
                    for (int i = 0; i < 8; i += 2) {
                        uint32_t b[4];
                        ldmatrix4t(b, sl + sw<D>(16 * j + 8 * (mt % 2) + lane % 8, cs * (SL / 8) + 8 * hf + i + mt / 2));
                        if (j == 0) {
                            mma0(tv[i], ph[j], b[0], b[1]);
                            mma0(tv[i + 1], ph[j], b[2], b[3]);
                        } else {
                            mma(tv[i], ph[j], b[0], b[1]);
                            mma(tv[i + 1], ph[j], b[2], b[3]);
                        }
                        if constexpr (P32) {
                            mma(tv[i], pr[j], b[0], b[1]);
                            mma(tv[i + 1], pr[j], b[2], b[3]);
                        }
                    }
                }
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    float* x = o[8 * hf + i];
                    x[0] = __fmaf_rn(x[0], a0, tv[i][0]);
                    x[1] = __fmaf_rn(x[1], a0, tv[i][1]);
                    x[2] = __fmaf_rn(x[2], a1, tv[i][2]);
                    x[3] = __fmaf_rn(x[3], a1, tv[i][3]);
                }
            }
            done(2 * t + 1);
        }
    }
    wait_group<0>();
    bool odd = false;
#pragma unroll
    for (int i = 0; i < NB; ++i) {
        const int d = cs * SL + 8 * i + 2 * qd;
        const float x0 = __fdiv_rn(o[i][0], l0), x1 = __fdiv_rn(o[i][1], l0);
        const float x2 = __fdiv_rn(o[i][2], l1), x3 = __fdiv_rn(o[i][3], l1);
        if (r0 < W) {
            *reinterpret_cast<uint32_t*>(out + ((int64_t)r0 * H + head) * D + d) = pack(x0, x1);
            odd |= !isfinite(x0) || !isfinite(x1);
        }
        if (r1 < W) {
            *reinterpret_cast<uint32_t*>(out + ((int64_t)r1 * H + head) * D + d) = pack(x2, x3);
            odd |= !isfinite(x2) || !isfinite(x3);
        }
    }
    return __syncthreads_or(odd);
}

// A block with a NaN or infinite output: recompute only affected heads, leaving every finite head's bits alone.
template <int D>
__device__ __noinline__ void odd(const __nv_bfloat16* q, const __nv_bfloat16* kc, const __nv_bfloat16* vc,
                                 __nv_bfloat16* out, const int* hi, int p0, int W, int H, int HK, int G, float scale,
                                 int win, int ring) {
    using K = Cfg<D>;
    const int HB = G % K::NG == 0 ? K::NG : 1, RB = K::NG / HB;
    const int hk = blockIdx.y / (G / HB), hg = blockIdx.y % (G / HB);
    const int row0 = blockIdx.x * 16 * RB;
    const auto kv = [&](int key, int which) {
        return (which == 0 ? kc : vc) + ((int64_t)(ring > 0 ? key % ring : key) * HK + hk) * D;
    };
    for (int i = 0; i < 16 * RB * HB; ++i) {
        const int r = row0 + i % (16 * RB), head = hk * G + hg * HB + i / (16 * RB);
        if (r >= W) continue;
        __nv_bfloat16* dst = out + ((int64_t)r * H + head) * D;
        const bool affected = tf_wrows_head_affected(reinterpret_cast<const uint16_t*>(dst), D, threadIdx.x, NT);
        // Every thread makes the same decision before slow_head's block barriers. rows() has published all stores.
        if (!__syncthreads_or(affected)) continue;
        const int p = p0 + r;
        slow_head<D>(q + ((int64_t)r * H + head) * D, kv, win > 0 ? max(p - win + 1, 0) : 0,
                     hi != nullptr ? hi[r] : p, (double)scale, dst, 0, D);
    }
}

template <int D, bool P32, int FAULT>
__device__ __forceinline__ void kernel(const __nv_bfloat16* q, const __nv_bfloat16* kc, const __nv_bfloat16* vc,
                                       __nv_bfloat16* out, const int* hi, int p0, int W, int H, int HK, int G,
                                       float scale, int win, int ring) {
    if (rows<D, P32, FAULT>(q, kc, vc, out, hi, p0, W, H, HK, G, scale, win, ring))
        odd<D>(q, kc, vc, out, hi, p0, W, H, HK, G, scale, win, ring);
}

} // namespace prompt

} // namespace tf_wrows
