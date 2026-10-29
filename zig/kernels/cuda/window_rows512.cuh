// window_rows.cuh's decode partials at head dim 512: K and V tiles in shared memory, a block 4 query heads.
#pragma once

#include "window_rows.cuh"

namespace tf_wrows {

// part's arithmetic for head dim 512: a block 4 heads, K and V tiles in shared memory, a warp a key run.
constexpr int TK = 32, NSG = 2;

// P: empty, or one Pg (window_pages.cu): key k of slot s at its page, tab[s * stride + k / tok] (no window).
template <int D, int FAULT, typename... P>
__device__ __forceinline__ void part_staged(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc,
                                            const __nv_bfloat16* __restrict__ vc, const int* __restrict__ pos,
                                            const int* __restrict__ slot, float* __restrict__ po, float* __restrict__ pm,
                                            float* __restrict__ pl, int h, int hk, int win, long long ring, float scale,
                                            P... pg) {
    constexpr int DPL = D / 32, HW = 4, C = D / 8; // a lane's dims and columns, heads a block (and warp), uint4 a key
    extern __shared__ uint4 tiles[];                       // [NSG][TK][C] K or V tiles; then the runs' partial o
    __shared__ float sp[CK * SP];
    __shared__ float mx[GMAX], ls[GMAX];
    float* ox = reinterpret_cast<float*>(tiles);
    const int64_t r = blockIdx.x;
    const int G = h / hk, HG = (G + HW - 1) / HW, c = blockIdx.z, chunks = gridDim.z;
    const int kh = blockIdx.y / HG, hg = blockIdx.y % HG; // the block's heads: hg HW.. of key head kh
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int p = pos[r];
    const int64_t base = (slot != nullptr ? static_cast<int64_t>(slot[r]) : 0) * ring;
    const int lo = win > 0 ? max(p - win + (FAULT == EDGE ? 0 : 1), 0) : 0;
    const int k0 = max((lo / CK + c) * CK, lo), k1 = min((lo / CK + c) * CK + CK - 1, p);
    const int64_t part = (r * hk + kh) * chunks + c;
    const int own = min(HW, G - hg * HW); // heads of this block
    if (k0 > k1) {
        if (threadIdx.x < own) pm[part * G + hg * HW + threadIdx.x] = -INFINITY, pl[part * G + hg * HW + threadIdx.x] = 0.0f;
        return;
    }
    const int nk = k1 - k0 + 1, nt = (nk + TK - 1) / TK;
    const int64_t r0 = k0 % ring;
    // warp kg: a tile's keys kg, kg + NW, ...; its keys in tile order are its run
    constexpr int WG = NW;
    const int kg = warp;
    constexpr bool busy = true;
    const auto stage = [&](const __nv_bfloat16* cache, int t) {
        uint4* dst = tiles + (t % NSG) * TK * C;
        for (int i = threadIdx.x; i < TK * C; i += NT) {
            const int jj = t * TK + i / C;
            if constexpr (sizeof...(P) != 0) { // a page a key: any tok, a tile may cross pages
                const Pg& g = (pg, ...);
                const int key = k0 + min(jj, nk - 1);
                const int64_t id = g.tab[(slot != nullptr ? slot[r] : 0) * g.stride + key / g.tok];
                cp16z(dst + i, cache + id * g.page + (static_cast<int64_t>(key % g.tok) * hk + kh) * D + (i % C) * 8,
                      jj < nk);
            } else {
                int64_t row = r0 + min(jj, nk - 1);
                if (row >= ring) row -= ring;
                cp16z(dst + i, cache + ((base + row) * hk + kh) * D + (i % C) * 8, jj < nk);
            }
        }
        commit();
    };
    const auto load = [&](const uint4* src, float (&x)[DPL]) {
#pragma unroll
        for (int i = 0; i < DPL / 8; ++i) {
            const uint4 u = src[i];
            const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
            for (int t = 0; t < 4; ++t) x[8 * i + 2 * t] = lo16(w[t]), x[8 * i + 2 * t + 1] = hi16(w[t]);
        }
    };
    float qv[HW][DPL];
#pragma unroll
    for (int hh = 0; hh < HW; ++hh) {
        const int hd = min(hg * HW + hh, G - 1);
        load(reinterpret_cast<const uint4*>(q + (r * h + kh * G + hd) * D + lane * DPL), qv[hh]);
    }
    for (int t = 0; t < NSG - 1; ++t)
        if (t < nt) stage(kc, t);
        else commit();
    for (int t = 0; t < nt; ++t) { // scores: a lane's DPL dims in an fma chain, then a xor tree over the lanes
        if (t + NSG - 1 < nt) stage(kc, t + NSG - 1);
        else commit();
        asm volatile("cp.async.wait_group %0;\n" ::"n"(NSG - 1));
        __syncthreads();
        if (busy) {
            for (int u0 = kg; u0 < TK; u0 += 4 * WG) { // four keys' chains side by side
                float kf[4][DPL], sc[4][HW];
#pragma unroll
                for (int x = 0; x < 4; ++x) {
                    const int u = min(u0 + x * WG, TK - 1);
                    load(tiles + (t % NSG) * TK * C + u * C + lane * (DPL / 8), kf[x]);
                }
#pragma unroll
                for (int x = 0; x < 4; ++x)
#pragma unroll
                    for (int hh = 0; hh < HW; ++hh) {
                        sc[x][hh] = 0.0f;
#pragma unroll
                        for (int i = 0; i < DPL; ++i) sc[x][hh] = __fmaf_rn(qv[hh][i], kf[x][i], sc[x][hh]);
                    }
#pragma unroll
                for (int o = 16; o > 0; o >>= 1)
#pragma unroll
                    for (int x = 0; x < 4; ++x)
#pragma unroll
                        for (int hh = 0; hh < HW; ++hh)
                            sc[x][hh] = __fadd_rn(sc[x][hh], __shfl_xor_sync(FULL, sc[x][hh], o));
                if (lane == 0)
#pragma unroll
                    for (int x = 0; x < 4; ++x) {
                        const int u = u0 + x * WG;
#pragma unroll
                        for (int hh = 0; hh < HW; ++hh)
                            if (u < TK && t * TK + u < nk && hg * HW + hh < G)
                                sp[(t * TK + u) * SP + hg * HW + hh] = __fmul_rn(sc[x][hh], scale);
                    }
            }
        }
        __syncthreads();
    }
    for (int t = 0; t < NSG - 1; ++t) // the values' first tiles load while the weights are formed
        if (t < nt) stage(vc, t);
        else commit();
    if (warp < own) { // a warp a head
        const int hd = hg * HW + warp;
        float m = -INFINITY;
        for (int jj = lane; jj < nk; jj += 32) m = fmaxf(m, sp[jj * SP + hd]);
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(FULL, m, o));
        float l = 0.0f;
        for (int jj = lane; jj < nk; jj += 32) {
            const float sv = sp[jj * SP + hd];
            const float e = sv == -INFINITY ? 0.0f : expf(__fsub_rn(sv, m));
            sp[jj * SP + hd] = e;
            l = __fadd_rn(l, e);
        }
        for (int o = 16; o > 0; o >>= 1) l = __fadd_rn(l, __shfl_xor_sync(FULL, l, o));
        if (lane == 0) mx[hd] = m, ls[hd] = l;
    }
    float acc[HW][DPL];
#pragma unroll
    for (int hh = 0; hh < HW; ++hh)
#pragma unroll
        for (int i = 0; i < DPL; ++i) acc[hh][i] = 0.0f;
    for (int t = 0; t < nt; ++t) { // o: a lane's DPL columns, an fma chain over its run's keys in order
        if (t + NSG - 1 < nt) stage(vc, t + NSG - 1);
        else commit();
        asm volatile("cp.async.wait_group %0;\n" ::"n"(NSG - 1));
        __syncthreads();
        if (busy) {
            for (int u = kg; u < TK && t * TK + u < nk; u += WG) {
                float vf[DPL];
                load(tiles + (t % NSG) * TK * C + u * C + lane * (DPL / 8), vf);
#pragma unroll
                for (int hh = 0; hh < HW; ++hh) {
                    if (hg * HW + hh >= G) break;
                    const float w = sp[(t * TK + u) * SP + hg * HW + hh];
#pragma unroll
                    for (int i = 0; i < DPL; ++i) acc[hh][i] = __fmaf_rn(w, vf[i], acc[hh][i]);
                }
            }
        }
        __syncthreads();
    }
    if (busy && kg > 0)
#pragma unroll
        for (int hh = 0; hh < HW; ++hh) {
            const int hd = hg * HW + hh;
            if (hd >= G) break;
#pragma unroll
            for (int i = 0; i < DPL; ++i) ox[((kg - 1) * HW + hh) * D + lane * DPL + i] = acc[hh][i];
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
                for (int k = 1; k < WG; ++k) a = __fadd_rn(a, ox[((k - 1) * HW + hh) * D + lane * DPL + i]);
                po[(part * G + hd) * D + lane * DPL + i] = a;
            }
        }
    }
    if (threadIdx.x < own) {
        const int hd = hg * HW + threadIdx.x;
        pm[part * G + hd] = mx[hd], pl[part * G + hd] = ls[hd];
    }
}

} // namespace tf_wrows
