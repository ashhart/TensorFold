// TF_GEMMA4_V_ONLY decode partials over pages: part_staged's arithmetic, a block a key head; each key tile made once.
#pragma once

#include "window_rows512.cuh"
#include "window_vonly.cuh"

namespace tf_wrows {

// part_staged<D, NONE>'s bits head for head, keys from values (vk_tile) once a tile for all of key head kh's groups.
template <int D>
__device__ __forceinline__ void part_vo(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ vc,
                                        const int* __restrict__ pos, const int* __restrict__ slot,
                                        float* __restrict__ po, float* __restrict__ pm, float* __restrict__ pl, int h,
                                        int hk, float scale, Vk vk, Pg g) {
    constexpr int DPL = D / 32, HW = 4, C = D / 8;
    extern __shared__ uint4 tiles[];
    __shared__ float sp[CK * SP];
    __shared__ float mx[GMAX], ls[GMAX];
    float* ox = reinterpret_cast<float*>(tiles);
    const int64_t r = blockIdx.x;
    const int G = h / hk, HG = (G + HW - 1) / HW, c = blockIdx.z, chunks = gridDim.z, kh = blockIdx.y;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int p = pos[r];
    const int k0 = c * CK, k1 = min(c * CK + CK - 1, p);
    const int64_t part = (r * hk + kh) * chunks + c;
    if (k0 > k1) {
        for (int i = threadIdx.x; i < G; i += NT) pm[part * G + i] = -INFINITY, pl[part * G + i] = 0.0f;
        return;
    }
    const int nk = k1 - k0 + 1, nt = (nk + TK - 1) / TK, ts = (slot != nullptr ? slot[r] : 0) * g.stride;
    constexpr int WG = NW;
    const int kg = warp;
    const auto stage = [&](int t) { // part_staged's paged stage of value tile t
        uint4* dst = tiles + (t % NSG) * TK * C;
        int64_t tile = -1;
        const int a = k0 + t * TK, b = k0 + min(t * TK + TK - 1, nk - 1);
        if (a / g.tok == b / g.tok)
            tile = g.tab[ts + a / g.tok] * g.page + (static_cast<int64_t>(a % g.tok) * hk + kh) * D;
        for (int i = threadIdx.x; i < TK * C; i += NT) {
            const int jj = t * TK + i / C;
            const int key = k0 + min(jj, nk - 1);
            if (tile >= 0) {
                cp16z(dst + i, vc + tile + (static_cast<int64_t>(key - k0 - t * TK) * hk * D) + (i % C) * 8, jj < nk);
                continue;
            }
            const int64_t id = g.tab[ts + key / g.tok];
            cp16z(dst + i, vc + id * g.page + (static_cast<int64_t>(key % g.tok) * hk + kh) * D + (i % C) * 8, jj < nk);
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
    for (int t = 0; t < NSG - 1; ++t)
        if (t < nt) stage(t);
        else commit();
    for (int t = 0; t < nt; ++t) { // scores: each tile's keys made once, then every head group over them
        if (t + NSG - 1 < nt) stage(t + NSG - 1);
        else commit();
        asm volatile("cp.async.wait_group %0;\n" ::"n"(NSG - 1));
        __syncthreads();
        vk_tile<D>(min(TK, nk - t * TK), k0 + t * TK, vk,
                   [&](int u, int d) { return smem_bf(tiles + (t % NSG) * TK * C) + u * D + d; });
        for (int hg = 0; hg < HG; ++hg) {
            float qv[HW][DPL];
#pragma unroll
            for (int hh = 0; hh < HW; ++hh) {
                const int hd = min(hg * HW + hh, G - 1);
                load(reinterpret_cast<const uint4*>(q + (r * h + kh * G + hd) * D + lane * DPL), qv[hh]);
            }
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
    for (int hd = warp; hd < G; hd += NW) { // a warp a head: part_staged's max and weights
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
    for (int hg = 0; hg < HG; ++hg) { // o: a group at a time over the values, staged again (L2 hits)
        for (int t = 0; t < NSG - 1; ++t)
            if (t < nt) stage(t);
            else commit();
        float acc[HW][DPL];
#pragma unroll
        for (int hh = 0; hh < HW; ++hh)
#pragma unroll
            for (int i = 0; i < DPL; ++i) acc[hh][i] = 0.0f;
        for (int t = 0; t < nt; ++t) {
            if (t + NSG - 1 < nt) stage(t + NSG - 1);
            else commit();
            asm volatile("cp.async.wait_group %0;\n" ::"n"(NSG - 1));
            __syncthreads();
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
            __syncthreads();
        }
        if (kg > 0)
#pragma unroll
            for (int hh = 0; hh < HW; ++hh) {
                if (hg * HW + hh >= G) break;
#pragma unroll
                for (int i = 0; i < DPL; ++i) ox[((kg - 1) * HW + hh) * D + lane * DPL + i] = acc[hh][i];
            }
        __syncthreads();
        if (kg == 0)
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
        __syncthreads(); // ox is the next group's tiles
    }
    for (int i = threadIdx.x; i < G; i += NT) pm[part * G + i] = mx[i], pl[part * G + i] = ls[i];
}

// merge_pg_odd over V-only pages: the float64 replay with each key from its value row (a warp's own, vk_row).
template <int D>
__device__ __noinline__ void merge_pg_odd_vo(const __nv_bfloat16* q, const __nv_bfloat16* vc, const int* pos,
                                             const int* slot, int h, int hk, float scale, Pg pg, Vk vk,
                                             __nv_bfloat16* oh, int64_t r, int kh, int hd) {
    __shared__ __nv_bfloat16 kr[NW][D];
    const int p = pos[r], G = h / hk, ts = (slot != nullptr ? slot[r] : 0) * pg.stride;
    const auto kv = [&](int key, int which) -> const __nv_bfloat16* {
        const int64_t pid = pg.tab[ts + key / pg.tok];
        const __nv_bfloat16* row = vc + pid * pg.page + (static_cast<int64_t>(key % pg.tok) * hk + kh) * D;
        return which == 0 ? vk_row<D>(row, key, vk, kr[threadIdx.x / 32]) : row;
    };
    const int c0 = blockIdx.z * NT % D;
    slow_head<D>(q + (r * h + kh * G + hd) * D, kv, 0, p, (double)scale, oh, c0, c0 + NT);
}

// merge<D, NONE, true> over V-only pages (no window): its sums in its order, merge_pg_odd_vo as its replay.
template <int D>
__device__ __forceinline__ void merge_vo(const float* __restrict__ po, const float* __restrict__ pm,
                                         const float* __restrict__ pl, __nv_bfloat16* __restrict__ out, int h, int hk,
                                         int chunks, const __nv_bfloat16* __restrict__ q,
                                         const __nv_bfloat16* __restrict__ vc, const int* __restrict__ pos,
                                         const int* __restrict__ slot, float scale, Pg pg, Vk vk) {
    const int64_t r = blockIdx.x;
    const int kh = blockIdx.y, G = h / hk;
    const int o = blockIdx.z * NT + threadIdx.x;
    const int hd = o / D, e = o % D;
    const int64_t part0 = (r * hk + kh) * chunks;
    float m = -INFINITY;
    for (int c = 0; c < chunks; ++c) m = fmaxf(m, pm[(part0 + c) * G + hd]);
    float acc = 0.0f, l = 0.0f;
    for (int c = 0; c < chunks; ++c) {
        const float mc = pm[(part0 + c) * G + hd];
        if (mc == -INFINITY) continue;
        const float a = expf(__fsub_rn(mc, m));
        acc = __fmaf_rn(a, po[((part0 + c) * G + hd) * D + e], acc);
        l = __fmaf_rn(a, pl[(part0 + c) * G + hd], l);
    }
    const float x = __fdiv_rn(acc, l);
    __nv_bfloat16* oh = out + (r * h + kh * G + hd) * D;
    oh[e] = __float2bfloat16_rn(x);
    if (__syncthreads_or(!isfinite(x))) merge_pg_odd_vo<D>(q, vc, pos, slot, h, hk, scale, pg, vk, oh, r, kh, hd);
}

} // namespace tf_wrows
