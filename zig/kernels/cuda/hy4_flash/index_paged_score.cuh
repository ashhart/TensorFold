// Hy4-flash compressed indexer: query M-RoPE and 16-head ReLU-weighted pool scoring on mma.m16n8k16.
#pragma once

#include "../common/qmm_frag.cuh"
#include "index_paged_common.cuh"

namespace tf_hy4_idx {

constexpr int kKQ = kDim / 16;
constexpr int kRD = 64;

__device__ __forceinline__ void ldm4(uint32_t (&r)[4], const void* p) { qmm_frag::ldmatrix4(r, p); }

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    qmm_frag::mma(d, a, b0, b1);
}

// ReLU preserving NaN (NaN < 0 is false); kBadNan uses fmaxf which turns NaN into 0.
__device__ __forceinline__ float relu_keep_nan(float d) { return d < 0.0f ? 0.0f : d; }

__device__ __forceinline__ int tile_at(int row, int u) { return row * kDim + ((u ^ (row & 7)) << 3); }

template <int WARPS, int NSUB, int STAGES, int RB>
struct ScoreCfg {
    static constexpr int BP = 16 * NSUB;
    static constexpr int TILE = BP * kDim;
    static constexpr int SMEM = STAGES * TILE * 2;
    static constexpr int THREADS = WARPS * 32;
    static_assert(RB <= WARPS && (WARPS % RB) == 0, "rows a CTA");
};

template <int F>
__device__ __forceinline__ float head_sum(const float (&hw)[4], float d0, float d1, float d2, float d3) {
    auto rl = [](float d) { return F == kBadNan ? fmaxf(d, 0.0f) : relu_keep_nan(d); };
    float s = step_fma<F>(hw[0], rl(d0), 0.0f);
    s = step_fma<F>(hw[1], rl(d1), s);
    s = step_fma<F>(hw[2], rl(d2), s);
    s = step_fma<F>(hw[3], rl(d3), s);
    return s;
}

struct RopeIn {
    const float2* CS;
    const int* POS3;
    int R;
};

template <int F>
__device__ __forceinline__ void q_mrope_body(const int* __restrict__ SEL, int n_sel, const bf16* __restrict__ IQ,
                                             const int* __restrict__ ROW_POS, const int* __restrict__ POS3, int R,
                                             const float2* __restrict__ CS, bf16* __restrict__ QI) {
    bump<F>(IQ);
    qmm_frag::grid_launch();
    const int i = blockIdx.x;
    if (i >= n_sel) return;
    const int r = SEL != nullptr ? SEL[i] : i, t = threadIdx.x, h = t >> 4, u = t & 15;
    const uint4 v = *reinterpret_cast<const uint4*>(IQ + (static_cast<size_t>(r) * kHeads + h) * kDim + u * 8);
    uint4 o = v;
    if (u >= 8) {
        const uint32_t w[4] = {v.x, v.y, v.z, v.w};
        uint32_t y[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int pr = 4 * (u - 8) + j;
            if (F == kBadTail && pr == kRopePairs - 1) {
                y[j] = w[j];
                continue;
            }
            const int pos = POS3 != nullptr ? POS3[mrope_plane(pr) * R + r] : ROW_POS[r];
            const float2 cs = CS[static_cast<size_t>(pos) * kRopePairs + pr];
            float y0, y1;
            rope_pair(bf_lo(w[j]), bf_hi(w[j]), cs.x, cs.y, y0, y1);
            if constexpr (F == kBadNan) {
                if (y0 != y0) y0 = 0.0f;
                if (y1 != y1) y1 = 0.0f;
            }
            y[j] = bf_pack(y0, y1);
        }
        o = make_uint4(y[0], y[1], y[2], y[3]);
    } else if constexpr (F == kBadNan) {
        const uint32_t w[4] = {v.x, v.y, v.z, v.w};
        uint32_t y[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float a = bf_lo(w[j]), b = bf_hi(w[j]);
            if (a != a) a = 0.0f;
            if (b != b) b = 0.0f;
            y[j] = bf_pack(a, b);
        }
        o = make_uint4(y[0], y[1], y[2], y[3]);
    }
    *reinterpret_cast<uint4*>(QI + (static_cast<size_t>(i) * kHeads + h) * kDim + u * 8) = o;
}

template <int WARPS, int NSUB, int STAGES, int RB, int F = kNone, bool ROPE = false>
__device__ __forceinline__ void score_rows(bf16* __restrict__ smem_tiles, int* __restrict__ meta,
                                           const int* __restrict__ SEL, int n_sel, const bf16* __restrict__ QI,
                                           const float* __restrict__ HW, const int* __restrict__ ROW_STREAM,
                                           const int* __restrict__ ROW_POS, const bf16* __restrict__ GKC, int gpool,
                                           float* __restrict__ SC, int gcap, RopeIn rope = RopeIn{nullptr, nullptr, 0},
                                           const uint32_t* __restrict__ PT = nullptr, int pt_stride = 0,
                                           int g_base = 0) {
    using C = ScoreCfg<WARPS, NSUB, STAGES, RB>;
    constexpr int BP = C::BP, THREADS = C::THREADS, CH = kDim / 8;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5, g = lane >> 2, q = lane & 3;
    bump<F>(QI);
    int* s_r = meta;
    int* s_own = meta + RB;
    int* s_str = meta + 2 * RB;
    const int i0 = blockIdx.y * RB;
    const int nb = min(RB, n_sel - i0);
    if (t < nb) {
        const int r = SEL != nullptr ? SEL[i0 + t] : (i0 + t);
        s_r[t] = r;
        s_own[t] = ROW_POS[r] / 4;
        s_str[t] = ROW_STREAM != nullptr ? ROW_STREAM[r] : 0;
    }
    __syncthreads();
    if constexpr (!ROPE) qmm_frag::grid_wait();
    qmm_frag::grid_launch();
    for (int a = 0; a < nb;) {
        int e = a + 1;
        while (e < nb && s_str[e] == s_str[a]) ++e;
        const int nr = e - a;
        int p2 = 1;
        while (p2 < nr) p2 <<= 1;
        const int wpr = WARPS / p2, slot = warp / wpr, sub0 = warp % wpr;
        const bool live = slot < nr;
        const int j = a + (live ? slot : 0);
        const int i = i0 + j, r = s_r[j], own = F == kBadTail ? (s_own[j] & ~15) : s_own[j];
        int nmax_g = 0;
        for (int k = a; k < e; ++k) nmax_g = max(nmax_g, s_own[k]);
        uint32_t bq[kKQ][2][2];
        {
            const bf16* qrow = QI + static_cast<size_t>(ROPE ? r : i) * kHeads * kDim;
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const uint32_t* hq = reinterpret_cast<const uint32_t*>(qrow + (nt * 8 + g) * kDim);
#pragma unroll
                for (int kq = 0; kq < kKQ; ++kq) {
                    bq[kq][nt][0] = hq[kq * 8 + q];
                    bq[kq][nt][1] = hq[kq * 8 + q + 4];
                }
            }
            if (ROPE) {
#pragma unroll
                for (int kq = kKQ / 2; kq < kKQ; ++kq)
#pragma unroll
                    for (int ee = 0; ee < 2; ++ee) {
                        const int pr = 8 * (kq - kKQ / 2) + q + 4 * ee;
                        const int pos = rope.POS3 != nullptr ? rope.POS3[mrope_plane(pr) * rope.R + r] : ROW_POS[r];
                        const float2 cs = rope.CS[static_cast<size_t>(pos) * kRopePairs + pr];
#pragma unroll
                        for (int nt = 0; nt < 2; ++nt) {
                            const uint32_t w = bq[kq][nt][ee];
                            float y0, y1;
                            rope_pair(bf_lo(w), bf_hi(w), cs.x, cs.y, y0, y1);
                            bq[kq][nt][ee] = bf_pack(y0, y1);
                        }
                    }
            }
        }
        float hw[4];
        hw[0] = HW[r * kHeads + 2 * q];
        hw[1] = HW[r * kHeads + 2 * q + 1];
        hw[2] = HW[r * kHeads + 8 + 2 * q];
        hw[3] = HW[r * kHeads + 9 + 2 * q];
        const int ns = (nmax_g + 15) >> 4;
        const int s0 = static_cast<int>(static_cast<int64_t>(blockIdx.x) * ns / gridDim.x);
        const int s1 = static_cast<int>(static_cast<int64_t>(blockIdx.x + 1) * ns / gridDim.x);
        const int ntile = (s1 - s0 + NSUB - 1) / NSUB;
        const int pend = min(s1 * 16, nmax_g);
        const bf16* keys = GKC + static_cast<size_t>(s_str[a]) * gpool * kDim;
        const uint32_t* s_pt = PT != nullptr ? (PT + static_cast<size_t>(s_str[a]) * pt_stride) : nullptr;
        float* sc = SC + static_cast<size_t>(i) * gcap;
        auto stage = [&](int buf, int k) {
            const int pb = (s0 + k * NSUB) * 16;
            bf16* tile = smem_tiles + buf * C::TILE;
            static_assert((BP * CH) % THREADS == 0, "whole staging steps");
#pragma unroll
            for (int it = 0; it < BP * CH / THREADS; ++it) {
                const int idx = t + it * THREADS, row = idx / CH, u = idx % CH, p = pb + row;
                const bool ok = p < pend;
                if (s_pt != nullptr) {
                    const unsigned up = static_cast<unsigned>(ok ? (g_base + p) : 0);
                    const size_t phys_p = ok ? (static_cast<size_t>(__ldg(s_pt + (up >> 6))) * 64u + (up & 63u)) : 0u;
                    tf_common::cp16z(tile + tile_at(row, u), GKC + phys_p * kDim + u * 8, ok);
                } else {
                    tf_common::cp16z(tile + tile_at(row, u), keys + static_cast<size_t>(ok ? p : 0) * kDim + u * 8, ok);
                }
            }
        };
#pragma unroll
        for (int k = 0; k < STAGES - 1; ++k) {
            if (k < ntile) stage(k, k);
            tf_common::cp_commit();
        }
        for (int k = 0; k < ntile; ++k) {
            tf_common::cp_wait<STAGES - 2>();
            __syncthreads();
            {
                const int kn = k + STAGES - 1;
                if (kn < ntile) stage(kn % STAGES, kn);
                tf_common::cp_commit();
            }
            const bf16* tile = smem_tiles + (k % STAGES) * C::TILE;
            const int sb = s0 + k * NSUB;
#pragma unroll 1
            for (int s = sub0; s < NSUB; s += wpr) {
                const int p0 = (sb + s) * 16;
                if (!live || sb + s >= s1 || p0 >= own) break;
                float acc[2][4] = {};
                const int lrow = s * 16 + ((lane >> 3) & 1) * 8 + (lane & 7);
#pragma unroll
                for (int kq = 0; kq < kKQ; ++kq) {
                    uint32_t a4[4];
                    ldm4(a4, tile + tile_at(lrow, kq * 2 + (lane >> 4)));
                    mma16816(acc[0], a4, bq[kq][0][0], bq[kq][0][1]);
                    mma16816(acc[1], a4, bq[kq][1][0], bq[kq][1][1]);
                    if constexpr (F == kBadSums) {
                        for (int m = 0; m < 2; ++m)
                            for (int ee = 0; ee < 4; ++ee) acc[m][ee] = __half2float(__float2half_rn(acc[m][ee]));
                    }
                }
                float sg = head_sum<F>(hw, acc[0][0], acc[0][1], acc[1][0], acc[1][1]);
                float sg8 = head_sum<F>(hw, acc[0][2], acc[0][3], acc[1][2], acc[1][3]);
                sg = add_f<F>(sg, __shfl_xor_sync(0xffffffffu, sg, 1));
                sg8 = add_f<F>(sg8, __shfl_xor_sync(0xffffffffu, sg8, 1));
                sg = add_f<F>(sg, __shfl_xor_sync(0xffffffffu, sg, 2));
                sg8 = add_f<F>(sg8, __shfl_xor_sync(0xffffffffu, sg8, 2));
                const int gg = p0 + g + (q & 1) * 8;
                if (q < 2 && gg < own) sc[gg] = (q & 1) ? sg8 : sg;
            }
        }
        __syncthreads();
        a = e;
    }
}

}  // namespace tf_hy4_idx
