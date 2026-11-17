// DeepSeek V4.1 release experts: CTA pair tiles through cp.async stages (gathered rows, packed weights, scales).
#pragma once

#include "release_moe.cuh"
#include "release_tile.cuh"

namespace rel {

// BM pairs by BN columns of NB matrices, KG K groups a stage (4 stages): WM x BN/32 warps of RM row tiles.
template <int BM_, int BN_, int NB_, int KG_>
struct ExCfg {
    static constexpr int BM = BM_, NB = NB_, BN = BN_, KG = KG_, ST = 4;
    static constexpr int WM = BM == 16 ? 1 : 2, WN = BN / 32, NT = 32 * WM * WN, RM = BM / 16 / WM;
    static constexpr int AR = 32 * KG, WR = 16 * KG;  // bytes a stage row: codes, packed weights
    static constexpr int A = BM * AR, W = BN * WR, SA = BM * KG, SW = BN * KG;
    static constexpr int NS = SA + NB * SW;
    static constexpr int STAGE = (A + NB * W + NS + 127) & ~127;
    static constexpr int ROWS = BM * 16;
    static constexpr int SMEM = ST * STAGE + ROWS;
    static constexpr int SPT = (NS + NT - 1) / NT;
    static_assert(KG == 4 || KG == 8, "four or eight groups a stage");
};

// A codes row's and a packed weight row's 16-byte chunk c, swizzled so 8- and 4-byte fragment reads do not conflict.
template <int KG>
__device__ __forceinline__ uint32_t a_chunk(int r, int c) {
    return (uint32_t)(r * 32 * KG + ((c ^ ((r & 3) << 1)) << 4));
}
template <int KG>
__device__ __forceinline__ uint32_t w_chunk(int col, int c) {
    const int x = KG == 4 ? ((col >> 1) & 3) : (col & 7);
    return (uint32_t)(col * 16 * KG + ((c ^ x) << 4));
}

// A pair tile's K groups through the stages into acc: row_of(i, codes, scales) names pair i's rows, W at column 0.
template <class C, class RowOf>
__device__ __forceinline__ void ex_cta(float (&acc)[C::NB][C::RM][4][4], RowOf row_of, int npairs,
                                       const uint8_t* const (&W)[C::NB], const uint8_t* const (&WS)[C::NB], int kb,
                                       int groups, int cvalid, uint8_t* smem, int tid, int wrow, int wcol) {
    constexpr int KG = C::KG;
    const uint8_t** rp = reinterpret_cast<const uint8_t**>(smem + C::ST * C::STAGE);
    const uint8_t** rs = rp + C::BM;
    if (tid < C::BM) {
        const uint8_t *c = nullptr, *s = nullptr;
        if (tid < npairs) row_of(tid, c, s);
        rp[tid] = c;
        rs[tid] = s;
    }
    __syncthreads();
    const uint32_t base = (uint32_t)__cvta_generic_to_shared(smem);
    const int stages = (groups + KG - 1) / KG;
    const Lane l = lane_of(tid & 31);
    // scale bytes pass through registers: fetched with a stage's codes, stored once this stage's mmas are issued
    uint32_t sv[C::SPT];
    auto load = [&](int st) {
        const int slot = st % C::ST, g = KG * st, ng = min(KG, groups - g);
        const uint32_t sA = base + slot * C::STAGE;
        for (int i = tid; i < C::BM * 2 * KG; i += C::NT) {
            const int r = i / (2 * KG), c = i % (2 * KG);
            if (c < 2 * ng) cp16(sA + a_chunk<KG>(r, c), rp[r] ? rp[r] + g * 32 + c * 16 : W[0], rp[r] != nullptr);
        }
#pragma unroll
        for (int b = 0; b < C::NB; ++b)
            for (int i = tid; i < C::BN * KG; i += C::NT) {
                const int col = i / KG, c = i % KG;
                const bool ok = col < cvalid;
                if (c < ng) cp16(sA + C::A + b * C::W + w_chunk<KG>(col, c), ok ? W[b] + col * kb + (g + c) * 16 : W[0], ok);
            }
#pragma unroll
        for (int j = 0; j < C::SPT; ++j) {
            const int i = tid + j * C::NT, s = i % KG;
            if (i < C::SA) {
                const uint8_t* row = rs[i / KG];
                sv[j] = (s < ng && row) ? row[g + s] : 127u;
            } else if (i < C::NS) {
                const int b = (i - C::SA) / C::SW, col = ((i - C::SA) % C::SW) / KG;
                sv[j] = (s < ng && col < cvalid) ? WS[b][(long long)col * groups + g + s] : 127u;
            }
        }
    };
    auto store = [&](int st) {
        uint8_t* S = smem + (st % C::ST) * C::STAGE + C::A + C::NB * C::W;
#pragma unroll
        for (int j = 0; j < C::SPT; ++j) {
            const int i = tid + j * C::NT;
            if (i < C::NS) S[i] = (uint8_t)sv[j];
        }
    };
    for (int st = 0; st < C::ST - 1; ++st) {
        if (st < stages) {
            load(st);
            store(st);
        }
        cp_commit();
    }
    for (int st = 0; st < stages; ++st) {
        cp_wait<C::ST - 2>();
        __syncthreads();
        const int next = st + C::ST - 1;
        if (next < stages) load(next);
        cp_commit();
        const uint8_t* sA = smem + (st % C::ST) * C::STAGE;
        const uint8_t* S = sA + C::A + C::NB * C::W;
        const int steps = min(KG, groups - KG * st);
#pragma unroll
        for (int k = 0; k < KG; ++k) {
            if (k >= steps) break;
            uint32_t a[C::RM][4], sa[C::RM];
#pragma unroll
            for (int i = 0; i < C::RM; ++i) {
                const int r = wrow + 16 * i + l.g, off = (l.t & 1) * 8, c = 2 * k + (l.t >> 1);
                const uint2 lo = *reinterpret_cast<const uint2*>(sA + a_chunk<KG>(r, c) + off);
                const uint2 hi = *reinterpret_cast<const uint2*>(sA + a_chunk<KG>(r + 8, c) + off);
                a[i][0] = lo.x;
                a[i][1] = hi.x;
                a[i][2] = lo.y;
                a[i][3] = hi.y;
                sa[i] = S[(r + ((l.t & 1) << 3)) * KG + k];
            }
#pragma unroll
            for (int b = 0; b < C::NB; ++b) {
                uint32_t bb[4][2], sb[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int col = wcol + 8 * j + l.g;
                    expand_e2m1(*reinterpret_cast<const uint32_t*>(sA + C::A + b * C::W + w_chunk<KG>(col, k) + 4 * l.t), bb[j]);
                    sb[j] = S[C::SA + b * C::SW + col * KG + k];
                }
#pragma unroll
                for (int i = 0; i < C::RM; ++i)
#pragma unroll
                    for (int j = 0; j < 4; ++j) mma_e2m1(acc[b][i][j], a[i], bb[j], sa[i], sb[j]);
            }
        }
        if (next < stages) store(next);
    }
    cp_wait<0>();
    __syncthreads();
}

// CTA tiles first, first + stride, ... over (active slot, BN columns); each loops its slot's BM-pair tiles.
template <class C, class F>
__device__ __forceinline__ void slot_ctas(const int* offs, const int* active, const int* nactive, int n, int first,
                                          int stride, F tile) {
    const int blocks = (n + C::BN - 1) / C::BN, tiles = *nactive * blocks;
    for (int t = first; t < tiles; t += stride) {
        const int slot = active[t / blocks], col0 = (t % blocks) * C::BN;
        const int lo = offs[slot], hi = offs[slot + 1];
        for (int at = lo; at < hi; at += C::BM) tile(slot, at, min(C::BM, hi - at), col0);
    }
}

// The routed gate/up through the stages: a CTA a (slot, BN columns), BM pairs at a time.
template <int BM, int BN, int KG>
__device__ __forceinline__ void gateup_ctas(const RelGateUp& p, uint8_t* smem, int tid, int first, int stride) {
    using C = ExCfg<BM, BN, 2, KG>;
    const int groups = p.k >> 5, kb = p.k >> 1, warp = tid >> 5;
    const int wrow = (warp / C::WN) * C::RM * 16, wcol = (warp % C::WN) * 32;
    const Lane l = lane_of(tid & 31);
    slot_ctas<C>(p.offs, p.active, p.nactive, p.n, first, stride, [&](int slot, int at, int np, int col0) {
        const long long w0 = (long long)slot * p.n + col0;
        const uint8_t* const W[2] = {p.w1 + w0 * kb, p.w3 + w0 * kb};
        const uint8_t* const WS[2] = {p.s1 + w0 * groups, p.s3 + w0 * groups};
        float acc[2][C::RM][4][4] = {};
        ex_cta<C>(acc, [&](int i, const uint8_t*& c, const uint8_t*& s) { gateup_row(p, at, i, c, s); }, np, W, WS, kb,
                  groups, min(BN, p.n - col0), smem, tid, wrow, wcol);
        if (col0 + wcol < p.n) gateup_store<C::RM>(p, acc, at, np, wrow, col0 + wcol, l);
    });
}

// The down through the stages: a CTA a (slot, BN columns), BM pairs at a time.
template <int BM, int BN, int KG>
__device__ __forceinline__ void down_ctas(const RelDown& p, uint8_t* smem, int tid, int first, int stride) {
    using C = ExCfg<BM, BN, 1, KG>;
    const int groups = p.k >> 5, kb = p.k >> 1, warp = tid >> 5;
    const int wrow = (warp / C::WN) * C::RM * 16, wcol = (warp % C::WN) * 32;
    const Lane l = lane_of(tid & 31);
    slot_ctas<C>(p.offs, p.active, p.nactive, p.n, first, stride, [&](int slot, int at, int np, int col0) {
        const long long w0 = (long long)slot * p.n + col0;
        const uint8_t* const W[1] = {p.w2 + w0 * kb};
        const uint8_t* const WS[1] = {p.s2 + w0 * groups};
        float acc[1][C::RM][4][4] = {};
        ex_cta<C>(acc, [&](int i, const uint8_t*& c, const uint8_t*& s) { down_row(p, at, i, c, s); }, np, W, WS, kb,
                  groups, min(BN, p.n - col0), smem, tid, wrow, wcol);
        if (col0 + wcol < p.n) down_store<C::RM>(p, acc, at, np, wrow, col0 + wcol, l);
    });
}

}  // namespace rel
