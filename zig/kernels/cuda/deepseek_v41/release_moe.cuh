// DeepSeek V4.1 release experts: building blocks for grouping, MXFP4 expert tiles, their epilogues and the combine.
#pragma once

#include "release_mma.cuh"

// The router's (row, k) pairs bucketed by the rank's expert slot in ascending (slot, row, k) order.
struct RelGroup {
    const int* ids;      // [m, topk] expert ids
    const int* slot_of;  // [nids] the rank's slot of an id, -1 when the rank does not hold it
    int* offs;           // [nslots + 1] each slot's first grouped pair
    int* pair_tok;       // [m * topk] grouped pair -> row * topk + k
    int* pair_of;        // [m * topk] row * topk + k -> grouped pair, -1 when unheld
    int* active;         // [nslots] slots holding a pair, ascending
    int* nactive;        // [1]
    int m;
    int topk;
    int nslots;
    int nids;
};

// Gate and up of every grouped pair: e2m1 w1, w3 [slot][n][k/2] with [slot][n][k/32] scales on the pair's row codes.
struct RelGateUp {
    const uint8_t* x;  // e4m3 codes of the rows (view), scales xs likewise
    long long x_off;
    long long x_ld;
    const uint8_t* xs;
    long long xs_off;
    long long xs_ld;
    const uint8_t* w1;
    const uint8_t* s1;
    const uint8_t* w3;
    const uint8_t* s3;
    const float* wts;  // [m * topk] route weights
    const int* pair_tok;
    const int* offs;
    const int* active;
    const int* nactive;
    uint8_t* codes;  // [pairs, n] the round trip of v, scales [pairs, n/32]
    uint8_t* scales;
    float* gsum;  // optional [pairs, n] FP32 sums (the contract's)
    float* usum;
    __nv_bfloat16* v;  // optional [pairs, n] bf16 v before its round trip
    int n;
    int k;
    int topk;
    int pad;
};

// w2 [slot][n][k/2] with [slot][n][k/32] scales on the grouped pairs' codes [pairs, k]: bf16 partials [pairs, n].
struct RelDown {
    const uint8_t* x;
    const uint8_t* xs;
    const uint8_t* w2;
    const uint8_t* s2;
    const int* offs;
    const int* active;
    const int* nactive;
    __nv_bfloat16* part;
    int n;
    int k;
};

namespace rel {

// The round trip of one row's 32 values held by a lane quad (8 a lane): the same steps as actq32, all lanes calling.
__device__ __forceinline__ void actq_quad(const float (&v)[8], uint8_t (&code)[8], uint32_t& scale) {
    int bad = 0;
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        bad |= isfinite(v[i]) ? 0 : 1;
        amax = fmaxf(amax, fabsf(v[i]));
    }
#pragma unroll
    for (int o = 1; o <= 2; o <<= 1) {
        bad |= __shfl_xor_sync(0xffffffffu, bad, o);
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    }
    if (bad) {
#pragma unroll
        for (int i = 0; i < 8; ++i) code[i] = 0x7f;
        scale = 0xff;
        return;
    }
    const int e = fp8_exponent(amax);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint32_t c = e4m3_code(fminf(fabsf(v[i]) * pow2(-e), 448.0f));
        code[i] = (uint8_t)(c | (signbit(v[i]) ? 0x80u : 0u));
    }
    scale = (uint32_t)(e + 127);
}

__device__ __forceinline__ uint2 ld64(const uint8_t* p) { return *reinterpret_cast<const uint2*>(p); }

// A lane's rows of a 16-row tile (g and g + 8): codes and scale rows, null for a padded row.
struct ExRows {
    const uint8_t *lo, *hi, *slo, *shi;
};

// A weight matrix at an expert's 32-column block: rows of k/2 packed bytes and k/32 scales.
struct ExW {
    const uint8_t *w, *s;
};

// Groups [g0, g1) of RM row tiles x 32 columns of NB matrices, ascending; lane t takes k 8t..8t+7 of each group.
template <int RM, int NB>
__device__ __forceinline__ void ex_kloop(float (&acc)[NB][RM][4][4], const ExRows (&r)[RM], const ExW (&w)[NB],
                                         int kb, int groups, int g0, int g1, Lane l) {
    for (int kg = g0; kg < g1; ++kg) {
        uint32_t a[RM][4], sa[RM];
#pragma unroll
        for (int i = 0; i < RM; ++i) {
            const uint2 z = make_uint2(0u, 0u);
            const uint2 lo = r[i].lo ? ld64(r[i].lo + kg * 32 + 8 * l.t) : z;
            const uint2 hi = r[i].hi ? ld64(r[i].hi + kg * 32 + 8 * l.t) : z;
            a[i][0] = lo.x;
            a[i][1] = hi.x;
            a[i][2] = lo.y;
            a[i][3] = hi.y;
            const uint8_t* s = (l.t & 1) ? r[i].shi : r[i].slo;
            sa[i] = s ? s[kg] : 127u;
        }
#pragma unroll
        for (int b = 0; b < NB; ++b) {
            uint32_t bb[4][2], sb[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const long long col = l.g + 8 * j;
                expand_e2m1(ld32(w[b].w + col * kb + kg * 16 + 4 * l.t), bb[j]);
                sb[j] = w[b].s[col * groups + kg];
            }
#pragma unroll
            for (int i = 0; i < RM; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) mma_e2m1(acc[b][i][j], a[i], bb[j], sa[i], sb[j]);
        }
    }
}

// Rows pair0.. (n_rows of them) of a tile read through `rows` (row index -> code row), RM row tiles.
template <int RM, class RowOf>
__device__ __forceinline__ void ex_rows(ExRows (&r)[RM], int n_rows, Lane l, RowOf row_of) {
#pragma unroll
    for (int i = 0; i < RM; ++i) {
        const int lo = 16 * i + l.g, hi = lo + 8;
        r[i].lo = r[i].slo = r[i].hi = r[i].shi = nullptr;
        if (lo < n_rows) row_of(lo, r[i].lo, r[i].slo);
        if (hi < n_rows) row_of(hi, r[i].hi, r[i].shi);
    }
}

// Gate/up's epilogue (RM row tiles from row0, 32 columns at col0): SwiGLU with the route weight, then the round trip.
template <int RM>
__device__ __forceinline__ void gateup_store(const RelGateUp& p, const float (&acc)[2][RM][4][4], int first, int npairs,
                                             int row0, int col0, Lane l) {
#pragma unroll
    for (int i = 0; i < RM; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = row0 + 16 * i + l.g + 8 * h;
            const bool valid = row < npairs;
            const float route = valid ? p.wts[p.pair_tok[first + row]] : 0.0f;
            float v[8];
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int c = 0; c < 2; ++c) v[2 * j + c] = swiglu(acc[0][i][j][2 * h + c], acc[1][i][j][2 * h + c], route, true);
            uint8_t code[8];
            uint32_t scale;
            actq_quad(v, code, scale);
            if (!valid) continue;
            const long long at = (long long)(first + row) * p.n + col0;
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int col = 8 * j + 2 * l.t + c;
                    p.codes[at + col] = code[2 * j + c];
                    if (p.gsum) p.gsum[at + col] = acc[0][i][j][2 * h + c];
                    if (p.usum) p.usum[at + col] = acc[1][i][j][2 * h + c];
                    if (p.v) p.v[at + col] = __float2bfloat16_rn(v[2 * j + c]);
                }
            if (l.t == 0) p.scales[(long long)(first + row) * (p.n >> 5) + (col0 >> 5)] = (uint8_t)scale;
        }
}

// Down's epilogue: bf16 partials of RM row tiles from row `row0` of the pair tile at columns col0..col0+31.
template <int RM>
__device__ __forceinline__ void down_store(const RelDown& p, const float (&acc)[1][RM][4][4], int first, int npairs,
                                           int row0, int col0, Lane l) {
#pragma unroll
    for (int i = 0; i < RM; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = row0 + 16 * i + l.g + 8 * h;
            if (row >= npairs) continue;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                __nv_bfloat16* o = p.part + (long long)(first + row) * p.n + col0 + 8 * j + 2 * l.t;
                o[0] = __float2bfloat16_rn(acc[0][i][j][2 * h]);
                o[1] = __float2bfloat16_rn(acc[0][i][j][2 * h + 1]);
            }
        }
}

// The code and scale rows of gate/up's pair i (its token's) and of down's (the pair's own).
__device__ __forceinline__ void gateup_row(const RelGateUp& p, int first, int i, const uint8_t*& c, const uint8_t*& s) {
    const long long row = p.pair_tok[first + i] / p.topk;
    c = p.x + p.x_off + row * p.x_ld;
    s = p.xs + p.xs_off + row * p.xs_ld;
}

__device__ __forceinline__ void down_row(const RelDown& p, int first, int i, const uint8_t*& c, const uint8_t*& s) {
    c = p.x + (long long)(first + i) * p.k;
    s = p.xs + (long long)(first + i) * (p.k >> 5);
}

// The routed gate/up of 16 grouped pairs (first, n of them) of one slot at columns col0..col0+31, from global memory.
__device__ __forceinline__ void gateup_tile(const RelGateUp& p, int slot, int first, int npairs, int col0, Lane l) {
    const int groups = p.k >> 5, kb = p.k >> 1;
    ExRows r[1];
    ex_rows<1>(r, npairs, l, [&](int i, const uint8_t*& c, const uint8_t*& s) { gateup_row(p, first, i, c, s); });
    const long long w0 = (long long)slot * p.n + col0;
    const ExW w[2] = {{p.w1 + w0 * kb, p.s1 + w0 * groups}, {p.w3 + w0 * kb, p.s3 + w0 * groups}};
    float acc[2][1][4][4] = {};
    ex_kloop<1, 2>(acc, r, w, kb, groups, 0, groups, l);
    gateup_store<1>(p, acc, first, npairs, 0, col0, l);
}

// The down of 16 grouped pairs of one slot at columns col0..col0+31, from global memory.
__device__ __forceinline__ void down_tile(const RelDown& p, int slot, int first, int npairs, int col0, Lane l) {
    const int groups = p.k >> 5, kb = p.k >> 1;
    ExRows r[1];
    ex_rows<1>(r, npairs, l, [&](int i, const uint8_t*& c, const uint8_t*& s) { down_row(p, first, i, c, s); });
    const long long w0 = (long long)slot * p.n + col0;
    const ExW w[1] = {{p.w2 + w0 * kb, p.s2 + w0 * groups}};
    float acc[1][1][4][4] = {};
    ex_kloop<1, 1>(acc, r, w, kb, groups, 0, groups, l);
    down_store<1>(p, acc, first, npairs, 0, col0, l);
}

// Warp tiles first, first + stride, ... over (active slot, 32-column block), each looping its slot's 16-pair tiles.
template <class F>
__device__ __forceinline__ void slot_tiles(const int* offs, const int* active, const int* nactive, int n, int first,
                                           int stride, F tile) {
    const int blocks = n >> 5, tiles = *nactive * blocks;
    for (int t = first; t < tiles; t += stride) {
        const int slot = active[t / blocks], col0 = (t % blocks) * 32;
        const int lo = offs[slot], hi = offs[slot + 1];
        for (int at = lo; at < hi; at += 16) tile(slot, at, min(16, hi - at), col0);
    }
}

// Row `row`'s output at column `col`: its top-k grouped partials in k order into one FP32 sum, then the shared one.
__device__ __forceinline__ float combine_row(const __nv_bfloat16* part, const int* pair_of, const __nv_bfloat16* shared,
                                             long long shared_ld, int n, int topk, long long row, int col) {
    float acc = 0.0f;
    for (int k = 0; k < topk; ++k) {
        const int at = pair_of[row * topk + k];
        if (at >= 0) acc = acc + __bfloat162float(part[(long long)at * n + col]);
    }
    if (shared) acc = acc + __bfloat162float(shared[row * shared_ld + col]);
    return acc;
}

}  // namespace rel
