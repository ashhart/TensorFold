// DeepSeek V4.1 release projections: CTA building blocks (cp.async stages, ldmatrix fragments, the K-slice pipeline).
#pragma once

#include "release_mma.cuh"

// a: FP8 codes (or BF16 x) at byte offset a_off, row stride a_ld elements; sa: a-scales likewise; w, ws as stored.
struct RelProj {
    const uint8_t* a;
    long long a_off;
    long long a_ld;
    const uint8_t* sa;
    long long sa_off;
    long long sa_ld;
    const uint8_t* w;
    const uint8_t* ws;
    void* y;
    long long y_ld;
    long long y_col;
    float* part;
    int m;
    int n;
    int k;
    int s;
    int f32;
    int pad;
};

namespace rel {

// A stage row holds KB bytes (KB / 32 K steps) in 16-byte chunks, swizzled so 8 rows of one chunk hit 8 bank groups.
template <int KB>
__device__ __forceinline__ uint32_t chunk_at(int r, int c) {
    static_assert(KB == 64 || KB == 128, "stage rows of 64 or 128 bytes");
    const int x = KB == 128 ? (r & 7) : ((r >> 1) & 3);
    return (uint32_t)(r * KB + ((c ^ x) << 4));
}

__device__ __forceinline__ void cp16(uint32_t dst, const void* src, bool full) {
    tf_common::cp16z(dst, src, full);
}

__device__ __forceinline__ void cp_commit() { tf_common::cp_commit(); }

template <int N>
__device__ __forceinline__ void cp_wait() {
    tf_common::cp_wait<N>();
}

__device__ __forceinline__ void ldsm4(uint32_t addr, uint32_t (&r)[4]) {
    qmm_frag::ldmatrix4(r, addr);
}

// `bytes` of each of `rows` global rows (stride ld) into a stage, thread tid of nt; rows from `valid` on are zero.
template <int KB>
__device__ __forceinline__ void stage_rows(uint32_t dst, const uint8_t* src, long long ld, int rows, int valid,
                                           int bytes, int tid, int nt) {
    constexpr int CH = KB / 16;
    const int chunks = bytes >> 4;
    for (int i = tid; i < rows * CH; i += nt) {
        const int r = i / CH, c = i % CH;
        if (c < chunks) cp16(dst + chunk_at<KB>(r, c), r < valid ? src + r * ld + c * 16 : src, r < valid);
    }
}

// Rows row0..row0+15 of step st (x4: rows 0-7 and 8-15 of bytes 0-15, then of bytes 16-31): the A fragment.
template <int KB>
__device__ __forceinline__ void frag_rows_shared(uint32_t base, int row0, int st, int lane, uint32_t (&f)[4]) {
    const int m = lane >> 3, r = row0 + (lane & 7) + ((m & 1) << 3);
    ldsm4(base + chunk_at<KB>(r, 2 * st + (m >> 1)), f);
}

// Columns col0..col0+15 of step st as two n8 tiles' B fragments.
template <int KB>
__device__ __forceinline__ void frag_cols_shared(uint32_t base, int col0, int st, int lane, uint32_t (&b0)[2],
                                                 uint32_t (&b1)[2]) {
    const int m = lane >> 3, r = col0 + (lane & 7) + ((m >> 1) << 3);
    uint32_t f[4];
    ldsm4(base + chunk_at<KB>(r, 2 * st + (m & 1)), f);
    b0[0] = f[0];
    b0[1] = f[1];
    b1[0] = f[2];
    b1[1] = f[3];
}

// A CTA tile of BM x BN outputs, WM x WN warps, ST stages of KB bytes a row; a warp holds RM x NW m16n8 tiles.
template <int BM_, int BN_, int WM_, int WN_, int ST_, int KB_, bool FP8_>
struct Cfg {
    static constexpr int BM = BM_, BN = BN_, WM = WM_, WN = WN_, ST = ST_, KB = KB_;
    static constexpr bool FP8 = FP8_;
    static constexpr int NT = 32 * WM * WN, RM = BM / WM / 16, NW = BN / WN / 8, SPS = KB / 32;
    static constexpr int NS = BM * SPS + BN / 32 * SPS;
    // stages start 128-byte aligned (an unaligned stage cost FP8 decode a fifth of its bandwidth on GB10)
    static constexpr int SB = FP8 ? ((NS + 127) & ~127) : 0;
    static constexpr int STAGE = (BM + BN) * KB + SB;
    static constexpr int SMEM = ST * STAGE;
    static constexpr int SPT = FP8 ? (NS + NT - 1) / NT : 1;
    static_assert(NW % 4 == 0 && RM >= 1 && ST >= 2 && STAGE % 128 == 0, "warp tiles span whole scale blocks");
};

// A stage's scale bytes (rows [BM][SPS], then weight blocks [BN/32][SPS]) for groups g.. (ng of them), in registers.
template <class C>
__device__ __forceinline__ void scales_fetch(uint32_t (&v)[C::SPT], const uint8_t* sa, long long sa_ld, int valid,
                                             const uint8_t* ws, int groups, int cvalid, int g, int ng, int tid) {
#pragma unroll
    for (int j = 0; j < C::SPT; ++j) {
        const int i = tid + j * C::NT, s = i % C::SPS;
        if (i < C::BM * C::SPS) {
            const int r = i / C::SPS;
            v[j] = (s < ng && r < valid) ? sa[r * sa_ld + g + s] : 127u;
        } else {
            const int b = (i - C::BM * C::SPS) / C::SPS;
            v[j] = (s < ng && b * 32 < cvalid) ? ws[(long long)b * groups + g + s] : 127u;
        }
    }
}

template <class C>
__device__ __forceinline__ void scales_store(uint8_t* dst, const uint32_t (&v)[C::SPT], int tid) {
#pragma unroll
    for (int j = 0; j < C::SPT; ++j) {
        const int i = tid + j * C::NT;
        if (i < C::NS) dst[i] = (uint8_t)v[j];
    }
}

// One CTA tile (rows row0.., columns col0.., K slice `slice`) through ST cp.async stages; smem holds C::SMEM bytes.
template <class C>
__device__ __forceinline__ void proj_tile(const RelProj& p, int row0, int col0, int slice, uint8_t* smem, int tid) {
    constexpr int es = C::FP8 ? 1 : 2, KB = C::KB;
    const int groups = p.k >> 5;
    const int2 gr = slice_groups(groups, slice, p.s);
    const long long lda = p.a_ld * es, ldw = (long long)p.k * es;
    const int total = (gr.y - gr.x) * 32 * es, stages = (total + KB - 1) / KB;
    const int valid = min(C::BM, p.m - row0), cvalid = min(C::BN, p.n - col0);
    const uint8_t* A = p.a + p.a_off + row0 * lda + gr.x * 32 * es;
    const uint8_t* W = p.w + col0 * ldw + gr.x * 32 * es;
    const uint8_t* SA = C::FP8 ? p.sa + p.sa_off + row0 * p.sa_ld : nullptr;
    const uint8_t* WS = C::FP8 ? p.ws + (long long)(col0 >> 5) * groups : nullptr;
    const uint32_t base = (uint32_t)__cvta_generic_to_shared(smem);
    const int lane = tid & 31, warp = tid >> 5;
    const Lane l = lane_of(lane);
    const int wrow = (warp / C::WN) * C::RM * 16, wcol = (warp % C::WN) * C::NW * 8;
    float acc[C::RM][C::NW][4] = {};
    // scale bytes go through registers: fetched with a stage's codes, stored once this stage's mmas are issued
    uint32_t sv[C::SPT];
    auto load = [&](int st) {
        const int slot = st % C::ST, bytes = min(KB, total - st * KB);
        stage_rows<KB>(base + slot * C::STAGE, A + st * KB, lda, C::BM, valid, bytes, tid, C::NT);
        stage_rows<KB>(base + slot * C::STAGE + C::BM * KB, W + st * KB, ldw, C::BN, cvalid, bytes, tid, C::NT);
        if constexpr (C::FP8) {
            scales_fetch<C>(sv, SA, p.sa_ld, valid, WS, groups, cvalid, gr.x + C::SPS * st, bytes >> 5, tid);
        }
    };
    auto scales = [&](int st) { return smem + (st % C::ST) * C::STAGE + (C::BM + C::BN) * KB; };
    for (int st = 0; st < C::ST - 1; ++st) {
        if (st < stages) {
            load(st);
            if constexpr (C::FP8) scales_store<C>(scales(st), sv, tid);
        }
        cp_commit();
    }
    for (int st = 0; st < stages; ++st) {
        cp_wait<C::ST - 2>();
        __syncthreads();
        const int next = st + C::ST - 1;
        if (next < stages) load(next);
        cp_commit();
        const int steps = min(C::SPS, (total - st * KB) >> 5);
        const uint32_t sA = base + (st % C::ST) * C::STAGE, sW = sA + C::BM * KB;
        const uint8_t* S = scales(st);
#pragma unroll
        for (int k = 0; k < C::SPS; ++k) {
            if (k >= steps) break;
            uint32_t a[C::RM][4], b[C::NW][2], sa[C::RM], sb[C::NW / 4];
#pragma unroll
            for (int i = 0; i < C::RM; ++i) frag_rows_shared<KB>(sA, wrow + 16 * i, k, lane, a[i]);
#pragma unroll
            for (int j = 0; j < C::NW / 2; ++j) frag_cols_shared<KB>(sW, wcol + 16 * j, k, lane, b[2 * j], b[2 * j + 1]);
            if constexpr (C::FP8) {
#pragma unroll
                for (int i = 0; i < C::RM; ++i) sa[i] = S[(wrow + 16 * i + l.g + ((l.t & 1) << 3)) * C::SPS + k];
#pragma unroll
                for (int q = 0; q < C::NW / 4; ++q) sb[q] = S[C::BM * C::SPS + ((wcol >> 5) + q) * C::SPS + k];
            }
            mma_step<C::RM, C::NW, C::FP8>(acc, a, b, sa, sb);
        }
        if constexpr (C::FP8) {
            if (next < stages) scales_store<C>(scales(next), sv, tid);
        }
    }
    cp_wait<0>();
    __syncthreads();
    store_acc<C::RM, C::NW>(acc, p.y, p.y_ld, p.y_col, p.part, p.m, p.n, p.s, slice, p.f32 != 0, row0 + wrow,
                            col0 + wcol, l);
}

// Tiles first, first + stride, ... (row tiles fastest, then columns, then slices): a CTA a tile or a persistent loop.
template <class C>
__device__ __forceinline__ void proj_tiles(const RelProj& p, uint8_t* smem, int tid, int first, int stride) {
    const int rows = (p.m + C::BM - 1) / C::BM, cols = (p.n + C::BN - 1) / C::BN, tiles = rows * cols * p.s;
    for (int t = first; t < tiles; t += stride) {
        const int rest = t / rows;
        proj_tile<C>(p, (t % rows) * C::BM, (rest % cols) * C::BN, rest / cols, smem, tid);
    }
}

// A 16 x 32 warp tile over groups [g0, g1) from global memory (A, SA: first row; W: first column; WS: its scale row).
template <bool FP8>
__device__ __forceinline__ void kslice_global(float (&acc)[1][4][4], const uint8_t* A, long long lda, int valid,
                                              const uint8_t* W, long long ldw, const uint8_t* SA, long long sa_ld,
                                              const uint8_t* WS, int g0, int g1, Lane l) {
    constexpr int es = FP8 ? 1 : 2;
    for (int kg = g0; kg < g1; ++kg)
#pragma unroll
        for (int h = 0; h < es; ++h) {
            const int kb = (kg * es + h) * 32;
            uint32_t a[1][4], b[4][2], sa[1] = {0}, sb[1] = {0};
            frag_rows_global(A, lda, valid, kb, l, a[0]);
#pragma unroll
            for (int j = 0; j < 4; ++j) frag_cols_global(W + 8 * j * ldw, ldw, kb, l, b[j]);
            if constexpr (FP8) {
                sa[0] = scale_rows_global(SA, sa_ld, valid, kg, l);
                sb[0] = WS[kg];
            }
            mma_step<1, 4, FP8>(acc, a, b, sa, sb);
        }
}

// The same tiles at 16 x 32 a warp straight from global memory (no stages): warps first, first + stride, ...
template <bool FP8>
__device__ __forceinline__ void warp_tiles(const RelProj& p, int first, int stride, int lane) {
    constexpr int es = FP8 ? 1 : 2;
    const Lane l = lane_of(lane);
    const int groups = p.k >> 5, rows = (p.m + 15) / 16, cols = p.n / 32, tiles = rows * cols * p.s;
    const long long lda = p.a_ld * es, ldw = (long long)p.k * es;
    for (int t = first; t < tiles; t += stride) {
        const int row0 = (t % rows) * 16, rest = t / rows, col0 = (rest % cols) * 32, slice = rest / cols;
        const int2 gr = slice_groups(groups, slice, p.s);
        float acc[1][4][4] = {};
        kslice_global<FP8>(acc, p.a + p.a_off + row0 * lda, lda, min(16, p.m - row0), p.w + col0 * ldw, ldw,
                           FP8 ? p.sa + p.sa_off + row0 * p.sa_ld : nullptr, p.sa_ld,
                           FP8 ? p.ws + (long long)(col0 >> 5) * groups : nullptr, gr.x, gr.y, l);
        store_acc<1, 4>(acc, p.y, p.y_ld, p.y_col, p.part, p.m, p.n, p.s, slice, p.f32 != 0, row0, col0, l);
    }
}

}  // namespace rel
