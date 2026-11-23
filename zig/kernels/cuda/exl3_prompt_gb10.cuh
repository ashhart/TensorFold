// Copyright 2026 Mia's AI Lab (MiaAI-Lab), Apache-2.0. SM121 port of patch 0078; see THIRD_PARTY_NOTICES.md.
#pragma once
__device__ __forceinline__ uint32_t lds32(uint32_t a) {
    uint32_t v;
    asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(v) : "r"(a));
    return v;
}
__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm2(uint32_t (&r)[2], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
__device__ __forceinline__ void cp16s(uint32_t dst, const void* src, int n) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(n));
}
// Fold the MCG mask and XOR into one LOP3 per state.
__device__ __forceinline__ uint32_t mcg2(uint32_t s0, uint32_t s1) {
    const uint32_t m = 0x8FFF8FFFu;
    uint32_t x0, x1;
    asm("lop3.b32 %0, %1, %2, 0x3B603B60, 0x6a;\n" : "=r"(x0) : "r"(s0 * 0xCBAC1FEDu), "r"(m));
    asm("lop3.b32 %0, %1, %2, 0x3B603B60, 0x6a;\n" : "=r"(x1) : "r"(s1 * 0xCBAC1FEDu), "r"(m));
    const uint32_t lo = __byte_perm(x0, x1, 0x5410), hi = __byte_perm(x0, x1, 0x7632);
    const half2 r = __hadd2(*reinterpret_cast<const half2*>(&lo), *reinterpret_cast<const half2*>(&hi));
    return *reinterpret_cast<const uint32_t*>(&r);
}
// Decode the lane and its predecessor into the same four A registers as decode_lane<8>.
__device__ __forceinline__ void decode_a8(uint32_t w, uint32_t p, uint32_t (&a)[4]) {
    const uint32_t s = __funnelshift_r(w, p, 20), t = w >> 4;
    a[0] = mcg2(__byte_perm(s, 0u, 0x4421), (s >> 4) & 0xffffu);
    a[1] = mcg2(__byte_perm(t, 0u, 0x4421), __byte_perm(w, 0u, 0x4421));
    a[2] = mcg2(s & 0xffffu, w >> 16);
    a[3] = mcg2(__byte_perm(t, 0u, 0x4410), w & 0xffffu);
}
// A thread fixes the bounds and addresses for its member rows and trellis tiles once per expert item.
template <int MATS, int XR, int K2, int NBA>
struct TLoads {
    static constexpr int TW = tile_words<K2>(), PIECES = TW / 4;
    static constexpr int XPIECES = XR * (TBK / 8), XPT = (XPIECES + TT - 1) / TT;
    static constexpr int WPIECES = 16 * TKT * PIECES, WPT = (WPIECES + TT - 1) / TT;
    const half* xs[XPT];
    uint32_t xd[XPT];
    int xn[XPT];
    bool xon[XPT];
    const uint32_t* ws[WPT];
    uint32_t wd[WPT];
    bool won[WPT];
    size_t wstep;
    __device__ __forceinline__ void init(const half* X0, const half* X1, const uint32_t* T0, const uint32_t* T1, const int* rows_sh,
            int K, int NTILES, int cb, int xdiv, uint32_t sbase) {
#pragma unroll
        for (int j = 0; j < XPT; ++j) {
            const int idx = (int)threadIdx.x + j * TT;
            const int m = idx / (TPM * (TBK / 8)), rem = idx % (TPM * (TBK / 8)), i = rem / (TBK / 8), part = rem % (TBK / 8);
            const bool in = idx < XPIECES;
            const int r = in ? rows_sh[i] : -1;
            xs[j] = (m ? X1 : X0) + (size_t)(r < 0 ? 0 : (r / xdiv)) * K + part * 8;
            xn[j] = r >= 0 ? 16 : 0;
            xon[j] = in && i < NBA * 8;
            xd[j] = sbase + (uint32_t)(((m * TPM + i) * TXS + part * 8) * 2);
        }
#pragma unroll
        for (int j = 0; j < WPT; ++j) {
            const int idx = (int)threadIdx.x + j * TT;
            won[j] = idx < WPIECES;
            const int q = won[j] ? idx / (TKT * PIECES) : 0, kk = (idx / PIECES) % TKT, part = idx % PIECES;
            int wm, nt;
            ttile<MATS>(q, cb, wm, nt);
            ws[j] = (wm ? T1 : T0) + ((size_t)kk * NTILES + nt) * TW + part * 4;
            wd[j] = sbase + (uint32_t)(XR * TXS * 2 + ((q * TKT + kk) * TW + part * 4) * 4);
        }
        wstep = (size_t)TKT * NTILES * TW;
    }
    __device__ __forceinline__ void issue(int c, uint32_t off) const {
#pragma unroll
        for (int j = 0; j < XPT; ++j)
            if (xon[j]) cp16s(xd[j] + off, xs[j] + (size_t)c * TBK, xn[j]);
#pragma unroll
        for (int j = 0; j < WPT; ++j)
            if (won[j]) cp16s(wd[j] + off, ws[j] + (size_t)c * wstep, 16);
    }
};

// Decode the next four-bit K tile while the current tile issues its MMAs, in ascending K order.
template <int MATS, int XR, int K2, int NBA, int PS>
__device__ __forceinline__ void pipelined_main(const half* __restrict__ X0, const half* __restrict__ X1, const uint64_t* __restrict__ tp0,
        const uint64_t* __restrict__ tp1, const int* rows_sh, int e, int K, int NTILES, int cb, int xdiv, unsigned char* smem,
        float (&acc)[TMT][NBA][4]) {
    constexpr int TW = tile_words<K2>(), LW = lane_words<K2>();
    constexpr int SB = tstage_halves<XR, K2>() * 2, TB = TW * 4;   // a stage's bytes, a tile's bytes
    static_assert(PS >= 3 && PS * SB <= tsmem_bytes<K2>(), "a stage of lookahead in the kernel's shared buffer");
    static_assert(TKT % 2 == 0, "two A fragment sets: an even number of k tiles a stage");
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int mat, nt_unused;
    ttile<MATS>(warp * TMT, cb, mat, nt_unused);
    const int chunks = K / TBK;
    const uint32_t* T0 = reinterpret_cast<const uint32_t*>(tp0[e]);
    const uint32_t* T1 = reinterpret_cast<const uint32_t*>(tp1[e]);
    const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem);
#pragma unroll
    for (int i = 0; i < TMT; ++i)
#pragma unroll
        for (int j = 0; j < NBA; ++j)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;
    TLoads<MATS, XR, K2, NBA> ld;
    ld.init(X0, X1, T0, T1, rows_sh, K, NTILES, cb, xdiv, sbase);
#pragma unroll
    for (int s = 0; s < PS - 1; ++s) {
        if (s < chunks) ld.issue(s, (uint32_t)(s * SB));
        cp_commit();
    }
    // ldmatrix rows (members) and columns (k) of this lane; this warp's first tile; the lane's words in a tile
    const uint32_t xlane = (uint32_t)((((XR == TPM ? 0 : mat * TPM) + (lane & 7) + ((lane >> 4) & 1) * 8) * TXS + ((lane >> 3) & 1) * 8) * 2);
    const uint32_t wwarp = (uint32_t)(XR * TXS * 2 + warp * TMT * TKT * TB);
    uint32_t wo[LW];
    {
        int word, offset;
        lane_start<K2>(lane, word, offset);
#pragma unroll
        for (int i = 0; i < LW; ++i) wo[i] = (uint32_t)(((word + i) % TW) * 4);
    }
    auto decode = [&](uint32_t tile, uint32_t (&a)[4]) {
        uint32_t w[LW];
#pragma unroll
        for (int i = 0; i < LW; ++i) w[i] = lds32(tile + wo[i]);
        static_assert(K2 == 8);
        decode_a8(w[1], w[0], a);
    };
    cp_wait<PS - 2>();
    __syncthreads();   // stage 0
    uint32_t A[2][TMT][4];
#pragma unroll
    for (int i = 0; i < TMT; ++i) decode(sbase + wwarp + (uint32_t)(i * TKT * TB), A[0][i]);
    uint32_t sc = sbase, sn = sbase + SB;   // this stage's and the next stage's buffers
    uint32_t fill = (uint32_t)((PS - 1) * SB);   // the buffer the next load goes to (an offset)
    for (int c = 0; c < chunks; ++c) {
        cp_wait<PS - 3>();   // stage c + 1 landed: its first k tile decodes in this stage
        __syncthreads();     // and every warp is past stage c - 1, whose buffer the load below takes
        if (c + PS - 1 < chunks) ld.issue(c + PS - 1, fill);
        cp_commit();
        fill = fill + SB == (uint32_t)(PS * SB) ? 0u : fill + SB;
#pragma unroll
        for (int kk = 0; kk < TKT; ++kk) {
            const int cur = kk & 1, nxt = cur ^ 1;
            const uint32_t src = kk + 1 < TKT ? sc + (uint32_t)((kk + 1) * TB) : sn;   // the next k tile's words
#pragma unroll
            for (int i = 0; i < TMT; ++i) decode(src + wwarp + (uint32_t)(i * TKT * TB), A[nxt][i]);
            const uint32_t xk = sc + xlane + (uint32_t)(kk * 32);
#pragma unroll
            for (int jp = 0; jp < (NBA + 1) / 2; ++jp) {
                const uint32_t xa = xk + (uint32_t)(jp * 16 * TXS * 2);
                if (2 * jp + 1 < NBA) {
                    uint32_t r[4];
                    ldsm4(r, xa);
                    const uint32_t bl[2] = {r[0], r[1]}, bh[2] = {r[2], r[3]};
#pragma unroll
                    for (int i = 0; i < TMT; ++i) {
                        mma16816(acc[i][2 * jp], A[cur][i], bl);
                        mma16816(acc[i][2 * jp + 1], A[cur][i], bh);
                    }
                } else {
                    uint32_t r[2];
                    ldsm2(r, xa);
#pragma unroll
                    for (int i = 0; i < TMT; ++i) mma16816(acc[i][2 * jp], A[cur][i], r);
                }
            }
        }
        sc = sn;
        sn = sn + SB == sbase + (uint32_t)(PS * SB) ? sbase : sn + SB;
    }
    cp_wait<0>();
    __syncthreads();   // the stages become the epilogue's sums
}
