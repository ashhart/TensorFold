// x3gm v3: gm2_kernel's items, chains and epilogue (x3gm.cu) with claim-ahead items, LEAN decode and MTL 1 warps.

#define DSV41_X3GM_NO_INSTANCES
#include "x3gm.cu"

namespace dsv41_x3gm {  // each element: one mma chain over ascending k from +0.0, gm_kernel's epilogue: v2's bits

// The 16-bit state at bit `OFF` of a 32-bit run.
template <int OFF>
__device__ __forceinline__ uint32_t window(uint32_t mm) {
    if constexpr (OFF == 0) return mm & 0xffffu;
    else if constexpr (OFF == 8) return __byte_perm(mm, 0u, 0x4421);
    else if constexpr (OFF == 16) return mm >> 16;
    else return (mm >> OFF) & 0xffffu;
}

template <int K2, int G, int J>
__device__ __forceinline__ void windows32(uint32_t mm, uint32_t (&st)[8]) {
    if constexpr (J < Fmt<K2>::GV) {
        st[G * Fmt<K2>::GV + J] = window<Fmt<K2>::off(J)>(mm);
        windows32<K2, G, J + 1>(mm, st);
    }
}

// decode_a's A fragment (same states), read from 32 bits when a run's highest window ends within them
template <int K2>
__device__ __forceinline__ void decode_a3(const uint32_t* tile, const LaneMap<K2>& m, uint32_t (&a)[4]) {
    constexpr int GV = Fmt<K2>::GV, NG = Fmt<K2>::NG, SPAN = Fmt<K2>::off(0) + 16;
    uint32_t st[8];
#pragma unroll
    for (int g = 0; g < NG; ++g) {
        const uint32_t whi = tile[m.hi[g]], wlo = tile[m.lo[g]];
        if constexpr (SPAN <= 32) {
            const uint32_t mm = __funnelshift_r(whi, wlo, m.sh[g]);   // low 32 bits of (wlo:whi) >> sh, sh 0..31
            if (g == 0) windows32<K2, 0, 0>(mm, st);
            else if (g == 1) windows32<K2, 1, 0>(mm, st);
        } else {
            const uint64_t mm = ((((uint64_t)wlo) << 32) | whi) >> m.sh[g];
#pragma unroll
            for (int j = 0; j < GV; ++j) st[g * GV + j] = (uint32_t)(mm >> Fmt<K2>::off(j)) & 0xffffu;
        }
    }
    a[0] = cb_pair<MUL1>(st[0], st[1]);
    a[1] = cb_pair<MUL1>(st[4], st[5]);
    a[2] = cb_pair<MUL1>(st[2], st[3]);
    a[3] = cb_pair<MUL1>(st[6], st[7]);
}

template <int K2, int LEAN>
__device__ __forceinline__ void decode3(const uint32_t* tile, const LaneMap<K2>& m, uint32_t (&a)[4]) {
    if constexpr (LEAN) decode_a3<K2>(tile, m, a);
    else decode_a<K2>(tile, m, a);
}

// kloop2 with LEAN's decode and `side(s)` after the barrier that opens stage s (stages 0 .. S - 1)
template <int N8, int MATS, int XM, int K2, int MTL, int NG, int KS, int NSA, int NGR, int LEAN, typename LoadFn,
          typename SideFn>
__device__ __forceinline__ void kloop3(float (&acc)[MTL][NG][4], const unsigned char* smem, int S, int xmat, int mat,
                                       int slice, int lr, int lc, const LaneMap<K2>& map, LoadFn&& load_stage,
                                       SideFn&& side) {
    using C = Cfg<MATS, XM, K2, MTL, NG, KS, NSA, NGR>;
    constexpr int BM = C::BM, LDA = C::LDA, CPR = C::CPR, TW = C::TW, PAIRS = N8 / 2;
#pragma unroll
    for (int s = 0; s < NSA - 1; ++s) {
        if (s < S) load_stage(s);
        cp_commit();
    }
    cp_wait<NSA - 2>();
    __syncthreads();
    for (int s = 0; s < S; ++s) {
        side(s);
        if (s + NSA - 1 < S) load_stage(s + NSA - 1);
        cp_commit();
        const unsigned char* st = smem + (size_t)(s % NSA) * C::STAGE;
        const half* A = reinterpret_cast<const half*>(st) + (size_t)xmat * BM * LDA;
        const uint32_t* Wd = reinterpret_cast<const uint32_t*>(st + C::A_BYTES) + (size_t)mat * KS * NB * TW +
                             slice * MTL * TW;
#pragma unroll
        for (int kk = 0; kk < KS; ++kk) {
            uint32_t af[MTL][4];
#pragma unroll
            for (int l = 0; l < MTL; ++l) decode3<K2, LEAN>(Wd + (kk * NB + l) * TW, map, af[l]);
            const int chunk = swz<CPR>(lr, kk * 2 + lc);
#pragma unroll
            for (int np = 0; np < PAIRS; ++np) {
                uint32_t b[4];
                ldmatrix4(b, A + (16 * np + lr) * LDA + chunk * 8);
                const uint32_t b01[2] = {b[0], b[1]}, b23[2] = {b[2], b[3]};
#pragma unroll
                for (int l = 0; l < MTL; ++l) {
                    mma16816(acc[l][2 * np], af[l], b01);
                    mma16816(acc[l][2 * np + 1], af[l], b23);
                }
            }
            if constexpr ((N8 & 1) != 0) {
                uint32_t b[2];
                ldmatrix2(b, A + (16 * PAIRS + lr) * LDA + chunk * 8);
#pragma unroll
                for (int l = 0; l < MTL; ++l) mma16816(acc[l][2 * PAIRS], af[l], b);
            }
        }
        cp_wait<NSA - 2>();
        __syncthreads();
    }
}

// gm2_item at LEAN's decode, `side` run inside its k loop
template <int MATS, int XM, int K2, int MTL, int NG, int KS, int NSA, int NGR, int LEAN, typename SideFn>
__device__ __forceinline__ void gm3_item(const Args& a, unsigned char* smem, const int* rows_sh, int e, int cnt,
                                         int nb, SideFn&& side) {
    using C = Cfg<MATS, XM, K2, MTL, NG, KS, NSA, NGR>;
    constexpr int THREADS = C::THREADS, BM = C::BM, LDA = C::LDA, LDE = C::LDE, ER = C::ER, CPR = C::CPR,
                  TW = C::TW, W = C::W;
    constexpr int ACH = XM * BM * CPR, WCH = MATS * KS * NB * K2;
    constexpr int APT = (ACH + THREADS - 1) / THREADS, WPT = (WCH + THREADS - 1) / THREADS;
    float* ep = reinterpret_cast<float*>(smem);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int mat = warp / C::WPM, slice = warp % C::WPM;
    const int xmat = XM == 1 ? 0 : mat;
    const int g = lane >> 2, t = lane & 3;
    const int K = a.K, N = a.N, KT = K >> 4, NTILES = N >> 4, S = KT / KS;
    const size_t se = (size_t)KT * NTILES * TW;
    const int lr = (lane & 7) + ((lane >> 4) << 3), lc = (lane >> 3) & 1;
    const LaneMap<K2> map(lane);
    const uint32_t* te0 = a.tp0 ? reinterpret_cast<const uint32_t*>(a.tp0[e]) : a.t0 + (size_t)e * se;
    const uint32_t* te1 = a.tp1 ? reinterpret_cast<const uint32_t*>(a.tp1[e]) : a.t1 + (size_t)e * se;
    const int cnt16 = (cnt + 15) & ~15;

    int a_src[APT], w_src[WPT];
#pragma unroll
    for (int j = 0; j < APT; ++j) {
        const int i = threadIdx.x + j * THREADS, r = (i / CPR) % BM, c = i % CPR;
        int v = -1;
        if (i < ACH && r < cnt16) v = rows_sh[r] < 0 ? -2 : rows_sh[r] * K + c * 8;
        a_src[j] = v;
    }
#pragma unroll
    for (int j = 0; j < WPT; ++j) {
        const int i = threadIdx.x + j * THREADS, q = i % K2, n = (i / K2) % NB, kk = (i / (K2 * NB)) % KS;
        w_src[j] = (kk * NTILES + nb * NB + n) * TW + q * 4;
    }
    auto load_stage = [&](int s) {
        unsigned char* st = smem + (size_t)(s % NSA) * C::STAGE;
        half* ra = reinterpret_cast<half*>(st);
        uint32_t* rw = reinterpret_cast<uint32_t*>(st + C::A_BYTES);
        const int k0 = s * KS * 16;
#pragma unroll
        for (int j = 0; j < APT; ++j) {
            const int i = threadIdx.x + j * THREADS, v = a_src[j];
            if (v != -1) {
                const int m = i / (BM * CPR), r = (i / CPR) % BM, c = i % CPR;
                cp_async16(ra + ((size_t)m * BM + r) * LDA + swz<CPR>(r, c) * 8, (m ? a.x1 : a.x0) + (v < 0 ? 0 : v + k0),
                           v >= 0);
            }
        }
        const size_t ws = (size_t)s * KS * NTILES * TW;
#pragma unroll
        for (int j = 0; j < WPT; ++j) {
            const int i = threadIdx.x + j * THREADS;
            if (WCH % THREADS == 0 || i < WCH) {
                const int q = i % K2, n = (i / K2) % NB, kk = (i / (K2 * NB)) % KS, m = i / (K2 * NB * KS);
                cp_async16(rw + ((m * KS + kk) * NB + n) * TW + q * 4, (m ? te1 : te0) + ws + w_src[j], true);
            }
        }
    };

    float acc[MTL][NG][4];
#pragma unroll
    for (int l = 0; l < MTL; ++l)
#pragma unroll
        for (int n = 0; n < NG; ++n)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[l][n][c] = 0.f;

    const int n8 = (cnt + 7) >> 3;
#define GM3_K(G)                                                                                                     \
    kloop3<G, MATS, XM, K2, MTL, NG, KS, NSA, NGR, LEAN>(acc, smem, S, xmat, mat, slice, lr, lc, map, load_stage, side)
    switch (n8 <= 8 ? n8 : (n8 + 1) & ~1) {
        case 1: GM3_K(1); break;
        case 2: GM3_K(2); break;
        case 3: GM3_K(3); break;
        case 4: GM3_K(4); break;
        case 5: if constexpr (NG >= 6) GM3_K(5); break;
        case 6: if constexpr (NG >= 6) GM3_K(6); break;
        case 7: if constexpr (NG >= 8) GM3_K(7); break;
        case 8: if constexpr (NG >= 8) GM3_K(8); break;
        case 10: if constexpr (NG >= 10) GM3_K(10); break;
        case 12: if constexpr (NG >= 12) GM3_K(12); break;
        case 14: if constexpr (NG >= 14) GM3_K(14); break;
        default: if constexpr (NG >= 16) GM3_K(16); break;
    }
#undef GM3_K

    // epilogue: gm_kernel's
#pragma unroll
    for (int rd = 0; rd < C::ROUNDS; ++rd) {
        if (rd) __syncthreads();
        if (ER * rd < cnt) {
#pragma unroll
            for (int n = 0; n < NGR; ++n) {
                const int ng = rd * NGR + n;
                float* E = ep + ((size_t)mat * ER + 8 * n + 2 * t) * LDE + slice * 16 * MTL + g;
#pragma unroll
                for (int l = 0; l < MTL; ++l) {
                    E[l * 16] = acc[l][ng][0];
                    E[LDE + l * 16] = acc[l][ng][1];
                    E[l * 16 + 8] = acc[l][ng][2];
                    E[LDE + l * 16 + 8] = acc[l][ng][3];
                }
            }
        }
        __syncthreads();
        for (int r = warp; r < ER && ER * rd + r < cnt; r += W) {
            const int row = rows_sh[ER * rd + r];
            const int c0 = 4 * lane;
            const size_t col = (size_t)e * N + nb * 128 + c0;
            float v[4];
            *reinterpret_cast<float4*>(v) = *reinterpret_cast<const float4*>(ep + (size_t)r * LDE + c0);
            tf_exl3::fwht128(v, lane);
            if constexpr (MATS == 2) {
                float u[4];
                *reinterpret_cast<float4*>(u) = *reinterpret_cast<const float4*>(ep + ((size_t)ER + r) * LDE + c0);
                tf_exl3::fwht128(u, lane);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float gg = fminf(v[j] * HAD * __half2float(a.sv0[col + j]), a.limit);
                    const float uu = fminf(fmaxf(u[j] * HAD * __half2float(a.sv1[col + j]), -a.limit), a.limit);
                    const float act = gg / (1.f + expf(-gg)) * uu;
                    v[j] = act * __half2float(a.sd[col + j]);
                }
                tf_exl3::fwht128(v, lane);
                half2* o = reinterpret_cast<half2*>(a.xd + (size_t)row * N + nb * 128 + c0);
                o[0] = __halves2half2(__float2half_rn(v[0] * HAD), __float2half_rn(v[1] * HAD));
                o[1] = __halves2half2(__float2half_rn(v[2] * HAD), __float2half_rn(v[3] * HAD));
            } else {
                float4 o;
                o.x = v[0] * HAD * __half2float(a.sv0[col + 0]);
                o.y = v[1] * HAD * __half2float(a.sv0[col + 1]);
                o.z = v[2] * HAD * __half2float(a.sv0[col + 2]);
                o.w = v[3] * HAD * __half2float(a.sv0[col + 3]);
                *reinterpret_cast<float4*>(a.y + (size_t)row * N + nb * 128 + c0) = o;
            }
        }
    }
}

__device__ __forceinline__ int ld_relaxed(const int* p) {
    int v;
    asm volatile("ld.global.relaxed.gpu.b32 %0, [%1];\n" : "=r"(v) : "l"(p));
    return v;
}

// One v3 launch over the passes of every width (gm2_kernel's), the next item claimed and read during this one
template <int MATS, int XM, int MTL, int NG, int KS, int NSA, int NGR, int LEAN>
__global__ void __launch_bounds__(Smem2<MATS, XM, MTL, NG, KS, NSA, NGR>::THREADS,
                                  Smem2<MATS, XM, MTL, NG, KS, NSA, NGR>::MINB) gm3_kernel(const Args a) {
    constexpr int BM = NG * 8, THREADS = Smem2<MATS, XM, MTL, NG, KS, NSA, NGR>::THREADS, AHEAD = 5;
    constexpr int RPT = (BM + THREADS - 1) / THREADS;          // member rows a thread reads ahead
    extern __shared__ __align__(16) unsigned char smem[];
    __shared__ int rows_sh[2][BM];
    __shared__ int meta_sh[2][4];                               // item, expert, first index, members
    const int NBLK = a.N >> 7, total = a.npass[0] * NBLK;
    int next = blockIdx.x;
    // the first item, claimed and read as gm2_kernel does
    if (threadIdx.x == 0) {
        const int item = a.ticket ? atomicAdd(a.ticket, 1) : next;
        meta_sh[0][0] = item;
        if (item < total) {
            const int tt = item / NBLK;
            meta_sh[0][1] = a.pe[tt];
            meta_sh[0][2] = a.poff[tt];
            meta_sh[0][3] = a.pcnt[tt];
        }
    }
    next += gridDim.x;
    __syncthreads();
    if (meta_sh[0][0] < total)
        for (int i = threadIdx.x; i < BM; i += THREADS) rows_sh[0][i] = i < meta_sh[0][3] ? a.order[meta_sh[0][2] + i] : -1;
    int cur = 0;
    for (;;) {
        __syncthreads();                                        // this item's rows and meta are written
        const int item = meta_sh[cur][0];
        if (item >= total) break;
        const int nb = item % NBLK, e = meta_sh[cur][1], cnt = meta_sh[cur][3];
        const int nx = cur ^ 1;
        // the next item, one dependent step a stage: claim; pass entry; publish; member rows; publish
        int claim = 0, pe_v = 0, off_v = 0, cnt_v = 0, rows_v[RPT];
        auto side = [&](int s) {
            if (s == 0) {
                if (threadIdx.x == 0) claim = a.ticket ? atomicAdd(a.ticket, 1) : next;
                next += gridDim.x;
            } else if (s == 1) {
                if (threadIdx.x == 0 && claim < total) {
                    const int tt = claim / NBLK;
                    pe_v = ld_relaxed(a.pe + tt);
                    off_v = ld_relaxed(a.poff + tt);
                    cnt_v = ld_relaxed(a.pcnt + tt);
                }
            } else if (s == 2) {
                if (threadIdx.x == 0) {
                    meta_sh[nx][0] = claim;
                    meta_sh[nx][1] = pe_v;
                    meta_sh[nx][2] = off_v;
                    meta_sh[nx][3] = cnt_v;
                }
            } else if (s == 3) {
                const bool live = meta_sh[nx][0] < total;
                const int o = meta_sh[nx][2], c = meta_sh[nx][3];
#pragma unroll
                for (int j = 0; j < RPT; ++j) {
                    const int i = threadIdx.x + j * THREADS;
                    rows_v[j] = live && i < BM && i < c ? ld_relaxed(a.order + o + i) : -1;
                }
            } else if (s == 4) {
#pragma unroll
                for (int j = 0; j < RPT; ++j) {
                    const int i = threadIdx.x + j * THREADS;
                    if (i < BM) rows_sh[nx][i] = rows_v[j];
                }
            }
        };
        switch (a.k2e[e]) {                                     // block-uniform
#define GM3_CASE(K)                                                                                                   \
    case K:                                                                                                           \
        gm3_item<MATS, XM, K, MTL, NG, KS, Ring2<MATS, XM, K, MTL, NG, KS, NSA, NGR>::value, NGR, LEAN>(             \
            a, smem, rows_sh[cur], e, cnt, nb, side);                                                                 \
        break;
            GM2_WIDTHS(GM3_CASE)
#undef GM3_CASE
            default: break;
        }
        // a k loop shorter than AHEAD stages: the rest of the steps now, a barrier between each
        const int S = (a.K >> 4) / KS;
        for (int s = S; s < AHEAD; ++s) {
            __syncthreads();
            side(s);
        }
        cur = nx;
    }
}

}  // namespace dsv41_x3gm

// v3's tiles x LEAN: gate/up at v2's GU0/GU1 and 1-column-tile warps, down at DN0 and 8-warp tiles; a Zig test picks
#define DSV41_X3GM3(MATS, XM, MTL, NG, KS, NSA, NGR, LEAN) \
    template __global__ void dsv41_x3gm::gm3_kernel<MATS, XM, MTL, NG, KS, NSA, NGR, LEAN>(const dsv41_x3gm::Args);
DSV41_X3GM3(2, 2, 2, 8, 2, 4, 4, 0)
DSV41_X3GM3(2, 2, 2, 8, 2, 4, 4, 1)
DSV41_X3GM3(2, 2, 1, 8, 2, 4, 4, 1)
DSV41_X3GM3(2, 2, 1, 8, 4, 3, 4, 1)
DSV41_X3GM3(2, 1, 2, 8, 4, 3, 4, 1)
DSV41_X3GM3(2, 1, 1, 8, 4, 3, 4, 1)
DSV41_X3GM3(1, 1, 2, 8, 4, 4, 8, 0)
DSV41_X3GM3(1, 1, 2, 8, 4, 4, 8, 1)
DSV41_X3GM3(1, 1, 1, 8, 4, 4, 8, 0)
DSV41_X3GM3(1, 1, 1, 8, 4, 4, 8, 1)
DSV41_X3GM3(1, 1, 1, 8, 2, 6, 8, 1)

// v2's gate/up at the 128-member tile GU2 (TF_DSV41_GM_GU2), never built by x3gm.cu's bindings; tiles move data only
template __global__ void dsv41_x3gm::gm2_kernel<2, 2, GM_GU2>(const dsv41_x3gm::Args);
