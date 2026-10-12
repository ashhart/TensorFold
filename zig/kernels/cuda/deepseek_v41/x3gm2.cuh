// x3gm.cu's v2 kernel (TF_DSV41_GM_V2): gm2_item, kloop2 and gm2_kernel, inside namespace dsv41_x3gm
#pragma once
// x3gm v2 (TF_DSV41_GM_V2): gm_kernel's items, chains and bits with less issue work (member groups, width per item)

__device__ __forceinline__ void ldsm2(uint32_t (&a)[2], const void* p) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(a[0]), "=r"(a[1]) : "r"(s));
}

// The K loop of one item with N8 active n8 member groups (acc[l][0 .. N8 - 1]; the rest stay 0.0, never stored)
template <int N8, int MATS, int XM, int K2, int MTL, int NG, int KS, int NSA, int NGR, typename LoadFn>
__device__ __forceinline__ void kloop2(float (&acc)[MTL][NG][4], const unsigned char* smem, int S, int xmat, int mat,
                                       int slice, int lr, int lc, const LaneMap<K2>& map, LoadFn&& load_stage) {
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
            for (int l = 0; l < MTL; ++l) decode_a<K2>(Wd + (kk * NB + l) * TW, map, af[l]);
            const int chunk = swz<CPR>(lr, kk * 2 + lc);
#pragma unroll
            for (int np = 0; np < PAIRS; ++np) {
                uint32_t b[4];
                ldsm4(b, A + (16 * np + lr) * LDA + chunk * 8);
                const uint32_t b01[2] = {b[0], b[1]}, b23[2] = {b[2], b[3]};
#pragma unroll
                for (int l = 0; l < MTL; ++l) {
                    mma16816(acc[l][2 * np], af[l], b01);
                    mma16816(acc[l][2 * np + 1], af[l], b23);
                }
            }
            if constexpr ((N8 & 1) != 0) {                // members 16 PAIRS .. + 7 (x2: lanes 0-15's addresses)
                uint32_t b[2];
                ldsm2(b, A + (16 * PAIRS + lr) * LDA + chunk * 8);
#pragma unroll
                for (int l = 0; l < MTL; ++l) mma16816(acc[l][2 * PAIRS], af[l], b);
            }
        }
        cp_wait<NSA - 2>();
        __syncthreads();
    }
}

// One item (pass x 128-column block) at width K2, rows_sh filled by the caller; otherwise gm_kernel's body
template <int MATS, int XM, int K2, int MTL, int NG, int KS, int NSA, int NGR>
__device__ __forceinline__ void gm2_item(const Args& a, unsigned char* smem, const int* rows_sh, int e, int cnt,
                                         int nb) {
    using C = Cfg<MATS, XM, K2, MTL, NG, KS, NSA, NGR>;
    constexpr int THREADS = C::THREADS, BM = C::BM, LDA = C::LDA, LDE = C::LDE, ER = C::ER, CPR = C::CPR,
                  TW = C::TW, W = C::W;
    constexpr int ACH = XM * BM * CPR, WCH = MATS * KS * NB * K2;    // 16-byte chunks a stage: member rows, words
    constexpr int APT = (ACH + THREADS - 1) / THREADS, WPT = (WCH + THREADS - 1) / THREADS;
    float* ep = reinterpret_cast<float*>(smem);               // [MATS][ER][LDE] after the K loop (aliases the ring)
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
    const int cnt16 = (cnt + 15) & ~15;                       // rows past it are never read

    // this thread's chunks: row chunk element offset at k 0 (-1: none, -2: zero fill), word chunk offset at stage 0
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

    static_assert(NG <= 16 && NG % 2 == 0, "n8 groups");
    const int n8 = (cnt + 7) >> 3;                            // >= 1 (a pass has members)
#define GM2_K(G) kloop2<G, MATS, XM, K2, MTL, NG, KS, NSA, NGR>(acc, smem, S, xmat, mat, slice, lr, lc, map, load_stage)
    switch (n8 <= 8 ? n8 : (n8 + 1) & ~1) {                   // 1..8, then even counts (NG 16)
        case 1: GM2_K(1); break;
        case 2: GM2_K(2); break;
        case 3: GM2_K(3); break;
        case 4: GM2_K(4); break;
        case 5: if constexpr (NG >= 6) GM2_K(5); break;
        case 6: if constexpr (NG >= 6) GM2_K(6); break;
        case 7: if constexpr (NG >= 8) GM2_K(7); break;
        case 8: if constexpr (NG >= 8) GM2_K(8); break;
        case 10: if constexpr (NG >= 10) GM2_K(10); break;
        case 12: if constexpr (NG >= 12) GM2_K(12); break;
        case 14: if constexpr (NG >= 14) GM2_K(14); break;
        default: if constexpr (NG >= 16) GM2_K(16); break;
    }
#undef GM2_K

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
                for (int j = 0; j < 4; ++j) {                 // upstream gateup_epilogue, ACT_F32
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
                float4 o;                                     // upstream down_epilogue
                o.x = v[0] * HAD * __half2float(a.sv0[col + 0]);
                o.y = v[1] * HAD * __half2float(a.sv0[col + 1]);
                o.z = v[2] * HAD * __half2float(a.sv0[col + 2]);
                o.w = v[3] * HAD * __half2float(a.sv0[col + 3]);
                *reinterpret_cast<float4*>(a.y + (size_t)row * N + nb * 128 + c0) = o;
            }
        }
    }
}

// The widths a v2 launch dispatches and, per width, the ring depth (cut where it would cost a second CTA an SM)
#define GM2_WIDTHS(X) X(3) X(4) X(5) X(6) X(7) X(8) X(10)
constexpr int GM2_NWIDTHS = 7;
template <int MATS, int XM, int K2, int MTL, int NG, int KS, int NSA, int NGR>
struct Ring2 {
    static constexpr bool fits2(int n) {
        return 2 * (Cfg<MATS, XM, K2, MTL, NG, KS, 2, NGR>::STAGE * (size_t)n + 1536) <= 102400;
    }
    static constexpr int cut(int n) { return n <= 2 || fits2(n) ? n : cut(n - 1); }
    static constexpr int value = Cfg<MATS, XM, 4, MTL, NG, KS, NSA, NGR>::MINB == 2 ? cut(NSA) : NSA;
};
template <int MATS, int XM, int MTL, int NG, int KS, int NSA, int NGR>
struct Smem2 {
#define GM2_SM(K) Cfg<MATS, XM, K, MTL, NG, KS, Ring2<MATS, XM, K, MTL, NG, KS, NSA, NGR>::value, NGR>::SMEM,
    static constexpr size_t list[GM2_NWIDTHS] = {GM2_WIDTHS(GM2_SM)};
#undef GM2_SM
    static constexpr size_t max_from(int i) {
        return i == GM2_NWIDTHS ? 0 : (list[i] > max_from(i + 1) ? list[i] : max_from(i + 1));
    }
    static constexpr size_t SMEM = max_from(0);
    static constexpr int THREADS = Cfg<MATS, XM, 4, MTL, NG, KS, NSA, NGR>::THREADS;
    static constexpr int MINB = 2 * (SMEM + 1536) <= 102400 && THREADS <= 256 && MTL * NG <= 16 ? 2 : 1;
};

// One v2 launch over the passes of every width (uniform stack: one width; ragged: each expert its own, a.k2e)
template <int MATS, int XM, int MTL, int NG, int KS, int NSA, int NGR>
__global__ void __launch_bounds__(Smem2<MATS, XM, MTL, NG, KS, NSA, NGR>::THREADS,
                                  Smem2<MATS, XM, MTL, NG, KS, NSA, NGR>::MINB) gm2_kernel(const Args a) {
    constexpr int BM = NG * 8, THREADS = Smem2<MATS, XM, MTL, NG, KS, NSA, NGR>::THREADS;
    extern __shared__ __align__(16) unsigned char smem[];
    __shared__ int rows_sh[BM];
    __shared__ int item_sh;
    const int NBLK = a.N >> 7, total = a.npass[0] * NBLK;
    int next = blockIdx.x;
    for (;;) {
        __syncthreads();                                      // the last item is done with rows_sh and shared memory
        if (threadIdx.x == 0) item_sh = a.ticket ? atomicAdd(a.ticket, 1) : next;
        next += gridDim.x;
        __syncthreads();
        const int item = item_sh;
        if (item >= total) break;
        const int tt = item / NBLK, nb = item % NBLK;
        const int e = a.pe[tt], off = a.poff[tt], cnt = a.pcnt[tt];
        for (int i = threadIdx.x; i < BM; i += THREADS) rows_sh[i] = i < cnt ? a.order[off + i] : -1;
        __syncthreads();
        switch (a.k2e[e]) {                                   // block-uniform
#define GM2_CASE(K)                                                                                                   \
    case K:                                                                                                           \
        gm2_item<MATS, XM, K, MTL, NG, KS, Ring2<MATS, XM, K, MTL, NG, KS, NSA, NGR>::value, NGR>(a, smem, rows_sh, e, \
                                                                                                  cnt, nb);           \
        break;
            GM2_WIDTHS(GM2_CASE)
#undef GM2_CASE
            default: break;                                   // never: the host checks every width is in K2S
        }
    }
}
