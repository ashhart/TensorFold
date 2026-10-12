// Timing ablations of x3gm.cu's gm_kernel: ABL 0 = the kernel, 1-7 each drop one part; ABL > 0 outputs are junk.
#pragma once
namespace dsv41_x3gm {
template <int ABL, int MATS, int XM, int K2, int MTL, int NG, int KS, int NSA, int NGR>
__global__ void __launch_bounds__(Cfg<MATS, XM, K2, MTL, NG, KS, NSA, NGR>::THREADS,
                                  Cfg<MATS, XM, K2, MTL, NG, KS, NSA, NGR>::MINB) gm_ablate(const Args a) {
    using C = Cfg<MATS, XM, K2, MTL, NG, KS, NSA, NGR>;
    constexpr int THREADS = C::THREADS, BM = C::BM, LDA = C::LDA, LDE = C::LDE, ER = C::ER, CPR = C::CPR,
                  TW = C::TW, W = C::W;
    extern __shared__ __align__(16) unsigned char smem[];
    float* ep = reinterpret_cast<float*>(smem);               // [MATS][ER][LDE] after the K loop (aliases the ring)
    __shared__ int rows_sh[BM];
    __shared__ int item_sh;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int mat = warp / C::WPM, slice = warp % C::WPM;     // columns slice 16 MTL .. + 16 MTL of the item
    const int xmat = XM == 1 ? 0 : mat;
    const int g = lane >> 2, t = lane & 3;
    const int K = a.K, N = a.N, KT = K >> 4, NTILES = N >> 4, S = KT / KS, NBLK = N >> 7;
    const size_t se = (size_t)KT * NTILES * TW;               // words of one expert's matrix
    const int total = a.npass[0] * NBLK;
    const int lr = (lane & 7) + ((lane >> 4) << 3), lc = (lane >> 3) & 1;
    const LaneMap<K2> map(lane);

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
        const uint32_t* te0 = a.tp0 ? reinterpret_cast<const uint32_t*>(a.tp0[e]) : a.t0 + (size_t)e * se;
        const uint32_t* te1 = a.tp1 ? reinterpret_cast<const uint32_t*>(a.tp1[e]) : a.t1 + (size_t)e * se;
        for (int i = threadIdx.x; i < BM; i += THREADS) rows_sh[i] = i < cnt ? a.order[off + i] : -1;
        __syncthreads();
        const int cnt16 = (cnt + 15) & ~15;                   // rows past it are never read (warp-uniform skip)

        // stage s: member rows (zero past the count), then the item's words of k tiles KS s .. KS s + KS - 1, block nb
        auto load_stage = [&](int s) {
            unsigned char* st = smem + (size_t)(s % NSA) * C::STAGE;
            half* ra = reinterpret_cast<half*>(st);
            const int k0 = s * KS * 16;
            for (int i = threadIdx.x; i < XM * BM * CPR && !(ABL == 4 && s >= NSA); i += THREADS) {
                const int m = i / (BM * CPR), r = (i / CPR) % BM, c = i % CPR;
                if (r < cnt16) {
                    const int row = rows_sh[r];
                    const half* src = (m ? a.x1 : a.x0) + (size_t)(row < 0 ? 0 : row) * K + k0 + c * 8;
                    cp_async16(ra + ((size_t)m * BM + r) * LDA + swz<CPR>(r, c) * 8, src, row >= 0);
                }
            }
            uint32_t* rw = reinterpret_cast<uint32_t*>(st + C::A_BYTES);
            constexpr int WCH = MATS * KS * NB * K2;          // 16-byte chunks: K2 a tile
            if (ABL == 3 && s >= NSA) return;
            for (int i = threadIdx.x; i < WCH; i += THREADS) {
                const int q = i % K2, n = (i / K2) % NB, kk = (i / (K2 * NB)) % KS, m = i / (K2 * NB * KS);
                const uint32_t* src = (m ? te1 : te0) + ((size_t)(s * KS + kk) * NTILES + nb * NB + n) * TW + q * 4;
                cp_async16(rw + ((m * KS + kk) * NB + n) * TW + q * 4, src, true);
            }
        };

        float acc[MTL][NG][4];
#pragma unroll
        for (int l = 0; l < MTL; ++l)
#pragma unroll
            for (int n = 0; n < NG; ++n)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[l][n][c] = 0.f;

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
                for (int l = 0; l < MTL; ++l) {
                    if constexpr (ABL == 1) {
                        const uint32_t* tw = Wd + (kk * NB + l) * TW;
                        af[l][0] = tw[lane % TW] & 0x3bff3bffu; af[l][1] = af[l][0] ^ 0x1u;
                        af[l][2] = af[l][0] ^ 0x2u; af[l][3] = af[l][0] ^ 0x4u;
                    } else {
                        decode_a<K2>(Wd + (kk * NB + l) * TW, map, af[l]);
                    }
                }
                const int chunk = swz<CPR>(lr, kk * 2 + lc);
#pragma unroll
                for (int np = 0; np < NG / 2; ++np) {
                    if (16 * np < cnt) {                      // warp-uniform
                        uint32_t b[4];
                        ldsm4(b, A + (16 * np + lr) * LDA + chunk * 8);
                        const uint32_t b01[2] = {b[0], b[1]}, b23[2] = {b[2], b[3]};
#pragma unroll
                        for (int l = 0; l < MTL; ++l) {
                            if constexpr (ABL == 2) {
                                acc[l][2 * np][0] += __uint_as_float((af[l][0] ^ b01[0]) & 0x3f7fffffu);
                                acc[l][2 * np + 1][0] += __uint_as_float((af[l][1] ^ b23[1]) & 0x3f7fffffu);
                            } else {
                                mma16816(acc[l][2 * np], af[l], b01);
                                mma16816(acc[l][2 * np + 1], af[l], b23);
                            }
                        }
                    }
                }
            }
            cp_wait<NSA - 2>();
            __syncthreads();
        }

        // epilogue: ER members a round, accumulators -> ep[mat][member][column], then one warp a member row
        if constexpr (ABL == 5) {                             // no epilogue: one store keeps the chains live
            if (acc[0][0][0] == 1234.5f) a.y[threadIdx.x] = acc[MTL - 1][NG - 1][3];
            continue;
        }
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
                if constexpr (ABL != 7) tf_exl3::fwht128(v, lane);
                if constexpr (MATS == 2) {
                    float u[4];
                    *reinterpret_cast<float4*>(u) = *reinterpret_cast<const float4*>(ep + ((size_t)ER + r) * LDE + c0);
                    if constexpr (ABL != 7) tf_exl3::fwht128(u, lane);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {                 // upstream gateup_epilogue, ACT_F32
                        const float gg = fminf(v[j] * HAD * (ABL == 6 ? 1.f : __half2float(a.sv0[col + j])), a.limit);
                        const float uu = fminf(fmaxf(u[j] * HAD * (ABL == 6 ? 1.f : __half2float(a.sv1[col + j])), -a.limit), a.limit);
                        const float act = gg / (1.f + expf(-gg)) * uu;
                        v[j] = act * (ABL == 6 ? 1.f : __half2float(a.sd[col + j]));
                    }
                    if constexpr (ABL != 7) tf_exl3::fwht128(v, lane);
                    half2* o = reinterpret_cast<half2*>(a.xd + (size_t)row * N + nb * 128 + c0);
                    o[0] = __halves2half2(__float2half_rn(v[0] * HAD), __float2half_rn(v[1] * HAD));
                    o[1] = __halves2half2(__float2half_rn(v[2] * HAD), __float2half_rn(v[3] * HAD));
                } else {
                    float4 o;                                     // upstream down_epilogue
                    o.x = v[0] * HAD * (ABL == 6 ? 1.f : __half2float(a.sv0[col + 0]));
                    o.y = v[1] * HAD * (ABL == 6 ? 1.f : __half2float(a.sv0[col + 1]));
                    o.z = v[2] * HAD * (ABL == 6 ? 1.f : __half2float(a.sv0[col + 2]));
                    o.w = v[3] * HAD * (ABL == 6 ? 1.f : __half2float(a.sv0[col + 3]));
                    *reinterpret_cast<float4*>(a.y + (size_t)row * N + nb * 128 + c0) = o;
                }
            }
        }
    }
}

}  // namespace dsv41_x3gm
