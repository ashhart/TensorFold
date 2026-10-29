// Copy of src/tensorfold/families/deepseek_v41/cuda/x3gm.cu (blob 7aed2b027afc): sass_check.sh



// x3gm: routed EXL3 prefill experts (gate/up, down); each output is one mma chain over ascending k from +0, any tiling

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "../exl3_decode.cuh"
#include "experts_grouped.cuh"
#include "common/cp_async.cuh"

namespace dsv41_x3gm {

using tf_exl3x::cb_pair;
using tf_exl3x::Fmt;
using tf_exl3x::LaneMap;
using tf_exl3x::mma16816;

constexpr float HAD = 0.08838834764831845f;          // 1 / sqrt(128)
constexpr int NB = 8;                                 // column tiles an item: 128 columns, one Hadamard block
constexpr int MUL1 = 2;

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool ok) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(ok ? 16 : 0));
}
using tf_common::cp_commit;
using tf_common::cp_wait;
__device__ __forceinline__ void ldsm4(uint32_t (&a)[4], const void* p) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(s));
}

// the 16-byte chunk c of member row r in a stage (CPR chunks a row): 8 rows of an ldmatrix phase in 8 bank groups
template <int CPR>
__device__ __forceinline__ int swz(int r, int c) {
    static_assert(CPR == 8 || CPR == 4, "rows of 64 or 32 halves");
    return CPR == 8 ? (c ^ (r & 7)) : (c ^ ((r >> 1) & 3));
}

// A 16x16 W_q tile from shared words as the A fragment of W^T: upstream decode_tile, registers reordered
template <int K2>
__device__ __forceinline__ void decode_a(const uint32_t* tile, const LaneMap<K2>& m, uint32_t (&a)[4]) {
    constexpr int GV = Fmt<K2>::GV, NG = Fmt<K2>::NG;
    uint32_t st[8];
#pragma unroll
    for (int g = 0; g < NG; ++g) {
        const uint32_t whi = tile[m.hi[g]], wlo = tile[m.lo[g]];
        const uint64_t mm = ((((uint64_t)wlo) << 32) | whi) >> m.sh[g];
#pragma unroll
        for (int j = 0; j < GV; ++j) st[g * GV + j] = (uint32_t)(mm >> Fmt<K2>::off(j)) & 0xffffu;
    }
    a[0] = cb_pair<MUL1>(st[0], st[1]);      // b0[0]: k 2t, 2t+1 of column g      -> A row g,     k 2t..
    a[1] = cb_pair<MUL1>(st[4], st[5]);      // b1[0]: k 2t, 2t+1 of column g + 8  -> A row g + 8, k 2t..
    a[2] = cb_pair<MUL1>(st[2], st[3]);      // b0[1]: k 2t+8, 2t+9 of column g    -> A row g,     k 2t+8..
    a[3] = cb_pair<MUL1>(st[6], st[7]);      // b1[1]                              -> A row g + 8, k 2t+8..
}

template <int MATS, int XM, int K2, int MTL, int NG, int KS, int NSA, int NGR>
struct Cfg {
    static constexpr int BM = NG * 8, WPM = NB / MTL, W = MATS * WPM, THREADS = W * 32;
    static constexpr int CPR = KS * 2, LDA = KS * 16, TW = 4 * K2;
    static constexpr int LDE = 132, ER = NGR * 8, ROUNDS = NG / NGR;
    static constexpr size_t A_BYTES = (size_t)XM * BM * LDA * sizeof(half);
    static constexpr size_t W_BYTES = (size_t)MATS * KS * NB * TW * sizeof(uint32_t);
    static constexpr size_t STAGE = A_BYTES + W_BYTES;
    static constexpr size_t RING = (size_t)NSA * STAGE;
    static constexpr size_t EPI = (size_t)MATS * ER * LDE * sizeof(float);
    static constexpr size_t SMEM = RING > EPI ? RING : EPI;
    // CTAs an SM the registers are capped for: 2 when two rings fit, <= 8 warps, <= 64 accumulators a lane; else 1
    static constexpr int MINB = 2 * (SMEM + 1536) <= 102400 && THREADS <= 256 && MTL * NG <= 16 ? 2 : 1;
    static_assert(NG % 2 == 0 && NG % NGR == 0 && NGR % 2 == 0 && NB % MTL == 0, "tiling");
    static_assert(A_BYTES % 16 == 0 && (TW * 4) % 16 == 0, "16-byte stages");
    static_assert(XM == 1 || XM == MATS, "one shared input or one a matrix");
};

struct Args {
    const half* x0;           // [P, K] rows of matrix 0 (gate / down input)
    const half* x1;           // [P, K] rows of matrix 1 (up) when XM == 2
    const uint32_t* t0;       // [E, K/16, N/16, 4 K2] trellis words, stacked (when tp0 is null)
    const uint32_t* t1;
    const int64_t* tp0;       // [E] expert e's words [K/16, N/16, 4 K2] (a ragged stack), or null
    const int64_t* tp1;
    const int* k2e;           // [E] each expert's K2 (gm2_kernel; the launch's width table)
    const int* order;         // pair indices by expert (stable: rows ascending)
    const int* pe;            // pass -> expert
    const int* poff;          // pass -> first index into order
    const int* pcnt;          // pass -> members (<= BM)
    const int* npass;         // [1] passes in all
    int* ticket;              // [1] zeroed, or null (static stride)
    const half* sv0;          // gate/up: svh_g [E, N]; down: svh_d [E, N]
    const half* sv1;          // svh_u [E, N]
    const half* sd;           // suh_d [E, N]
    half* xd;                 // gate/up out [P, N]
    float* y;                 // down out [P, N]
    int K, N;
    float limit;
};

template <int MATS, int XM, int K2, int MTL, int NG, int KS, int NSA, int NGR>
__global__ void __launch_bounds__(Cfg<MATS, XM, K2, MTL, NG, KS, NSA, NGR>::THREADS,
                                  Cfg<MATS, XM, K2, MTL, NG, KS, NSA, NGR>::MINB) gm_kernel(const Args a) {
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

        // stage s: member rows' k [16 KS s, + 16 KS) (zero past the count), then the item's k tile words, block nb
        auto load_stage = [&](int s) {
            unsigned char* st = smem + (size_t)(s % NSA) * C::STAGE;
            half* ra = reinterpret_cast<half*>(st);
            const int k0 = s * KS * 16;
            for (int i = threadIdx.x; i < XM * BM * CPR; i += THREADS) {
                const int m = i / (BM * CPR), r = (i / CPR) % BM, c = i % CPR;
                if (r < cnt16) {
                    const int row = rows_sh[r];
                    const half* src = (m ? a.x1 : a.x0) + (size_t)(row < 0 ? 0 : row) * K + k0 + c * 8;
                    cp_async16(ra + ((size_t)m * BM + r) * LDA + swz<CPR>(r, c) * 8, src, row >= 0);
                }
            }
            uint32_t* rw = reinterpret_cast<uint32_t*>(st + C::A_BYTES);
            constexpr int WCH = MATS * KS * NB * K2;          // 16-byte chunks: K2 a tile
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
                for (int l = 0; l < MTL; ++l) decode_a<K2>(Wd + (kk * NB + l) * TW, map, af[l]);
                const int chunk = swz<CPR>(lr, kk * 2 + lc);
#pragma unroll
                for (int np = 0; np < NG / 2; ++np) {
                    if (16 * np < cnt) {                      // warp-uniform
                        uint32_t b[4];
                        ldsm4(b, A + (16 * np + lr) * LDA + chunk * 8);
                        const uint32_t b01[2] = {b[0], b[1]}, b23[2] = {b[2], b[3]};
#pragma unroll
                        for (int l = 0; l < MTL; ++l) {
                            mma16816(acc[l][2 * np], af[l], b01);
                            mma16816(acc[l][2 * np + 1], af[l], b23);
                        }
                    }
                }
            }
            cp_wait<NSA - 2>();
            __syncthreads();
        }

        // epilogue: ER members a round, accumulators -> ep[mat][member][column], then one warp a member row
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
}

#include "x3gm2.cuh"

// Rotated routed inputs: out_m[p] = fp16((x[p / slots] * suh_m[e]) H / sqrt 128), upstream rot_in's arithmetic
template <typename TIN> __device__ __forceinline__ float to_f(TIN v);
template <> __device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <> __device__ __forceinline__ float to_f<half>(half v) { return __half2float(v); }

template <typename TIN, int MATS>
__global__ void __launch_bounds__(256) rot_kernel(const TIN* __restrict__ x, int x_stride, const int* __restrict__ pick,
                                                  const half* __restrict__ suh0, const half* __restrict__ suh1,
                                                  half* __restrict__ out0, half* __restrict__ out1, int K, int slots,
                                                  int items) {
    const int item = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (item >= items) return;                                // warp-uniform
    const int KB = K >> 7, p = item / KB, blk = item % KB, lane = threadIdx.x & 31;
    const int e = pick[p];
    const TIN* xr = x + (size_t)(p / slots) * x_stride + blk * 128 + 4 * lane;
    float xv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) xv[j] = to_f<TIN>(xr[j]);
#pragma unroll
    for (int m = 0; m < MATS; ++m) {
        const half* suh = (m ? suh1 : suh0) + (size_t)e * K + blk * 128 + 4 * lane;
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = xv[j] * __half2float(suh[j]);
        tf_exl3::fwht128(v, lane);
        half* o = (m ? out1 : out0) + (size_t)p * K + blk * 128 + 4 * lane;
#pragma unroll
        for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD);
    }
}

// Tests: W_q [K, N] fp16 of one trellis through decode_a (words staged in shared memory as the gm kernels stage them).
template <int K2>
__global__ void __launch_bounds__(32) dequant_kernel(const uint32_t* __restrict__ T, half* __restrict__ out, int N) {
    constexpr int TW = 4 * K2;
    __shared__ __align__(16) uint32_t tile[TW];
    const int kt = blockIdx.x, nt = blockIdx.y, lane = threadIdx.x, NTILES = N >> 4;
    for (int i = lane; i < TW; i += 32) tile[i] = T[((size_t)kt * NTILES + nt) * TW + i];
    __syncwarp();
    const LaneMap<K2> map(lane);
    uint32_t af[4];
    decode_a<K2>(tile, map, af);
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int q = 0; q < 4; ++q) {                             // A row (q & 1 ? g + 8 : g) = W column; k 2t (+8)
        const half2 h = *reinterpret_cast<const half2*>(&af[q]);
        const int col = nt * 16 + g + 8 * (q & 1), row = kt * 16 + 2 * t + 8 * (q >> 1);
        out[(size_t)row * N + col] = __low2half(h);
        out[(size_t)(row + 1) * N + col] = __high2half(h);
    }
}


// Tile table (MTL, NG, KS, NSA, NGR), mirrored by the host tile lists (a test parses these lines); 2.. = 4K tiles
#define GM_GU0 2, 8, 4, 3, 4
#define GM_GU1 2, 8, 2, 4, 4
#define GM_GU2 1, 16, 2, 4, 4
#define GM_GU3 1, 16, 4, 3, 4
#define GM_GU4 2, 16, 2, 3, 4
#define GM_GU5 2, 8, 2, 6, 4
#define GM_DN0 2, 8, 4, 4, 8
#define GM_DN1 1, 8, 4, 4, 8
#define GM_DN2 1, 16, 4, 4, 8
#define GM_DN3 1, 16, 2, 6, 8
#define GM_DN4 2, 16, 4, 3, 16
#define GM_NGU 6
#define GM_NDN 5


}  // namespace dsv41_x3gm

#ifndef DSV41_X3GM_NO_INSTANCES
// Instances the bindings launch: gate/up (XM 1 shared input, 2 own) GM_GU0-5, down GM_DN0-4, K2 3-8 and 10
#define DSV41_X3GM_GU(XM, K2, CFG) template __global__ void dsv41_x3gm::gm_kernel<2, XM, K2, CFG>(const dsv41_x3gm::Args);
#define DSV41_X3GM_DN(K2, CFG) template __global__ void dsv41_x3gm::gm_kernel<1, 1, K2, CFG>(const dsv41_x3gm::Args);
// v2 (gm2_kernel: every width in one launch): gate/up GM_GU0 / GM_GU1 at either input mode, down GM_DN0
template __global__ void dsv41_x3gm::gm2_kernel<2, 1, GM_GU0>(const dsv41_x3gm::Args);
template __global__ void dsv41_x3gm::gm2_kernel<2, 1, GM_GU1>(const dsv41_x3gm::Args);
template __global__ void dsv41_x3gm::gm2_kernel<2, 2, GM_GU0>(const dsv41_x3gm::Args);
template __global__ void dsv41_x3gm::gm2_kernel<2, 2, GM_GU1>(const dsv41_x3gm::Args);
template __global__ void dsv41_x3gm::gm2_kernel<1, 1, GM_DN0>(const dsv41_x3gm::Args);
DSV41_X3GM_GU(1, 3, GM_GU0)
DSV41_X3GM_GU(1, 3, GM_GU1)
DSV41_X3GM_GU(1, 3, GM_GU2)
DSV41_X3GM_GU(1, 3, GM_GU3)
DSV41_X3GM_GU(1, 3, GM_GU4)
DSV41_X3GM_GU(1, 3, GM_GU5)
DSV41_X3GM_GU(2, 3, GM_GU0)
DSV41_X3GM_GU(2, 3, GM_GU1)
DSV41_X3GM_GU(2, 3, GM_GU2)
DSV41_X3GM_GU(2, 3, GM_GU3)
DSV41_X3GM_GU(2, 3, GM_GU4)
DSV41_X3GM_GU(2, 3, GM_GU5)
DSV41_X3GM_DN(3, GM_DN0)
DSV41_X3GM_DN(3, GM_DN1)
DSV41_X3GM_DN(3, GM_DN2)
DSV41_X3GM_DN(3, GM_DN3)
DSV41_X3GM_DN(3, GM_DN4)
DSV41_X3GM_GU(1, 4, GM_GU0)
DSV41_X3GM_GU(1, 4, GM_GU1)
DSV41_X3GM_GU(1, 4, GM_GU2)
DSV41_X3GM_GU(1, 4, GM_GU3)
DSV41_X3GM_GU(1, 4, GM_GU4)
DSV41_X3GM_GU(1, 4, GM_GU5)
DSV41_X3GM_GU(2, 4, GM_GU0)
DSV41_X3GM_GU(2, 4, GM_GU1)
DSV41_X3GM_GU(2, 4, GM_GU2)
DSV41_X3GM_GU(2, 4, GM_GU3)
DSV41_X3GM_GU(2, 4, GM_GU4)
DSV41_X3GM_GU(2, 4, GM_GU5)
DSV41_X3GM_DN(4, GM_DN0)
DSV41_X3GM_DN(4, GM_DN1)
DSV41_X3GM_DN(4, GM_DN2)
DSV41_X3GM_DN(4, GM_DN3)
DSV41_X3GM_DN(4, GM_DN4)
DSV41_X3GM_GU(1, 5, GM_GU0)
DSV41_X3GM_GU(1, 5, GM_GU1)
DSV41_X3GM_GU(1, 5, GM_GU2)
DSV41_X3GM_GU(1, 5, GM_GU3)
DSV41_X3GM_GU(1, 5, GM_GU4)
DSV41_X3GM_GU(1, 5, GM_GU5)
DSV41_X3GM_GU(2, 5, GM_GU0)
DSV41_X3GM_GU(2, 5, GM_GU1)
DSV41_X3GM_GU(2, 5, GM_GU2)
DSV41_X3GM_GU(2, 5, GM_GU3)
DSV41_X3GM_GU(2, 5, GM_GU4)
DSV41_X3GM_GU(2, 5, GM_GU5)
DSV41_X3GM_DN(5, GM_DN0)
DSV41_X3GM_DN(5, GM_DN1)
DSV41_X3GM_DN(5, GM_DN2)
DSV41_X3GM_DN(5, GM_DN3)
DSV41_X3GM_DN(5, GM_DN4)
DSV41_X3GM_GU(1, 6, GM_GU0)
DSV41_X3GM_GU(1, 6, GM_GU1)
DSV41_X3GM_GU(1, 6, GM_GU2)
DSV41_X3GM_GU(1, 6, GM_GU3)
DSV41_X3GM_GU(1, 6, GM_GU4)
DSV41_X3GM_GU(1, 6, GM_GU5)
DSV41_X3GM_GU(2, 6, GM_GU0)
DSV41_X3GM_GU(2, 6, GM_GU1)
DSV41_X3GM_GU(2, 6, GM_GU2)
DSV41_X3GM_GU(2, 6, GM_GU3)
DSV41_X3GM_GU(2, 6, GM_GU4)
DSV41_X3GM_GU(2, 6, GM_GU5)
DSV41_X3GM_DN(6, GM_DN0)
DSV41_X3GM_DN(6, GM_DN1)
DSV41_X3GM_DN(6, GM_DN2)
DSV41_X3GM_DN(6, GM_DN3)
DSV41_X3GM_DN(6, GM_DN4)
DSV41_X3GM_GU(1, 7, GM_GU0)
DSV41_X3GM_GU(1, 7, GM_GU1)
DSV41_X3GM_GU(1, 7, GM_GU2)
DSV41_X3GM_GU(1, 7, GM_GU3)
DSV41_X3GM_GU(1, 7, GM_GU4)
DSV41_X3GM_GU(1, 7, GM_GU5)
DSV41_X3GM_GU(2, 7, GM_GU0)
DSV41_X3GM_GU(2, 7, GM_GU1)
DSV41_X3GM_GU(2, 7, GM_GU2)
DSV41_X3GM_GU(2, 7, GM_GU3)
DSV41_X3GM_GU(2, 7, GM_GU4)
DSV41_X3GM_GU(2, 7, GM_GU5)
DSV41_X3GM_DN(7, GM_DN0)
DSV41_X3GM_DN(7, GM_DN1)
DSV41_X3GM_DN(7, GM_DN2)
DSV41_X3GM_DN(7, GM_DN3)
DSV41_X3GM_DN(7, GM_DN4)
DSV41_X3GM_GU(1, 8, GM_GU0)
DSV41_X3GM_GU(1, 8, GM_GU1)
DSV41_X3GM_GU(1, 8, GM_GU2)
DSV41_X3GM_GU(1, 8, GM_GU3)
DSV41_X3GM_GU(1, 8, GM_GU4)
DSV41_X3GM_GU(1, 8, GM_GU5)
DSV41_X3GM_GU(2, 8, GM_GU0)
DSV41_X3GM_GU(2, 8, GM_GU1)
DSV41_X3GM_GU(2, 8, GM_GU2)
DSV41_X3GM_GU(2, 8, GM_GU3)
DSV41_X3GM_GU(2, 8, GM_GU4)
DSV41_X3GM_GU(2, 8, GM_GU5)
DSV41_X3GM_DN(8, GM_DN0)
DSV41_X3GM_DN(8, GM_DN1)
DSV41_X3GM_DN(8, GM_DN2)
DSV41_X3GM_DN(8, GM_DN3)
DSV41_X3GM_DN(8, GM_DN4)
DSV41_X3GM_GU(1, 10, GM_GU0)
DSV41_X3GM_GU(1, 10, GM_GU1)
DSV41_X3GM_GU(1, 10, GM_GU2)
DSV41_X3GM_GU(1, 10, GM_GU3)
DSV41_X3GM_GU(1, 10, GM_GU4)
DSV41_X3GM_GU(1, 10, GM_GU5)
DSV41_X3GM_GU(2, 10, GM_GU0)
DSV41_X3GM_GU(2, 10, GM_GU1)
DSV41_X3GM_GU(2, 10, GM_GU2)
DSV41_X3GM_GU(2, 10, GM_GU3)
DSV41_X3GM_GU(2, 10, GM_GU4)
DSV41_X3GM_GU(2, 10, GM_GU5)
DSV41_X3GM_DN(10, GM_DN0)
DSV41_X3GM_DN(10, GM_DN1)
DSV41_X3GM_DN(10, GM_DN2)
DSV41_X3GM_DN(10, GM_DN3)
DSV41_X3GM_DN(10, GM_DN4)
#define DSV41_X3GM_ROT(T, M) template __global__ void dsv41_x3gm::rot_kernel<T, M>(const T* __restrict__, int, \
    const int* __restrict__, const half* __restrict__, const half* __restrict__, half* __restrict__, \
    half* __restrict__, int, int, int);
DSV41_X3GM_ROT(__nv_bfloat16, 1)
DSV41_X3GM_ROT(__nv_bfloat16, 2)
DSV41_X3GM_ROT(half, 1)
DSV41_X3GM_ROT(half, 2)
template __global__ void dsv41_x3gm::dequant_kernel<3>(const uint32_t* __restrict__, half* __restrict__, int);
template __global__ void dsv41_x3gm::dequant_kernel<4>(const uint32_t* __restrict__, half* __restrict__, int);
template __global__ void dsv41_x3gm::dequant_kernel<5>(const uint32_t* __restrict__, half* __restrict__, int);
template __global__ void dsv41_x3gm::dequant_kernel<6>(const uint32_t* __restrict__, half* __restrict__, int);
template __global__ void dsv41_x3gm::dequant_kernel<7>(const uint32_t* __restrict__, half* __restrict__, int);
template __global__ void dsv41_x3gm::dequant_kernel<8>(const uint32_t* __restrict__, half* __restrict__, int);
template __global__ void dsv41_x3gm::dequant_kernel<10>(const uint32_t* __restrict__, half* __restrict__, int);
#endif  // DSV41_X3GM_NO_INSTANCES
