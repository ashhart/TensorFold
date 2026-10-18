// Copy of src/tensorfold/families/deepseek_v41/cuda/pfdense.cuh (blob 7757ae8b9b9f): sass_check.sh



// pfdense: prefill's dense EXL3 GEMM with the trellis decoded inside it, bit-exact to unpack + Triton's _gemm epilogue

#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "../exl3_decode.cuh"
#include "pfdense_args.h"

namespace dsv41_pfd {

using namespace tf_exl3;

// -- configurations: all give the same bits (one fp32 mma.m16n8k16 chain per output, k16 slices ascending from +0) ---
template <int BM_, int WM_, int WN_, int KS_, int NST_>
struct Cfg {
    static constexpr int BM = BM_, WM = WM_, WN = WN_, KS = KS_, NST = NST_;
    static constexpr int NW = WM * WN, THREADS = NW * 32;
    static constexpr int MT = BM / (WM * 16);             // m16 tiles a warp
    static constexpr int NT = 8 / WN;                     // n16 tiles (one trellis tile each) a warp
    static constexpr int CPR = 2 * KS, LDA = 16 * KS;     // 16-byte chunks / halves of a stage's xh row
    static constexpr int EPR = BM / WM;                   // epilogue rows a round (one warp row's rows)
    static constexpr int GPW = MT >= WN ? MT / WN : 1;    // 16-row groups a warp in its round
    static constexpr int NSPLIT = WN >= MT ? WN / MT : 1; // column parts of a group
    static constexpr int EN8 = 16 / NSPLIT;               // n8 tiles of a warp's epilogue part
    static_assert(BM % (WM * 16) == 0 && MT >= 1 && 8 % WN == 0, "warp grid");
    static_assert(MT % WN == 0 || WN % MT == 0, "epilogue units");
    static_assert(KS == 2 || KS == 4 || KS == 8, "k steps a stage");
    static_assert((KS * 8) % NW == 0, "a stage's tiles split evenly over the warps");
    static_assert(NST >= 2, "stages");
};

template <int K2, bool LANES, class C>
struct Lay {
    static constexpr int TW = 4 * K2;                      // words of a tile
    static constexpr int A_BYTES = C::BM * C::LDA * 2;
    static constexpr int W_BYTES = C::KS * 8 * TW * 4;     // 128 K2 bytes a k step
    static constexpr int STAGE = A_BYTES + W_BYTES;
    static constexpr int RING = C::NST * STAGE;
    static constexpr int BBUF = C::KS * 8 * 32 * 16;       // decoded B fragments of a stage: 512 B a tile
    static constexpr int MAIN = RING + 2 * BBUF;
    static constexpr int EPI = C::EPR * 128 * 2 * 2;       // top and rest, bf16 [EPR][128] each
    static constexpr int SMEM = MAIN > EPI ? MAIN : EPI;
    static constexpr int TPW = C::KS * 8 / C::NW;          // tiles a warp decodes a stage
    static constexpr int MINB = 2 * (SMEM + 1024) <= 102400 ? 2 : 1;
    static_assert(A_BYTES % 16 == 0 && STAGE % 16 == 0, "16-byte stages");
    static_assert(!LANES || (TPW % 2 == 0 && C::KS % 2 == 0 && K2 % 2 == 0 && K2 <= 12),
                  "lanes: tile pairs of 2-step groups, K2 4-12");
};

// the configuration list: pfdense_args.h (DSV41_PFD_CFGS)

// -- small device helpers -----------------------------------------------------------------------------------------
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool ok) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void bar_sync() { asm volatile("bar.sync 0;\n" ::: "memory"); }
__device__ __forceinline__ void ldsm4(uint32_t (&a)[4], const void* p) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(s));
}
// mma.m16n8k16, bf16 inputs, fp32 accumulators (the epilogue's Hadamard chain): d += a @ b
__device__ __forceinline__ void mma16816_bf16(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// chunk c of row r of a stage's xh rows (CPR chunks a row): an ldmatrix phase's 8 rows in 8 bank groups
template <int CPR>
__device__ __forceinline__ int swz(int r, int c) {
    if constexpr (CPR == 4) return c ^ ((r >> 1) & 3);
    else return c ^ (r & 7);
}
// the epilogue's [rows][128] bf16 planes: 16 chunks a row, the low 3 bits XORed with the row
__device__ __forceinline__ int esw(int r, int c) { return c ^ (r & 7); }

// -- decode: stage tiles into B fragments; strips / stored via decode.cuh's decode_lane (upstream unpack_kernel's) --
template <int K2>
__device__ __forceinline__ uint4 frag_strips(const uint32_t* tile, int lane) {
    uint32_t w[lane_words<K2>()];
    load_lane_words<K2>(tile, lane, w);
    uint32_t b0[2], b1[2];
    decode_lane<K2, CB_MUL1>(w, lane, b0, b1);
    return make_uint4(b0[0], b0[1], b1[0], b1[1]);
}

// lanes (dense3.cu): lane L's words at ((w / 4) * 32 + L) * 4 + w % 4; tiles j, j + 1 by 64-bit funnel shifts
template <int K2>
__device__ __forceinline__ void frag_lanes_pair(const uint32_t* grp, int gs, int j, int lane, uint4& f0, uint4& f1) {
    constexpr int n = 4 * K2, GW = 2 * K2;
    const int p = (gs * 8 + j) * n, wi = p >> 5, bo = p & 31;
    const int c0 = wi >> 2, off = wi & 3;
    const uint4 u = *reinterpret_cast<const uint4*>(grp + (c0 * 32 + lane) * 4);
    const uint4 v = c0 + 1 < GW / 4 ? *reinterpret_cast<const uint4*>(grp + ((c0 + 1) * 32 + lane) * 4)
                                     : make_uint4(0u, 0u, 0u, 0u);
    const uint32_t x0 = off == 0 ? u.x : off == 1 ? u.y : off == 2 ? u.z : u.w;
    const uint32_t x1 = off == 0 ? u.y : off == 1 ? u.z : off == 2 ? u.w : v.x;
    const uint32_t x2 = off == 0 ? u.z : off == 1 ? u.w : off == 2 ? v.x : v.y;
    const uint32_t x3 = off == 0 ? u.w : off == 1 ? v.x : off == 2 ? v.y : v.z;
    const uint64_t hi = ((uint64_t)x0 << 32) | x1, lo = ((uint64_t)x2 << 32) | x3;
    const uint64_t sh = bo ? ((hi << bo) | (lo >> (64 - bo))) : hi;
    const uint64_t sl = lo << bo;
    const uint64_t o0 = sh >> (64 - n);
    const uint64_t o1 = ((sh << n) | (sl >> (64 - n))) >> (64 - n);
    const uint32_t tails = (uint32_t)(o0 & 0xffffu) | ((uint32_t)(o1 & 0xffffu) << 16);
    const uint32_t pre = __shfl_sync(0xffffffffu, tails, (lane + 31) & 31);
    uint32_t b[2][4];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const uint64_t o = h ? o1 : o0;
        const uint64_t pp = h ? (pre >> 16) : (pre & 0xffffu);
        uint32_t s[8];
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            const int sft = (7 - q) * K2 / 2;                // the window of value 8 lane + q: bits [sft, sft + 16)
            uint64_t w = o >> sft;
            if (n - sft < 16) w |= pp << (n - sft);
            s[q] = (uint32_t)(w & 0xffffu);
        }
        b[h][0] = decode2<CB_MUL1>(s[0], s[1]);
        b[h][1] = decode2<CB_MUL1>(s[2], s[3]);
        b[h][2] = decode2<CB_MUL1>(s[4], s[5]);
        b[h][3] = decode2<CB_MUL1>(s[6], s[7]);
    }
    f0 = make_uint4(b[0][0], b[0][1], b[0][2], b[0][3]);
    f1 = make_uint4(b[1][0], b[1][1], b[1][2], b[1][3]);
}

// warp `warp` decodes tiles warp * TPW .. + TPW - 1 (tile kk * 8 + j of the stage) of the raw stage W into D
template <int K2, bool LANES, class C>
__device__ __forceinline__ void decode_stage(const uint32_t* W, uint4* D, int warp, int lane) {
    using L = Lay<K2, LANES, C>;
    if constexpr (!LANES) {
#pragma unroll
        for (int i = 0; i < L::TPW; ++i) {
            const int ti = warp * L::TPW + i;
            D[ti * 32 + lane] = frag_strips<K2>(W + ti * L::TW, lane);
        }
    } else {
#pragma unroll
        for (int i = 0; i < L::TPW; i += 2) {
            const int ti = warp * L::TPW + i, kk = ti >> 3, j = ti & 7;
            uint4 f0, f1;
            frag_lanes_pair<K2>(W + (kk >> 1) * (64 * K2), kk & 1, j, lane, f0, f1);
            D[ti * 32 + lane] = f0;
            D[(ti + 1) * 32 + lane] = f1;
        }
    }
}

// H's B fragment, k16 step kb, n8 tile nb (H[k][n] = (-1)^popc(k & n), bf16 +-1); P: this lane's pair at kb = nb = 0
__device__ __forceinline__ uint32_t had_base(int lane) {
    const int g = lane >> 2, t = lane & 3;
    const uint32_t lo = (__popc((2 * t) & g) & 1) ? 0xBF80u : 0x3F80u;
    const uint32_t hi = (__popc((2 * t + 1) & g) & 1) ? 0xBF80u : 0x3F80u;
    return lo | (hi << 16);
}
__device__ __forceinline__ void had_frag(uint32_t P, int kb, int nb, uint32_t (&b)[2]) {
    const uint32_t s = (__popc(kb & (nb >> 1)) & 1) ? 0x80008000u : 0u;
    b[0] = P ^ s;
    b[1] = b[0] ^ ((nb & 1) ? 0x80008000u : 0u);
}

// the finished pair of outputs (columns col, col + 1 of one row): Triton's y * SCALE * svh (+ bias as one fma)
__device__ __forceinline__ void store2(const Args& a, long long row, int col, float y0, float y1) {
    const half2 sv = *reinterpret_cast<const half2*>(static_cast<const half*>(a.svh) + col);
    const float s0 = __low2float(sv), s1 = __high2float(sv);
    float v0 = __fmul_rn(y0, HAD_SCALE), v1 = __fmul_rn(y1, HAD_SCALE);
    if (a.bias != nullptr) {
        const half2 bb = *reinterpret_cast<const half2*>(static_cast<const half*>(a.bias) + col);
        v0 = __fmaf_rn(v0, s0, __low2float(bb));
        v1 = __fmaf_rn(v1, s1, __high2float(bb));
    } else {
        v0 = __fmul_rn(v0, s0);
        v1 = __fmul_rn(v1, s1);
    }
    const long long at = row * a.o_stride + col;
    if (a.out_type == OUT_F32) {
        *reinterpret_cast<float2*>(static_cast<float*>(a.out) + at) = make_float2(v0, v1);
    } else if (a.out_type == OUT_BF16) {
        *reinterpret_cast<__nv_bfloat162*>(static_cast<__nv_bfloat16*>(a.out) + at) = __floats2bfloat162_rn(v0, v1);
    } else {
        *reinterpret_cast<half2*>(static_cast<half*>(a.out) + at) = __floats2half2_rn(v0, v1);
    }
}

// -- the kernel -----------------------------------------------------------------------------------------------------
template <int K2, bool LANES, class C>
__global__ void __launch_bounds__(C::THREADS, (Lay<K2, LANES, C>::MINB)) pfd_kernel(const Args a) {
    using L = Lay<K2, LANES, C>;
    constexpr int BM = C::BM, KS = C::KS, NST = C::NST, CPR = C::CPR, LDA = C::LDA, MT = C::MT, NT = C::NT;
    constexpr int TW = L::TW;
    extern __shared__ __align__(128) unsigned char smem[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int wm = warp / C::WN, wn = warp % C::WN;
    const int g = lane >> 2, t = lane & 3;
    const int M = a.M, K = a.K, NB = a.N >> 7, S = K / (16 * KS);

    // Triton _gemm's raster: GROUP row tiles a column sweep
    const int pid = blockIdx.x, nm = (M + BM - 1) / BM;
    const int per = a.group * NB, first = (pid / per) * a.group;
    const int rows = min(nm - first, a.group);
    const int pm = first + (pid % per) % rows, pn = (pid % per) / rows;
    const int m0 = pm * BM;
    const uint32_t* Tb = a.T + (size_t)pn * a.stride_nb;
    const half* xh = static_cast<const half*>(a.xh);
    unsigned char* bdec = smem + L::RING;

    // stage s's xh rows into ring slot s % NST (rows >= M: zero-filled, never stored)
    auto load_a = [&](int s) {
        half* A = reinterpret_cast<half*>(smem + (s % NST) * L::STAGE);
        const int k0 = s * KS * 16;
        for (int i = threadIdx.x; i < BM * CPR; i += C::THREADS) {
            const int r = i / CPR, c = i % CPR;
            const bool ok = m0 + r < M;
            cp_async16(A + r * LDA + swz<CPR>(r, c) * 8, xh + (size_t)(ok ? m0 + r : 0) * K + k0 + c * 8, ok);
        }
    };
    // stage s's words (KS k steps x 8 tiles, as stored) into ring slot `slot`
    auto load_w = [&](int s, int slot) {
        uint32_t* W = reinterpret_cast<uint32_t*>(smem + slot * L::STAGE + L::A_BYTES);
        for (int i = threadIdx.x; i < KS * 8 * K2; i += C::THREADS) {
            const int kk = i / (8 * K2), q = i % (8 * K2);
            cp_async16(W + kk * 8 * TW + q * 4, Tb + (size_t)(s * KS + kk) * a.stride_k + q * 4, true);
        }
    };
    auto words_of = [&](int slot) {
        return reinterpret_cast<const uint32_t*>(smem + slot * L::STAGE + L::A_BYTES);
    };
    auto bbuf = [&](int b) { return reinterpret_cast<uint4*>(bdec + b * L::BBUF); };

    float acc[MT][2 * NT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < 2 * NT; ++j)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;

    // prologue: W(0) in slot NST - 1 (free until iteration 0 refills it), then groups G_s = {A(s), W(s + 1)}
    load_w(0, NST - 1);
    cp_commit();
#pragma unroll
    for (int s = 0; s < NST - 1; ++s) {
        if (s < S) load_a(s);
        if (s + 1 < S) load_w(s + 1, s);
        cp_commit();
    }
    cp_wait<NST - 1>();
    bar_sync();
    decode_stage<K2, LANES, C>(words_of(NST - 1), bbuf(0), warp, lane);

#pragma unroll 1
    for (int s = 0; s < S; ++s) {
        cp_wait<NST - 2>();                                  // G_s landed: A(s), W(s + 1)
        bar_sync();                                          // ... for every thread; B(s) decoded; iteration s - 1 done
        {
            const int f = s + NST - 1;                       // refill the slot iteration s - 1 used
            if (f < S) load_a(f);
            if (f + 1 < S) load_w(f + 1, f % NST);
            cp_commit();
        }
        if (s + 1 < S) decode_stage<K2, LANES, C>(words_of(s % NST), bbuf((s + 1) & 1), warp, lane);
        const half* A = reinterpret_cast<const half*>(smem + (s % NST) * L::STAGE);
        const uint4* D = bbuf(s & 1);
#pragma unroll
        for (int kk = 0; kk < KS; ++kk) {
            uint32_t af[MT][4];
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                const int r = (wm * MT + mt) * 16 + (lane & 15);
                ldsm4(af[mt], A + r * LDA + swz<CPR>(r, 2 * kk + (lane >> 4)) * 8);
            }
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const uint4 bb = D[(kk * 8 + wn * NT + nt) * 32 + lane];
                const uint32_t b0[2] = {bb.x, bb.y}, b1[2] = {bb.z, bb.w};
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    mma16816(acc[mt][2 * nt], af[mt], b0);
                    mma16816(acc[mt][2 * nt + 1], af[mt], b1);
                }
            }
        }
    }
    cp_wait<0>();
    bar_sync();                                              // the ring and the B buffers become the epilogue's planes

    // epilogue, round wm: this warp row's rows. Earlier rounds: wait them out (2 barriers each)
    __nv_bfloat16* Et = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* Er = Et + C::EPR * 128;
    for (int r = 0; r < wm; ++r) {
        bar_sync();
        bar_sync();
    }
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int j = 0; j < 2 * NT; ++j) {
            const int col = 16 * (wn * NT + (j >> 1)) + 8 * (j & 1) + 2 * t;
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                const float c0 = acc[mt][j][2 * hh], c1 = acc[mt][j][2 * hh + 1];
                const __nv_bfloat162 top = __floats2bfloat162_rn(c0, c1);
                const __nv_bfloat162 rest = __floats2bfloat162_rn(__fsub_rn(c0, __low2float(top)),
                                                                  __fsub_rn(c1, __high2float(top)));
                const int row = mt * 16 + g + 8 * hh;
                const int at = row * 128 + esw(row, col >> 3) * 8 + (col & 7);
                *reinterpret_cast<__nv_bfloat162*>(Et + at) = top;
                *reinterpret_cast<__nv_bfloat162*>(Er + at) = rest;
            }
        }
    bar_sync();
    {
        const uint32_t P = had_base(lane);
        const int part = wn % C::NSPLIT;
#pragma unroll
        for (int gi = 0; gi < C::GPW; ++gi) {
            const int grp = (wn / C::NSPLIT) * C::GPW + gi;
            float y[C::EN8][4];
#pragma unroll
            for (int e = 0; e < C::EN8; ++e)
#pragma unroll
                for (int c = 0; c < 4; ++c) y[e][c] = 0.f;
#pragma unroll
            for (int q = 0; q < 16; ++q) {                   // top's k16 steps 0-7, then rest's 0-7: one chain
                const int kb = q & 7;
                const int row = grp * 16 + (lane & 15);
                uint32_t af[4];
                ldsm4(af, (q < 8 ? Et : Er) + row * 128 + esw(row, 2 * kb + (lane >> 4)) * 8);
#pragma unroll
                for (int e = 0; e < C::EN8; ++e) {
                    uint32_t hb[2];
                    had_frag(P, kb, part * C::EN8 + e, hb);
                    mma16816_bf16(y[e], af, hb);
                }
            }
#pragma unroll
            for (int e = 0; e < C::EN8; ++e) {
                const int col = pn * 128 + 8 * (part * C::EN8 + e) + 2 * t;
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    const int row = m0 + wm * C::EPR + grp * 16 + g + 8 * hh;
                    if (row < M) store2(a, row, col, y[e][2 * hh], y[e][2 * hh + 1]);
                }
            }
        }
    }
    bar_sync();
    for (int r = wm + 1; r < C::WM; ++r) {
        bar_sync();
        bar_sync();
    }
}

// Tests: W_q [K, N] fp16 read back through the kernel's decode; one CTA a (k step pair, 128-column block), 4 warps
template <int K2, bool LANES>
__global__ void __launch_bounds__(128) pfd_dequant_kernel(const uint32_t* __restrict__ T, long long stride_k,
                                                          long long stride_nb, half* __restrict__ out, int N) {
    using C = Cfg<64, 2, 2, 2, 2>;
    using L = Lay<K2, LANES, C>;
    __shared__ __align__(16) uint32_t W[2 * 8 * 4 * K2];
    __shared__ __align__(16) uint4 D[2 * 8 * 32];
    const int ks = blockIdx.x, pn = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int i = threadIdx.x; i < 2 * 8 * K2; i += 128) {
        const int kk = i / (8 * K2), q = i % (8 * K2);
        const uint4 v = *reinterpret_cast<const uint4*>(T + (size_t)pn * stride_nb + (size_t)(2 * ks + kk) * stride_k
                                                          + q * 4);
        *reinterpret_cast<uint4*>(W + kk * 8 * L::TW + q * 4) = v;
    }
    __syncthreads();
    decode_stage<K2, LANES, C>(W, D, warp, lane);
    __syncthreads();
    const int g = lane >> 2, t = lane & 3;
    for (int ti = warp; ti < 16; ti += 4) {
        const uint4 f = D[ti * 32 + lane];
        const uint32_t r[4] = {f.x, f.y, f.z, f.w};
        const int kk = ti >> 3, j = ti & 7;
#pragma unroll
        for (int q = 0; q < 4; ++q) {                       // b0[0], b0[1], b1[0], b1[1]
            const int k = (2 * ks + kk) * 16 + 2 * t + 8 * (q & 1), n = pn * 128 + j * 16 + g + 8 * (q >> 1);
            out[(size_t)k * N + n] = __ushort_as_half((unsigned short)(r[q] & 0xffffu));
            out[(size_t)(k + 1) * N + n] = __ushort_as_half((unsigned short)(r[q] >> 16));
        }
    }
}

// -- host side (CUDA runtime only; the bindings are pfdense.cpp) ---------------------------------------------------
template <int K2, bool LANES, class C>
inline void launch(const Args& a, cudaStream_t stream) {
    using L = Lay<K2, LANES, C>;
    auto kern = pfd_kernel<K2, LANES, C>;
    static bool ready = false;
    if (!ready) {
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, L::SMEM);
        ready = true;
    }
    const int nm = (a.M + C::BM - 1) / C::BM;
    kern<<<(unsigned)(nm * (a.N >> 7)), C::THREADS, L::SMEM, stream>>>(a);
}

template <int K2, bool LANES, class C>
inline int occupancy() {
    using L = Lay<K2, LANES, C>;
    auto kern = pfd_kernel<K2, LANES, C>;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, L::SMEM);
    int n = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kern, C::THREADS, L::SMEM);
    return n;
}

// [BM, WM, WN, KS, NST, threads, smem, blocks an SM] of configuration `cfg` at (K2, LANES); -1s for an unknown id
template <int K2, bool LANES>
inline void describe(int cfg, long long (&o)[8]) {
    for (auto& v : o) v = -1;
#define DSV41_PFD_DESC(ID, BM, WM, WN, KS, NST)                                                                    \
    if (cfg == ID) {                                                                                                \
        using C = Cfg<BM, WM, WN, KS, NST>;                                                                         \
        o[0] = BM; o[1] = WM; o[2] = WN; o[3] = KS; o[4] = NST; o[5] = C::THREADS;                                 \
        o[6] = Lay<K2, LANES, C>::SMEM; o[7] = occupancy<K2, LANES, C>();                                          \
    }
    DSV41_PFD_CFGS(DSV41_PFD_DESC)
#undef DSV41_PFD_DESC
}

template <int K2, bool LANES>
inline bool run(const Args& a, int cfg, cudaStream_t stream) {
#define DSV41_PFD_RUN(ID, BM, WM, WN, KS, NST)                                                                     \
    if (cfg == ID) {                                                                                                \
        launch<K2, LANES, Cfg<BM, WM, WN, KS, NST>>(a, stream);                                                    \
        return true;                                                                                                \
    }
    DSV41_PFD_CFGS(DSV41_PFD_RUN)
#undef DSV41_PFD_RUN
    return false;
}

template <int K2, bool LANES>
inline void dequant(const uint32_t* T, long long stride_k, long long stride_nb, void* out, int K, int N,
                    cudaStream_t stream) {
    pfd_dequant_kernel<K2, LANES><<<dim3((unsigned)(K / 32), (unsigned)(N / 128)), 128, 0, stream>>>(
        T, stride_k, stride_nb, static_cast<half*>(out), N);
}

}  // namespace dsv41_pfd

// every kernel of a width, explicitly (the sm_121 compile test builds the .cu files alone)
#define DSV41_PFD_INST(ID, BM, WM, WN, KS, NST) \
    template __global__ void dsv41_pfd::pfd_kernel<DSV41_PFD_K2, DSV41_PFD_LN, dsv41_pfd::Cfg<BM, WM, WN, KS, NST>>( \
        const dsv41_pfd::Args);

// the entry points of width K2 (pfdense_args.h): strips / stored always, lanes when LANES_OK
#define DSV41_PFD_ENTRY(K2_, RUN_LANES, DESC_LANES, DQ_LANES)                                                       \
    namespace dsv41_pfd {                                                                                           \
    bool run_k##K2_(const Args& a, bool lanes, int cfg, cudaStream_t s) {                                           \
        if (lanes) return RUN_LANES;                                                                                \
        return run<K2_, false>(a, cfg, s);                                                                          \
    }                                                                                                               \
    void describe_k##K2_(bool lanes, int cfg, long long (&o)[8]) {                                                  \
        if (lanes) { DESC_LANES; } else describe<K2_, false>(cfg, o);                                              \
    }                                                                                                               \
    bool dequant_k##K2_(bool lanes, const uint32_t* T, long long sk, long long snb, void* out, int K, int N,         \
                        cudaStream_t s) {                                                                           \
        if (lanes) return DQ_LANES;                                                                                 \
        dequant<K2_, false>(T, sk, snb, out, K, N, s);                                                              \
        return true;                                                                                                \
    }                                                                                                               \
    }
#define DSV41_PFD_WIDTH(K2_)                                                                                        \
    DSV41_PFD_ENTRY(K2_, run<K2_ DSV41_PFD_COMMA true>(a, cfg, s), describe<K2_ DSV41_PFD_COMMA true>(cfg, o),      \
                    (dequant<K2_ DSV41_PFD_COMMA true>(T, sk, snb, out, K, N, s), true))
#define DSV41_PFD_WIDTH_STRIPS(K2_)                                                                                 \
    DSV41_PFD_ENTRY(K2_, false, for (auto& v : o) v = -1, false)
#define DSV41_PFD_COMMA ,
