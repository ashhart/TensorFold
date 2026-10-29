#pragma once

// gfx11 WMMA prefill GEMM (BF16, m >= 16) over dequantized weights: LDS holds bf16(code * scale + bias) instead of
// the codes, so the matrix core sums the whole K extent in fp32 and no group's scale or bias is folded by hand.
// Its sums differ from wmma_gemm_tile's: every weight rounds to bf16 once, over the layer's table values.

#include "common/wmma.hpp"
#include "tiles/epilogue.hpp"
#include "tiles/gemm.hpp"
#include "tiles/plan.hpp"

namespace tf {
namespace rocm {

// Two neighbouring codes of a column as the BF16 weights they stand for: nearest-even of code * scale + bias.
template <class T>
__device__ __forceinline__ uint32_t dq_pair(typename T::pair p, float scale, float bias) {
    const float lo = fmaf(T::lo(p), scale, bias);
    const float hi = fmaf(T::hi(p), scale, bias);
    const bf16x2 w{narrow<T>(lo), narrow<T>(hi)};
    return __builtin_bit_cast(uint32_t, w);
}

// Dec reads the words and group terms, Act the rows of x, T (a matrix-core Dot) multiplies and Epi stores.
template <class Dec, class Act, class T, class Epi>
__device__ __forceinline__ void wmma_gemm_dq_tile(typename Dec::Args& a) {
#if TF_DEVICE_WMMA_GFX11
    if (!Dec::take_item(a, blockIdx.z)) return;
    const int bx = blockIdx.x;
    const int m0 = blockIdx.y * kGemmM;
    if (m0 >= a.m) return;
    constexpr int kLdw = kGemmLd / 2;
    __shared__ __attribute__((aligned(16))) uint32_t xs[2][kGemmM * kLdw];
    __shared__ __attribute__((aligned(16))) uint32_t ws[2][kGemmN * kLdw];

    const int tid = threadIdx.x;
    const int n0 = bx * kGemmN;
    const int words_row = Dec::template row_words<int>(a);
    const int groups = a.k / a.group;
    const int stages = a.k / kGemmK;
    const int per = a.group / kGemmK;
    const int rows_left = a.m - m0;
    const int line = tid & (kGemmN - 1);
    const int side = __builtin_amdgcn_readfirstlane(tid >> 7);
    const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
    const int lane = tid & 31;

    // Staging as in gemm_tile: half a column's codes and half a row of x a thread, two stages a fetch.
    const int col_s = n0 + line < a.n ? n0 + line : a.n - 1;
    const uint32_t* wsrc = Dec::words(a) + static_cast<long long>(col_s) * words_row;
    const int xrow = tid >> 1;
    const int xhalf = tid & 1;
    const bool xlive = wave * 16 < rows_left;
    const int row_s = m0 + xrow < a.m ? m0 + xrow : 0;
    const typename Act::elem* xrow_ptr = Act::row(a, row_s);
    const u32x4* xsrc = reinterpret_cast<const u32x4*>(xrow_ptr);
    uint32_t wreg[2][Dec::kWords];
    u32x4 xreg[4];
    typename Dec::Term term[2];
    auto fetch = [&](int st0) {
        const int st1 = st0 + 1 < stages ? st0 + 1 : stages - 1;
        const int st = st0 < stages ? st0 : stages - 1;
        Dec::template load<true>(wsrc + st * Dec::kWords, wreg[0]);
        Dec::template load<true>(wsrc + st1 * Dec::kWords, wreg[1]);
        const int sx_ = xhalf ? st1 : st;
        if (xlive) {
#pragma unroll
            for (int v = 0; v < 4; ++v) xreg[v] = xsrc[sx_ * 4 + v];
        }
        const long long base = static_cast<long long>(col_s) * groups;
        term[0] = Dec::term(a, base + st / per);
        term[1] = Dec::term(a, base + st1 / per);
    };
    // Stage st (parity J) from `buf`: this thread's 16 weights of its column, and its row of x if any.
    auto stash = [&]<int J>(std::integral_constant<int, J>, int st, int buf) {
        const float scale = Dec::scale_value(a, term[J]);
        const float bias = Dec::bias_value(a, term[J]);
        u32x4* dst = reinterpret_cast<u32x4*>(ws[buf] + line * kLdw + side * 8);
        auto decode = [&]<int S>(std::integral_constant<int, S>) {
#pragma unroll
            for (int v = 0; v < 2; ++v) {
                uint32_t d[4];
#pragma unroll
                for (int h = 0; h < 4; ++h) {
                    const int t = S * 16 + v * 8 + h * 2;
                    d[h] = dq_pair<T>(Dec::template pair<T>(wreg[J], t), scale, bias);
                }
                dst[v] = u32x4{d[0], d[1], d[2], d[3]};
            }
        };
        if (side == 0) decode(std::integral_constant<int, 0>{}); else decode(std::integral_constant<int, 1>{});
        if (xhalf == J && xlive) {
            u32x4* xdst = reinterpret_cast<u32x4*>(xs[buf] + xrow * kLdw);
#pragma unroll
            for (int v = 0; v < 4; ++v) xdst[v] = xreg[v];
        }
    };

    // Wave (wm, wn) owns rows 32 wm.., columns 64 wn.. (2 x 4 tiles); a lane: column lane % 16, rows 2 i + lane / 16.
    const int wm = wave & 3;
    const int wn = wave >> 2;
    const int rl = lane & 15;
    const int rh = lane >> 4;
    const bool live0 = wm * 32 < rows_left;
    const bool live1 = wm * 32 + 16 < rows_left;
    f32x8 acc[2][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) acc[mt][nt] = f32x8{0, 0, 0, 0, 0, 0, 0, 0};
    }
    fetch(0);
    stash(std::integral_constant<int, 0>{}, 0, 0);
    __syncthreads();
    auto step = [&]<int P>(std::integral_constant<int, P>, int s) {
        constexpr int cur = P;
        if constexpr (P == 1) fetch(s + 1);
        if (live0) {
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                bf16x16 bf[4];
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) bf[nt] = frag16(ws[cur] + (wn * 64 + nt * 16 + rl) * kLdw + t * 8);
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    if (mt == 1 && !live1) continue;
                    const bf16x16 af = frag16(xs[cur] + (wm * 32 + mt * 16 + rl) * kLdw + t * 8);
#pragma unroll
                    for (int nt = 0; nt < 4; ++nt) acc[mt][nt] = T::mma(af, bf[nt], acc[mt][nt]);
                }
            }
        }
        stash(std::integral_constant<int, P ^ 1>{}, s + 1, cur ^ 1);
        __syncthreads();
    };
    for (int s = 0; s < stages; s += 2) {
        step(std::integral_constant<int, 0>{}, s);
        if (s + 1 < stages) step(std::integral_constant<int, 1>{}, s + 1);
    }
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = m0 + wm * 32 + mt * 16 + 2 * i + rh;
            if (row >= a.m) continue;
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const int col = n0 + wn * 64 + nt * 16 + rl;
                if (col < a.n) Epi::store(a, row, col, acc[mt][nt][i]);
            }
        }
    }
#else
    (void)a;
    __builtin_trap();
#endif
}

}  // namespace rocm
}  // namespace tf
