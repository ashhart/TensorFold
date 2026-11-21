#pragma once

// gfx11 WMMA prefill GEMM for a routed item of few rows: RM x 16 rows a block, all 8 waves split the 128 columns.

#include "common/wmma.hpp"
#include "tiles/gemm.hpp"
#include "tiles/gemm_kp.hpp"
#include "tiles/plan.hpp"

namespace tf {
namespace rocm {

// The most rows a launch's cut sends to K-parallel passes, the rows a pass takes and the columns a lane takes in it.
constexpr int kItemFewRows = 8;
constexpr int kItemFewPass = 2;
constexpr int kItemFewCols = 2;

// Same staging, stage order and per-element sums as wmma_gemm_tile, so each output equals that tile's byte for byte.
template <class Dec, class Act, class T, class Epi, int RM, class Few>
__device__ __forceinline__ void wmma_item_tile(typename Dec::Args& a) {
#if TF_DEVICE_WMMA_GFX11
    if (!Dec::take_item(a, blockIdx.z)) return;
    constexpr int kRows = RM * 16;
    const int bx = blockIdx.x;
    const int m0 = blockIdx.y * kRows;
    if (m0 >= a.m) return;
    constexpr int kLdw = kGemmLd / 2;
    __shared__ __attribute__((aligned(16))) uint32_t xs[2][kRows * kLdw];
    __shared__ __attribute__((aligned(16))) uint32_t ws[2][kGemmN * kLdw];
    __shared__ float sx[2][kRows];
    __shared__ float sc[3][kGemmN];
    __shared__ float bi[3][kGemmN];

    const int tid = threadIdx.x;
    const int n0 = bx * kGemmN;

    // An item within the launch's cut takes K-parallel passes with Few's Dot and epilogue: same bytes, no matrix core.
    if (a.m <= (a.route.few < kItemFewRows ? a.route.few : kItemFewRows)) {
        constexpr int CB = kItemFewCols;
        constexpr int R = kItemFewPass;
        // a pass's partials, a row a lane: each wave's own slice of the two code stages, which this path leaves unused
        constexpr int kLane = (R * CB + 1) + (R + 1) + 2 * (CB + 1);
        static_assert(8 * 32 * kLane <= 2 * kGemmN * kLdw, "the few-row passes' partials fit the tile's code stages");
        const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
        const auto lds = [wave](int lane) {
            float* const row = reinterpret_cast<float*>(&ws[0][0]) + (wave * 32 + lane) * kLane;
            return KpRows{row, row + R * CB + 1, row + R * CB + 1 + R + 1, row + R * CB + 1 + R + 1 + CB + 1};
        };
        // The 8 waves walk the block's columns side by side, CB a lane set a pass, R rows of the item a pass.
        const int span = CB * (32 / kp_lanes(a.k / a.group));
        const int end = n0 + kGemmN < a.n ? n0 + kGemmN : a.n;
        for (int colw = n0 + wave * span; colw < end; colw += 8 * span) {
            gemm_kp_wave<Dec, Act, typename Few::Dot, typename Few::Epi, CB, R, true>(a, colw, 0, R, lds);
            if constexpr (!Few::every_pass) break;  // the contract's control stops after a wave's first
        }
        return;
    }
    const int words_row = Dec::template row_words<int>(a);
    const int groups = a.k / a.group;
    const int stages = a.k / kGemmK;
    const int per = a.group / kGemmK;
    const int rows_left = a.m - m0;
    const int line = tid & (kGemmN - 1);
    const int side = __builtin_amdgcn_readfirstlane(tid >> 7);
    const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
    const int lane = tid & 31;

    // Weights as in wmma_gemm_tile; x is staged by the first 2 * kRows threads, two stages a fetch.
    const int col_s = n0 + line < a.n ? n0 + line : a.n - 1;
    const uint32_t* wsrc = Dec::words(a) + static_cast<long long>(col_s) * words_row;
    const int xrow = tid >> 1;
    const int xhalf = tid & 1;
    const bool xlive = xrow < kRows && (xrow & ~15) < rows_left;
    const bool sums = side == 1 && line < kRows;
    const int row_s = xrow < kRows && m0 + xrow < a.m ? m0 + xrow : 0;
    const typename Act::elem* xrow_ptr = Act::row(a, row_s);
    const u32x4* xsrc = reinterpret_cast<const u32x4*>(xrow_ptr);
    uint32_t wreg[2][Dec::kWords];
    u32x4 xreg[4];
    typename Dec::Term term[2];
    float run = 0.f;
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
    auto stash = [&]<int J>(std::integral_constant<int, J>, int st, int buf) {
        u32x4* dst = reinterpret_cast<u32x4*>(ws[buf] + line * kLdw + side * 8);
        auto decode = [&]<int S>(std::integral_constant<int, S>) {
#pragma unroll
            for (int v = 0; v < 2; ++v) {
                uint32_t d[4];
#pragma unroll
                for (int h = 0; h < 4; ++h) {
                    const int t = S * 16 + v * 8 + h * 2;
                    d[h] = __builtin_bit_cast(uint32_t, Dec::template pair<T>(wreg[J], t));
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
        if (side == 0 && st % per == per - 1) {
            sc[(st / per) % 3][line] = Dec::scale_value(a, term[J]);
            bi[(st / per) % 3][line] = Dec::bias_value(a, term[J]);
        }
    };
    auto sumx = [&](int st, int buf) {
        if (st % per == 0) run = 0.f;
        const u32x4* src = reinterpret_cast<const u32x4*>(xs[buf] + line * kLdw);
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            const u32x4 u = src[v];
            const uint32_t q[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
            for (int h = 0; h < 4; ++h) {
                const typename T::pair p = __builtin_bit_cast(typename T::pair, q[h]);
                run += T::lo(p);
                run += T::hi(p);
            }
        }
        if (st % per == per - 1) sx[(st / per) & 1][line] = run;
    };

    // Wave w owns columns 16 w..; a lane: column lane % 16, rows 2 i + lane / 16 of each of the RM row tiles.
    const int rl = lane & 15;
    const int rh = lane >> 4;
    const int cw = wave * 16;
    f32x8 acc[RM];
    f32x8 d[RM];
#pragma unroll
    for (int mt = 0; mt < RM; ++mt) {
        acc[mt] = f32x8{0, 0, 0, 0, 0, 0, 0, 0};
        d[mt] = acc[mt];
    }
    const f32x8 zero = {0, 0, 0, 0, 0, 0, 0, 0};
    auto apply_bias = [&](int slot, int g) {
        const float bias = bi[slot][cw + rl];
#pragma unroll
        for (int mt = 0; mt < RM; ++mt) {
            if (mt * 16 >= rows_left) continue;
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[mt][i] = Dec::fold_bias(acc[mt][i], sx[g][mt * 16 + 2 * i + rh], bias);
        }
    };
    fetch(0);
    stash(std::integral_constant<int, 0>{}, 0, 0);
    __syncthreads();
    auto step = [&]<int P>(std::integral_constant<int, P>, int s) {
        constexpr int cur = P;
        if (s > 0 && (s - 1) % per == per - 1) apply_bias(((s - 1) / per) % 3, ((s - 1) / per) & 1);
        if constexpr (P == 1) fetch(s + 1);
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            const bf16x16 bf = frag16(ws[cur] + (cw + rl) * kLdw + t * 8);
            const bool first = t == 0 && s % per == 0;
#pragma unroll
            for (int mt = 0; mt < RM; ++mt) {
                if (mt * 16 >= rows_left) continue;
                const bf16x16 af = frag16(xs[cur] + (mt * 16 + rl) * kLdw + t * 8);
                d[mt] = T::mma(af, bf, first ? zero : d[mt]);
            }
        }
        if (s % per == per - 1) {
            const float scale = sc[(s / per) % 3][cw + rl];
#pragma unroll
            for (int mt = 0; mt < RM; ++mt) {
                if (mt * 16 >= rows_left) continue;
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[mt][i] = Dec::fold_scale(acc[mt][i], d[mt][i], scale);
            }
        }
        if (sums) sumx(s, cur);
        stash(std::integral_constant<int, P ^ 1>{}, s + 1, cur ^ 1);
        __syncthreads();
    };
    for (int s = 0; s < stages; s += 2) {
        step(std::integral_constant<int, 0>{}, s);
        if (s + 1 < stages) step(std::integral_constant<int, 1>{}, s + 1);
    }
    if ((stages - 1) % per == per - 1) apply_bias(((stages - 1) / per) % 3, ((stages - 1) / per) & 1);
    const int col = n0 + cw + rl;
    if (col >= a.n) return;
#pragma unroll
    for (int mt = 0; mt < RM; ++mt) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = m0 + mt * 16 + 2 * i + rh;
            if (row < a.m) Epi::store(a, row, col, acc[mt][i]);
        }
    }
#else
    // Never launched off gfx11; the lanes' own id keeps the stub a kernel the limits records can name.
    if (threadIdx.x < 100u) __builtin_trap();
#endif
}

}  // namespace rocm
}  // namespace tf
