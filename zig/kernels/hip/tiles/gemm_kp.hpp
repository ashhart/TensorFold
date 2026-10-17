#pragma once

// Few-row prefill GEMM, a lane a column's group: outputs match affine_gemm_block bit for bit (same chains, fold order).

#include "common/dot2.hpp"
#include "tiles/gemm.hpp"
#include "tiles/plan.hpp"

namespace tf {
namespace rocm {

// The lanes a column takes, a group each: 32, or 16 or 8 when it has few groups (a wave then takes more columns).
__host__ __device__ inline int kp_lanes(int groups) { return groups > 16 ? 32 : groups > 8 ? 16 : 8; }

// CB columns a lane, R rows a pass, WAVES waves a block; passes span <= RB adjacent row blocks to share weights.
template <class Dec, class Act, class T, class Epi, int CB, int R, int WAVES, int RB, bool LOOP>
__device__ __forceinline__ void gemm_kp_tile(typename Dec::Args& a) {
    const int blocks = (a.m + R - 1) / R < RB ? (a.m + R - 1) / R : RB;  // of the launch: a routed item has its own rows
    if (!Dec::take_item(a, blockIdx.z)) return;
    constexpr int NO = R * CB;  // outputs of a lane set
    constexpr int PS = NO + 1;
    constexpr int OPL = (4 * NO + 31) / 32;  // outputs a lane folds, at most
    const int groups = a.k / a.group;
    const int gp = kp_lanes(groups);  // the lanes of a set; a wave has 32 / gp sets
    const int sets = 32 / gp;
    const int colw = ((blockIdx.x / blocks) * WAVES + (threadIdx.x >> 5)) * CB * sets;
    if (colw >= a.n) return;
    __shared__ float part[WAVES][32][PS];
    __shared__ float spart[WAVES][32][R + 1];
    __shared__ float scl[WAVES][32][CB + 1];
    __shared__ float bil[WAVES][32][CB + 1];

    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x >> 5;
    const int per = a.group / 32;
    const int rounds = (groups + gp - 1) / gp;
    const int gi = lane & (gp - 1);
    const int col0 = colw + (lane / gp) * CB;  // this lane's columns
    const long long words_row = Dec::template row_words<long long>(a);
    const uint32_t* wsrc[CB];
    long long tb[CB];
#pragma unroll
    for (int c = 0; c < CB; ++c) {
        const int col = col0 + c < a.n ? col0 + c : a.n - 1;
        wsrc[c] = Dec::words(a) + col * words_row;
        tb[c] = static_cast<long long>(col) * groups;
    }
    auto pass = [&](int row0) {
        const typename Act::elem* xr[R];
#pragma unroll
        for (int r = 0; r < R; ++r) xr[r] = Act::row(a, row0 + r < a.m ? row0 + r : 0);
        float acc[OPL];
#pragma unroll
        for (int t = 0; t < OPL; ++t) acc[t] = 0.f;

        // Stage s of group g's code words for every column, plus its scale and bias (groups past the last repeat it).
        uint32_t w[CB][Dec::kWords];
        uint32_t wn[CB][Dec::kWords];
        typename Dec::Term term[CB];
        auto fetch = [&](uint32_t (&dst)[CB][Dec::kWords], int g, int s) {
            const int gl = g < groups ? g : groups - 1;
#pragma unroll
            for (int c = 0; c < CB; ++c) Dec::template load<true>(wsrc[c] + (static_cast<long long>(gl) * per + s) * Dec::kWords, dst[c]);
        };
        auto tables = [&](int g) {
            const int gl = g < groups ? g : groups - 1;
#pragma unroll
            for (int c = 0; c < CB; ++c) term[c] = Dec::term(a, tb[c] + gl);
        };
        fetch(w, gi, 0);
        tables(gi);
        for (int round = 0; round < rounds; ++round) {
            const int g = round * gp + gi;
            const int gl = g < groups ? g : groups - 1;
            float d[R][CB];
            float sx[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                sx[r] = 0.f;
#pragma unroll
                for (int c = 0; c < CB; ++c) d[r][c] = 0.f;
            }
            for (int s = 0; s < per; ++s) {
                // the next stage's loads (the next round's first) go out before this stage's dots
                if (s + 1 < per) fetch(wn, g, s + 1);
                else if (round + 1 < rounds) fetch(wn, g + gp, 0);
                typename T::pair wd[CB][16];
#pragma unroll
                for (int c = 0; c < CB; ++c) {
#pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        wd[c][i] = Dec::template pair<T>(w[c], 2 * i);
                    }
                }
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const u32x4* src = reinterpret_cast<const u32x4*>(xr[r] + static_cast<long long>(gl) * a.group + s * 32);
                    uint32_t xn[16];
#pragma unroll
                    for (int v = 0; v < 4; ++v) {
                        const u32x4 u = src[v];
                        xn[4 * v] = u.x;
                        xn[4 * v + 1] = u.y;
                        xn[4 * v + 2] = u.z;
                        xn[4 * v + 3] = u.w;
                    }
                    float run = sx[r];
#pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        const typename T::pair xp = __builtin_bit_cast(typename T::pair, xn[i]);
                        run += T::lo(xp);
                        run += T::hi(xp);
                    }
                    sx[r] = run;
#pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        const typename T::pair xp = __builtin_bit_cast(typename T::pair, xn[i]);
#pragma unroll
                        for (int c = 0; c < CB; ++c) d[r][c] = T::dot(xp, wd[c][i], d[r][c]);
                    }
                }
#pragma unroll
                for (int c = 0; c < CB; ++c) {
#pragma unroll
                    for (int i = 0; i < Dec::kWords; ++i) w[c][i] = wn[c][i];
                }
            }
            // this group's results to LDS, then the round's groups in order
#pragma unroll
            for (int r = 0; r < R; ++r) {
                spart[wave][lane][r] = sx[r];
#pragma unroll
                for (int c = 0; c < CB; ++c) part[wave][lane][r * CB + c] = d[r][c];
            }
#pragma unroll
            for (int c = 0; c < CB; ++c) {
                scl[wave][lane][c] = Dec::scale_value(a, term[c]);
                bil[wave][lane][c] = Dec::bias_value(a, term[c]);
            }
            if (round + 1 < rounds) tables(g + gp);
            __syncwarp();
            const int count = groups - round * gp < gp ? groups - round * gp : gp;
#pragma unroll
            for (int t = 0; t < OPL; ++t) {
                const int os = lane + 32 * t;
                if (os >= NO * sets) continue;
                const int base = (os / NO) * gp;  // the first lane of the output's set
                const int o = os % NO;
                const int r = o / CB;
                const int c = o % CB;
                float v = acc[t];
                for (int gg = 0; gg < count; ++gg) {
                    v = Dec::fold_scale(v, part[wave][base + gg][o], scl[wave][base + gg][c]);
                    v = Dec::fold_bias(v, spart[wave][base + gg][r], bil[wave][base + gg][c]);
                }
                acc[t] = v;
            }
            __syncwarp();
        }
#pragma unroll
        for (int t = 0; t < OPL; ++t) {
            const int os = lane + 32 * t;
            if (os >= NO * sets) continue;
            const int o = os % NO;
            const int row = row0 + o / CB;
            const int col = colw + (os / NO) * CB + o % CB;
            if (row < a.m && col < a.n) Epi::store(a, row, col, acc[t]);
        }
    };
    if constexpr (LOOP) {
        for (int row0 = (blockIdx.x % blocks) * R; row0 < a.m; row0 += blocks * R) pass(row0);
    } else {
        pass((blockIdx.x % blocks) * R);
    }
}

}  // namespace rocm
}  // namespace tf
