#pragma once

// Streaming decode tile for 1 to 8 rows: a lane owns a 32-code chunk of a column; no LDS, lanes fold by shuffle.

#include <type_traits>
#include <utility>

#include "common/dot2.hpp"
#include "common/vec.hpp"
#include "tiles/plan.hpp"

namespace tf {
namespace rocm {

constexpr int kStreamWaves = 4;

// Tile shapes that exist: CB columns a lane times R rows at most 16 accumulators.
constexpr bool stream_shape(int rows, int cb) { return cb * rows <= 16; }
constexpr int kStreamRows = 16;  // the most rows a launch takes

// The pair (half `parity` of a, half `parity` of b).
__device__ inline uint32_t stream_pick(uint32_t a, uint32_t b, int parity) {
    return __builtin_amdgcn_perm(b, a, parity ? 0x07060302u : 0x05040100u);
}

// x of one row in the order of the code pairs: pair I takes the halves of the codes it is multiplied with.
template <class Dec, typename T, int... I>
__device__ inline void stream_x_pairs(const uint32_t (&xn)[16], typename T::pair (&xp)[16],
                                      std::integer_sequence<int, I...>) {
    (([&] {
         constexpr int lo = Dec::template lo<I>();
         constexpr int hi = lo + Dec::gap();
         if constexpr (Dec::kGapPairs) {
             xp[I] = __builtin_bit_cast(typename T::pair, stream_pick(xn[lo >> 1], xn[hi >> 1], lo & 1));
         } else {
             xp[I] = __builtin_bit_cast(typename T::pair, xn[I]);
         }
     }()),
     ...);
}

// One column's chunk against RG rows: each code pair is made once and dotted with every row's x.
template <class Dec, typename T, int RG, int... I>
__device__ inline void stream_dots(const typename T::pair (&xp)[RG][16], const uint32_t (&w)[Dec::kWords], float (&d)[RG],
                                   std::integer_sequence<int, I...>) {
    (([&] {
         const typename T::pair q = Dec::template stream_pair<T, I>(w);
#pragma unroll
         for (int r = 0; r < RG; ++r) d[r] = T::dot(xp[r][I], q, d[r]);
     }()),
     ...);
}

// Epi::kPair: a lane's CB columns are CB / 2 gate columns and the same up columns; out is silu(gate) * up.
template <class Dec, class Act, class T, class Epi, int R, int CB, bool WIDE>
__device__ inline void stream_body(const typename Dec::Args& a, int lpc_log2, int gshift, int bx, int pair_cols, float limit) {
    using pair = typename T::pair;
    constexpr bool PAIR = Epi::kPair;
    constexpr int RG = R < 4 ? R : 4;  // rows a pass keeps in registers
    constexpr int NG = R / RG;
    constexpr uint32_t kOne = T::ones;
    const int row0 = blockIdx.y * R;  // a block's rows: more than 8 rows are 8-row blocks on the grid's y
    const int lpc = 1 << lpc_log2;
    const int lane = threadIdx.x & 31;
    const int j = lane & (lpc - 1);
    const int slot = lane >> lpc_log2;
    constexpr int CH = PAIR ? CB / 2 : CB;  // output columns a lane
    const int cols = PAIR ? pair_cols : a.n;
    const int per_wave = (32 >> lpc_log2) * CH;
    const int col0 = (bx * kStreamWaves + (threadIdx.x >> 5)) * per_wave + slot * CH;
    const int nch = a.k >> 5;
    const long long words_row = Dec::template row_words<long long>(a);
    const long long groups = a.k / a.group;
    const uint32_t* wp[CB];
    long long sb[CB];
#pragma unroll
    for (int c = 0; c < CB; ++c) {
        const int cj = PAIR ? c % CH : c;
        const long long col = (col0 + cj < cols ? col0 + cj : cols - 1) + (PAIR && c >= CH ? pair_cols : 0);
        wp[c] = Dec::words(a) + col * words_row;
        sb[c] = col * groups;
    }
    const typename Act::elem* xr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) xr[r] = Act::row(a, row0 + r < a.m ? row0 + r : 0);
    float acc[R][CB];
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int c = 0; c < CB; ++c) acc[r][c] = 0.f;
    }

    const typename Dec::StreamTerms terms(a);

    // The loads of one round: every column's chunk and its scale and bias, none waited for before the dots.
    auto fetch = [&](int base, uint32_t (&w)[CB][Dec::kWords], unsigned short (&sr)[CB], unsigned short (&br)[CB]) {
        const int q = base + j < nch ? base + j : nch - 1;
#pragma unroll
        for (int c = 0; c < CB; ++c) {
            Dec::template load<WIDE>(wp[c] + static_cast<long long>(q) * Dec::kWords, w[c]);
            const long long at = sb[c] + (q >> gshift);
            sr[c] = terms.scale_raw(at);
            br[c] = terms.bias_raw(at);
        }
    };
    // The dots of one round. A lane past the row's last chunk read the last one: its scale and bias are zero.
    auto work = [&](int base, const uint32_t (&w)[CB][Dec::kWords], const unsigned short (&sr)[CB],
                    const unsigned short (&br)[CB]) {
        const bool live = base + j < nch;
        const int q = live ? base + j : nch - 1;
        float sc[CB];
        float bi[CB];
#pragma unroll
        for (int c = 0; c < CB; ++c) {
            sc[c] = live ? terms.widen(sr[c]) : 0.f;
            bi[c] = live ? terms.widen(br[c]) : 0.f;
        }
#pragma unroll
        for (int g = 0; g < NG; ++g) {
            if (row0 + g * RG >= a.m) break;
            pair xp[RG][16];
            float sx[RG];
#pragma unroll
            for (int r = 0; r < RG; ++r) {
                uint32_t xn[16];
                const u32x4* src =
                    reinterpret_cast<const u32x4*>(xr[g * RG + r] + static_cast<long long>(q) * 32);
#pragma unroll
                for (int v = 0; v < 4; ++v) {
                    const u32x4 t = src[v];
                    xn[4 * v] = t.x;
                    xn[4 * v + 1] = t.y;
                    xn[4 * v + 2] = t.z;
                    xn[4 * v + 3] = t.w;
                }
                float s = 0.f;
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    s = T::dot(__builtin_bit_cast(pair, xn[i]), __builtin_bit_cast(pair, kOne), s);
                }
                sx[r] = s;
                stream_x_pairs<Dec, T>(xn, xp[r], std::make_integer_sequence<int, 16>{});
            }
#pragma unroll
            for (int c = 0; c < CB; ++c) {
                float d[RG];
#pragma unroll
                for (int r = 0; r < RG; ++r) d[r] = 0.f;
                stream_dots<Dec, T, RG>(xp, w[c], d, std::make_integer_sequence<int, 16>{});
#pragma unroll
                for (int r = 0; r < RG; ++r) {
                    acc[g * RG + r][c] = Dec::fold_scale(acc[g * RG + r][c], d[r], sc[c]);
                    acc[g * RG + r][c] = Dec::fold_bias(acc[g * RG + r][c], sx[r], bi[c]);
                }
            }
        }
    };

    // Columns stagger their start round by group of 8 to spread memory channels; a column's sum order ignores shape.
    const int rounds = (nch + lpc - 1) >> lpc_log2;
    const int rot = rounds > 1 ? (col0 >> 3) % rounds : 0;
    auto at = [&](int r) {
        const int ri = r + rot >= rounds ? r + rot - rounds : r + rot;
        return ri << lpc_log2;
    };
    uint32_t wa[CB][Dec::kWords];
    unsigned short sa[CB], ba[CB];
    if constexpr (R == 1) {
        uint32_t wb[CB][Dec::kWords];
        unsigned short sbb[CB], bbb[CB];
        fetch(at(0), wa, sa, ba);
        for (int r = 0; r < rounds; r += 2) {
            if (r + 1 < rounds) fetch(at(r + 1), wb, sbb, bbb);
            work(at(r), wa, sa, ba);
            if (r + 1 >= rounds) break;
            if (r + 2 < rounds) fetch(at(r + 2), wa, sa, ba);
            work(at(r + 1), wb, sbb, bbb);
        }
    } else {
        // several rows: the dots cover the loads' latency, and registers go to the rows' x
        for (int r = 0; r < rounds; ++r) {
            fetch(at(r), wa, sa, ba);
            work(at(r), wa, sa, ba);
        }
    }

#pragma unroll
    for (int r = 0; r < R; ++r) {
        float v[CB];
#pragma unroll
        for (int c = 0; c < CB; ++c) {
            v[c] = acc[r][c];
            for (int d = lpc >> 1; d > 0; d >>= 1) v[c] += __shfl_xor(v[c], d, 32);
        }
        if (j != 0 || row0 + r >= a.m) continue;
#pragma unroll
        for (int c = 0; c < CH; ++c) {
            const int col = col0 + c;
            if (col >= cols) continue;
            Epi::put(a, out_row(a, row0 + r), col, v, c, pair_cols, limit);
        }
    }
}

// Item z of the plan (or plain product, or a group's side) at CB columns a lane, R rows; chunk group is q >> gshift.
template <class Dec, class Act, class T, class Epi, int R, int CB, class Sides>
__device__ inline void stream_tile(typename Dec::Args& a, const Sides& sides, int lpc_log2, int gshift) {
    int bx = blockIdx.x;
    if (sides.count > 0) {
        int s = 0;
        while (s + 1 < sides.count && bx >= sides.first[s + 1]) ++s;
        bx -= sides.first[s];
        Dec::bind(a, sides, s);
        a.n = sides.n[s];
        if (sides.out_half) {
            a.out16 = static_cast<__half*>(sides.out[s]);
        } else {
            a.out = static_cast<float*>(sides.out[s]);
        }
    }
    if (!Dec::take_item(a, blockIdx.z) || static_cast<int>(blockIdx.y) * R >= a.m) return;
    if constexpr (Dec::kWideBytes == 4) {
        stream_body<Dec, Act, T, Epi, R, CB, false>(a, lpc_log2, gshift, bx, sides.pair_cols, sides.limit);
    } else {
        // every chunk is wide-aligned when the words and the row stride are
        if (Dec::wide_aligned(a)) {
            stream_body<Dec, Act, T, Epi, R, CB, true>(a, lpc_log2, gshift, bx, sides.pair_cols, sides.limit);
        } else {
            stream_body<Dec, Act, T, Epi, R, CB, false>(a, lpc_log2, gshift, bx, sides.pair_cols, sides.limit);
        }
    }
}

}  // namespace rocm
}  // namespace tf
