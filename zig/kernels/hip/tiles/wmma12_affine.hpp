#pragma once

// gfx12 WMMA prefill GEMM over affine (MLX) weights: y = x . W^T for BF16 x and BITS-bit codes with a scale and a bias
// a group. Codes are below 256, exact in BF16, so a group's WMMA sums take the codes as they are; at the group's end
// each output folds them with the group's scale, then the row's sum of x with its bias (Dec::fold_scale, fold_bias).
// The block stages x as it is and the codes decoded to BF16, in the same double-buffered LDS as the dense tile
// (tiles/wmma12_step.hpp). A K step is 32 codes, the smallest group, so a step never crosses a group.
// A row's sum of x is the dot2 tiles' chain, element by element in k order (tiles/matrix_gemm.hpp does the same), and
// an output's groups fold in k order, so a row's bytes depend on neither m nor the block. They are not the dot2 tiles'
// bytes: the WMMA group sum is one fp32 ULP off the dot2 chain on some outputs (registry.Family.matrix12).

#include <type_traits>

#include "common/vec.hpp"
#include "common/wmma.hpp"
#include "tiles/plan.hpp"
#include "tiles/wmma12_gemm.hpp"
#include "tiles/wmma12_step.hpp"

namespace tf {
namespace rocm {

// 128 x 128 outputs, 2 x 4 waves of 64 x 32, K steps of 32 padded to 40 halves: 40 KiB of staged rows.
using W12Affine = W12Geo<128, 128, 32, 2, 4>;
// The blocks of few rows (a routed item's, or a short product): 64 rows on 2 x 4 waves, and 32 rows on 1 x 4 waves
// whose threads each decode a whole column. A row's bytes are the same on all three.
using W12Affine64 = W12Geo<64, 128, 32, 2, 4>;
using W12Affine32 = W12Geo<32, 128, 32, 1, 4>;

// Words [LO, LO + N) of a column's 32-code chunk of BITS-bit codes, MLX's packing: as wide as their offsets allow
// (a chunk starts BITS words into a column whose rows are whole chunks, so an even BITS keeps 8-byte offsets even).
template <int BITS, int LO, int N>
__device__ __forceinline__ void w12_chunk_words(const uint32_t* src, uint32_t (&w)[BITS]) {
    if constexpr (BITS % 4 == 0 && LO % 4 == 0 && N % 4 == 0) {
#pragma unroll
        for (int i = 0; i < N / 4; ++i) {
            const u32x4 v = reinterpret_cast<const u32x4*>(src + LO)[i];
            w[LO + 4 * i] = v.x, w[LO + 4 * i + 1] = v.y, w[LO + 4 * i + 2] = v.z, w[LO + 4 * i + 3] = v.w;
        }
    } else if constexpr (BITS % 2 == 0) {
        // an odd first or last word alone, the even pairs between as 8-byte loads (6 bits: 3 words in 2 loads)
        constexpr int A = LO % 2, P = (N - A) / 2;
        if constexpr (A == 1) w[LO] = src[LO];
#pragma unroll
        for (int i = 0; i < P; ++i) {
            const uint2 v = reinterpret_cast<const uint2*>(src + LO + A)[i];
            w[LO + A + 2 * i] = v.x, w[LO + A + 2 * i + 1] = v.y;
        }
        if constexpr ((N - A) % 2 == 1) w[LO + N - 1] = src[LO + N - 1];
    } else {
#pragma unroll
        for (int i = 0; i < N; ++i) w[LO + i] = src[LO + i];
    }
}

// One G::BM x G::BN block of an item of the plan (or the plain product). Operand layouts as in wmma12_gemm_tile; Dec
// reads a column's codes and group terms, Act the rows of x, Epi stores.
// Rows past m stage zeros and columns past n the last column; neither is stored.
template <class G, class Dec, class Act, class Epi, class Step = W12Step, class Fault = W12Exact>
__device__ __forceinline__ void wmma12_affine_tile(typename Dec::Args& a) {
#if TF_DEVICE_WMMA_GFX12
    constexpr int BM = G::BM, BN = G::BN, BK = G::BK, Threads = G::Threads;
    constexpr int TM = BM / G::WM / 16, TN = BN / G::WN / 16;  // 16 x 16 products a wave
    constexpr int PR = BK / 8;                                  // 16-byte pieces a staged row
    constexpr int XP = BM * PR, XPT = XP / Threads;            // x pieces a stage, a thread
    constexpr int CT = Threads / BN;                            // threads a column of codes
    constexpr int WPT = PR / CT;                                // pieces of a column a thread decodes
    static_assert(BK == 32, "a K step is one 32-code chunk, so it never crosses a group");
    // a contract control folds each row's bias with the next row's sum of x
    constexpr bool stale = requires { requires Fault::stale_sums; };
    static_assert(XP % Threads == 0 && Threads % BN == 0 && PR % CT == 0, "every thread stages whole pieces");
    __shared__ __attribute__((aligned(16))) unsigned short xs[2][BM * G::Ld];
    __shared__ __attribute__((aligned(16))) unsigned short ws[2][BN * G::Ld];
    __shared__ __attribute__((aligned(16))) float sx[2][BM];  // a row's sum of x over a group, by the group's parity
    __shared__ __attribute__((aligned(8))) float2 sb[2][BN];  // a column's scale and bias of that group, one load

    // the launch's block must be the geometry's: fewer threads would leave staged rows and columns unwritten
    if (blockDim.x != Threads) __builtin_trap();
    if (!Dec::take_item(a, blockIdx.z)) return;
    int bx, by;
    w12_band(bx, by);
    const int m0 = by * BM, n0 = bx * BN;
    if (m0 >= a.m) return;
    const int tid = threadIdx.x, lane = tid & 31;
    const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
    const int wr = wave / G::WN, wc = wave % G::WN;
    // a wave's 16-row tiles past m (a routed item's few rows, a short product) skip their loads, products and folds;
    // the test is the wave's, so it branches uniformly, and a live row's chain is untouched
    const int rows_left = a.m - m0;
    bool live[TM];
#pragma unroll
    for (int i = 0; i < TM; ++i) live[i] = wr * (BM / G::WM) + 16 * i < rows_left;

    const int groups = a.k / a.group;
    const int steps = a.k / BK;
    const int per_log2 = __builtin_ctz(a.group / BK);  // groups are 32, 64 or 128 codes: one, two or four steps
    const int per = 1 << per_log2;
    auto group_of = [&](int s) { return s >> per_log2; };
    auto group_end = [&](int s) { return (s & (per - 1)) == per - 1; };

    // x: piece p = tid + q Threads is chunk p % PR of row staged_row(p): a wave's 8 rows in the order 0 4 1 5 2 6 3
    // 7, so the 8 lanes of a 16-byte LDS store pass hold rows 4 apart (320 B) and hit 8 bank groups, not 7.
    static_assert(Threads % 32 == 0 && PR == 4, "a wave stages whole groups of 8 rows of 4 chunks");
    auto staged_row = [](int p) {
        const int rr = (p & 31) >> 2;
        return 8 * (p >> 5) + ((rr & 1) << 2) + (rr >> 1);
    };
    const u32x4* xsrc[XPT];
    bool xlive[XPT];
#pragma unroll
    for (int q = 0; q < XPT; ++q) {
        const int p = tid + q * Threads;
        const int r = m0 + staged_row(p);
        xsrc[q] = reinterpret_cast<const u32x4*>(Act::row(a, r < a.m ? r : 0)) + p % PR;
        xlive[q] = r < a.m;
    }
    // codes: column tid % BN of the block, its pieces from WPT side; side is a wave's (uniform), so the decode below
    // indexes the chunk's words by constants and keeps them in registers. Side 0 keeps the column's terms.
    const int wcol = tid % BN;
    const int side = __builtin_amdgcn_readfirstlane(tid / BN);
    const bool terms = side == 0;
    const int col_s = n0 + wcol < a.n ? n0 + wcol : a.n - 1;
    const uint32_t* wsrc = Dec::words(a) + static_cast<long long>(col_s) * Dec::template row_words<int>(a);
    const long long tbase = static_cast<long long>(col_s) * groups;

    u32x4 rx[XPT] = {};  // rows past m keep zeros: they feed only rows that are never stored
    // only the words of a side's 16 codes are loaded; the others stay 0 and are never read
    uint32_t rw[Dec::kWords] = {};
    typename Dec::Term term;
    // the last BM threads (whole waves) each keep one row's sum of x, read back from the staged rows
    const bool summer = tid >= Threads - BM;
    const int srow = tid - (Threads - BM);
    float run = 0.f;
    auto fetch = [&](int s) {
#pragma unroll
        for (int q = 0; q < XPT; ++q) if (xlive[q]) rx[q] = xsrc[q][s * PR];
        // side S decodes codes 16 S .. 16 S + 15: the words that hold their bits
        constexpr int B = Dec::kBits, lo1 = 16 * B / 32, hi0 = (16 * B - 1) / 32;
        if constexpr (CT == 1) Dec::template load<true>(wsrc + s * Dec::kWords, rw);
        else if (side == 0) w12_chunk_words<B, 0, hi0 + 1>(wsrc + s * Dec::kWords, rw);
        else w12_chunk_words<B, lo1, B - lo1>(wsrc + s * Dec::kWords, rw);
        if (terms && group_end(s)) term = Dec::term(a, tbase + group_of(s));
    };
    auto stash = [&](int s, int buf) {
#pragma unroll
        for (int q = 0; q < XPT; ++q) {
            const int p = tid + q * Threads;
            w12_store<G>(xs[buf], staged_row(p), p % PR, rx[q]);
        }
        auto decode = [&]<int S>(std::integral_constant<int, S>) {
#pragma unroll
            for (int c = 0; c < WPT; ++c) {
                uint32_t d[4];
#pragma unroll
                for (int h = 0; h < 4; ++h) {
                    d[h] = __builtin_bit_cast(uint32_t, Dec::template pair<DotBF16>(rw, 8 * (WPT * S + c) + 2 * h));
                }
                w12_store<G>(ws[buf], wcol, WPT * S + c, u32x4{d[0], d[1], d[2], d[3]});
            }
        };
        static_assert(CT == 1 || CT == 2, "one or two sides of a column's codes");
        if constexpr (CT == 1) decode(std::integral_constant<int, 0>{});
        else if (side == 0) decode(std::integral_constant<int, 0>{});
        else decode(std::integral_constant<int, 1>{});
        if (terms && group_end(s)) sb[buf][wcol] = make_float2(Dec::scale_value(a, term), Dec::bias_value(a, term));
    };

    f32x8 acc[TM][TN], d[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = d[i][j] = f32x8{};
    const f32x8 zero = {};
    const int arow = wr * (BM / G::WM) + (lane & 15), bcol = wc * (BN / G::WN) + (lane & 15);
    const int koff = 8 * (lane >> 4);
    // a group's bias folds a step after its scale, once its row sums are past a barrier; it waits in registers, as
    // the group's terms in LDS are restaged before then
    float bias[TN];
    auto fold_bias = [&](int g) {
#pragma unroll
        for (int i = 0; i < TM; ++i) {
            if (!live[i]) continue;
            float s_x[8];
            const int r0 = wr * (BM / G::WM) + 16 * i + koff;
#pragma unroll
            for (int e = 0; e < 8; ++e) s_x[e] = sx[g & 1][stale ? (r0 + e + 1) % BM : r0 + e];
#pragma unroll
            for (int j = 0; j < TN; ++j)
#pragma unroll
                for (int e = 0; e < 8; ++e) acc[i][j][e] = Dec::fold_bias(acc[i][j][e], s_x[e], bias[j]);
        }
    };
    fetch(0);
    stash(0, 0);
    w12_sync<Fault>();
    for (int s = 0; s < steps; ++s) {
        const int buf = s & 1;
        if (s + 1 < steps) fetch(s + 1);  // the next step's global loads first, ahead of the bias fold
        if (s > 0 && group_end(s - 1)) fold_bias(group_of(s - 1));
        // a group's first product starts from zero on a uniform branch, not a select an accumulator
        const bool first = (s & (per - 1)) == 0;
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            bf16x8 bv[TN];
#pragma unroll
            for (int j = 0; j < TN; ++j) bv[j] = w12_load<G, Fault>(ws[buf], bcol + 16 * j, (kk + koff) / 8);
            // one 16-row tile at a time: its operand, then its TN products, so a single A fragment is live
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                if (!live[i]) continue;
                const bf16x8 av = w12_load<G, Fault>(xs[buf], arow + 16 * i, (kk + koff) / 8);
                if (kk == 0 && first) {
#pragma unroll
                    for (int j = 0; j < TN; ++j) d[i][j] = Step::mma(av, bv[j], zero);
                } else {
#pragma unroll
                    for (int j = 0; j < TN; ++j) d[i][j] = Step::mma(av, bv[j], d[i][j]);
                }
            }
        }
        if (group_end(s)) {
            float scale[TN];
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                const float2 t = sb[buf][bcol + 16 * j];
                scale[j] = t.x, bias[j] = t.y;
            }
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
#pragma unroll
                    for (int e = 0; e < 8; ++e)
                        if (live[i]) acc[i][j][e] = Dec::fold_scale(acc[i][j][e], d[i][j][e], scale[j]);
        }
        if (summer) {
            // the dot2 tiles' sum of x: each element in k order, continued across a group's steps
#pragma unroll
            for (int c = 0; c < PR; ++c) {
                const u32x4 u = *reinterpret_cast<const u32x4*>(&xs[buf][w12_at<G>(srow, c)]);
                const uint32_t q[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
                for (int h = 0; h < 4; ++h) {
                    run += __uint_as_float(q[h] << 16);
                    run += __uint_as_float(q[h] & 0xffff0000u);
                }
            }
            if (group_end(s)) {
                sx[group_of(s) & 1][srow] = run;
                run = 0.f;
            }
        }
        if (s + 1 < steps) stash(s + 1, buf ^ 1);
        w12_sync<Fault>();
    }
    fold_bias(group_of(steps - 1));
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int col = n0 + bcol + 16 * j;
            if (col >= a.n) continue;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int row = m0 + wr * (BM / G::WM) + 16 * i + koff + e;
                if (row < a.m) Epi::store(a, row, col, acc[i][j][e]);
            }
        }
    }
#else
    (void)a;
    __builtin_trap();
#endif
}

// The K-parallel block of few rows: 16 rows by BN = 16 TN columns, W waves that each take every W-th group of k.
// A wave reads its operands from global memory into its fragments, with no staging, and sums a group with the same
// products in the same order as the staged tiles; the block then folds a round's W groups in k order through LDS. So
// a row's bytes are the staged tiles', and a product of few rows has W times the loads in flight.
template <int W_, int TN_>
struct W12KpGeo {
    static constexpr int W = W_, TN = TN_, BM = 16, BN = 16 * TN_, Threads = 32 * W_;
};
// 8 waves of 32 columns: on gfx1201 it beats 4 waves of 64 wherever the registry can choose it (more of k in flight)
using W12AffineKp = W12KpGeo<8, 2>;

template <class G, class Dec, class Act, class Epi, class Step = W12Step, class Fault = W12Exact>
__device__ __forceinline__ void wmma12_affine_kp_tile(typename Dec::Args& a) {
#if TF_DEVICE_WMMA_GFX12
    constexpr int W = G::W, TN = G::TN, BN = G::BN, Threads = G::Threads;
    constexpr int OPT = 16 * BN / Threads;  // outputs a thread folds, all in one column
    constexpr int CPL = BN / 32;            // columns whose group terms a lane loads
    constexpr int KW = Dec::kWords;
    constexpr int LP = BN + 2;  // a part row's stride: the two halves of a lane's stores 16 banks apart
    constexpr bool stale = requires { requires Fault::stale_sums; };
    static_assert(Threads % BN == 0 && BN % 32 == 0, "a thread folds outputs of one column, a lane whole terms");
    __shared__ float part[W][16 * LP];  // a round's group sums, a wave's group a slot
    __shared__ float2 sbt[W][BN];       // their columns' scales and biases
    __shared__ float sxs[W][16];        // their rows' sums of x

    if (blockDim.x != Threads) __builtin_trap();
    if (!Dec::take_item(a, blockIdx.z)) return;
    const int n0 = blockIdx.x * BN;
    const int tid = threadIdx.x, lane = tid & 31;
    const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
    const int groups = a.k / a.group;
    const int per_log2 = __builtin_ctz(a.group / 32);  // 32-code chunks a group: one, two or four
    const int per = 1 << per_log2;
    const int rounds = (groups + W - 1) / W;
    const int chunks = rounds << per_log2;  // of this wave's groups, in order
    const int half = lane >> 4;              // A and B fragments hold k 8 half .. 8 half + 7 of each 16
    const int fcol = tid % BN;               // the column this thread folds
    // B: column lane % 16 of each 16-column tile (columns past n read the last and are never stored)
    const long long words_row = Dec::template row_words<long long>(a);
    const uint32_t* wsrc[TN];
#pragma unroll
    for (int j = 0; j < TN; ++j) {
        const int col = n0 + 16 * j + (lane & 15);
        wsrc[j] = Dec::words(a) + static_cast<long long>(col < a.n ? col : a.n - 1) * words_row;
    }
    long long tb[CPL];
#pragma unroll
    for (int c = 0; c < CPL; ++c) {
        const int col = n0 + lane + 32 * c;
        tb[c] = static_cast<long long>(col < a.n ? col : a.n - 1) * groups;
    }
    // group of chunk t of this wave's sequence: (t / per) W + wave, the last group's past the end
    auto group_at = [&](int t) {
        const int g = (t >> per_log2) * W + wave;
        return g < groups ? g : groups - 1;
    };
    // codes 8 (P + half) .. + 7 of a chunk as BF16, the staged tiles' codes and conversion (Dec::pair_of): the half
    // picks its piece's two words and shift by a select between constants, so the chunk stays in registers
    auto piece = [&]<int P>(const uint32_t (&ws)[KW], std::integral_constant<int, P>) {
        constexpr int B = Dec::kBits;
        constexpr int b0 = 8 * P * B, b1 = 8 * (P + 1) * B;  // the first bit of piece P, and of piece P + 1
        static_assert((b0 & 31) + 8 * B <= 64 && (b1 & 31) + 8 * B <= 64, "a piece's codes fit a 64-bit window");
        const uint32_t lo = half ? ws[b1 >> 5] : ws[b0 >> 5];
        const uint32_t hi = half ? ws[(b1 + 8 * B - 1) >> 5] : ws[(b0 + 8 * B - 1) >> 5];
        const uint64_t v = ((static_cast<uint64_t>(hi) << 32) | lo) >> (half ? b1 & 31 : b0 & 31);
        auto code = [&](int i) { return static_cast<uint32_t>(v >> (i * B)) & ((1u << B) - 1u); };
        uint32_t d[4];
#pragma unroll
        for (int h = 0; h < 4; ++h) d[h] = __builtin_bit_cast(uint32_t, Dec::template pair_of<DotBF16>(code(2 * h), code(2 * h + 1)));
        return __builtin_bit_cast(bf16x8, u32x4{d[0], d[1], d[2], d[3]});
    };
    constexpr std::integral_constant<int, 0> k0{};
    constexpr std::integral_constant<int, 2> k16{};
    // the block's 16-row passes over an item's rows; blocks past the item's rows have no pass
    for (int m0 = blockIdx.y * 16; m0 < a.m; m0 += gridDim.y * 16) {
        // A: row lane % 16 of the pass (rows past m read row 0 and are never stored)
        const int xr = m0 + (lane & 15);
        const u32x4* xrow = reinterpret_cast<const u32x4*>(Act::row(a, xr < a.m ? xr : 0));
        u32x4 xa[2], xn[2];
        uint32_t w[TN][KW], wn[TN][KW];
        typename Dec::Term term[CPL], termn[CPL];
        // chunk t's operands, and at a group's first chunk the group's terms, which fold at its last
        auto fetch = [&](int t, u32x4 (&x)[2], uint32_t (&ws)[TN][KW], typename Dec::Term (&tm)[CPL]) {
            const int g = group_at(t);
            const long long chunk = (static_cast<long long>(g) << per_log2) + (t & (per - 1));
            x[0] = xrow[4 * chunk + half];
            x[1] = xrow[4 * chunk + 2 + half];
#pragma unroll
            for (int j = 0; j < TN; ++j) Dec::template load<true>(wsrc[j] + chunk * KW, ws[j]);
            if ((t & (per - 1)) == 0) {
#pragma unroll
                for (int q = 0; q < CPL; ++q) tm[q] = Dec::term(a, tb[q] + g);
            }
        };

        float acc[OPT];
#pragma unroll
        for (int i = 0; i < OPT; ++i) acc[i] = 0.f;
        f32x8 d[TN];
#pragma unroll
        for (int j = 0; j < TN; ++j) d[j] = f32x8{};
        const f32x8 zero = {};
        float run = 0.f;  // lanes 0 .. 15: their row's sum of x over the group, the dot2 tiles' chain
        fetch(0, xa, w, term);
        for (int t = 0; t < chunks; ++t) {
            const int c = t & (per - 1);
            if (t + 1 < chunks) fetch(t + 1, xn, wn, termn);
            // the chunk's 32 elements of x in k order: this lane's 8, the other half's 8, then the same at k 16
            const u32x4 o0 = {__shfl_xor(xa[0].x, 16, 32), __shfl_xor(xa[0].y, 16, 32), __shfl_xor(xa[0].z, 16, 32),
                              __shfl_xor(xa[0].w, 16, 32)};
            const u32x4 o1 = {__shfl_xor(xa[1].x, 16, 32), __shfl_xor(xa[1].y, 16, 32), __shfl_xor(xa[1].z, 16, 32),
                              __shfl_xor(xa[1].w, 16, 32)};
            if (half == 0) {
                const u32x4 seq[4] = {xa[0], o0, xa[1], o1};
#pragma unroll
                for (int v = 0; v < 4; ++v) {
                    const uint32_t q[4] = {seq[v].x, seq[v].y, seq[v].z, seq[v].w};
#pragma unroll
                    for (int h = 0; h < 4; ++h) {
                        run += __uint_as_float(q[h] << 16);
                        run += __uint_as_float(q[h] & 0xffff0000u);
                    }
                }
            }
            // a group's first product starts from zero on a uniform branch, not a select an accumulator
            const bf16x8 a0 = __builtin_bit_cast(bf16x8, xa[0]), a1 = __builtin_bit_cast(bf16x8, xa[1]);
            if (c == 0) {
#pragma unroll
                for (int j = 0; j < TN; ++j) d[j] = Step::mma(a0, piece(w[j], k0), zero);
            } else {
#pragma unroll
                for (int j = 0; j < TN; ++j) d[j] = Step::mma(a0, piece(w[j], k0), d[j]);
            }
#pragma unroll
            for (int j = 0; j < TN; ++j) d[j] = Step::mma(a1, piece(w[j], k16), d[j]);
            if (t + 1 < chunks) {
                xa[0] = xn[0], xa[1] = xn[1];
#pragma unroll
                for (int j = 0; j < TN; ++j)
#pragma unroll
                    for (int i = 0; i < KW; ++i) w[j][i] = wn[j][i];
            }
            if (c != per - 1) continue;
            // the round's groups to LDS, then each output folds them in k order
#pragma unroll
            for (int j = 0; j < TN; ++j)
#pragma unroll
                for (int e = 0; e < 8; ++e) part[wave][(8 * half + e) * LP + 16 * j + (lane & 15)] = d[j][e];
#pragma unroll
            for (int q = 0; q < CPL; ++q) {
                sbt[wave][lane + 32 * q] = make_float2(Dec::scale_value(a, term[q]), Dec::bias_value(a, term[q]));
                if (t + 1 < chunks) term[q] = termn[q];
            }
            if (half == 0) sxs[wave][lane] = run;
            run = 0.f;
            w12_sync<Fault>();
            const int round = t >> per_log2;
            const int count = groups - round * W < W ? groups - round * W : W;
#pragma unroll
            for (int gg = 0; gg < W; ++gg) {
                if (gg >= count) break;
                const float2 sb = sbt[gg][fcol];
#pragma unroll
                for (int i = 0; i < OPT; ++i) {
                    const int r = (tid + i * Threads) / BN;
                    acc[i] = Dec::fold_scale(acc[i], part[gg][r * LP + fcol], sb.x);
                    acc[i] = Dec::fold_bias(acc[i], sxs[gg][stale ? (r + 1) % 16 : r], sb.y);
                }
            }
            w12_sync<Fault>();
        }
#pragma unroll
        for (int i = 0; i < OPT; ++i) {
            const int row = m0 + (tid + i * Threads) / BN, col = n0 + fcol;
            if (row < a.m && col < a.n) Epi::store(a, row, col, acc[i]);
        }
    }
#else
    (void)a;
    __builtin_trap();
#endif
}

}  // namespace rocm
}  // namespace tf
