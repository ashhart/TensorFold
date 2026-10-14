// Dot2 tiles for every width and both activation types: one row, 2-8 rows, the GEMM tile, and routed plans.

#include "tiles/dot2.hpp"
#include "tiles/gemm.hpp"
#include "tiles/gemm_kp.hpp"
#include "tiles/stream.hpp"
#include "tiles/matrix_gemm.hpp"
#include "quant/mlx_tiles.hpp"

namespace tf {
namespace rocm {

// Groups a wave owns each round. Fewer rows leave room for more loads in flight; at most 16 code words a lane.
constexpr int lane_groups(int bits, int rows, int pieces) {
    int per = rows <= 2 ? 4 : rows <= 4 ? 2 : 1;
    while (per > 1 && per * pieces * bits > 8) per /= 2;
    return per;
}

template <typename T, int BITS, int ROWS, int PIECES>
__global__ void __launch_bounds__(kLaneCols * kLaneWaves) affine_dot2_lanes(Affine a) {
    if (!take_item(a, blockIdx.z)) return;
    constexpr int kGroup = 32 * PIECES;
    constexpr int kPer = lane_groups(BITS, ROWS, PIECES);
    constexpr int kRound = kLaneWaves * kPer;
    __shared__ typename T::elem xs[ROWS][kRound * kGroup];
    __shared__ float part[kRound][ROWS][2][kLaneCols];
    const int lane = threadIdx.x % kLaneCols;
    const int wave = threadIdx.x / kLaneCols;
    const int col = blockIdx.x * kLaneCols + lane;
    const int groups = a.k / kGroup;
    const int words_row = a.k * BITS / 32;
    // A lane past N reads the last column and stores nothing, so every lane reaches every barrier.
    const uint32_t* wrow = a.words + static_cast<long long>(col < a.n ? col : a.n - 1) * words_row;
    const typename T::elem* x = static_cast<const typename T::elem*>(a.x);
    float acc[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) acc[r] = 0.f;

    for (int g0 = 0; g0 < groups; g0 += kRound) {
        constexpr int kSpan = kRound * kGroup;
        for (int e = threadIdx.x; e < ROWS * kSpan; e += blockDim.x) {
            const int r = e / kSpan;
            const int kk = e - r * kSpan;
            const int gk = g0 * kGroup + kk;
            xs[r][kk] = (r < a.m && gk < a.k) ? x[x_row(a, r) * a.k + gk] : T::zero();
        }
        // 6- and 8-bit lanes load every code word of their groups up front; narrower widths load a piece at a time.
        constexpr bool kAhead = BITS >= 6;
        uint32_t w[kPer][PIECES][BITS];
        if constexpr (kAhead) {
#pragma unroll
            for (int j = 0; j < kPer; ++j) {
                const int g = g0 + wave * kPer + j;
                if (g < groups) {
#pragma unroll
                    for (int p = 0; p < PIECES; ++p) load_piece<BITS>(wrow + (g * PIECES + p) * BITS, w[j][p]);
                }
            }
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < kPer; ++j) {
            const int local = wave * kPer + j;
            if (g0 + local >= groups) continue;
            float dot[ROWS];
            float sum[ROWS];
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                dot[r] = 0.f;
                sum[r] = 0.f;
            }
#pragma unroll
            for (int p = 0; p < PIECES; ++p) {
                if constexpr (!kAhead) load_piece<BITS>(wrow + ((g0 + local) * PIECES + p) * BITS, w[j][p]);
                const int xk = local * kGroup + 32 * p;
#pragma unroll
                for (int t = 0; t < 32; t += 2) {
                    const typename T::pair q = T::two(MlxDecoder<BITS>::template code<T>(w[j][p], t),
                                                      MlxDecoder<BITS>::template code<T>(w[j][p], t + 1));
#pragma unroll
                    for (int r = 0; r < ROWS; ++r) {
                        const typename T::pair xp = T::two(xs[r][xk + t], xs[r][xk + t + 1]);
                        dot[r] = T::dot(xp, q, dot[r]);
                        sum[r] += T::lo(xp);
                        sum[r] += T::hi(xp);
                    }
                }
            }
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                part[local][r][0][lane] = dot[r];
                part[local][r][1][lane] = sum[r];
            }
        }
        __syncthreads();
        if (wave == 0 && col < a.n) {
            const int round = groups - g0 < kRound ? groups - g0 : kRound;
            for (int j = 0; j < round; ++j) {
                const long long at = static_cast<long long>(col) * groups + g0 + j;
                const float scale = a.scale[at];
                const float bias = a.bias[at];
#pragma unroll
                for (int r = 0; r < ROWS; ++r) {
                    acc[r] = fmaf(part[j][r][0][lane], scale, acc[r]);
                    acc[r] = fmaf(part[j][r][1][lane], bias, acc[r]);
                }
            }
        }
        // The next round restages xs: wave 0's fold above only reads part.
    }
    if (wave == 0 && col < a.n) {
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            if (r >= a.m) continue;
            if (a.out16 != nullptr) {
                a.out16[out_row(a, r) * a.n + col] = __float2half_rn(acc[r]);
            } else {
                a.out[out_row(a, r) * a.n + col] = acc[r];
            }
        }
    }
}

// One-row tile: a lane owns a group, so a wave reads a weight row contiguously; lane c folds column c in order.
constexpr int kRowWaves = 4;
constexpr int kRowCols = 32;

template <typename T, int BITS, int PIECES>
__global__ void __launch_bounds__(32 * kRowWaves) affine_dot2_row(Affine a, int lpc) {
    if (!take_item(a, blockIdx.z)) return;
    constexpr int kGroup = 32 * PIECES;
    __shared__ float dots[kRowWaves][kRowCols][33];
    __shared__ float sums[kRowWaves][32];
    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x >> 5;
    const int slots = 32 / lpc;                 // columns a pass
    const int slot = lane / lpc;
    const int j = lane % lpc;
    const int col0 = (blockIdx.x * kRowWaves + wave) * kRowCols;
    const int mine = col0 + lane;               // the column this lane folds
    const int groups = a.k / kGroup;
    const long long words_row = static_cast<long long>(a.k) * BITS / 32;
    const typename T::pair* x = reinterpret_cast<const typename T::pair*>(
        static_cast<const typename T::elem*>(a.x) + x_row(a, 0) * a.k);
    float acc = 0.f;
    for (int g0 = 0; g0 < groups; g0 += lpc) {
        const int g = g0 + j;
        const bool live = g < groups;
        typename T::pair xv[PIECES * 16];
        float sum = 0.f;
#pragma unroll
        for (int i = 0; i < PIECES * 16; ++i) {
            xv[i] = live ? x[g * (kGroup / 2) + i] : T::two(T::zero(), T::zero());
            sum += T::lo(xv[i]);
            sum += T::hi(xv[i]);
        }
        if (slot == 0) sums[wave][j] = sum;
#pragma unroll 2
        for (int pass = 0; pass < kRowCols / slots; ++pass) {
            const int c = pass * slots + slot;
            const int col = col0 + c;
            float dot = 0.f;
            if (live && col < a.n) {
                const uint32_t* wrow = a.words + static_cast<long long>(col) * words_row +
                                       static_cast<long long>(g) * PIECES * BITS;
#pragma unroll
                for (int p = 0; p < PIECES; ++p) {
                    uint32_t w[BITS];
                    load_piece<BITS>(wrow + p * BITS, w);
#pragma unroll
                    for (int t = 0; t < 32; t += 2) {
                        const typename T::pair q =
                            T::two(MlxDecoder<BITS>::template code<T>(w, t), MlxDecoder<BITS>::template code<T>(w, t + 1));
                        dot = T::dot(xv[p * 16 + t / 2], q, dot);
                    }
                }
            }
            dots[wave][c][j] = dot;
        }
        __syncthreads();
        if (mine < a.n) {
            const int round = groups - g0 < lpc ? groups - g0 : lpc;
            for (int i = 0; i < round; ++i) {
                const long long at = static_cast<long long>(mine) * groups + g0 + i;
                acc = fmaf(dots[wave][lane][i], a.scale[at], acc);
                acc = fmaf(sums[wave][i], a.bias[at], acc);
            }
        }
        __syncthreads();
    }
    if (mine < a.n) {
        if (a.out16 != nullptr) {
            a.out16[out_row(a, 0) * a.n + mine] = __float2half_rn(acc);
        } else {
            a.out[out_row(a, 0) * a.n + mine] = acc;
        }
    }
}

// GEMM tile: 128 x 128, a thread 8 rows x 8 columns strided by 16, K through double-buffered LDS 32 codes a stage.
constexpr int kBlockM = 128;
constexpr int kBlockN = 128;
constexpr int kBlockK = 32;
constexpr int kBlockLd = kBlockK + 2;  // 17 words a row: the 16 column threads of a wave read 16 banks

template <typename T, int BITS>
__device__ inline void stage_block(typename T::elem* xs, typename T::elem* ws, const Affine& a, int m0, int n0, int k0,
                                   int words_row) {
    const typename T::elem* x = static_cast<const typename T::elem*>(a.x);
    // x: 128 rows x 32 halves, 16 halves (two 16-byte loads) a thread.
    {
        const int r = threadIdx.x >> 1;
        const int half = (threadIdx.x & 1) * 16;
        const int gr = m0 + r;
        uint4 lo = make_uint4(0u, 0u, 0u, 0u);
        uint4 hi = lo;
        if (gr < a.m) {
            const uint4* src = reinterpret_cast<const uint4*>(x + x_row(a, gr) * a.k + k0 + half);
            lo = src[0];
            hi = src[1];
        }
        uint32_t* dst = reinterpret_cast<uint32_t*>(xs + r * kBlockLd + half);
        dst[0] = lo.x; dst[1] = lo.y; dst[2] = lo.z; dst[3] = lo.w;
        dst[4] = hi.x; dst[5] = hi.y; dst[6] = hi.z; dst[7] = hi.w;
    }
    // Codes: one 32-code piece a column, threads 0..127.
    if (threadIdx.x < kBlockN) {
        const int c = threadIdx.x;
        const int gc = n0 + c;
        typename T::elem* dst = ws + c * kBlockLd;
        if (gc < a.n) {
            uint32_t w[BITS];
            load_piece<BITS>(a.words + static_cast<long long>(gc) * words_row + k0 / 32 * BITS, w);
#pragma unroll
            for (int t = 0; t < 32; t += 2) {
                *reinterpret_cast<typename T::pair*>(dst + t) =
                    T::two(MlxDecoder<BITS>::template code<T>(w, t), MlxDecoder<BITS>::template code<T>(w, t + 1));
            }
        } else {
#pragma unroll
            for (int t = 0; t < 32; t += 2) {
                *reinterpret_cast<typename T::pair*>(dst + t) = T::two(T::zero(), T::zero());
            }
        }
    }
}

template <typename T, int BITS>
__global__ void __launch_bounds__(256, 1) affine_dot2_block(Affine a) {
    if (!take_item(a, blockIdx.z)) return;
    if (static_cast<int>(blockIdx.y) * kBlockM >= a.m) return;
    __shared__ typename T::elem xs[2][kBlockM * kBlockLd];
    __shared__ typename T::elem ws[2][kBlockN * kBlockLd];
    const int tx = threadIdx.x & 15;
    const int ty = threadIdx.x >> 4;
    const int m0 = blockIdx.y * kBlockM;
    const int n0 = blockIdx.x * kBlockN;
    const int words_row = a.k * BITS / 32;
    const int groups = a.k / a.group;
    const int stages = a.k / kBlockK;
    const int per = a.group / kBlockK;
    float acc[8][8];
    float dot[8][8];
    float sumx[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        sumx[i] = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            acc[i][j] = 0.f;
            dot[i][j] = 0.f;
        }
    }
    stage_block<T, BITS>(xs[0], ws[0], a, m0, n0, 0, words_row);
    __syncthreads();
    for (int s = 0; s < stages; ++s) {
        const int cur = s & 1;
        if (s + 1 < stages) stage_block<T, BITS>(xs[cur ^ 1], ws[cur ^ 1], a, m0, n0, (s + 1) * kBlockK, words_row);
        const typename T::elem* xb = xs[cur];
        const typename T::elem* wb = ws[cur];
#pragma unroll 4
        for (int t = 0; t < kBlockK; t += 2) {
            typename T::pair xv[8];
            typename T::pair wv[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                xv[i] = *reinterpret_cast<const typename T::pair*>(xb + (ty + 16 * i) * kBlockLd + t);
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                wv[j] = *reinterpret_cast<const typename T::pair*>(wb + (tx + 16 * j) * kBlockLd + t);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) {
#pragma unroll
                for (int j = 0; j < 8; ++j) dot[i][j] = T::dot(xv[i], wv[j], dot[i][j]);
                sumx[i] += T::lo(xv[i]);
                sumx[i] += T::hi(xv[i]);
            }
        }
        if (s % per == per - 1) {
            const int g = s / per;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int col = n0 + tx + 16 * j;
                const long long at = static_cast<long long>(col < a.n ? col : 0) * groups + g;
                const float scale = a.scale[at];
                const float bias = a.bias[at];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    acc[i][j] = fmaf(dot[i][j], scale, acc[i][j]);
                    acc[i][j] = fmaf(sumx[i], bias, acc[i][j]);
                    dot[i][j] = 0.f;
                }
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) sumx[i] = 0.f;
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = m0 + ty + 16 * i;
        if (row >= a.m) continue;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int col = n0 + tx + 16 * j;
            if (col < a.n) a.out[out_row(a, row) * a.n + col] = acc[i][j];
        }
    }
}

// The kernels launches/affine.zig resolves, for the build's activation type: bf16 where the GPU has its dot2.
#if TF_DOT2_BF16
using Dot = DotBF16;
#else
using Dot = DotF16;
#endif

#define TF_LANES(BITS, ROWS)                                                    \
    template __global__ void affine_dot2_lanes<Dot, BITS, ROWS, 1>(Affine);     \
    template __global__ void affine_dot2_lanes<Dot, BITS, ROWS, 2>(Affine);     \
    template __global__ void affine_dot2_lanes<Dot, BITS, ROWS, 4>(Affine);
#define TF_STREAM(BITS, R, CB)                                                                     \
    template __global__ void affine_dot2_stream<Dot, BITS, R, CB, false>(Affine, StreamSides, int, int); \
    template __global__ void affine_dot2_stream<Dot, BITS, R, CB, true>(Affine, StreamSides, int, int);
#if TF_MATRIX
#define TF_WMMA(BITS) template __global__ void affine_wmma_gemm<BITS>(Affine);
#else
#define TF_WMMA(BITS)
#endif
#define TF_TILES(BITS)                                                                       \
    template __global__ void affine_dot2_block<Dot, BITS>(Affine);                           \
    template __global__ void affine_gemm_block<Dot, BITS, GemmShape<Dot>::rt>(Affine);       \
    template __global__ void affine_gemm_kp<Dot, BITS, 8, 1, 2, 2, false>(Affine);           \
    template __global__ void affine_gemm_kp<Dot, BITS, 4, 4, 2, 4, true>(Affine);            \
    template __global__ void affine_gemm_kp<Dot, BITS, 4, 8, 2, 4, false>(Affine);           \
    template __global__ void affine_gemm_kp<Dot, BITS, 4, 2, 2, 4, true>(Affine);            \
    template __global__ void affine_dot2_row<Dot, BITS, 1>(Affine, int);                     \
    template __global__ void affine_dot2_row<Dot, BITS, 2>(Affine, int);                     \
    template __global__ void affine_dot2_row<Dot, BITS, 4>(Affine, int);                     \
    TF_LANES(BITS, 1) TF_LANES(BITS, 2) TF_LANES(BITS, 4) TF_LANES(BITS, kLaneRows)          \
    TF_STREAM(BITS, 1, 2) TF_STREAM(BITS, 1, 4) TF_STREAM(BITS, 1, 8)                        \
    TF_STREAM(BITS, 2, 2) TF_STREAM(BITS, 2, 4) TF_STREAM(BITS, 2, 8)                        \
    TF_STREAM(BITS, 4, 2) TF_STREAM(BITS, 4, 4)                                               \
    TF_WMMA(BITS)

TF_TILES(2)
TF_TILES(3)
TF_TILES(4)
TF_TILES(5)
TF_TILES(6)
TF_TILES(8)

}  // namespace rocm
}  // namespace tf
