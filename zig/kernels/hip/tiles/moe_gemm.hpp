#pragma once

// Flash Next's routed expert products of a prompt on the gfx11 matrix cores, affine_wmma_gemm's arithmetic and bytes.

// Inspired by mlx-serve's moe_prefill_cuda.zig: gate and up rows in one tile, SwiGLU epilogue, register prefetch.

#include <hip/hip_bfloat16.h>

#include <type_traits>
#include <utility>

#include "common/wmma.hpp"
#include "quant/mlx_decoder.hpp"
#include "tiles/plan.hpp"

namespace tf {
namespace rocm {

namespace moe_tile {
constexpr int kBM = 64;       // rows a pass
constexpr int kBN = 128;      // staged weight rows a block
constexpr int kThreads = 256;
}  // namespace moe_tile

struct MoeGateUp {
    static constexpr int kK = 2560;  // hidden
    static constexpr int kNI = 640;  // expert width: gate rows [0, 640), up rows [640, 1280) of an expert
    static constexpr int kN = 2 * kNI;
    static constexpr int kTG = 8;    // groups a term load covers
    static constexpr bool kAct = true;
    static constexpr int kBlocks = kNI / 64;
    // windows of loads held in registers (4: the fastest of 2, 4 and 8 on gfx1151)
    static constexpr int kAhead = 4;
    // staged row `line` of block x: gate column 64 x + line, then the up column 640 + 64 x + line - 64
    __device__ static int col(int bx, int line) { return line < 64 ? 64 * bx + line : kNI + 64 * bx + line - 64; }
};

struct MoeDown {
    static constexpr int kK = 640;   // expert width
    static constexpr int kN = 2560;  // hidden
    static constexpr int kTG = 4;    // 20 groups a row: spans of 4
    static constexpr bool kAct = false;
    static constexpr int kBlocks = kN / 128;
    // one window of loads ahead (2 is no faster on gfx1151)
    static constexpr int kAhead = 1;
    __device__ static int col(int bx, int line) { return 128 * bx + line; }
};

// KW: K a window, 64 or 128; a staged row is KW + 8 bf16 (8 lanes' 16-byte reads hit 32 banks). AHEAD: windows ahead.
template <class S, int BITS, int KW = 64, int AHEAD = 1>
__device__ __forceinline__ void wmma_moe_tile(Affine a, float limit) {
#if TF_DEVICE_WMMA_GFX11
    using namespace moe_tile;
    static_assert(BITS == 4, "a 32-code group is one 16-byte chunk");
    static_assert(KW == 128 || KW == 64, "4 or 2 groups a window");
    constexpr int kK = S::kK;
    constexpr int kGW = KW / 32;               // groups a window
    constexpr int kWindows = kK / KW;
    constexpr int kLdw = (KW + 8) / 2;
    constexpr int kWChunks = kBN * kGW / 256;  // 16-byte weight chunks a thread a window
    constexpr int kXPer = KW / 8;              // 16-byte x pieces a row
    constexpr int kXChunks = kBM * kXPer / 256;
    constexpr int kTG = S::kTG;
    constexpr int kTermEvery = kTG / kGW;      // windows a term load covers
    constexpr int kTW = kTG / 2;               // words a thread's term load
    static_assert(kK % KW == 0 && (kK / 32) % kTG == 0 && kTG % kGW == 0, "whole windows and term spans");
    static_assert(kWindows % AHEAD == 0, "a window's register slot is a compile-time constant");
    using Dec = MlxDecoder<BITS>;
    using T = WmmaBF16;
    using TermLoad = std::conditional_t<kTW == 4, uint4, uint2>;
    if (!Dec::take_item(a, blockIdx.z)) return;
    __shared__ __attribute__((aligned(16))) uint32_t xs[kBM * kLdw];
    __shared__ __attribute__((aligned(16))) uint32_t ws[kBN * kLdw];
    __shared__ float sx[kGW][kBM];
    __shared__ float sc[kTG][kBN];
    __shared__ float bi[kTG][kBN];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
    const int bx = blockIdx.x;
    constexpr int kWordsRow = kK * BITS / 32;
    constexpr int kGroups = kK / 32;

    // weights: chunk i of thread t is row (256 i + t) / kGW, group t % kGW of the window
    const int wq = tid % kGW;
    const uint32_t* wsrc[kWChunks];
#pragma unroll
    for (int i = 0; i < kWChunks; ++i) wsrc[i] = a.words + static_cast<long long>(S::col(bx, (256 * i + tid) / kGW)) * kWordsRow + wq * BITS;
    // terms of row t % 128, kTG groups a load: the scales (t < 128) or the biases
    const int trow = tid & (kBN - 1);
    const bool thigh = tid >= kBN;
    const long long tcol = static_cast<long long>(S::col(bx, trow)) * kGroups;
    const TermLoad* tsrc = reinterpret_cast<const TermLoad*>(static_cast<const uint16_t*>(thigh ? a.bias.p : a.scale.p) + tcol);
    // x: chunk i of thread t is row (256 i + t) / kXPer, 16-byte piece t % kXPer (group piece / 4, quad lane t % 4)
    const int xc = tid % kXPer;

    // wave (wm, wn): rows 32 wm.., staged rows 16 wn.. and 64 + 16 wn..; a lane: column lane % 16, rows 2 i + lane / 16
    const int wm = wave & 1;
    const int wn = wave >> 1;
    const int rl = lane & 15;
    const int rh = lane >> 4;
    const int lines[2] = {wn * 16 + rl, 64 + wn * 16 + rl};
    const f32x8 zero = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int h0 = 0; h0 < a.m; h0 += kBM) {
        const int rows_left = a.m - h0 < kBM ? a.m - h0 : kBM;
        const u32x4* xsrc[kXChunks];
#pragma unroll
        for (int i = 0; i < kXChunks; ++i) {
            const int r = (256 * i + tid) / kXPer;
            const int rs = h0 + (r < rows_left ? r : rows_left - 1);
            xsrc[i] = reinterpret_cast<const u32x4*>(static_cast<const __bf16*>(a.x) + x_row(a, rs) * kK) + xc;
        }
        const bool live0 = wm * 32 < rows_left;
        const bool live1 = wm * 32 + 16 < rows_left;
        u32x4 wreg[AHEAD][kWChunks];
        u32x4 xreg[AHEAD][kXChunks];
        TermLoad treg[AHEAD];
        f32x8 acc[2][2];
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
            for (int j = 0; j < 2; ++j) acc[mt][j] = zero;
        }
        // Every load is unconditional (a branch forces vmcnt(0) at a stash); terms load with a span's first window
        constexpr bool kTermsStatic = AHEAD % kTermEvery == 0;
        auto load = [&]<int R, bool TERMS = true>(std::integral_constant<int, R>, int w, std::bool_constant<TERMS> = {}) {
            const int wl = w < kWindows ? w : kWindows - 1;
#pragma unroll
            for (int i = 0; i < kWChunks; ++i) wreg[R][i] = *reinterpret_cast<const u32x4*>(wsrc[i] + wl * kGW * BITS);
#pragma unroll
            for (int i = 0; i < kXChunks; ++i) xreg[R][i] = xsrc[i][wl * kXPer];
            if constexpr (TERMS) treg[R] = tsrc[wl / kTermEvery];
        };
        auto stash = [&]<int R>(std::integral_constant<int, R>, int w) {
#pragma unroll
            for (int i = 0; i < kWChunks; ++i) {
                const int row = (256 * i + tid) / kGW;
                const uint32_t wv[BITS] = {wreg[R][i].x, wreg[R][i].y, wreg[R][i].z, wreg[R][i].w};
                u32x4* dst = reinterpret_cast<u32x4*>(ws + row * kLdw + wq * 16);
#pragma unroll
                for (int v = 0; v < 4; ++v) {
                    uint32_t q[4];
#pragma unroll
                    for (int h = 0; h < 4; ++h) q[h] = __builtin_bit_cast(uint32_t, Dec::template pair<T>(wv, v * 8 + h * 2));
                    dst[v] = u32x4{q[0], q[1], q[2], q[3]};
                }
            }
#pragma unroll
            for (int i = 0; i < kXChunks; ++i) {
                const int row = (256 * i + tid) / kXPer;
                *reinterpret_cast<u32x4*>(xs + row * kLdw + xc * 4) = xreg[R][i];
                // the group's sum in order: quad lane 0 adds its 8 values and hands the sum on, ..., lane 3 stores it
                const uint32_t q[4] = {xreg[R][i].x, xreg[R][i].y, xreg[R][i].z, xreg[R][i].w};
                float run = 0.f;
#pragma unroll
                for (int hop = 0; hop < 4; ++hop) {
                    const float in = __shfl_up(run, 1, 4);
                    if ((tid & 3) == hop) {
                        run = hop == 0 ? 0.f : in;
#pragma unroll
                        for (int h = 0; h < 4; ++h) {
                            const typename T::pair p = __builtin_bit_cast(typename T::pair, q[h]);
                            run += T::lo(p);
                            run += T::hi(p);
                        }
                    }
                }
                if ((tid & 3) == 3) sx[xc >> 2][row] = run;
            }
            if (w % kTermEvery == 0) {
                const uint32_t* tv = reinterpret_cast<const uint32_t*>(&treg[R]);
                float(*dst)[kBN] = thigh ? bi : sc;
#pragma unroll
                for (int j = 0; j < kTG; ++j) dst[j][trow] = __uint_as_float(((tv[j >> 1] >> (16 * (j & 1))) & 0xffffu) << 16);
            }
        };

        lds_barrier();
        [&]<int... R>(std::integer_sequence<int, R...>) {
            (load(std::integral_constant<int, R>{}, R, std::bool_constant<!kTermsStatic || R % kTermEvery == 0>{}), ...);
        }(std::make_integer_sequence<int, AHEAD>{});
        stash(std::integral_constant<int, 0>{}, 0);
        load(std::integral_constant<int, 0>{}, AHEAD);
        lds_barrier();
        // Window w: compute it, stash window w + 1 from its slot, refill the slot with window w + 1 + AHEAD.
        auto window = [&]<int J>(std::integral_constant<int, J>, int w) {
            constexpr int next = (J + 1) % AHEAD;
            if (live0) {
#pragma unroll
                for (int gi = 0; gi < kGW; ++gi) {
                    f32x8 d[2][2];
#pragma unroll
                    for (int t = 0; t < 2; ++t) {
                        bf16x16 bf[2];
#pragma unroll
                        for (int j = 0; j < 2; ++j) bf[j] = frag16(ws + lines[j] * kLdw + gi * 16 + t * 8);
#pragma unroll
                        for (int mt = 0; mt < 2; ++mt) {
                            if (mt == 1 && !live1) continue;
                            const bf16x16 af = frag16(xs + (wm * 32 + mt * 16 + rl) * kLdw + gi * 16 + t * 8);
#pragma unroll
                            for (int j = 0; j < 2; ++j) d[mt][j] = T::mma(af, bf[j], t == 0 ? zero : d[mt][j]);
                        }
                    }
                    const int gt = (kGW * w + gi) % kTG;
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt) {
                        if (mt == 1 && !live1) continue;
                        float s_x[8];
#pragma unroll
                        for (int i = 0; i < 8; ++i) s_x[i] = sx[gi][wm * 32 + mt * 16 + 2 * i + rh];
#pragma unroll
                        for (int j = 0; j < 2; ++j) {
                            const float scale = sc[gt][lines[j]];
                            const float bias = bi[gt][lines[j]];
#pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                acc[mt][j][i] = Dec::fold_scale(acc[mt][j][i], d[mt][j][i], scale);
                                acc[mt][j][i] = Dec::fold_bias(acc[mt][j][i], s_x[i], bias);
                            }
                        }
                    }
                }
            }
            lds_barrier();
            // past the last window the stash rewrites LDS no later window reads
            stash(std::integral_constant<int, next>{}, w + 1);
            load(std::integral_constant<int, next>{}, w + 1 + AHEAD, std::bool_constant<!kTermsStatic || (J + 1) % kTermEvery == 0>{});
            lds_barrier();
        };
        for (int w0 = 0; w0 < kWindows; w0 += AHEAD) {
            [&]<int... J>(std::integer_sequence<int, J...>) {
                (window(std::integral_constant<int, J>{}, w0 + J), ...);
            }(std::make_integer_sequence<int, AHEAD>{});
        }

#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int row = wm * 32 + mt * 16 + 2 * i + rh;
                if (row >= rows_left) continue;
                const long long orow = out_row(a, h0 + row);
                if constexpr (S::kAct) {
                    // moe_act_kernel on the pair's fp32 sums: clamp when limit > 0, g / (1 + exp(-g)) * u, hip_bfloat16
                    float g = acc[mt][0][i];
                    float u = acc[mt][1][i];
                    if (limit > 0.f) {
                        g = fminf(g, limit);
                        u = fminf(fmaxf(u, -limit), limit);
                    }
                    const hip_bfloat16 v(g / (1.f + expf(-g)) * u);
                    reinterpret_cast<uint16_t*>(a.out16)[orow * S::kNI + 64 * bx + wn * 16 + rl] = v.data;
                } else {
#pragma unroll
                    for (int j = 0; j < 2; ++j) a.out[orow * S::kN + S::col(bx, lines[j])] = acc[mt][j][i];
                }
            }
        }
    }
#else
    (void)a;
    (void)limit;
    __builtin_trap();
#endif
}

}  // namespace rocm
}  // namespace tf
