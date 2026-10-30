#pragma once

// numerics=inexact: plain 4-bit group-32 MLX products, codes dequantized to bf16 in LDS, one WMMA chain over K.

// Block 256 x 128, 8 waves of 64 x 64: Strata's kernel_bk64 layout (github.com/Niko1221/Strata, MIT, wmma_gemm.cu).

#include "common/wmma.hpp"
#include "quant/mlx.hpp"
#include "tiles/epilogue.hpp"

namespace tf {
namespace rocm {

namespace dense_inexact {
constexpr int kBM = 256;
constexpr int kBN = 128;
constexpr int kXPer = kBM * 2 / 256;  // a thread's 64-byte x groups a window
constexpr int kKW = 64;
constexpr int kLd = kKW + 8;  // bf16 a staged row
constexpr int kThreads = 256;

__device__ __forceinline__ uint32_t round_bf16(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u + (((u >> 16) & 1u) + 0x7fffu)) >> 16;
}

// The 8 codes of a word as 4 BF16 pairs of fma(scale, q, bias), in order.
__device__ __forceinline__ u32x4 dequant8(uint32_t w, float s, float b) {
    uint32_t p[4];
#pragma unroll
    for (int h = 0; h < 4; ++h) {
        const float lo = fmaf(s, static_cast<float>((w >> (8 * h)) & 0xfu), b);
        const float hi = fmaf(s, static_cast<float>((w >> (8 * h + 4)) & 0xfu), b);
        p[h] = round_bf16(lo) | (round_bf16(hi) << 16);
    }
    return u32x4{p[0], p[1], p[2], p[3]};
}
}  // namespace dense_inexact

// K a multiple of 64; fp32 out, or bf16 with out16. BF16W: a.words are bf16 (n, K) rows, copied, not dequantized.
template <int K, bool BF16W = false>
__device__ __forceinline__ void wmma_dense_inexact_tile(const Affine& a) {
#if TF_DEVICE_WMMA_GFX11
    using namespace dense_inexact;
    using T = WmmaBF16;
    constexpr int kWindows = K / kKW;
    constexpr int kGroups = K / 32;
    constexpr int kWordsRow = K / 8;
    static_assert(K % kKW == 0, "whole windows");
    __shared__ __attribute__((aligned(16))) __bf16 xs[kBM * kLd];
    __shared__ __attribute__((aligned(16))) __bf16 ws[kBN * kLd];
    const int num_n = (a.n + kBN - 1) / kBN;
    const int m0 = blockIdx.x / num_n * kBM;
    const int n0 = blockIdx.x % num_n * kBN;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
    const int rl = lane & 15;
    const int rh = lane >> 4;
    // a thread's loads: group q of x rows r and r + 128; group g of column n0 + tid / 2's codes, its scale and bias
    const u32x4* xsrc[kXPer];
    int xr[kXPer];
    const int xq = tid & 1;
#pragma unroll
    for (int i = 0; i < kXPer; ++i) {
        xr[i] = (256 * i + tid) >> 1;
        const int row = m0 + xr[i] < a.m ? m0 + xr[i] : a.m - 1;
        xsrc[i] = reinterpret_cast<const u32x4*>(static_cast<const __bf16*>(a.x) + static_cast<long long>(row) * K) + xq * 4;
    }
    const int wcol = tid >> 1;
    const int wg = tid & 1;
    const int col = n0 + wcol < a.n ? n0 + wcol : a.n - 1;
    const uint4* wsrc = reinterpret_cast<const uint4*>(a.words + static_cast<long long>(col) * kWordsRow) + wg;
    const uint16_t* ssrc = static_cast<const uint16_t*>(a.scale.p) + static_cast<long long>(col) * kGroups + wg;
    const uint16_t* bsrc = static_cast<const uint16_t*>(a.bias.p) + static_cast<long long>(col) * kGroups + wg;
    u32x4 xreg[kXPer][4];
    uint4 wreg;
    uint16_t sreg;
    uint16_t breg;
    // BF16W: the column's 32 bf16 of window w at 16-byte loads 8 w + 4 wg ..
    const u32x4* w16src = reinterpret_cast<const u32x4*>(reinterpret_cast<const __bf16*>(a.words) + static_cast<long long>(col) * K) + wg * 4;
    u32x4 w16reg[BF16W ? 4 : 1];
    // unconditional loads (a window past the last repeats it): no vmcnt(0) at the stash
    auto load = [&](int w) {
        const int wl = w < kWindows ? w : kWindows - 1;
#pragma unroll
        for (int i = 0; i < kXPer; ++i) {
#pragma unroll
            for (int v = 0; v < 4; ++v) xreg[i][v] = xsrc[i][wl * 8 + v];
        }
        if constexpr (BF16W) {
#pragma unroll
            for (int v = 0; v < 4; ++v) w16reg[v] = w16src[wl * 8 + v];
        } else {
            wreg = wsrc[2 * wl];
            sreg = ssrc[2 * wl];
            breg = bsrc[2 * wl];
        }
    };
    auto stash = [&]() {
#pragma unroll
        for (int i = 0; i < kXPer; ++i) {
            u32x4* xd = reinterpret_cast<u32x4*>(xs + xr[i] * kLd + xq * 32);
#pragma unroll
            for (int v = 0; v < 4; ++v) xd[v] = xreg[i][v];
        }
        u32x4* wd = reinterpret_cast<u32x4*>(ws + wcol * kLd + wg * 32);
        if constexpr (BF16W) {
#pragma unroll
            for (int v = 0; v < 4; ++v) wd[v] = w16reg[v];
        } else {
            const float s = __uint_as_float(static_cast<uint32_t>(sreg) << 16);
            const float b = __uint_as_float(static_cast<uint32_t>(breg) << 16);
            const uint32_t wv[4] = {wreg.x, wreg.y, wreg.z, wreg.w};
#pragma unroll
            for (int v = 0; v < 4; ++v) wd[v] = dequant8(wv[v], s, b);
        }
    };
    // wave (wm, wn): rows 64 wm.., columns 64 wn..; a lane: column lane % 16, rows 2 i + lane / 16
    const int wm = wave & 3;
    const int wn = wave >> 2;
    f32x8 acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = f32x8{0, 0, 0, 0, 0, 0, 0, 0};
    }
    load(0);
    stash();
    load(1);
    lds_barrier();
    for (int w = 0; w < kWindows; ++w) {
#pragma unroll
        for (int ks = 0; ks < kKW; ks += 16) {
            bf16x16 bf[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) bf[j] = frag16(reinterpret_cast<const uint32_t*>(ws + (wn * 64 + j * 16 + rl) * kLd + ks));
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const bf16x16 af = frag16(reinterpret_cast<const uint32_t*>(xs + (wm * 64 + i * 16 + rl) * kLd + ks));
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = T::mma(af, bf[j], acc[i][j]);
            }
        }
        lds_barrier();
        // past the last window the stash rewrites LDS no later window reads
        stash();
        load(w + 2);
        lds_barrier();
    }
    auto put = [&](auto store) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int row = m0 + wm * 64 + i * 16 + 2 * e + rh;
                if (row >= a.m) continue;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int c = n0 + wn * 64 + j * 16 + rl;
                    if (c < a.n) store(static_cast<long long>(row) * a.n + c, acc[i][j][e]);
                }
            }
        }
    };
    if (a.out16 != nullptr) {
        auto* out = reinterpret_cast<typename T::elem*>(a.out16);
        put([&](long long at, float v) { out[at] = narrow<T>(v); });
    } else {
        put([&](long long at, float v) { a.out[at] = v; });
    }
#else
    (void)a;
    __builtin_trap();
#endif
}

}  // namespace rocm
}  // namespace tf
