#pragma once

// The gfx12 WMMA pieces below a tile's K loop, shared by the tiles that stage BF16 operands in LDS: the block
// geometry and its staging layout, the LDS-only barrier, the 16-deep BF16 step, the operand loads, and the staging
// the contract's controls break. A tile keeps its own K loop and epilogue (tiles/wmma12_gemm.hpp).
// tiles/wmma12_gemm_k32_ref.hpp does not include this header: it is a frozen copy that the served block must equal.

#include "common/arch.hpp"
#include "common/vec.hpp"
#include "common/wmma.hpp"

namespace tf {
namespace rocm {

// A block of BM x BN outputs, WM x WN waves of (BM / WM) x (BN / WN), K steps of BK through double-buffered LDS.
// Staged rows are padded by 8 halves when that fits 64 KiB; past it they are unpadded with their 16-byte chunks
// XOR-swizzled by row (w12_at), which spreads the readers' banks without the pad.
template <int BM_, int BN_, int BK_, int WM_, int WN_>
struct W12Geo {
    static constexpr int BM = BM_, BN = BN_, BK = BK_, WM = WM_, WN = WN_;
    static constexpr bool Padded = 2 * (BM + BN) * (BK + 8) * 2 <= 64 * 1024;
    static constexpr int Ld = Padded ? BK + 8 : BK;  // halves a staged row, 16-byte aligned
    static constexpr int Threads = 32 * WM * WN;
    static_assert(BM % (16 * WM) == 0 && BN % (16 * WN) == 0, "a wave's sub-tile is whole 16 x 16 products");
    static_assert(BK % 16 == 0, "a K step is whole 16-deep products");
    static_assert(Padded || BK % 64 == 0, "an unpadded row holds whole groups of 8 chunks to swizzle");
    static_assert(2 * (BM + BN) * Ld * 2 <= 64 * 1024, "double-buffered x and W rows fit 64 KiB of LDS");
};

typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));

// The half offset of 16-byte chunk c of staged row r: in place when padded, else at c ^ (r % 8).
template <class G>
__device__ __forceinline__ int w12_at(int r, int c) {
    if constexpr (G::Padded) return r * G::Ld + 8 * c;
    else return r * G::Ld + 8 * (c ^ (r % 8));
}

// The served step, a 16-deep BF16 product into fp32 sums; the contract's control swaps in a less exact one.
struct W12Step {
#if TF_DEVICE_WMMA_GFX12
    __device__ static f32x8 mma(bf16x8 a, bf16x8 b, f32x8 c) {
        return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
    }
#endif
};

// The served tile's staging: operand loads undo the stores' swizzle, and the K loop's barriers order LDS. The
// contract's controls each break one (a swizzle the loads ignore, a K loop without its barriers) and must go red.
struct W12Exact {
    static constexpr bool swizzled_loads = true, ordered = true;
};

// The K loop's barrier: LDS stores before it are seen by LDS loads after it; global memory is not ordered (none moves).
// A control whose K loop is unordered gets no barrier.
template <class Fault>
__device__ __forceinline__ void w12_sync() {
    if constexpr (Fault::ordered) {
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
    }
}

// Stores an operand's chunk c of staged row r where w12_load reads it.
template <class G>
__device__ __forceinline__ void w12_store(unsigned short* stage, int r, int c, u32x4 v) {
    *reinterpret_cast<u32x4*>(&stage[w12_at<G>(r, c)]) = v;
}

// An operand's chunk c of staged row r as the stores laid it out (a control's loads may ignore the swizzle).
template <class G, class Fault>
__device__ __forceinline__ bf16x8 w12_load(const unsigned short* stage, int r, int c) {
    const int at = Fault::swizzled_loads ? w12_at<G>(r, c) : r * G::Ld + 8 * c;
    return *reinterpret_cast<const bf16x8*>(&stage[at]);
}

}  // namespace rocm
}  // namespace tf
