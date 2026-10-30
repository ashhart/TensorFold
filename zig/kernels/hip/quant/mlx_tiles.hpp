#pragma once

// MLX affine kernels: each shared tile over the MLX decoder, identity encoder and Dot, named as the launchers look up.

#include <type_traits>

#include "quant/act.hpp"
#include "quant/mlx_decoder.hpp"
#include "tiles/dot2.hpp"
#include "tiles/epilogue.hpp"
#include "tiles/gemm.hpp"
#include "tiles/gemm_kp.hpp"
#include "tiles/dense_gemm.hpp"
#include "tiles/dense_inexact.hpp"
#include "tiles/matrix_gemm.hpp"
#include "tiles/moe_gemm.hpp"
#include "tiles/stream.hpp"

namespace tf {
namespace rocm {

// ---- the prefill GEMM tile ----

template <typename T, int BITS, int RT>
__global__ void __launch_bounds__(256) affine_gemm_block(Affine a) {
    gemm_tile<MlxDecoder<BITS>, IdentityAct<T>, T, F32Out, RT>(a);
}

// ---- the K-parallel tiles of a short prompt ----

template <typename T, int BITS, int CB, int R, int WAVES, int RB, bool LOOP>
__global__ void __launch_bounds__(32 * WAVES) affine_gemm_kp(Affine a) {
    gemm_kp_tile<MlxDecoder<BITS>, IdentityAct<T>, T, F32Out, CB, R, WAVES, RB, LOOP>(a);
}

// ---- the prefill GEMM tile on the matrix cores of gfx11 ----

template <int BITS>
__global__ void __launch_bounds__(256) affine_wmma_gemm(Affine a) {
    wmma_gemm_tile<MlxDecoder<BITS>, BF16Act, WmmaBF16, F32Out>(a);
}

// ---- Qwen3.8-Flash-Next's routed expert products of a prompt (tiles/moe_gemm.hpp, gfx11 matrix cores) ----

// 64-wide K windows; each shape's ring depth is its own (MoeGateUp::kAhead, MoeDown::kAhead).

// gate and up with silu(gate) * up as the epilogue, into a.out16 (bf16)
template <int BITS>
__global__ void __launch_bounds__(moe_tile::kThreads) affine_wmma_moe_gu(Affine a, float limit) {
    wmma_moe_tile<MoeGateUp, BITS, 64, MoeGateUp::kAhead>(a, limit);
}

// down into a.out (fp32)
template <int BITS>
__global__ void __launch_bounds__(moe_tile::kThreads) affine_wmma_moe_down(Affine a) {
    wmma_moe_tile<MoeDown, BITS, 64, MoeDown::kAhead>(a, 0.f);
}

// numerics=inexact: the same with the codes dequantized to bf16 (gate+up: 128-row passes of 64 x 32 waves)
template <int BITS>
__global__ void __launch_bounds__(moe_tile::kThreads) affine_wmma_moe_gu_inexact(Affine a, float limit) {
    static_assert(BITS == 4, "4-bit codes");
    wmma_moe_inexact_tile<MoeGateUp, 2, 128, 64>(a, limit);
}

// down in gate+up's 128-row passes of 64 x 32 waves: a plan item of up to 128 pairs is one pass, dequantized once.
template <int BITS>
__global__ void __launch_bounds__(moe_tile::kThreads) affine_wmma_moe_down_inexact(Affine a) {
    static_assert(BITS == 4, "4-bit codes");
    wmma_moe_inexact_tile<MoeDown, 2, 128, 64>(a, 0.f);
}

// ---- a prompt's plain 4-bit group-32 products at Flash Next's K (tiles/dense_gemm.hpp, gfx11 matrix cores) ----

// K 2560 (the trunk's input projections), 6144 (out), 10240 (hc down), 320 (hc up); the bytes of affine_wmma_gemm
template <int K>
__global__ void __launch_bounds__(dense_tile::kThreads) affine_wmma_dense(Affine a) {
    wmma_dense_tile<K, DenseTune<K>::kAhead>(a);
}

// hc up (K 320) with tf_fn_hc_mix in the epilogue; tf_hc_gate: its gate rnd(sigmoid(g)) for every bf16 g, filled once.
__device__ uint16_t tf_hc_gate[1 << 16];
extern "C" __global__ void __launch_bounds__(256) tf_hc_gate_fill() {
    const int h = blockIdx.x * 256 + threadIdx.x;
    const float g = __uint_as_float(static_cast<uint32_t>(h) << 16);
    tf_hc_gate[h] = __builtin_bit_cast(uint16_t, narrow<WmmaBF16>(1.0f / (1.0f + expf(-g))));
}

extern "C" __global__ void __launch_bounds__(dense_tile::kThreads) tf_affine_dense_hcmix(Affine a, HcMix mix) {
    wmma_dense_tile<320, DenseTune<320>::kAhead, true>(a, mix, tf_hc_gate);
}

// numerics=inexact: the same products with the codes dequantized to bf16 in the tile, one WMMA chain (K 2560 and 6144).
template <int K>
__global__ void __launch_bounds__(dense_inexact::kThreads) affine_wmma_dense_inexact(Affine a) {
    wmma_dense_inexact_tile<K>(a);
}

// numerics=inexact: a prompt's MoE router logits (bf16 x, the router's bf16 rows) on the inexact dense tile, fp32 out.
template <int K>
__global__ void __launch_bounds__(dense_inexact::kThreads) affine_wmma_router(Affine a) {
    wmma_dense_inexact_tile<K, true>(a);
}

// ---- the decode stream tile ----

// Up to four products sharing x, K, width and group in one launch; side s owns blocks [first[s], first[s + 1]).
constexpr int kStreamSides = 4;

struct StreamSides {
    const uint32_t* words[kStreamSides];
    const void* scale[kStreamSides];
    const void* bias[kStreamSides];
    void* out[kStreamSides];
    int n[kStreamSides];
    int first[kStreamSides + 1];
    int count;  // 0 is the plain product of the Affine
    int out_half;
    int pair_cols;  // > 0: the (gate | up) rows of a stacked product with this many columns each, out is its activation
    float limit;    // the activation's clamp when above 0
    int row_blocks;  // > 1: the launch's row blocks are the grid's x and its column blocks its y
};

// Item z of the plan (or the plain product, or a group's side) at CB columns a lane, R rows; PAIR fuses the activation.
template <typename T, int BITS, int R, int CB, bool PAIR>
__global__ void __launch_bounds__(32 * kStreamWaves) affine_dot2_stream(Affine a, StreamSides sides, int lpc_log2,
                                                                       int gshift) {
    using Epi = std::conditional_t<PAIR, StreamSwiglu<T>, StreamRound<T>>;
    stream_tile<MlxDecoder<BITS>, IdentityAct<T>, T, Epi, R, CB>(a, sides, lpc_log2, gshift);
}

// The plain product on the held-codes body (stream_body's HELD): the same sums as affine_dot2_stream.
template <typename T, int BITS, int R, int CB>
__global__ void __launch_bounds__(32 * kStreamWaves) affine_dot2_stream_held(Affine a, StreamSides sides, int lpc_log2,
                                                                            int gshift) {
    stream_tile<MlxDecoder<BITS>, IdentityAct<T>, T, StreamRound<T>, R, CB, true>(a, sides, lpc_log2, gshift);
}

// A routed plan of 5 to 8 rows as a work list (stream_list): grid z blocks walk `items`; 16-byte code loads, so 4-bit.
template <typename T, int BITS, int R, int CB, bool PAIR>
__global__ void __launch_bounds__(32 * kStreamWaves) affine_dot2_stream_list(Affine a, StreamSides sides, int lpc_log2,
                                                                            int gshift, int items) {
    using Epi = std::conditional_t<PAIR, StreamSwiglu<T>, StreamRound<T>>;
    stream_list<MlxDecoder<BITS>, IdentityAct<T>, T, Epi, R, CB>(a, sides.pair_cols, sides.limit, lpc_log2, gshift, items);
}

// The plain product (or plan item) with Nemotron's relu^2 as the epilogue, into a.out16 in the activation type.
template <typename T, int BITS, int R, int CB>
__global__ void __launch_bounds__(32 * kStreamWaves) affine_dot2_stream_relu2(Affine a, StreamSides sides, int lpc_log2,
                                                                             int gshift) {
    stream_tile<MlxDecoder<BITS>, IdentityAct<T>, T, StreamRelu2<T>, R, CB>(a, sides, lpc_log2, gshift);
}

// Relu^2 products of one x in one launch: plan a (items on z) on the first `split` blocks, plain b past them at z 0.
template <typename T, int BITS, int R, int CB>
__global__ void __launch_bounds__(32 * kStreamWaves) affine_dot2_stream_relu2_pair(Affine a, Affine b, int split,
                                                                                  int lpc_log2, int gshift) {
    StreamSides sides{};
    if (static_cast<int>(blockIdx.x) < split) {
        stream_tile<MlxDecoder<BITS>, IdentityAct<T>, T, StreamRelu2<T>, R, CB>(a, sides, lpc_log2, gshift);
        return;
    }
    if (blockIdx.z != 0) return;
    sides.count = 1;
    sides.first[0] = split;
    sides.first[1] = gridDim.x;
    sides.words[0] = b.words;
    sides.scale[0] = b.scale.p;
    sides.bias[0] = b.bias.p;
    sides.out[0] = b.out16;
    sides.n[0] = b.n;
    sides.out_half = 1;
    stream_tile<MlxDecoder<BITS>, IdentityAct<T>, T, StreamRelu2<T>, R, CB>(b, sides, lpc_log2, gshift);
}


}  // namespace rocm
}  // namespace tf
