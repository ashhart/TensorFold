#pragma once

// MLX affine kernels: each shared tile over the MLX decoder, identity encoder and Dot, named as the launchers look up.

#include <type_traits>

#include "quant/act.hpp"
#include "quant/mlx_decoder.hpp"
#include "tiles/dot2.hpp"
#include "tiles/epilogue.hpp"
#include "tiles/gemm.hpp"
#include "tiles/gemm_kp.hpp"
#include "tiles/matrix_gemm.hpp"
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
};

// Item z of the plan (or the plain product, or a group's side) at CB columns a lane, R rows; PAIR fuses the activation.
template <typename T, int BITS, int R, int CB, bool PAIR>
__global__ void __launch_bounds__(32 * kStreamWaves) affine_dot2_stream(Affine a, StreamSides sides, int lpc_log2,
                                                                       int gshift) {
    using Epi = std::conditional_t<PAIR, StreamSwiglu<T>, StreamRound<T>>;
    stream_tile<MlxDecoder<BITS>, IdentityAct<T>, T, Epi, R, CB>(a, sides, lpc_log2, gshift);
}


}  // namespace rocm
}  // namespace tf
