#pragma once

// ModelOpt NVFP4 kernels: the shared tiles over Nvfp4Decoder, named as the launcher will look them up.

#include <type_traits>

#include "quant/act.hpp"
#include "quant/mlx_tiles.hpp"
#include "quant/nvfp4_decoder.hpp"
#include "tiles/dot2.hpp"
#include "tiles/epilogue.hpp"
#include "tiles/gemm.hpp"
#include "tiles/matrix_gemm.hpp"
#include "tiles/matrix_item.hpp"
#include "tiles/stream.hpp"

namespace tf {
namespace rocm {

// e2m1's codes are a nibble, eight to a word.
constexpr int kNvfp4Bits = 4;

// ---- the prefill GEMM tile ----

template <typename T, int RT>
__global__ void __launch_bounds__(256) nvfp4_gemm_block(Nvfp4 a) {
    gemm_tile<Nvfp4Decoder<kNvfp4Bits>, IdentityAct<T>, T, GemmOut<T>, RT>(a);
}

// ---- the prefill GEMM tile on the matrix cores of gfx11 ----

__global__ void __launch_bounds__(256) nvfp4_wmma_gemm(Nvfp4 a) {
    wmma_gemm_tile<Nvfp4Decoder<kNvfp4Bits>, BF16Act, WmmaBF16, GemmOut<WmmaBF16>>(a);
}

// ---- the same on RM x 16 rows a block, for a routed item of few rows ----

template <int RM>
__global__ void __launch_bounds__(256) nvfp4_wmma_item(Nvfp4 a) {
    wmma_item_tile<Nvfp4Decoder<kNvfp4Bits>, BF16Act, WmmaBF16, GemmOut<WmmaBF16>, RM>(a);
}

// ---- the decode stream tile ----

// Item z of the plan (or the plain product, or a group's side) at CB columns a lane, R rows; PAIR fuses the activation.
template <typename T, int R, int CB, bool PAIR>
__global__ void __launch_bounds__(32 * kStreamWaves) nvfp4_dot2_stream(Nvfp4 a, StreamSides sides, int lpc_log2,
                                                                     int gshift) {
    using Epi = std::conditional_t<PAIR, StreamSwiglu<T>, StreamRound<T>>;
    stream_tile<Nvfp4Decoder<kNvfp4Bits>, IdentityAct<T>, T, Epi, R, CB>(a, sides, lpc_log2, gshift);
}

}  // namespace rocm
}  // namespace tf
