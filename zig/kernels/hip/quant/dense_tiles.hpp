#pragma once

// BF16 dense kernels: the shared prefill GEMM tile over the BF16 decoder, named as the launchers look up.

#include "quant/act.hpp"
#include "quant/bf16_decoder.hpp"
#include "tiles/epilogue.hpp"
#include "tiles/gemm.hpp"
#include "tiles/wmma12_gemm.hpp"

namespace tf {
namespace rocm {

// y = x . W^T for BF16 x and BF16 W at any row count; a build without the BF16 dot2 keeps the symbol and traps.
template <typename T, int RT>
__global__ void __launch_bounds__(256) dense_gemm_block(Affine a) {
#if TF_DEVICE_BF16_DOT2
    gemm_tile<Bf16Decoder, IdentityAct<T>, T, GemmOut<T>, RT>(a);
#else
    (void)a;
    __builtin_trap();
#endif
}

// The same product on gfx12's WMMA tile: its own sum order (k order, 16 a step), so its own bound, not dot2's bytes.
// A build without gfx12 WMMA keeps the symbol and traps.
template <typename T, class G>
__global__ void __launch_bounds__(G::Threads) dense_wmma12_gemm(Affine a) {
    wmma12_gemm_tile<G, IdentityAct<T>, GemmOut<T>>(a);
}

// The same product over a packed gate/up weight, writing silu(gate) * up as tf_silu_mul would (act is n / 2 wide).
template <typename T, class G>
__global__ void __launch_bounds__(G::Threads) dense_wmma12_gemm_silu(Affine a) {
    wmma12_gemm_tile<G, IdentityAct<T>, SiluPairOut<T>>(a);
}

}  // namespace rocm
}  // namespace tf
