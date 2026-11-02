#pragma once

// BF16 dense kernels: the shared prefill GEMM tile over the BF16 decoder, named as the launchers look up.

#include "quant/act.hpp"
#include "quant/bf16_decoder.hpp"
#include "tiles/epilogue.hpp"
#include "tiles/gemm.hpp"

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

}  // namespace rocm
}  // namespace tf
