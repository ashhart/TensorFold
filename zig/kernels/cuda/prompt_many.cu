// One grid of prompt_lean's FP4 tiles over the NVFP4 projections of one bf16 input.
#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "common/qmm_frag.cuh"
#include "common/fp_pairs.cuh"

namespace tf_prompt_many {

using namespace qmm_frag;

// the tile both this image and prompt16.cu's prompt_lean compile (common/prompt_lean.cuh)
#include "common/prompt_lean.cuh"

using tf_common::fp4pair;

// prompt_lean's bits, its tiles spread over the parts
template <bool F32>
__device__ __forceinline__ void lean_many(
        const __nv_bfloat16* __restrict__ x, const Parts parts, int M, int K, int ldx, int group) {
    lean_tile_body<FP4, 128, 256, 1, 8, 3, F32>(x, LeanManyRaw{parts}, M, K, ldx, group);
}

}  // namespace tf_prompt_many

// The instances the Zig launcher loads by name: one grid, every part's own prompt_lean bits.
extern "C" __global__ void __launch_bounds__(256, 1) tf_prompt16_many(
        const __nv_bfloat16* __restrict__ x, const __grid_constant__ tf_prompt_many::Parts parts, int M, int K, int ldx, int group) {
    tf_prompt_many::lean_many<false>(x, parts, M, K, ldx, group);
}

extern "C" __global__ void __launch_bounds__(256, 1) tf_prompt16_many_f32(
        const __nv_bfloat16* __restrict__ x, const __grid_constant__ tf_prompt_many::Parts parts, int M, int K, int ldx, int group) {
    tf_prompt_many::lean_many<true>(x, parts, M, K, ldx, group);
}
