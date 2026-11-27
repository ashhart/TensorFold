// Whole BF16 Qwen prompts only: the original arithmetic with one consume/reuse fence per half item.
#pragma once
#include "common/attn_pipe_schedule.cuh"
namespace tf_prefill_attention {
extern "C" __global__ void __launch_bounds__(256, 1)
tf_qwen_prompt_pipeline(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc,
                       const __nv_bfloat16* __restrict__ vc, __nv_bfloat16* __restrict__ out,
                       int p0, int W, int H, int HK, int G, float scale, int CAP) {
    constexpr int D = 256, WARPS = 8, HPC = 2, NS = 2;
    constexpr bool P256 = false, HPAGED = false, causal = true, F8 = false, HM = true, IDS = false;
    constexpr int PN = 0, skip = 0, PTOK = 1;
    constexpr const int* PT = nullptr;
    constexpr long long PS = 0;
    const int HO = H;
    const float *ks = nullptr, *vs = nullptr;
    auto first_row = [] { return blockIdx.x * 16 * (WARPS / HPC); };
#define PATTN_PIPELINE 1
#include "prefill_attention_rows.cuh"
#undef PATTN_PIPELINE
}
}  // namespace tf_prefill_attention
