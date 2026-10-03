// FP8 prompts below compute capability 8.9 (no FP8 MMA): the entry points qmm.cpp binds, refusing by name.
#include <torch/extension.h>

void qmm_prefill8w_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, int,
                        int, bool, int, bool) {
    TORCH_CHECK(false, "FP8 prompts need compute capability 8.9 (FP8 MMA); this GPU takes bf16 prompts "
                "(drop --prefill-fp8)");
}

void qmm_prefill8_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                       const at::Tensor&, at::Tensor&, int, int, bool, int) {
    TORCH_CHECK(false, "FP8 prompts need compute capability 8.9 (FP8 MMA); this GPU takes bf16 prompts "
                "(drop --prefill-fp8)");
}
