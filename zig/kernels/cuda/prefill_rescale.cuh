// The opt-in whole-prompt path may skip an identity rescale only for an unchanged finite maximum.
#pragma once

#include <math.h>

namespace tf_prefill_attention {

__host__ __device__ __forceinline__ bool keep_prompt_rescale(float old_max, float new_max) {
#ifdef TF_QPROMPT_RESCALE_CONTROL
    return isfinite(old_max);  // contract_controls.cu only: drops the unchanged-maximum requirement
#else
    return old_max == new_max && isfinite(old_max);
#endif
}

}  // namespace tf_prefill_attention
