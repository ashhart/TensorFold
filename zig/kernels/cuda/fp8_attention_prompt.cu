// Native FP8 QK is an opt-in prompt operator; no decode entry point is replaced.
#include "common/fp8_attention_prompt.cuh"

#define TF_FP8_ATTENTION_ARGS                                                                                          \
    const __nv_bfloat16 *q, const uint8_t *k, const uint8_t *v, const float *ks, const float *vs, __nv_bfloat16 *out,  \
        int prefix, int rows, int heads, int kv_heads, int group, float scale, int capacity
#define TF_FP8_ATTENTION_INPUTS q, k, v, ks, vs, out, prefix, rows, heads, kv_heads, group, capacity, scale
#define TF_FP8_ATTENTION_WHOLE(NAME, TERMS)                                                                            \
    extern "C" __global__ __launch_bounds__(128, 1) void NAME(TF_FP8_ATTENTION_ARGS) {                                 \
        tf_fp8_attention::run<TERMS, false>({TF_FP8_ATTENTION_INPUTS, nullptr, 0, 0});                                 \
    }
#define TF_FP8_ATTENTION_PAGED(NAME, TERMS)                                                                            \
    extern "C" __global__ __launch_bounds__(128, 1) void NAME(TF_FP8_ATTENTION_ARGS, const int *pages,                 \
                                                              int64_t page_bytes, int page_tokens) {                   \
        tf_fp8_attention::run<TERMS, true>({TF_FP8_ATTENTION_INPUTS, pages, page_bytes, page_tokens});                 \
    }
TF_FP8_ATTENTION_WHOLE(tf_fp8q_attention_whole, 1)
TF_FP8_ATTENTION_WHOLE(tf_fp8q_attention_whole_residual, 2)
TF_FP8_ATTENTION_PAGED(tf_fp8q_attention_paged, 1)
TF_FP8_ATTENTION_PAGED(tf_fp8q_attention_paged_residual, 2)

extern "C" __global__ __launch_bounds__(128, 1) void tf_fp8q_attention_probe(TF_FP8_ATTENTION_ARGS, float *scores) {
    tf_fp8_attention::run<1, false, true>({TF_FP8_ATTENTION_INPUTS, nullptr, 0, 0}, scores);
}
extern "C" __global__ __launch_bounds__(128, 1) void tf_fp8q_attention_probe_narrow(TF_FP8_ATTENTION_ARGS,
                                                                                    float *scores) {
    tf_fp8_attention::run<1, false, true, true>({TF_FP8_ATTENTION_INPUTS, nullptr, 0, 0}, scores);
}
