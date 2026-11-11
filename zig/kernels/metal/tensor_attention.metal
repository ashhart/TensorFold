
#define IM_ATTN(NAME, TYPE, CAUSAL, DIM) \
[[max_total_threads_per_threadgroup(128)]] [[kernel]] void NAME( \
    const device TYPE* q [[buffer(0)]], const device TYPE* k [[buffer(1)]], const device TYPE* v [[buffer(2)]], \
    const device int* p [[buffer(3)]], const device float* f [[buffer(4)]], device TYPE* o [[buffer(5)]], \
    uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]]) { \
    tfp::attention_nax<TYPE, false, false, CAUSAL, DIM>(q, k, v, o, p, f, tg, sg, lane); \
}
IM_ATTN(im_attention_bf16_causal, bfloat, true, 128)
IM_ATTN(im_attention_bf16_full, bfloat, false, 128)
IM_ATTN(im_attention_f32_causal, float, true, 128)
IM_ATTN(im_attention_f32_full, float, false, 128)
IM_ATTN(im_attention_f32_causal32, float, true, 32)
IM_ATTN(im_attention_f32_full32, float, false, 32)
IM_ATTN(im_attention_bf16_causal32, bfloat, true, 32)
IM_ATTN(im_attention_bf16_full32, bfloat, false, 32)
IM_ATTN(im_attention_bf16_causal64, bfloat, true, 64)
IM_ATTN(im_attention_bf16_full64, bfloat, false, 64)
IM_ATTN(im_attention_f32_causal64, float, true, 64)
IM_ATTN(im_attention_f32_full64, float, false, 64)
IM_ATTN(im_attention_f16_full64, half, false, 64)
IM_ATTN(im_attention_f16_full32, half, false, 32)
IM_ATTN(im_attention_bf16_causal96, bfloat, true, 96)
IM_ATTN(im_attention_bf16_full96, bfloat, false, 96)
IM_ATTN(im_attention_f32_causal96, float, true, 96)
IM_ATTN(im_attention_f32_full96, float, false, 96)
IM_ATTN(im_attention_bf16_causal256, bfloat, true, 256)
IM_ATTN(im_attention_bf16_full256, bfloat, false, 256)
IM_ATTN(im_attention_f32_causal256, float, true, 256)
IM_ATTN(im_attention_f32_full256, float, false, 256)

[[kernel, max_total_threads_per_threadgroup(128)]] void im_attention_f32_window64(
    const device float* q [[buffer(0)]], const device float* k [[buffer(1)]], const device float* v [[buffer(2)]],
    const device int* p [[buffer(3)]], const device float* f [[buffer(4)]], device float* o [[buffer(5)]],
    uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    tfp::attention_nax<float, false, false, true, 64, 128>(q, k, v, o, p, f, tg, sg, lane);
}

[[kernel, max_total_threads_per_threadgroup(128)]] void im_attention_f32_window512(
    const device float* q [[buffer(0)]], const device float* k [[buffer(1)]], const device float* v [[buffer(2)]],
    const device int* p [[buffer(3)]], const device float* f [[buffer(4)]], device float* o [[buffer(5)]],
    uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    tfp::attention_nax<float, false, false, true, 64, 512>(q, k, v, o, p, f, tg, sg, lane);
}
