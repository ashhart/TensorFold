// GLM keeps its existing image, entry points, bounds and compiler flags.
#include "common/kda.cuh"

extern "C" __global__ void __launch_bounds__(1024) tf_glm_kda_chain(
        int H, const __nv_bfloat16* __restrict__ P, int p_stride, int b_off, const __nv_bfloat16* __restrict__ A,
        int a_stride, const __nv_bfloat16* __restrict__ G, int g_stride, const __nv_bfloat16* __restrict__ cs,
        const float* __restrict__ cw, const float* __restrict__ state_in, const float* __restrict__ a_log,
        const float* __restrict__ dt_bias, const __nv_bfloat16* __restrict__ norm_w, float eps, float lower, int rows,
        __nv_bfloat16* __restrict__ out, float* __restrict__ state_out, float* __restrict__ k_save,
        __nv_bfloat16* __restrict__ v_save, float* __restrict__ g_save, float* __restrict__ b_save) {
    chain_kernel_body(
        H, P, p_stride, b_off, A, a_stride, G, g_stride, cs, cw, state_in, a_log, dt_bias, norm_w, eps, lower, rows,
        out, state_out, k_save, v_save, g_save, b_save);
}

extern "C" __global__ void __launch_bounds__(1024) tf_glm_kda_replay(
        int H, const float* __restrict__ state_in, const float* __restrict__ k_save,
        const __nv_bfloat16* __restrict__ v_save, const float* __restrict__ g_save, const float* __restrict__ b_save,
        int rows, float* __restrict__ state_out) {
    replay_kernel_body(H, state_in, k_save, v_save, g_save, b_save, rows, state_out);
}

extern "C" __global__ void __launch_bounds__(1024) tf_glm_kda_replay_layers(
        int H, const float* __restrict__ state_in, size_t state_stride, const float* __restrict__ k_save,
        const __nv_bfloat16* __restrict__ v_save, const float* __restrict__ g_save, const float* __restrict__ b_save,
        size_t kv_stride, size_t b_stride, int rows, float* __restrict__ state_out) {
    replay_layers_kernel_body(
        H, state_in, state_stride, k_save, v_save, g_save, b_save, kv_stride, b_stride, rows, state_out);
}

extern "C" __global__ void __launch_bounds__(512) tf_glm_kda_prep(
        int H, const __nv_bfloat16* __restrict__ P, int p_stride, int b_off, const __nv_bfloat16* __restrict__ A,
        int a_stride, const __nv_bfloat16* __restrict__ cs, const float* __restrict__ cw,
        const float* __restrict__ a_log, const float* __restrict__ dt_bias, float lower, float* __restrict__ q_out,
        float* __restrict__ k_save, __nv_bfloat16* __restrict__ v_save, float* __restrict__ g_save,
        float* __restrict__ b_save) {
    prep_kernel_body(
        H, P, p_stride, b_off, A, a_stride, cs, cw, a_log, dt_bias, lower, q_out, k_save, v_save, g_save, b_save);
}

extern "C" __global__ void __launch_bounds__(256) tf_glm_kda_step(
        int H, const float* __restrict__ state_in, const float* __restrict__ q_in, const float* __restrict__ k_save,
        const __nv_bfloat16* __restrict__ v_save, const float* __restrict__ g_save, const float* __restrict__ b_save,
        int rows, __nv_bfloat16* __restrict__ y_out, float* __restrict__ state_out) {
    step_kernel_body<8, 16>(H, state_in, q_in, k_save, v_save, g_save, b_save, rows, y_out, state_out);
}

extern "C" __global__ void __launch_bounds__(128) tf_glm_kda_out(
        int H, const __nv_bfloat16* __restrict__ y_in, const __nv_bfloat16* __restrict__ G, int g_stride,
        const __nv_bfloat16* __restrict__ norm_w, float eps, __nv_bfloat16* __restrict__ out) {
    out_kernel_body(H, y_in, G, g_stride, norm_w, eps, out);
}
