// Planted contract controls for Hy4-flash FP8 Engram kernels (fatbin_contract_controls only).
#include "hy4_flash/engram_fp8.cuh"

#define HY4_ENGRAM_FP8_EMBED_CTL(NAME, CONTROL)                                                                        \
    extern "C" __global__ void __launch_bounds__(hy4_engram_fp8_impl::kThreads)                                        \
        NAME(const unsigned char* raw, const unsigned short* head_norms, float eps, unsigned short* out, int pairs,    \
             int heads, int dim, int repeat) {                                                                         \
        hy4_engram_fp8_impl::engram_embed<CONTROL>(raw, head_norms, eps, out, pairs, heads, dim, repeat);              \
    }
HY4_ENGRAM_FP8_EMBED_CTL(tf_ctl_hy4_engram_fp8_embed_bad_nan, 1)
HY4_ENGRAM_FP8_EMBED_CTL(tf_ctl_hy4_engram_fp8_embed_bad_repeat, 2)
HY4_ENGRAM_FP8_EMBED_CTL(tf_ctl_hy4_engram_fp8_embed_bad_tail, 3)
HY4_ENGRAM_FP8_EMBED_CTL(tf_ctl_hy4_engram_fp8_embed_bad_sums, 4)

#define HY4_ENGRAM_FP8_BIAS_CTL(NAME, CONTROL)                                                                         \
    extern "C" __global__ void __launch_bounds__(hy4_engram_fp8_impl::kThreads)                                        \
        NAME(const float* x, const unsigned short* bias, float* out, int rows, int cols, int repeat) {                 \
        hy4_engram_fp8_impl::engram_bias<CONTROL>(x, bias, out, rows, cols, repeat);                                   \
    }
HY4_ENGRAM_FP8_BIAS_CTL(tf_ctl_hy4_engram_fp8_bias_bad_nan, 1)
HY4_ENGRAM_FP8_BIAS_CTL(tf_ctl_hy4_engram_fp8_bias_bad_repeat, 2)
HY4_ENGRAM_FP8_BIAS_CTL(tf_ctl_hy4_engram_fp8_bias_bad_tail, 3)

#define HY4_ENGRAM_FP8_GATE_CTL(NAME, CONTROL)                                                                         \
    extern "C" __global__ void __launch_bounds__(hy4_engram_fp8_impl::kThreads)                                        \
        NAME(const unsigned short* streams_in, const float* proj, const unsigned short* norm_keys,                     \
             const unsigned short* norm_queries, const unsigned short* pre_gamma,                                      \
             const unsigned short* stream_scales, const unsigned char* mask, float eps, unsigned short* engram_out,    \
             unsigned short* streams_out, int hidden, int repeat) {                                                    \
        hy4_engram_fp8_impl::engram_gate<CONTROL>(streams_in, proj, norm_keys, norm_queries, pre_gamma, stream_scales, \
                                                  mask, eps, engram_out, streams_out, hidden, repeat);                 \
    }
HY4_ENGRAM_FP8_GATE_CTL(tf_ctl_hy4_engram_fp8_gate_bad_nan, 1)
HY4_ENGRAM_FP8_GATE_CTL(tf_ctl_hy4_engram_fp8_gate_bad_repeat, 2)
HY4_ENGRAM_FP8_GATE_CTL(tf_ctl_hy4_engram_fp8_gate_bad_tail, 3)
HY4_ENGRAM_FP8_GATE_CTL(tf_ctl_hy4_engram_fp8_gate_bad_sums, 4)

#undef HY4_ENGRAM_FP8_EMBED_CTL
#undef HY4_ENGRAM_FP8_BIAS_CTL
#undef HY4_ENGRAM_FP8_GATE_CTL
