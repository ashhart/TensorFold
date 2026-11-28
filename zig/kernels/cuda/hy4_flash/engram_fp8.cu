// Production Hy4-flash FP8 Engram CUDA kernels: 100-byte row decode, rank-0 bias, and 4-stream gated residual.
#include "engram_fp8.cuh"

using namespace hy4_engram_fp8_impl;

extern "C" __global__ void __launch_bounds__(kThreads)
    hy4_engram_fp8_embed(const unsigned char* raw, const unsigned short* head_norms, float eps, unsigned short* out,
                         int pairs, int heads, int dim, int repeat) {
    engram_embed<0>(raw, head_norms, eps, out, pairs, heads, dim, repeat);
}

extern "C" __global__ void __launch_bounds__(kThreads)
    hy4_engram_fp8_bias(const float* x, const unsigned short* bias, float* out, int rows, int cols, int repeat) {
    engram_bias<0>(x, bias, out, rows, cols, repeat);
}

extern "C" __global__ void __launch_bounds__(kThreads)
    hy4_engram_fp8_gate(const unsigned short* streams_in, const float* proj, const unsigned short* norm_keys,
                        const unsigned short* norm_queries, const unsigned short* pre_gamma,
                        const unsigned short* stream_scales, const unsigned char* mask, float eps,
                        unsigned short* engram_out, unsigned short* streams_out, int hidden, int repeat) {
    engram_gate<0>(streams_in, proj, norm_keys, norm_queries, pre_gamma, stream_scales, mask, eps, engram_out,
                   streams_out, hidden, repeat);
}
