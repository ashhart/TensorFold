// Prefill attention for long prompts: 64 query rows share a 64-key tile; dot2 products and online softmax in fp32.

#include "attention/prefill.hpp"

// The served tile masks each key past a row's edge only where dot2 rounds a -0 lane apart (RDNA 3.5).
constexpr bool kMasked = TF_DOT2_SIGNED_ZERO != 0;

extern "C" __global__ void __launch_bounds__(kThreads, 1)
tf_fa_wide_f16(const float* q, const __half* k, const __half* v, float* out, int qlen, int span, int heads,
               int kv_heads, int d, float scale, int q_pos0, long long k_sb, long long k_sh, long long k_ss,
               long long v_sb, long long v_sh, long long v_ss) {
    fa_dot2<__half, false, true, kMasked, false>(q, k, v, out, qlen, span, heads, kv_heads, d, scale, q_pos0, k_sb, k_sh, k_ss, v_sb, v_sh, v_ss, nullptr, 0);
}

extern "C" __global__ void __launch_bounds__(kThreads, 1)
tf_fa_wide_bf16(const float* q, const hip_bfloat16* k, const hip_bfloat16* v, float* out, int qlen, int span,
                int heads, int kv_heads, int d, float scale, int q_pos0, long long k_sb, long long k_sh,
                long long k_ss, long long v_sb, long long v_sh, long long v_ss) {
    fa_dot2<hip_bfloat16, false, true, kMasked, false>(q, k, v, out, qlen, span, heads, kv_heads, d, scale, q_pos0, k_sb, k_sh, k_ss, v_sb,
                                 v_sh, v_ss, nullptr, 0);
}

// Full attention for one image. The strides and signature match the contiguous causal tile.
extern "C" __global__ void __launch_bounds__(kThreads, 1)
tf_fa_full_bf16(const float* q, const hip_bfloat16* k, const hip_bfloat16* v, float* out, int qlen, int span,
                int heads, int kv_heads, int d, float scale, int q_pos0, long long k_sb, long long k_sh,
                long long k_ss, long long v_sb, long long v_sh, long long v_ss) {
    fa_dot2<hip_bfloat16, false, false, false, false>(q, k, v, out, qlen, span, heads, kv_heads, d, scale, q_pos0, k_sb,
                                      k_sh, k_ss, v_sb, v_sh, v_ss, nullptr, 0);
}

// The same tile over a stream's pages: `k` and `v` are a layer's pools, `table` the stream's page ids.
extern "C" __global__ void __launch_bounds__(kThreads, 1)
tf_fa_paged_f16(const float* q, const __half* k, const __half* v, float* out, int qlen, int span, int heads,
                int kv_heads, int d, float scale, int q_pos0, const unsigned* table, int count) {
    fa_dot2<__half, true, true, kMasked, false>(q, k, v, out, qlen, span, heads, kv_heads, d, scale, q_pos0, 0, 0, 0, 0, 0, 0, table, count);
}

extern "C" __global__ void __launch_bounds__(kThreads, 1)
tf_fa_paged_bf16(const float* q, const hip_bfloat16* k, const hip_bfloat16* v, float* out, int qlen, int span,
                 int heads, int kv_heads, int d, float scale, int q_pos0, const unsigned* table, int count) {
    fa_dot2<hip_bfloat16, true, true, kMasked, false>(q, k, v, out, qlen, span, heads, kv_heads, d, scale, q_pos0, 0, 0, 0, 0, 0, 0, table, count);
}

// Edge pass: a block reruns masked when a key it masks holds a non-finite value; a trap where the tile is masked.
template <typename Cache, bool PAGED, typename... A>
__device__ __forceinline__ void fa_edge(A... a) {
#if TF_DOT2_SIGNED_ZERO
    __builtin_trap();
#else
    fa_dot2<Cache, PAGED, true, true, true>(a...);
#endif
}

extern "C" __global__ void __launch_bounds__(kThreads, 1)
tf_fa_wide_edge_f16(const float* q, const __half* k, const __half* v, float* out, int qlen, int span, int heads,
                    int kv_heads, int d, float scale, int q_pos0, long long k_sb, long long k_sh, long long k_ss,
                    long long v_sb, long long v_sh, long long v_ss) {
    fa_edge<__half, false>(q, k, v, out, qlen, span, heads, kv_heads, d, scale, q_pos0, k_sb, k_sh, k_ss, v_sb, v_sh,
                           v_ss, nullptr, 0);
}

extern "C" __global__ void __launch_bounds__(kThreads, 1)
tf_fa_wide_edge_bf16(const float* q, const hip_bfloat16* k, const hip_bfloat16* v, float* out, int qlen, int span,
                     int heads, int kv_heads, int d, float scale, int q_pos0, long long k_sb, long long k_sh,
                     long long k_ss, long long v_sb, long long v_sh, long long v_ss) {
    fa_edge<hip_bfloat16, false>(q, k, v, out, qlen, span, heads, kv_heads, d, scale, q_pos0, k_sb, k_sh, k_ss, v_sb,
                                 v_sh, v_ss, nullptr, 0);
}

extern "C" __global__ void __launch_bounds__(kThreads, 1)
tf_fa_paged_edge_f16(const float* q, const __half* k, const __half* v, float* out, int qlen, int span, int heads,
                     int kv_heads, int d, float scale, int q_pos0, const unsigned* table, int count) {
    fa_edge<__half, true>(q, k, v, out, qlen, span, heads, kv_heads, d, scale, q_pos0, 0, 0, 0, 0, 0, 0, table, count);
}

extern "C" __global__ void __launch_bounds__(kThreads, 1)
tf_fa_paged_edge_bf16(const float* q, const hip_bfloat16* k, const hip_bfloat16* v, float* out, int qlen, int span,
                      int heads, int kv_heads, int d, float scale, int q_pos0, const unsigned* table, int count) {
    fa_edge<hip_bfloat16, true>(q, k, v, out, qlen, span, heads, kv_heads, d, scale, q_pos0, 0, 0, 0, 0, 0, 0, table,
                                count);
}
