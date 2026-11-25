// Hy4-flash's absorbed gated MLA on CUDA (docs/recipes/hy4-flash.md, decision 1): hy4_mla.cuh's bodies as served.

#include "mla_paged.cuh"

// q_nope absorbed through kv_b's key rows into fp32 latent queries; grid (heads, 4, rows / 16), 128 threads.
extern "C" __global__ void __launch_bounds__(128)
    tf_hy4_mla_absorb(const __nv_bfloat16* q, int64_t qs, const __nv_bfloat16* kvb, float* ql, int rows, int heads) {
  tf_hy4::absorb<tf_hy4::kNone>(q, qs, kvb, ql, rows, heads);
}

// Attention over the latent cache with sinks (window, span or list); grid (rows, heads / 8), 256 threads, kAttnSmem.
extern "C" __global__ void __launch_bounds__(256)
    tf_hy4_mla_attn(const float* ql, const __nv_bfloat16* qr, const __nv_bfloat16* cache, const int* pos,
                    const int* idx, const float* sinks, float* ol, int heads, int window, int nidx, int ring,
                    float scale) {
  tf_hy4::attn<tf_hy4::kNone>(ql, qr, cache, pos, idx, sinks, ol, heads, window, nidx, ring, scale);
}

// The weighted latent through kv_b's value rows, times the sigmoid gate; grid (heads, 8, rows / 16), 128 threads.
extern "C" __global__ void __launch_bounds__(128)
    tf_hy4_mla_value(const float* ol, const __nv_bfloat16* kvb, const __nv_bfloat16* gate, int64_t gs,
                     __nv_bfloat16* out, int64_t os, int rows, int heads) {
  tf_hy4::value<tf_hy4::kNone>(ol, kvb, gate, gs, out, os, rows, heads);
}

// The ranks' o_proj partials summed (the all-reduce seam), rounded to bf16, then post_proj_norm; a row a block of 256.
extern "C" __global__ void __launch_bounds__(256)
    tf_hy4_mla_post(const float* parts, int64_t plane, int world, const __nv_bfloat16* w, __nv_bfloat16* out,
                    int width, float eps) {
  tf_hy4::post<tf_hy4::kNone>(parts, plane, world, w, out, width, eps);
}

// tf_hy4_mla_attn with row r's own cache at caches[r] (batched decode); grid (rows, heads / 8), 256 threads, kAttnSmem.
extern "C" __global__ void __launch_bounds__(256)
    tf_hy4_mla_attn_rows(const float* ql, const __nv_bfloat16* qr, const unsigned long long* caches, const int* pos,
                         const int* idx, const float* sinks, float* ol, int heads, int window, int nidx, int ring,
                         float scale) {
  tf_hy4::attn_rows<tf_hy4::kNone>(ql, qr, caches, pos, idx, sinks, ol, heads, window, nidx, ring, scale);
}
