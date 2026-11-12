// MLA over a 512-wide latent with 64 roped columns and 128-wide heads (Ling-3.1), on common/mla.cuh's bodies.
#include <cuda_bf16.h>
#include <stdint.h>

#include "common/mla.cuh"

// The 16-head chunk tiles over 576-wide q and keys (the latent, then 64 roped columns), in wide_smem<16, 64>() bytes.
extern "C" __global__ void __launch_bounds__(256)
    tf_mla_chunks_r16(const __nv_bfloat16* qa, const __nv_bfloat16* lc, const int* pos, const int* tok, const int* cnt,
                      float* po, float* pm, float* pl, int64_t rows, int64_t heads, int64_t w, float scale, int sparse,
                      const int* pt, int64_t pstep) {
  glm_attn::chunks_wide<16, false, 64>(qa, lc, pos, tok, cnt, po, pm, pl, rows, heads, w, scale, sparse, pt, pstep);
}

// tf_mla_chunks_r16 over a split cache: block z is chunk z * world + rank of the positions, page z of `lc`.
extern "C" __global__ void __launch_bounds__(256)
    tf_mla_chunks_split_r16(const __nv_bfloat16* qa, const __nv_bfloat16* lc, const int* pos, float* po, float* pm,
                            float* pl, int64_t rows, int64_t heads, float scale, int world, int rank) {
  glm_attn::chunks_wide<16, false, 64, true>(qa, lc, pos, nullptr, nullptr, po, pm, pl, rows, heads, 0, scale, 0,
                                             nullptr, 0, nullptr, world, rank);
}

// One rank's chunks of a split cache merged into one (o, m, l in fp32), block (row, head) of 128.
extern "C" __global__ void __launch_bounds__(128)
    tf_mla_merge_part(const float* po, const float* pm, const float* pl, float* qo, float* qm, float* ql, int64_t rows,
                      int64_t heads, int nch) {
  tf_mla::merge_part(po, pm, pl, qo, qm, ql, rows, heads, nch);
}

// q_nope (the first 128 of each head's qs columns) absorbed into qa rows qaw wide (the latent, then the roped q).
extern "C" __global__ void __launch_bounds__(128)
    tf_mla_absorb_q_128(const __nv_bfloat16* q, const __nv_bfloat16* wk, __nv_bfloat16* qa, int64_t rows,
                        int64_t heads, int64_t rb, int64_t qs, int64_t qaw) {
  tf_mla::absorb_q<128>(q, wk, qa, rows, heads, rb, qs, qaw);
}

// The latent output expanded into 128-wide v heads.
extern "C" __global__ void __launch_bounds__(128)
    tf_mla_expand_v_128(const __nv_bfloat16* ol, const __nv_bfloat16* wv, __nv_bfloat16* out, int64_t rows,
                        int64_t heads, int64_t rb) {
  tf_mla::expand_v<128>(ol, wv, out, rows, heads, rb);
}

// The 64 roped columns (GPT-J pairs) at pos + row, cos and sin times YaRN's attention factor.
extern "C" __global__ void __launch_bounds__(32)
    tf_mla_rope(const __nv_bfloat16* x, int64_t xs, int64_t xh, __nv_bfloat16* out, int64_t os, int64_t oh,
                const int* pos, const float* inv_freq, float scale) {
  tf_mla::rope(x, xs, xh, out, os, oh, pos, inv_freq, scale);
}

// The head-wise output gate, block (row, head) of dv threads.
extern "C" __global__ void tf_mla_gate(__nv_bfloat16* o, int64_t heads, const __nv_bfloat16* g, int64_t gs) {
  tf_mla::gate(o, heads, g, gs);
}
