// ERNIE 5.5's HCA attention: core's chunk tiles over 512-wide q and keys (a key row is its value), as listed per row.
#include <cuda_bf16.h>
#include <stdint.h>

#include "common/mla.cuh"

// 16 heads a block, a row's listed keys 512 a chunk, partials (o, m, l) for mla's tf_mla_merge_sink; wide_smem<16>.
extern "C" __global__ void __launch_bounds__(256)
    tf_ernie55_hca_chunks(const __nv_bfloat16* qa, const __nv_bfloat16* lc, const int* pos, const int* tok,
                          const int* cnt, float* po, float* pm, float* pl, int64_t rows, int64_t heads, int64_t w,
                          float scale, int sparse, const int* pt, int64_t pstep) {
  glm_attn::chunks_wide<16>(qa, lc, pos, tok, cnt, po, pm, pl, rows, heads, w, scale, sparse, pt, pstep);
}
