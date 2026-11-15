// Planted control for embed_rows.cu's tf_embed_rows_tiles4: the tied head's tiles read with codes out of order.
#pragma once

#include "embed_affine.cuh"

// Gate 3: embed_rows.cu's tf_embed_rows_tiles4 with `keep`'s rounding points and code order; the exact check goes red.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_embed_tiles4(const int* ids, const uint32_t* w,
                                                                       const __nv_bfloat16* s, const __nv_bfloat16* b,
                                                                       __nv_bfloat16* out, int d, int npad, float scale,
                                                                       int keep) {
    tf_embed_affine::tile4_rows(ids, w, s, b, out, d, npad, scale, keep);
}
