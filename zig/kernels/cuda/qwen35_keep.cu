// Keeping a verified window's path (cuda_cache.zig): the recurrences take its logged updates and its keys move down.

#include "qwen35.cuh"

using namespace tf_q35;

// apply, apply_pend and conv: main's kernels, then their _table copies (qwen35_keep_state.cuh, TF_Q35_K)
#define TF_Q35_TABLE 0
#include "qwen35_keep_state.cuh"
#undef TF_Q35_TABLE
#define TF_Q35_TABLE 1
#include "qwen35_keep_state.cuh"
#undef TF_Q35_TABLE

// The path's keys and values move down to the slots after the cache; row j lands at or below j, never on one unread.
extern "C" __global__ void __launch_bounds__(256) tf_q35k_compact(Geo geo, Seg sg, Path path) {
    const int KVH = geo.kvh;
    const int layer = blockIdx.x, h = blockIdx.y;
    __nv_bfloat16* kv = reinterpret_cast<__nv_bfloat16*>(sg.kv);
    const int d = threadIdx.x;
    for (int j = 0; j < path.n; j++) {
        const int from = sg.len + path.rows[j], to = sg.len + j;
        if (from != to) {
            kv[kv_row(sg, KVH, layer, 0, h, to) + d] = kv[kv_row(sg, KVH, layer, 0, h, from) + d];
            kv[kv_row(sg, KVH, layer, 1, h, to) + d] = kv[kv_row(sg, KVH, layer, 1, h, from) + d];
        }
        __syncthreads();
    }
}
