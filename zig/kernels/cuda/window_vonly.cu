// TF_GEMMA4_V_ONLY (Gemma 4 31B's global layers): V alone in the pages, each key made from its value in attention.

#include "window_rows.cuh"
#include "window_rows512.cuh"
#include "window_vonly_part.cuh"

using namespace tf_wrows;

// tf_wrows_part_512_pg's partials over V-only pages: a block a key head, each key from its value (vk_tile).
extern "C" __global__ void __launch_bounds__(NT)
    tf_wrows_part_512_pg_vo(const __nv_bfloat16* q, const __nv_bfloat16* vc, const int* pos, const int* slot,
                            float* po, float* pm, float* pl, int h, int hk, float scale, const int* tab,
                            long long page, int tok, int stride, const __nv_bfloat16* kw, const float* inv) {
    part_vo<512>(q, vc, pos, slot, po, pm, pl, h, hk, scale, Vk{kw, inv}, Pg{tab, page, tok, stride});
}

// tf_wrows_merge_512_pg over V-only pages: its sums in its order; a non-finite head's float64 replay makes its keys.
extern "C" __global__ void __launch_bounds__(NT)
    tf_wrows_merge_512_pg_vo(const float* po, const float* pm, const float* pl, __nv_bfloat16* out, int h, int hk,
                             int chunks, const __nv_bfloat16* q, const __nv_bfloat16* vc, const int* pos,
                             const int* slot, float scale, const int* tab, long long page, int tok, int stride,
                             const __nv_bfloat16* kw, const float* inv) {
    merge_vo<512>(po, pm, pl, out, h, hk, chunks, q, vc, pos, slot, scale, Pg{tab, page, tok, stride}, Vk{kw, inv});
}

// Rows v [n, row] into the V pages alone: tf_wrows_put's v half.
extern "C" __global__ void __launch_bounds__(128) tf_wrows_put_v(const __nv_bfloat16* v, __nv_bfloat16* vc,
                                                                 const int* pos, const int* slot, const int* tab,
                                                                 long long page, int tok, int stride, int row) {
    const int64_t r = blockIdx.x;
    const int p = pos[r];
    const int64_t id = tab[slot[r] * stride + p / tok];
    const int64_t at = id * page + static_cast<int64_t>(p % tok) * row;
    for (int e = threadIdx.x * 8; e < row; e += 128 * 8)
        *reinterpret_cast<uint4*>(vc + at + e) = *reinterpret_cast<const uint4*>(v + r * row + e);
}

// Keys k0 + [0, gridDim.x) of key head blockIdx.y made from their values into ks, in vc's page layout (a table row).
extern "C" __global__ void __launch_bounds__(256)
    tf_vonly_keys(const __nv_bfloat16* vc, __nv_bfloat16* ks, const int* tab, long long page, int tok, int hk,
                  int k0, const __nv_bfloat16* kw, const float* inv) {
    constexpr int D = 512;
    const int key = k0 + blockIdx.x, kh = blockIdx.y, d = threadIdx.x;
    const int64_t at = static_cast<int64_t>(tab[key / tok]) * page + (static_cast<int64_t>(key % tok) * hk + kh) * D;
    __nv_bfloat16 e[2] = {vc[at + d], vc[at + d + D / 2]};
    vk_pair<D>(&e[0], &e[1], d, key, Vk{kw, inv});
    ks[at + d] = e[0];
    ks[at + d + D / 2] = e[1];
}
