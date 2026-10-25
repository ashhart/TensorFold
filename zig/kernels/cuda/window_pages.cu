// window_rows.cuh over paged caches (Dory's full-attention layers, head dim 256): the kernels, and the rows' writer.

#include "window_rows.cuh"

using namespace tf_wrows;

// Dory's full-attention caches in pages (head dim 256, no window): kc/vc a layer's part of page 0, tab a row of ids.
extern "C" __global__ void __launch_bounds__(NT) tf_wrows_part_256_pg(
    const __nv_bfloat16* q, const __nv_bfloat16* kc, const __nv_bfloat16* vc, const int* pos, const int* slot,
    float* po, float* pm, float* pl, int h, int hk, int win, long long ring, float scale, const int* tab,
    long long page, int tok, int stride) {
    part<256, NONE, true>(q, kc, vc, pos, slot, po, pm, pl, h, hk, win, ring, scale, Pg{tab, page, tok, stride});
}

extern "C" __global__ void __launch_bounds__(NT) tf_wrows_merge_256_pg(
    const float* po, const float* pm, const float* pl, __nv_bfloat16* out, int h, int hk, int chunks,
    const __nv_bfloat16* q, const __nv_bfloat16* kc, const __nv_bfloat16* vc, const int* pos, const int* slot, int win,
    long long ring, float scale, const int* tab, long long page, int tok, int stride) {
    merge<256, NONE, true>(po, pm, pl, out, h, hk, chunks, q, kc, vc, pos, slot, win, ring, scale,
                           Pg{tab, page, tok, stride});
}

// The prompt kernel's tab is its stream's own row of page ids.
extern "C" __global__ void __launch_bounds__(NT, 1) tf_wrows_prompt_256_pg(
    const __nv_bfloat16* q, const __nv_bfloat16* kc, const __nv_bfloat16* vc, __nv_bfloat16* out, const int* hi,
    int p0, int W, int H, int HK, int G, float scale, int win, int ring, const int* tab, long long page, int tok) {
    prompt::kernel<256, false, NONE, true>(q, kc, vc, out, hi, p0, W, H, HK, G, scale, win, ring,
                                           Pg{tab, page, tok, 0});
}

// Rows k, v [n, row] (row = hk * D elements) into the pages at their positions: a block a row, 16 bytes a thread.
extern "C" __global__ void __launch_bounds__(128) tf_wrows_put(const __nv_bfloat16* k, const __nv_bfloat16* v,
                                                               __nv_bfloat16* kc, __nv_bfloat16* vc, const int* pos,
                                                               const int* slot, const int* tab, long long page,
                                                               int tok, int stride, int row) {
    const int64_t r = blockIdx.x;
    const int p = pos[r];
    const int64_t id = tab[slot[r] * stride + p / tok];
    const int64_t at = id * page + static_cast<int64_t>(p % tok) * row;
    for (int e = threadIdx.x * 8; e < row; e += 128 * 8) {
        *reinterpret_cast<uint4*>(kc + at + e) = *reinterpret_cast<const uint4*>(k + r * row + e);
        *reinterpret_cast<uint4*>(vc + at + e) = *reinterpret_cast<const uint4*>(v + r * row + e);
    }
}
