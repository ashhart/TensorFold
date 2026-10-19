// window_attention's rows at head dim 256 (window_rows.cuh): decode partials and merge, prompt tiles.

#include "window_rows.cuh"

using namespace tf_wrows;

extern "C" __global__ void __launch_bounds__(NT) tf_wrows_part_256(const __nv_bfloat16* q, const __nv_bfloat16* kc,
                                                                   const __nv_bfloat16* vc, const int* pos,
                                                                   const int* slot, float* po, float* pm, float* pl,
                                                                   int h, int hk, int win, long long ring, float scale) {
    part<256, NONE>(q, kc, vc, pos, slot, po, pm, pl, h, hk, win, ring, scale);
}

extern "C" __global__ void __launch_bounds__(NT) tf_wrows_merge_256(const float* po, const float* pm, const float* pl,
                                                                    __nv_bfloat16* out, int h, int hk, int chunks,
                                                                    const __nv_bfloat16* q, const __nv_bfloat16* kc,
                                                                    const __nv_bfloat16* vc, const int* pos,
                                                                    const int* slot, int win, long long ring,
                                                                    float scale) {
    merge<256, NONE>(po, pm, pl, out, h, hk, chunks, q, kc, vc, pos, slot, win, ring, scale);
}

extern "C" __global__ void __launch_bounds__(NT, 1) tf_wrows_prompt_256(const __nv_bfloat16* q,
                                                                        const __nv_bfloat16* kc,
                                                                        const __nv_bfloat16* vc, __nv_bfloat16* out,
                                                                        const int* hi, int p0, int W, int H, int HK,
                                                                        int G, float scale, int win, int ring) {
    prompt::kernel<256, false, NONE>(q, kc, vc, out, hi, p0, W, H, HK, G, scale, win, ring);
}

extern "C" __global__ void __launch_bounds__(NT, 1) tf_wrows_prompt_256_p32(const __nv_bfloat16* q,
                                                                            const __nv_bfloat16* kc,
                                                                            const __nv_bfloat16* vc,
                                                                            __nv_bfloat16* out, const int* hi, int p0,
                                                                            int W, int H, int HK, int G, float scale,
                                                                            int win, int ring) {
    prompt::kernel<256, true, NONE>(q, kc, vc, out, hi, p0, W, H, HK, G, scale, win, ring);
}
