// Explicit optional FP8 tiles; quantization and all weight allocations remain in the original module.
#include "fp8_prompt_tiles.cuh"

#define TF_FP8_TILE(NAME, BM, BN, WM, WN, F32, ROW) \
extern "C" __global__ __launch_bounds__(256) void NAME( \
    const uint8_t* x, const uint8_t* weight, void* output, int m, int n, int k, int npad, float alpha, \
    const float* row_scales, const float* weight_scales) { \
    tf_fp8_tiles::product<BM, BN, WM, WN, F32, ROW>(x, weight, output, m, n, k, npad, alpha, row_scales, weight_scales); \
}
#define TF_FP8_GEOMETRY(NAME, BM, BN, WM, WN) \
TF_FP8_TILE(NAME, BM, BN, WM, WN, false, false) \
TF_FP8_TILE(NAME##_f32, BM, BN, WM, WN, true, false) \
TF_FP8_TILE(NAME##_row, BM, BN, WM, WN, false, true) \
TF_FP8_TILE(NAME##_row_f32, BM, BN, WM, WN, true, true)

TF_FP8_GEOMETRY(tf_fp8_tile128x64, 128, 64, 4, 2)
TF_FP8_GEOMETRY(tf_fp8_tile128x128, 128, 128, 4, 2)
TF_FP8_GEOMETRY(tf_fp8_tile64x64, 64, 64, 2, 4)
#undef TF_FP8_GEOMETRY
#undef TF_FP8_TILE
