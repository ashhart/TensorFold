// Unlimited-OCR-2's image-encoder kernels (uocr_vision.cuh) with one planted fault each; the named gate must go red.
#pragma once
#include "uocr_vision.cuh"

// Gate 3: the convolution rows offset by the padding the wrong way.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_im2col_pad_down(const bf16* x, bf16* y, int channels,
                                                                              int in_side, int out_side, int span,
                                                                              int stride, int pad, int cstride,
                                                                              int pstride) {
  tf_uocr::im2col<tf_uocr::UV_PAD_DOWN>(x, y, channels, in_side, out_side, span, stride, pad, cstride, pstride);
}

// Gate 3: each window's rows and columns swapped.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_window_transposed(const bf16* x, bf16* y, int side,
                                                                                int padded, int window, int channels) {
  tf_uocr::window_rows<tf_uocr::UV_TRANSPOSED>(x, y, side, padded, window, channels);
}

// Gate 3: the window rows replace the map's instead of adding to them.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_unwindow_no_add(bf16* x, const bf16* win, int side,
                                                                              int padded, int window, int channels) {
  tf_uocr::unwindow_add<tf_uocr::UV_NO_ADD>(x, win, side, padded, window, channels);
}

// Gate 3: the relative offsets mirrored (key minus query).
extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_rel_mirrored(const bf16* qkv, const bf16* rh,
                                                                           const bf16* rw, bf16* out_h, bf16* out_w,
                                                                           int groups, int side, int heads, int width) {
  tf_uocr::rel_pos<tf_uocr::UV_MIRRORED>(qkv, rh, rw, out_h, out_w, groups, side, heads, width);
}

// Gate 4: the relative-position dot summed in half.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_rel_half_sums(const bf16* qkv, const bf16* rh,
                                                                            const bf16* rw, bf16* out_h, bf16* out_w,
                                                                            int groups, int side, int heads, int width) {
  tf_uocr::rel_pos<tf_uocr::UV_HALF_SUMS>(qkv, rh, rw, out_h, out_w, groups, side, heads, width);
}

// Gate 3: the row term read at the key's column.
extern "C" __global__ void __launch_bounds__(128) tf_ctl_uocr_attention_row_by_column(const bf16* qkv, const bf16* rel_h,
                                                                                     const bf16* rel_w, bf16* o,
                                                                                     int groups, int len, int heads,
                                                                                     int width, int side, float scale) {
  tf_uocr::attention<tf_uocr::UV_ROW_BY_COLUMN>(qkv, rel_h, rel_w, o, groups, len, heads, width, side, scale);
}

// Gate 1: a NaN score weighs 0.
extern "C" __global__ void __launch_bounds__(128) tf_ctl_uocr_attention_nan_skipped(const bf16* qkv, const bf16* rel_h,
                                                                                   const bf16* rel_w, bf16* o,
                                                                                   int groups, int len, int heads,
                                                                                   int width, int side, float scale) {
  tf_uocr::attention<tf_uocr::UV_NAN_SKIPPED>(qkv, rel_h, rel_w, o, groups, len, heads, width, side, scale);
}
