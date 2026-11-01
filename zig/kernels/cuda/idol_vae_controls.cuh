// Idol's image encoder controls (contract_controls.cu): idol_vae.cuh's bodies each built with one planted flaw.
#pragma once
#include "idol_vae.cuh"

extern "C" __global__ void __launch_bounds__(256) tf_ctl_iv_conv_in_half(const float* x, const float* w, const float* b,
                                                                         float* y, int rows, int cols) {
  tf_iv::conv_in<tf_iv::conv_in_half>(x, w, b, y, rows, cols);
}

#define TF_CTL_IV_CONV(name, flaw)                                                                                 \
  extern "C" __global__ void __launch_bounds__(256)                                                                \
      name(const float* X, const float* W, const float* B, const float* R, float* Y, int rows, int cols,           \
           int in_rows, int in_cols, int cin, int cout, int ldw, int taps, int stride, int pad, int add) {         \
    tf_iv::conv<flaw>(X, W, B, R, Y, rows, cols, in_rows, in_cols, cin, cout, ldw, taps, stride, pad, add);        \
  }
TF_CTL_IV_CONV(tf_ctl_iv_conv_half_in, tf_iv::conv_half_in)
TF_CTL_IV_CONV(tf_ctl_iv_conv_sym_pad, tf_iv::conv_sym_pad)
TF_CTL_IV_CONV(tf_ctl_iv_conv_half_sum, tf_iv::conv_half_sum)
#undef TF_CTL_IV_CONV

extern "C" __global__ void __launch_bounds__(256) tf_ctl_iv_norm_half(const float* x, const float* stats, const float* g,
                                                                      const float* beta, float* y, int pixels,
                                                                      int channels, int silu) {
  tf_iv::norm<tf_iv::norm_half>(x, stats, g, beta, y, pixels, channels, silu);
}

extern "C" __global__ void __launch_bounds__(512) tf_ctl_iv_group_part_half(const float* x, const float* stats,
                                                                            float* part, int pixels, int channels,
                                                                            int chunk, int chunks, int centered) {
  tf_iv::group_part<tf_iv::part_half>(x, stats, part, pixels, channels, chunk, chunks, centered);
}

extern "C" __global__ void __launch_bounds__(1024) tf_ctl_iv_group_finish_half(const float* part, float* stats,
                                                                               int pixels, int channels, int chunks,
                                                                               int centered, float eps) {
  tf_iv::group_finish<tf_iv::finish_half>(part, stats, pixels, channels, chunks, centered, eps);
}

extern "C" __global__ void __launch_bounds__(256) tf_ctl_iv_softmax_half(float* s, int n, float scale) {
  tf_iv::softmax<tf_iv::softmax_half>(s, n, scale);
}

extern "C" __global__ void __launch_bounds__(256) tf_ctl_iv_bias_swap(const float* acc, const float* b, const float* r,
                                                                      float* y, int rows, int cols, int by_row,
                                                                      int add) {
  tf_iv::bias<tf_iv::bias_swap>(acc, b, r, y, rows, cols, by_row, add);
}

#define TF_CTL_IV_TOKENS(name, flaw)                                                                               \
  extern "C" __global__ void __launch_bounds__(256) name(const float* moments, const float* noise, const float* m, \
                                                         const float* s, float* out, int rows, int cols, int pad) { \
    tf_iv::tokens<flaw>(moments, noise, m, s, out, rows, cols, pad);                                               \
  }
TF_CTL_IV_TOKENS(tf_ctl_iv_tokens_axes, tf_iv::tokens_axes)
TF_CTL_IV_TOKENS(tf_ctl_iv_tokens_clamp, tf_iv::tokens_clamp)
#undef TF_CTL_IV_TOKENS
