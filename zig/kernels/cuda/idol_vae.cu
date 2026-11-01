// Idol's image encoder on CUDA: idol_vae.cuh's bodies as the engine runs them (flaw none), fp32 throughout.
#include "idol_vae.cuh"

using namespace tf_iv;

extern "C" __global__ void __launch_bounds__(256) tf_iv_conv_in(const float* __restrict__ x, const float* __restrict__ w,
                                                                const float* __restrict__ b, float* __restrict__ y,
                                                                int rows, int cols) {
  conv_in<none>(x, w, b, y, rows, cols);
}

extern "C" __global__ void __launch_bounds__(256) tf_iv_conv(const float* __restrict__ X, const float* __restrict__ W,
                                                             const float* __restrict__ B, const float* __restrict__ R,
                                                             float* __restrict__ Y, int rows, int cols, int in_rows,
                                                             int in_cols, int cin, int cout, int ldw, int taps,
                                                             int stride, int pad, int add) {
  conv<none>(X, W, B, R, Y, rows, cols, in_rows, in_cols, cin, cout, ldw, taps, stride, pad, add);
}

extern "C" __global__ void __launch_bounds__(512) tf_iv_group_part(const float* __restrict__ x,
                                                                   const float* __restrict__ stats,
                                                                   float* __restrict__ part, int pixels, int channels,
                                                                   int chunk, int chunks, int centered) {
  group_part<none>(x, stats, part, pixels, channels, chunk, chunks, centered);
}

extern "C" __global__ void __launch_bounds__(1024) tf_iv_group_finish(const float* __restrict__ part,
                                                                      float* __restrict__ stats, int pixels,
                                                                      int channels, int chunks, int centered,
                                                                      float eps) {
  group_finish<none>(part, stats, pixels, channels, chunks, centered, eps);
}

extern "C" __global__ void __launch_bounds__(256) tf_iv_norm(const float* __restrict__ x, const float* __restrict__ stats,
                                                             const float* __restrict__ g, const float* __restrict__ beta,
                                                             float* __restrict__ y, int pixels, int channels, int silu) {
  norm<none>(x, stats, g, beta, y, pixels, channels, silu);
}

extern "C" __global__ void __launch_bounds__(256) tf_iv_bias(const float* acc, const float* __restrict__ b,
                                                             const float* r, float* y, int rows, int cols, int by_row,
                                                             int add) {
  bias<none>(acc, b, r, y, rows, cols, by_row, add);
}

extern "C" __global__ void __launch_bounds__(256) tf_iv_softmax(float* s, int n, float scale) {
  softmax<none>(s, n, scale);
}

extern "C" __global__ void __launch_bounds__(256) tf_iv_tokens(const float* __restrict__ moments,
                                                               const float* __restrict__ noise,
                                                               const float* __restrict__ m, const float* __restrict__ s,
                                                               float* __restrict__ out, int rows, int cols, int pad) {
  tokens<none>(moments, noise, m, s, out, rows, cols, pad);
}
