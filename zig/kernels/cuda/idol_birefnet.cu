// Idol's background remover (BiRefNet) on CUDA in fp32: idol_birefnet.cuh's bodies, instantiated plain (CTL_NONE).
#include "idol_birefnet.cuh"

using namespace tf_br;

extern "C" __global__ void __launch_bounds__(256) tf_br_conv(const float* __restrict__ x, const float* __restrict__ w,
                                                             const float* __restrict__ b, const float* __restrict__ post,
                                                             const float* r, const int* __restrict__ map, float* y,
                                                             int rows, int cols, int cin, int ldx, int cout, int ldw,
                                                             int k, int ldy, int yoff, int flags, int row0) {
  conv<CTL_NONE>(x, w, b, post, r, map, y, rows, cols, cin, ldx, cout, ldw, k, ldy, yoff, flags, row0);
}

extern "C" __global__ void __launch_bounds__(256) tf_br_norm(const float* __restrict__ x, const int* __restrict__ map,
                                                             const float* __restrict__ g, const float* __restrict__ beta,
                                                             float* __restrict__ y, int width, int mode, int src_w,
                                                             int ldy, int yoff, float eps) {
  norm<CTL_NONE>(x, map, g, beta, y, width, mode, src_w, ldy, yoff, eps);
}

extern "C" __global__ void __launch_bounds__(192) tf_br_window(const float* __restrict__ qkv,
                                                               const float* __restrict__ table, float* __restrict__ o,
                                                               int width, int heads, int rows, int cols, int shift,
                                                               float scale) {
  window<CTL_NONE>(qkv, table, o, width, heads, rows, cols, shift, scale);
}

// Pixels on grid x (blockDim.y a block), channels on grid y (blockDim.x a block).
extern "C" __global__ void __launch_bounds__(256) tf_br_resize(const float* __restrict__ x, float* __restrict__ y, int in_h,
                                                               int in_w, int out_h, int out_w, int channels, int ldx,
                                                               int xoff, int ldy, int yoff) {
  const int pix = blockIdx.x * blockDim.y + threadIdx.y;
  const int ch = blockIdx.y * blockDim.x + threadIdx.x;
  resize<CTL_NONE>(x, y, in_h, in_w, out_h, out_w, channels, ldx, xoff, ldy, yoff, pix, ch);
}

// Channel ch * a * b + i * b + j (0 past channels * a * b): mode 0 a x b patches, mode 1 image2patches' tiles.
extern "C" __global__ void __launch_bounds__(256) tf_br_tiles(const float* __restrict__ src, float* __restrict__ dst,
                                                              int out_h, int out_w, int src_w, int channels, int a, int b,
                                                              int mode, int cpad) {
  const int pix = blockIdx.x * blockDim.y + threadIdx.y;
  const int c = blockIdx.y * blockDim.x + threadIdx.x;
  if (c >= cpad || pix >= out_h * out_w) return;
  const size_t o = (size_t)pix * cpad + c;
  const int ab = a * b;
  if (c >= channels * ab) {
    dst[o] = 0.0f;
    return;
  }
  const int ch = c / ab, i = (c % ab) / b, j = c % b;
  const int oy = pix / out_w, ox = pix % out_w;
  const int sy = mode == 0 ? oy * a + i : i * out_h + oy;
  const int sx = mode == 0 ? ox * b + j : j * out_w + ox;
  dst[o] = src[(size_t)(sy * src_w + sx) * channels + ch];
}

// y [pixels, cout] = conv3x3(x) + b of a 3-channel picture x [pixels, 3], zero padded; w [27, cout] by (tap, channel).
extern "C" __global__ void __launch_bounds__(256) tf_br_conv3(const float* __restrict__ x, const float* __restrict__ w,
                                                              const float* __restrict__ b, float* __restrict__ y, int rows,
                                                              int cols, int cout) {
  const int pix = blockIdx.x * blockDim.y + threadIdx.y;
  const int o = blockIdx.y * blockDim.x + threadIdx.x;
  if (o >= cout || pix >= rows * cols) return;
  const int py = pix / cols, px = pix % cols;
  float s = 0.0f;
#pragma unroll
  for (int ky = 0; ky < 3; ++ky)
#pragma unroll
    for (int kx = 0; kx < 3; ++kx) {
      const int iy = py + ky - 1, ix = px + kx - 1;
      if (iy < 0 || ix < 0 || iy >= rows || ix >= cols) continue;
      const float* v = x + (size_t)(iy * cols + ix) * 3;
#pragma unroll
      for (int c = 0; c < 3; ++c) s = __fmaf_rn(v[c], w[((ky * 3 + kx) * 3 + c) * cout + o], s);
    }
  y[(size_t)pix * cout + o] = s + b[o];
}

// Chunk rows on grid x (blockDim.y a block), taps on grid y (blockDim.x a block).
extern "C" __global__ void __launch_bounds__(256) tf_br_deform(const float* __restrict__ x, const float* __restrict__ offs,
                                                               float* __restrict__ col, int rows, int cols, int k,
                                                               int first, int count, int ldo) {
  const int r = blockIdx.x * blockDim.y + threadIdx.y;
  const int tap = blockIdx.y * blockDim.x + threadIdx.x;
  deform<CTL_NONE>(x, offs, col, rows, cols, k, first, count, ldo, r, tap);
}

extern "C" __global__ void __launch_bounds__(256) tf_br_pool(const float* __restrict__ x, float* __restrict__ mean,
                                                             int pixels, int channels) {
  pool<CTL_NONE>(x, mean, pixels, channels);
}

// x [pixels, channels] *= sigmoid(g . w + b), g [pixels, 16] (the gradient attention's 1x1 to one channel).
extern "C" __global__ void __launch_bounds__(256) tf_br_gate(const float* __restrict__ g, const float* __restrict__ w,
                                                             const float* __restrict__ b, float* __restrict__ x,
                                                             int pixels, int channels) {
  const int pix = blockIdx.x * blockDim.y + threadIdx.y;
  const int c = blockIdx.y * blockDim.x + threadIdx.x;
  if (c >= channels || pix >= pixels) return;
  float s = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) s = __fmaf_rn(g[(size_t)pix * 16 + i], w[i], s);
  const size_t o = (size_t)pix * channels + c;
  x[o] = x[o] * sigmoid(s + b[0]);
}

// logits = conv_out1 of [bilinear (align_corners) of a [in, cp] | q [pixels, cq]], w [cp + cq].
extern "C" __global__ void __launch_bounds__(256) tf_br_head(const float* __restrict__ a, const float* __restrict__ q,
                                                             const float* __restrict__ w, const float* __restrict__ b,
                                                             float* __restrict__ y, int in_h, int in_w, int out_h,
                                                             int out_w, int cp, int cq) {
  const int at = blockIdx.x * 256 + threadIdx.x;
  if (at >= out_h * out_w) return;
  const int oy = at / out_w, ox = at % out_w;
  const float fy = ((float)(in_h - 1) / (float)(out_h - 1)) * (float)oy;
  const float fx = ((float)(in_w - 1) / (float)(out_w - 1)) * (float)ox;
  const int y0 = min((int)fy, in_h - 1), x0 = min((int)fx, in_w - 1);
  const int y1 = min(y0 + 1, in_h - 1), x1 = min(x0 + 1, in_w - 1);
  const float ly = fminf(fmaxf(fy - (float)y0, 0.0f), 1.0f);
  const float lx = fminf(fmaxf(fx - (float)x0, 0.0f), 1.0f);
  const float* s00 = a + (size_t)(y0 * in_w + x0) * cp;
  const float* s01 = a + (size_t)(y0 * in_w + x1) * cp;
  const float* s10 = a + (size_t)(y1 * in_w + x0) * cp;
  const float* s11 = a + (size_t)(y1 * in_w + x1) * cp;
  float s = 0.0f;
  for (int c = 0; c < cp; ++c) {
    const float top = (1.0f - lx) * s00[c] + lx * s01[c];
    const float bottom = (1.0f - lx) * s10[c] + lx * s11[c];
    s = __fmaf_rn((1.0f - ly) * top + ly * bottom, w[c], s);
  }
  const float* qr = q + (size_t)at * cq;
  for (int c = 0; c < cq; ++c) s = __fmaf_rn(qr[c], w[cp + c], s);
  y[at] = s + b[0];
}
