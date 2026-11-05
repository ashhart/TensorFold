// Gate 3 and 4 controls for uocr.cu's moves: each kernel with one index, padding or sum rule broken.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace tf_ctl_uocr_moves {
typedef __nv_bfloat16 bf16;
__device__ __forceinline__ float ld(const bf16* p) { return __bfloat162float(*p); }
__device__ __forceinline__ bf16 to(float x) { return __float2bfloat16_rn(x); }

// tf_uocr_im2col; PAD: the outside reads 1, SWAP: ky and kx swapped.
template <int PAD, int SWAP>
__device__ __forceinline__ void im2col(const bf16* x, bf16* y, int in_h, int in_w, int channels, int k, int stride, int pad,
                                       int out_h, int out_w, int chw) {
  const int width = channels * k * k;
  const int col = blockIdx.y * 256 + threadIdx.x, row = blockIdx.x;
  if (col >= width || row >= out_h * out_w) return;
  const int c = col / (k * k), ky = SWAP ? col % k : (col / k) % k, kx = SWAP ? (col / k) % k : col % k;
  const int iy = (row / out_w) * stride + ky - pad;
  const int ix = (row % out_w) * stride + kx - pad;
  bf16 v = to(PAD ? 1.0f : 0.0f);
  if (iy >= 0 && ix >= 0 && iy < in_h && ix < in_w)
    v = chw ? x[((size_t)c * in_h + iy) * in_w + ix] : x[((size_t)iy * in_w + ix) * channels + c];
  y[(size_t)row * width + col] = v;
}

// tf_uocr_window; SWAP: a window's rows and columns swapped, PAD: the padding reads 1.
template <int SWAP, int PAD>
__device__ __forceinline__ void window(const bf16* x, bf16* y, int side, int padded, int window, int channels) {
  const int c = blockIdx.y * 256 + threadIdx.x, r = blockIdx.x;
  if (c >= channels || r >= padded * padded) return;
  const int per = padded / window, w2 = window * window;
  const int win = r / w2, in = r % w2;
  const int gy = (win / per) * window + (SWAP ? in % window : in / window);
  const int gx = (win % per) * window + (SWAP ? in / window : in % window);
  y[(size_t)r * channels + c] = (gy < side && gx < side) ? x[((size_t)gy * side + gx) * channels + c] : to(PAD ? 1.0f : 0.0f);
}

// tf_uocr_rel_pos; SWAP: rel_h and rel_w positions swapped, HALF: the 64 products summed in half.
template <int SWAP, int HALF>
__device__ __forceinline__ void rel_pos(const bf16* qkv, const bf16* tab_h, const bf16* tab_w, bf16* rel_h, bf16* rel_w,
                                        int groups, int side, int width) {
  const int L = side * side;
  const int x = threadIdx.x, row = blockIdx.x, h = blockIdx.y;
  if (x >= 2 * side || row >= groups * L) return;
  const int g = row / L, qi = row % L, qh = qi / side, qw = qi % side;
  const bool rows = x < side;
  const int kk = rows ? x : x - side;
  const int at = SWAP ? (rows ? qw : qh) : (rows ? qh : qw);
  const bf16* q = qkv + (size_t)row * 3 * width + h * 64;
  const bf16* r = (rows ? tab_h : tab_w) + (size_t)(at - kk + side - 1) * 64;
  float s = 0.0f;
  if (HALF) {
    __half hs = __float2half(0.0f);
    for (int d = 0; d < 64; d++) hs = __hadd(hs, __float2half(ld(q + d) * ld(r + d)));
    s = __half2float(hs);
  } else {
    for (int d = 0; d < 64; d++) s += ld(q + d) * ld(r + d);
  }
  bf16* out = rows ? rel_h : rel_w;
  out[(((size_t)h * groups + g) * L + qi) * side + kk] = to(s);
}
}  // namespace tf_ctl_uocr_moves

extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_im2col_pad_one(const __nv_bfloat16* x, __nv_bfloat16* y, int in_h, int in_w,
                                                                             int channels, int k, int stride, int pad, int out_h,
                                                                             int out_w, int chw) {
  tf_ctl_uocr_moves::im2col<1, 0>(x, y, in_h, in_w, channels, k, stride, pad, out_h, out_w, chw);
}

extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_im2col_kykx(const __nv_bfloat16* x, __nv_bfloat16* y, int in_h, int in_w,
                                                                          int channels, int k, int stride, int pad, int out_h,
                                                                          int out_w, int chw) {
  tf_ctl_uocr_moves::im2col<0, 1>(x, y, in_h, in_w, channels, k, stride, pad, out_h, out_w, chw);
}

extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_window_swap(const __nv_bfloat16* x, __nv_bfloat16* y, int side, int padded,
                                                                          int window, int channels) {
  tf_ctl_uocr_moves::window<1, 0>(x, y, side, padded, window, channels);
}

extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_window_pad_one(const __nv_bfloat16* x, __nv_bfloat16* y, int side,
                                                                             int padded, int window, int channels) {
  tf_ctl_uocr_moves::window<0, 1>(x, y, side, padded, window, channels);
}

// tf_uocr_unwindow_add with a window's rows and columns swapped.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_unwindow_add_swap(__nv_bfloat16* x, const __nv_bfloat16* win, int side,
                                                                                int padded, int window, int channels) {
  using namespace tf_ctl_uocr_moves;
  const int c = blockIdx.y * 256 + threadIdx.x, r = blockIdx.x;
  if (c >= channels || r >= side * side) return;
  const int gy = r / side, gx = r % side;
  const int per = padded / window;
  const int w = (gy / window) * per + gx / window, in = (gx % window) * window + gy % window;
  const size_t i = (size_t)r * channels + c;
  x[i] = to(ld(x + i) + ld(win + ((size_t)w * window * window + in) * channels + c));
}

extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_rel_pos_swap(const __nv_bfloat16* qkv, const __nv_bfloat16* tab_h,
                                                                           const __nv_bfloat16* tab_w, __nv_bfloat16* rel_h,
                                                                           __nv_bfloat16* rel_w, int groups, int side, int width) {
  tf_ctl_uocr_moves::rel_pos<1, 0>(qkv, tab_h, tab_w, rel_h, rel_w, groups, side, width);
}

extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_rel_pos_half(const __nv_bfloat16* qkv, const __nv_bfloat16* tab_h,
                                                                           const __nv_bfloat16* tab_w, __nv_bfloat16* rel_h,
                                                                           __nv_bfloat16* rel_w, int groups, int side, int width) {
  tf_ctl_uocr_moves::rel_pos<0, 1>(qkv, tab_h, tab_w, rel_h, rel_w, groups, side, width);
}

// tf_uocr_concat with the left row one column wider.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_concat_wide(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* y,
                                                                          int rows, int left, int right) {
  const int width = left + right;
  const int c = blockIdx.y * 256 + threadIdx.x, r = blockIdx.x;
  if (c >= width || r >= rows) return;
  y[(size_t)r * width + c] = c <= left ? a[(size_t)r * left + c] : b[(size_t)r * right + c - left];
}

// tf_uocr_clip_embed with the class row taking position 1.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_clip_embed_pos1(const __nv_bfloat16* cls, const __nv_bfloat16* sam,
                                                                              const __nv_bfloat16* pos, __nv_bfloat16* y, int width) {
  using namespace tf_ctl_uocr_moves;
  const int c = blockIdx.y * 256 + threadIdx.x, r = blockIdx.x;
  if (c >= width) return;
  const float v = r == 0 ? ld(cls + c) : ld(sam + (size_t)(r - 1) * width + c);
  y[(size_t)r * width + c] = to(v + ld(pos + (size_t)(r > 1 ? r : 1) * width + c));
}

// tf_uocr_embed writing image tokens' rows too, from the table's image-token row.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_embed_images(const unsigned* ids, const __nv_bfloat16* embed,
                                                                           __nv_bfloat16* x, unsigned image_token) {
  const int c = blockIdx.y * 256 + threadIdx.x;
  const unsigned id = ids[blockIdx.x];
  if (c >= 1280) return;
  x[(size_t)blockIdx.x * 1280 + c] = embed[(size_t)id * 1280 + c];
}

// tf_uocr_gather with odd rows reading the even row's index.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_gather_even(const __nv_bfloat16* x, const unsigned* rows, __nv_bfloat16* y) {
  const int c = blockIdx.y * 256 + threadIdx.x;
  if (c >= 1280) return;
  y[(size_t)blockIdx.x * 1280 + c] = x[(size_t)rows[blockIdx.x & ~1u] * 1280 + c];
}

// tf_uocr_scatter with odd rows written to the even row's slot.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_scatter_even(const __nv_bfloat16* y, const unsigned* dest,
                                                                           __nv_bfloat16* slots) {
  const int c = blockIdx.y * 256 + threadIdx.x;
  if (c >= 1280) return;
  slots[(size_t)dest[blockIdx.x & ~1u] * 1280 + c] = y[(size_t)blockIdx.x * 1280 + c];
}
