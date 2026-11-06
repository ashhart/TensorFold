// Unlimited-OCR-2's image encoder steps on CUDA (uocr/vision.metal's).
#pragma once
#include <cuda_fp16.h>
#include "uocr_common.cuh"

// y = bf16(a + b) over `n` values (torch's bf16 add: both widened, one rounding).
extern "C" __global__ void __launch_bounds__(256) tf_uocr_add(const bf16* __restrict__ a, const bf16* __restrict__ b,
                                                              bf16* __restrict__ y, long long n) {
  const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
  if (i < n) y[i] = to(ld(a + i) + ld(b + i));
}

// nn.LayerNorm on bf16 rows: the mean and variance of the widened row, the affine in fp32, one rounding.
extern "C" __global__ void __launch_bounds__(256) tf_uocr_layer_norm(const bf16* __restrict__ x, const bf16* __restrict__ w,
                                                                     const bf16* __restrict__ b, bf16* __restrict__ y,
                                                                     int width, float eps) {
  __shared__ float part[33];
  const bf16* xr = x + (size_t)blockIdx.x * width;
  bf16* yr = y + (size_t)blockIdx.x * width;
  float s = 0.0f;
  for (int i = threadIdx.x; i < width; i += blockDim.x) s += ld(xr + i);
  const float mean = block_sum(s, part) / (float)width;
  float v = 0.0f;
  for (int i = threadIdx.x; i < width; i += blockDim.x) {
    const float d = ld(xr + i) - mean;
    v += d * d;
  }
  const float inv = 1.0f / sqrtf(block_sum(v, part) / (float)width + eps);
  for (int i = threadIdx.x; i < width; i += blockDim.x) yr[i] = to((ld(xr + i) - mean) * inv * ld(w + i) + ld(b + i));
}

// nn.GELU (erf) on bf16 values in place, in fp32, one rounding.
extern "C" __global__ void __launch_bounds__(256) tf_uocr_gelu(bf16* __restrict__ x, long long n) {
  const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
  if (i >= n) return;
  const float v = ld(x + i);
  x[i] = to(0.5f * v * (1.0f + erf_(v * 0.70710678118654752f)));
}

// CLIP's quick_gelu as its module runs it on bf16: bf16(x * bf16(sigmoid(bf16(1.702 * x)))), in place.
extern "C" __global__ void __launch_bounds__(256) tf_uocr_quick_gelu(bf16* __restrict__ x, long long n) {
  const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
  if (i >= n) return;
  const float v = ld(x + i);
  const float z = bf(1.702f * v);
  const float s = bf(1.0f / (1.0f + expf(-z)));
  x[i] = to(v * s);
}

// y [rows, left + right] = [a row | b row]: CLIP's rows (past its class row) beside SAM's.
extern "C" __global__ void __launch_bounds__(256) tf_uocr_concat(const bf16* __restrict__ a, const bf16* __restrict__ b,
                                                                 bf16* __restrict__ y, int rows, int left, int right) {
  const int width = left + right;
  const int c = blockIdx.y * 256 + threadIdx.x, r = blockIdx.x;
  if (c >= width || r >= rows) return;
  y[(size_t)r * width + c] = c < left ? a[(size_t)r * left + c] : b[(size_t)r * right + c - left];
}

// CLIP's embeddings on SAM's map: y[0] = bf16(cls + pos[0]), y[r] = bf16(sam[r - 1] + pos[r]); rows of `width`.
extern "C" __global__ void __launch_bounds__(256) tf_uocr_clip_embed(const bf16* __restrict__ cls, const bf16* __restrict__ sam,
                                                                     const bf16* __restrict__ pos, bf16* __restrict__ y,
                                                                     int width) {
  const int c = blockIdx.y * 256 + threadIdx.x, r = blockIdx.x;
  if (c >= width) return;
  const float v = r == 0 ? ld(cls + c) : ld(sam + (size_t)(r - 1) * width + c);
  y[(size_t)r * width + c] = to(v + ld(pos + (size_t)r * width + c));
}

namespace tf_uocr {

// A planted fault for the contract's controls (contract_uocr_vision.cuh); the engine's kernels use UV_EXACT.
enum VisionFault {
  UV_EXACT,
  UV_PAD_DOWN,
  UV_TRANSPOSED,
  UV_NO_ADD,
  UV_MIRRORED,
  UV_HALF_SUMS,
  UV_ROW_BY_COLUMN,
  UV_NAN_SKIPPED,
};

// Convolution rows [out_side^2, channels span^2], columns (c, ky, kx); map value (c, y, x) at c cs + (y side + x) ps.
template <int F>
__device__ __forceinline__ void im2col(const bf16* __restrict__ x, bf16* __restrict__ y, int channels, int in_side,
                                       int out_side, int span, int stride, int pad, int cstride, int pstride) {
  const int taps = span * span, width = channels * taps;
  const int col = blockIdx.y * 256 + threadIdx.x, r = blockIdx.x;
  if (col >= width || r >= out_side * out_side) return;
  const int c = col / taps, t = col % taps;
  const int iy = (r / out_side) * stride + t / span + (F == UV_PAD_DOWN ? pad : -pad);
  const int ix = (r % out_side) * stride + t % span - pad;
  const bool inside = iy >= 0 && ix >= 0 && iy < in_side && ix < in_side;
  y[(size_t)r * width + col] = inside ? x[(size_t)c * cstride + ((size_t)iy * in_side + ix) * pstride] : to(0.0f);
}

// The map's rows [side^2, channels] as the padded map's window x window squares, both in row order; padding rows are 0.
template <int F>
__device__ __forceinline__ void window_rows(const bf16* __restrict__ x, bf16* __restrict__ y, int side, int padded,
                                            int window, int channels) {
  const int ch = blockIdx.y * 256 + threadIdx.x, at = blockIdx.x;
  if (ch >= channels || at >= padded * padded) return;
  const int area = window * window, across = padded / window;
  const int w = at / area, i = at % area;
  const int a = F == UV_TRANSPOSED ? i % window : i / window, b = F == UV_TRANSPOSED ? i / window : i % window;
  const int r = (w / across) * window + a, c = (w % across) * window + b;
  y[(size_t)at * channels + ch] = r < side && c < side ? x[((size_t)r * side + c) * channels + ch] : to(0.0f);
}

// The windows' rows back onto the map, added to it in place: x = bf16(x + window row); padding rows are dropped.
template <int F>
__device__ __forceinline__ void unwindow_add(bf16* __restrict__ x, const bf16* __restrict__ win, int side, int padded,
                                             int window, int channels) {
  const int ch = blockIdx.y * 256 + threadIdx.x, at = blockIdx.x;
  if (ch >= channels || at >= side * side) return;
  const int r = at / side, c = at % side, across = padded / window;
  const int w = (r / window) * across + c / window, i = (r % window) * window + c % window;
  const size_t to_ = (size_t)at * channels + ch;
  const float v = ld(win + ((size_t)w * window * window + i) * channels + ch);
  x[to_] = to(F == UV_NO_ADD ? v : __fadd_rn(ld(x + to_), v));
}

// Relative-position terms: for query i at (a, b) of its group, rel_h[h, i, j] = bf16(q . Rh[a - j + side - 1]); w by b.
template <int F>
__device__ __forceinline__ void rel_pos(const bf16* __restrict__ qkv, const bf16* __restrict__ rh,
                                        const bf16* __restrict__ rw, bf16* __restrict__ out_h, bf16* __restrict__ out_w,
                                        int groups, int side, int heads, int width) {
  const int len = side * side, rows = groups * len, head = blockIdx.y;
  const long long idx = (long long)blockIdx.x * 256 + threadIdx.x;
  if (idx >= (long long)rows * 2 * side || head >= heads) return;
  const int row = (int)(idx / (2 * side)), jj = (int)(idx % (2 * side));
  const bool across = jj >= side;
  const int j = across ? jj - side : jj, i = row % len, pos = across ? i % side : i / side;
  const int from = F == UV_MIRRORED ? j + side - 1 - pos : pos + side - 1 - j;
  const bf16* q = qkv + (size_t)row * 3 * width + head * 64;
  const bf16* r = (across ? rw : rh) + (size_t)from * 64;
  float s = 0.0f;
  for (int c = 0; c < 64; c++) {
    s = fmaf(ld(q + c), ld(r + c), s);
    if (F == UV_HALF_SUMS) s = __half2float(__float2half_rn(s));
  }
  (across ? out_w : out_h)[((size_t)head * rows + row) * side + j] = to(s);
}

#define UV_QROWS 8    // query rows a warp
#define UV_KEYS 32    // keys a step
#define UV_HEAD 64    // head width
#define UV_BLOCK 512  // keys a softmax block: its own running top score
#define UV_BLOCKS 8   // blocks a row at most (4,096 keys)

// A score as the softmax sees it: scale * (q . k) plus its query's two relative-position terms summed in bf16.
template <int F>
__device__ __forceinline__ float logit(float dot, int key, int row, int head, int groups, int len, int side, float scale,
                                       const bf16* __restrict__ rel_h, const bf16* __restrict__ rel_w) {
  float s = __fmul_rn(scale, dot);
  if (side > 0) {
    const size_t at = ((size_t)head * groups * len + row) * side;
    const float th = ld(rel_h + at + (F == UV_ROW_BY_COLUMN ? key % side : key / side)), tw = ld(rel_w + at + key % side);
    s = __fadd_rn(s, bf(__fadd_rn(th, tw)));
  }
  return s;
}

// This lane's 8 keys of a step: q . k in one fp32 fma chain over the head, keys from `ks` (rows of UV_HEAD + 1).
__device__ __forceinline__ void step_dots(const float* q, const float* ks, int part, float* dots) {
#pragma unroll
  for (int j = 0; j < 8; j++) {
    const float* k = ks + (part * 8 + j) * (UV_HEAD + 1);
    float s = 0.0f;
#pragma unroll
    for (int d = 0; d < UV_HEAD; d++) s = fmaf(q[d], k[d], s);
    dots[j] = s;
  }
}

// Attention in groups of len rows, 512-key blocks: fp32 weights on each block's running top, bf16 ones times values.
template <int F>
__device__ __forceinline__ void attention(const bf16* __restrict__ qkv, const bf16* __restrict__ rel_h,
                                          const bf16* __restrict__ rel_w, bf16* __restrict__ o, int groups, int len,
                                          int heads, int width, int side, float scale) {
  __shared__ float ks[UV_KEYS * (UV_HEAD + 1)];
  __shared__ float vs[UV_KEYS * UV_HEAD];
  __shared__ float stages[4 * UV_QROWS * UV_KEYS];
  const int t = threadIdx.x, lane = t & 31, wp = t >> 5;
  float* stage = stages + wp * UV_QROWS * UV_KEYS;
  const int head = blockIdx.y, base = blockIdx.z * len, stride = 3 * width;
  const int q0 = blockIdx.x * 4 * UV_QROWS + wp * UV_QROWS;
  // 4 warps of 8 query rows; lane: row lane / 4 of its warp's 8, keys (lane % 4) * 8 .. + 8 of each step
  const int my = lane >> 2, part = lane & 3, qr = min(q0 + my, len - 1), row = base + qr;
  // this lane's query, rows past the group as zeros
  float q[UV_HEAD];
  const bool live = q0 + my < len;
#pragma unroll
  for (int d = 0; d < UV_HEAD; d++) q[d] = live ? ld(qkv + (size_t)row * stride + head * UV_HEAD + d) : 0.0f;
  float dots[8];
  float top[UV_BLOCKS];
#pragma unroll
  for (int b = 0; b < UV_BLOCKS; b++) top[b] = -INFINITY;
  for (int k0 = 0; k0 < len; k0 += UV_KEYS) {
    __syncthreads();
    for (int e = t; e < UV_KEYS * UV_HEAD; e += 128) {
      const int key = k0 + e / UV_HEAD;
      ks[(e / UV_HEAD) * (UV_HEAD + 1) + e % UV_HEAD] =
          key < len ? ld(qkv + (size_t)(base + key) * stride + width + head * UV_HEAD + e % UV_HEAD) : 0.0f;
    }
    __syncthreads();
    step_dots(q, ks, part, dots);
    float m = -INFINITY;
    for (int j = 0; j < 8; j++) {
      const int key = k0 + part * 8 + j;
      if (key < len) m = fmaxf(m, logit<F>(dots[j], key, row, head, groups, len, side, scale, rel_h, rel_w));
    }
#pragma unroll
    for (int b = 0; b < UV_BLOCKS; b++)
      if (b == k0 / UV_BLOCK) top[b] = fmaxf(top[b], m);
  }
  // each block's top over the row's 4 lanes, then running: block b's is the largest of blocks 0 .. b
#pragma unroll
  for (int b = 0; b < UV_BLOCKS; b++) {
    top[b] = fmaxf(top[b], __shfl_xor_sync(0xffffffffu, top[b], 1));
    top[b] = fmaxf(top[b], __shfl_xor_sync(0xffffffffu, top[b], 2));
    if (b > 0) top[b] = fmaxf(top[b], top[b - 1]);
  }
  float acc[UV_HEAD / 4];
#pragma unroll
  for (int c = 0; c < UV_HEAD / 4; c++) acc[c] = 0.0f;
  float mass = 0.0f, cur = top[0];
  for (int k0 = 0; k0 < len; k0 += UV_KEYS) {
    const int b = k0 / UV_BLOCK;
    if (b > 0 && k0 % UV_BLOCK == 0) {
      // a new block: the row's sums so far times exp(previous top - this top)
      float next = cur;
#pragma unroll
      for (int i = 0; i < UV_BLOCKS; i++)
        if (i == b) next = top[i];
      const float c = expf(cur - next);
      cur = next;
      mass = __fmul_rn(mass, c);
#pragma unroll
      for (int i = 0; i < UV_HEAD / 4; i++) acc[i] = __fmul_rn(c, acc[i]);
    }
    __syncthreads();
    for (int e = t; e < UV_KEYS * UV_HEAD; e += 128) {
      const int key = k0 + e / UV_HEAD;
      const size_t at = (size_t)(base + key) * stride + head * UV_HEAD + e % UV_HEAD;
      ks[(e / UV_HEAD) * (UV_HEAD + 1) + e % UV_HEAD] = key < len ? ld(qkv + at + width) : 0.0f;
      vs[e] = key < len ? ld(qkv + at + 2 * width) : 0.0f;
    }
    __syncthreads();
    step_dots(q, ks, part, dots);
    for (int j = 0; j < 8; j++) {
      const int key = k0 + part * 8 + j;
      const bool use = key < len && (F != UV_NAN_SKIPPED || !isnan(dots[j]));
      const float w = use ? expf(logit<F>(dots[j], key, row, head, groups, len, side, scale, rel_h, rel_w) - cur) : 0.0f;
      mass += w;
      stage[my * UV_KEYS + part * 8 + j] = bf(w);
    }
    __syncwarp();
    // this lane's 16 output columns part, part + 4, ...: one fp32 fma chain over the keys
    for (int k = 0; k < UV_KEYS; k++) {
      const float p = stage[my * UV_KEYS + k];
#pragma unroll
      for (int c = 0; c < UV_HEAD / 4; c++) acc[c] = fmaf(p, vs[k * UV_HEAD + c * 4 + part], acc[c]);
    }
    __syncwarp();
  }
  mass += __shfl_xor_sync(0xffffffffu, mass, 1);
  mass += __shfl_xor_sync(0xffffffffu, mass, 2);
  if (!live) return;
  bf16* out = o + (size_t)row * width + head * UV_HEAD;
#pragma unroll
  for (int c = 0; c < UV_HEAD / 4; c++) out[c * 4 + part] = to(acc[c] / mass);
}

}  // namespace tf_uocr

extern "C" __global__ void __launch_bounds__(256) tf_uocr_im2col(const bf16* __restrict__ x, bf16* __restrict__ y,
                                                                 int channels, int in_side, int out_side, int span,
                                                                 int stride, int pad, int cstride, int pstride) {
  tf_uocr::im2col<UV_EXACT>(x, y, channels, in_side, out_side, span, stride, pad, cstride, pstride);
}

extern "C" __global__ void __launch_bounds__(256) tf_uocr_window(const bf16* __restrict__ x, bf16* __restrict__ y, int side,
                                                                 int padded, int window, int channels) {
  tf_uocr::window_rows<UV_EXACT>(x, y, side, padded, window, channels);
}

extern "C" __global__ void __launch_bounds__(256) tf_uocr_unwindow_add(bf16* __restrict__ x, const bf16* __restrict__ win,
                                                                       int side, int padded, int window, int channels) {
  tf_uocr::unwindow_add<UV_EXACT>(x, win, side, padded, window, channels);
}

extern "C" __global__ void __launch_bounds__(256) tf_uocr_rel_pos(const bf16* __restrict__ qkv, const bf16* __restrict__ rh,
                                                                  const bf16* __restrict__ rw, bf16* __restrict__ out_h,
                                                                  bf16* __restrict__ out_w, int groups, int side,
                                                                  int heads, int width) {
  tf_uocr::rel_pos<UV_EXACT>(qkv, rh, rw, out_h, out_w, groups, side, heads, width);
}

extern "C" __global__ void __launch_bounds__(128) tf_uocr_attention(const bf16* __restrict__ qkv,
                                                                    const bf16* __restrict__ rel_h,
                                                                    const bf16* __restrict__ rel_w, bf16* __restrict__ o,
                                                                    int groups, int len, int heads, int width, int side,
                                                                    float scale) {
  tf_uocr::attention<UV_EXACT>(qkv, rel_h, rel_w, o, groups, len, heads, width, side, scale);
}
