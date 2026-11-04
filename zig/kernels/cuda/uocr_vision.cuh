// Unlimited-OCR-2's image encoder steps on CUDA (uocr/vision.metal's).
#pragma once
#include "uocr_common.cuh"

// Convolution patches [out_h * out_w, c * k * k] in (c, ky, kx) order, from [c, h, w] (chw) or [h, w, c] input.
extern "C" __global__ void __launch_bounds__(256) tf_uocr_im2col(const bf16* __restrict__ x, bf16* __restrict__ y, int in_h,
                                                                 int in_w, int channels, int k, int stride, int pad,
                                                                 int out_h, int out_w, int chw) {
  const int width = channels * k * k;
  const int col = blockIdx.y * 256 + threadIdx.x, row = blockIdx.x;
  if (col >= width || row >= out_h * out_w) return;
  const int c = col / (k * k), ky = (col / k) % k, kx = col % k;
  const int iy = (row / out_w) * stride + ky - pad;
  const int ix = (row % out_w) * stride + kx - pad;
  bf16 v = to(0.0f);
  if (iy >= 0 && ix >= 0 && iy < in_h && ix < in_w)
    v = chw ? x[((size_t)c * in_h + iy) * in_w + ix] : x[((size_t)iy * in_w + ix) * channels + c];
  y[(size_t)row * width + col] = v;
}

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

// SAM's LayerNorm2d over a position's channels, op by op in bf16 as its module computes it in the input's dtype.
extern "C" __global__ void __launch_bounds__(256) tf_uocr_layer_norm2d(const bf16* __restrict__ x, const bf16* __restrict__ w,
                                                                       const bf16* __restrict__ b, bf16* __restrict__ y,
                                                                       int width, float eps) {
  __shared__ float part[33];
  const bf16* xr = x + (size_t)blockIdx.x * width;
  bf16* yr = y + (size_t)blockIdx.x * width;
  float s = 0.0f;
  for (int i = threadIdx.x; i < width; i += blockDim.x) s += ld(xr + i);
  const float u = bf(block_sum(s, part) / (float)width);
  float v = 0.0f;
  for (int i = threadIdx.x; i < width; i += blockDim.x) {
    const float d = bf(ld(xr + i) - u);
    v += bf(d * d);
  }
  const float var = bf(block_sum(v, part) / (float)width);
  const float den = bf(sqrtf(bf(var + eps)));
  for (int i = threadIdx.x; i < width; i += blockDim.x) {
    const float d = bf(bf(ld(xr + i) - u) / den);
    yr[i] = to(bf(ld(w + i) * d) + ld(b + i));
  }
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

// SAM's window partition: [side, side, C] zero-padded to `padded`, as [windows, window^2, C] row-major windows.
extern "C" __global__ void __launch_bounds__(256) tf_uocr_window(const bf16* __restrict__ x, bf16* __restrict__ y, int side,
                                                                 int padded, int window, int channels) {
  const int c = blockIdx.y * 256 + threadIdx.x, r = blockIdx.x;
  if (c >= channels || r >= padded * padded) return;
  const int per = padded / window, w2 = window * window;
  const int win = r / w2, in = r % w2;
  const int gy = (win / per) * window + in / window, gx = (win % per) * window + in % window;
  y[(size_t)r * channels + c] = (gy < side && gx < side) ? x[((size_t)gy * side + gx) * channels + c] : to(0.0f);
}

// The partition undone and cropped, then the residual: x[side, side, C] = bf16(x + windows' row).
extern "C" __global__ void __launch_bounds__(256) tf_uocr_unwindow_add(bf16* __restrict__ x, const bf16* __restrict__ win,
                                                                       int side, int padded, int window, int channels) {
  const int c = blockIdx.y * 256 + threadIdx.x, r = blockIdx.x;
  if (c >= channels || r >= side * side) return;
  const int gy = r / side, gx = r % side;
  const int per = padded / window;
  const int w = (gy / window) * per + gx / window, in = (gy % window) * window + gx % window;
  const size_t i = (size_t)r * channels + c;
  x[i] = to(ld(x + i) + ld(win + ((size_t)w * window * window + in) * channels + c));
}

// SAM's decomposed relative positions: q . R[q - k + side - 1] along h and w, each sum rounded to bf16.
extern "C" __global__ void __launch_bounds__(256) tf_uocr_rel_pos(const bf16* __restrict__ qkv, const bf16* __restrict__ tab_h,
                                                                  const bf16* __restrict__ tab_w, bf16* __restrict__ rel_h,
                                                                  bf16* __restrict__ rel_w, int groups, int side, int width) {
  const int L = side * side;
  const int x = threadIdx.x, row = blockIdx.x, h = blockIdx.y;
  if (x >= 2 * side || row >= groups * L) return;
  const int g = row / L, qi = row % L, qh = qi / side, qw = qi % side;
  const bool rows = x < side;
  const int kk = rows ? x : x - side;
  const bf16* q = qkv + (size_t)row * 3 * width + h * 64;
  const bf16* r = (rows ? tab_h : tab_w) + (size_t)((rows ? qh : qw) - kk + side - 1) * 64;
  float s = 0.0f;
  for (int d = 0; d < 64; d++) s += ld(q + d) * ld(r + d);
  bf16* out = rows ? rel_h : rel_w;
  out[(((size_t)h * groups + g) * L + qi) * side + kk] = to(s);
}

// Windowed or global attention, heads of 64: fp32 scores plus bf16(rel_h + rel_w), bf16 weights, one division.
#define UV_TILE 64
#define UV_PAD 66
extern "C" __global__ void __launch_bounds__(256) tf_uocr_attention(const bf16* __restrict__ qkv, const bf16* __restrict__ rel_h,
                                                                    const bf16* __restrict__ rel_w, bf16* __restrict__ o,
                                                                    int groups, int len, int width, int side, float scale) {
  __shared__ __align__(16) bf16 ks[UV_TILE * UV_PAD];
  __shared__ __align__(16) bf16 vs[UV_TILE * 64];
  const int tiles_per_group = (len + 7) / 8;
  const int g = blockIdx.x / tiles_per_group, h = blockIdx.y;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int qi = (blockIdx.x % tiles_per_group) * 8 + warp;
  const bool live = qi < len;
  const size_t stride = 3 * (size_t)width;
  const size_t row = (size_t)g * len + (live ? qi : 0);
  const bf16* kbase = qkv + (size_t)g * len * stride + width + h * 64;
  const bf16* vbase = kbase + width;
  float q[64];
  {
    const bf16* qr = qkv + row * stride + h * 64;
#pragma unroll
    for (int d = 0; d < 64; d += 8) load8(qr + d, q + d);
  }
  const size_t rel = (((size_t)h * groups + g) * len + (live ? qi : 0)) * (side > 0 ? side : 1);
  float m = -INFINITY, sum = 0.0f, a0 = 0.0f, a1 = 0.0f;
  for (int pass = 0; pass < 2; ++pass) {
    for (int t0 = 0; t0 < len; t0 += UV_TILE) {
      __syncthreads();
      // 64 keys x 64 dims, 8 bf16 a load: 512 loads over 256 threads
      for (int i = threadIdx.x; i < UV_TILE * 8; i += 256) {
        const int key = i >> 3, part = (i & 7) * 8;
        uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
        if (t0 + key < len) {
          kv = *reinterpret_cast<const uint4*>(kbase + (size_t)(t0 + key) * stride + part);
          if (pass == 1) vv = *reinterpret_cast<const uint4*>(vbase + (size_t)(t0 + key) * stride + part);
        }
        uint32_t* kd = reinterpret_cast<uint32_t*>(ks + key * UV_PAD + part);
        kd[0] = kv.x;
        kd[1] = kv.y;
        kd[2] = kv.z;
        kd[3] = kv.w;
        if (pass == 1) *reinterpret_cast<uint4*>(vs + key * 64 + part) = vv;
      }
      __syncthreads();
      if (!live) continue;
      float e[2];
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int key = lane + 32 * half, j = t0 + key;
        float s = -INFINITY;
        if (j < len) {
          const __nv_bfloat162* kr = reinterpret_cast<const __nv_bfloat162*>(ks + key * UV_PAD);
          float acc = 0.0f;
#pragma unroll
          for (int d = 0; d < 32; ++d) {
            const float2 f = __bfloat1622float2(kr[d]);
            acc += q[2 * d] * f.x;
            acc += q[2 * d + 1] * f.y;
          }
          s = acc * scale;
          if (side > 0) s += bf(ld(rel_h + rel + j / side) + ld(rel_w + rel + j % side));
        }
        e[half] = s;
      }
      if (pass == 0) {
        m = fmaxf(m, fmaxf(e[0], e[1]));
        continue;
      }
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const float x = t0 + lane + 32 * half < len ? expf(e[half] - m) : 0.0f;
        sum += x;
        e[half] = bf(x);
      }
      const int n = min(UV_TILE, len - t0);
      for (int k = 0; k < n; ++k) {
        const float w = __shfl_sync(0xffffffffu, e[k >> 5], k & 31);
        a0 += w * ld(vs + k * 64 + lane);
        a1 += w * ld(vs + k * 64 + lane + 32);
      }
    }
    if (pass == 0) m = warp_max(m);
  }
  if (!live) return;
  sum = warp_sum(sum);
  bf16* out = o + row * width + h * 64;
  out[lane] = to(a0 / sum);
  out[lane + 32] = to(a1 / sum);
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

