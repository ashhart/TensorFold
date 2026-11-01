// Idol's BiRefNet in fp32 as idol_birefnet.metal; bodies templated on a control (if constexpr), CTL_NONE shipped.
#pragma once

#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>

namespace tf_br {

enum Ctl : int {
  CTL_NONE = 0,
  CTL_CONV_HALF_BLOCKS = 1,  // br_conv adds its 256-term blocks in fp16
  CTL_CONV_RELU_MAX = 2,     // br_conv's ReLU by fmaxf, which drops a NaN
  CTL_NORM_HALF_SUM = 3,     // br_norm sums the row in fp16
  CTL_WINDOW_HALF_P = 4,     // br_window rounds P to fp16 before P V
  CTL_DEFORM_HALF_COL = 5,   // br_deform rounds its columns to fp16
  CTL_POOL_HALF_SUM = 6,     // br_pool sums in fp16
  CTL_RESIZE_HALF = 7,       // br_resize stages its samples in fp16
};

__device__ __forceinline__ float rh(float x) { return __half2float(__float2half_rn(x)); }

__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// A block of 256's total of every thread's `v`, returned to all: lanes, then the 8 warps in order.
__device__ __forceinline__ float block_sum(float v, float* part) {
  v = warp_sum(v);
  __syncthreads();
  if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = v;
  __syncthreads();
  float total = 0.0f;
#pragma unroll
  for (int w = 0; w < 8; ++w) total += part[w];
  return total;
}

constexpr int flag_bn = 1, flag_relu = 2, flag_gelu = 4, flag_add = 8, flag_map = 16;

// br_conv's tile: 64 pixels x 64 outputs a block of 256 threads (4 x 4 each), 32 terms a step, 256 terms a block.
constexpr int CBM = 64, CBN = 64, CBK = 32, CBLOCK = 256;

// y = conv_k(x) + b, then flags 1 BN, 2 ReLU, 4 GELU, 8 + r, 16 map; sums in ordered 256-term fp32 blocks.
template <int CTL>
__device__ __forceinline__ void conv(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b,
                                     const float* __restrict__ post, const float* r, const int* __restrict__ map,
                                     float* y, int rows, int cols, int cin, int ldx, int cout, int ldw, int k, int ldy,
                                     int yoff, int flags, int row0) {
  __shared__ __align__(16) float As[CBK][CBM + 4];
  __shared__ __align__(16) float Bs[CBK][CBN];
  const int t = threadIdx.x;
  const int pixels = rows * cols;
  const int m0 = blockIdx.x * CBM;
  const int n0 = blockIdx.y * CBN;
  const int pad = k / 2;
  const int K = k * k * cin;
  const int ty = t / 16, tx = t % 16;
  float acc[4][4], part[4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[i][j] = part[i][j] = 0.0f;
  for (int k0 = 0; k0 < K; k0 += CBK) {
    const int tap = k0 / cin;
    const int c0 = k0 % cin;
    // A: 64 pixels x 32 channels as 512 float4, two a thread, stored k-major
#pragma unroll
    for (int q = 0; q < 2; ++q) {
      const int e = t + q * 256;
      const int row = e / 8, kq = (e % 8) * 4;
      const int pix = m0 + row;
      const int oy = pix / cols, ox = pix % cols;
      const int iy = oy + tap / k - pad, ix = ox + tap % k - pad;
      float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      if (pix < pixels && iy >= 0 && ix >= 0 && iy < rows && ix < cols)
        v = *reinterpret_cast<const float4*>(x + (size_t)(iy * cols + ix) * ldx + c0 + kq);
      As[kq][row] = v.x;
      As[kq + 1][row] = v.y;
      As[kq + 2][row] = v.z;
      As[kq + 3][row] = v.w;
    }
    // B: 32 rows x 64 outputs as 512 float4
#pragma unroll
    for (int q = 0; q < 2; ++q) {
      const int e = t + q * 256;
      const int kk = e / 16, nq = (e % 16) * 4;
      *reinterpret_cast<float4*>(&Bs[kk][nq]) = *reinterpret_cast<const float4*>(w + (size_t)(k0 + kk) * ldw + n0 + nq);
    }
    __syncthreads();
#pragma unroll 8
    for (int kk = 0; kk < CBK; ++kk) {
      const float4 av = *reinterpret_cast<const float4*>(&As[kk][ty * 4]);
      const float4 bv = *reinterpret_cast<const float4*>(&Bs[kk][tx * 4]);
      const float am[4] = {av.x, av.y, av.z, av.w};
      const float bm[4] = {bv.x, bv.y, bv.z, bv.w};
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) part[i][j] = __fmaf_rn(am[i], bm[j], part[i][j]);
    }
    __syncthreads();
    if ((k0 + CBK) % CBLOCK == 0 || k0 + CBK >= K) {
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          if constexpr (CTL == CTL_CONV_HALF_BLOCKS) {
            acc[i][j] = rh(rh(acc[i][j]) + rh(part[i][j]));
          } else {
            acc[i][j] = __fadd_rn(acc[i][j], part[i][j]);
          }
          part[i][j] = 0.0f;
        }
    }
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int row = m0 + ty * 4 + i;
    if (row >= pixels) continue;
    int out = row0 + row;
    if (flags & flag_map) out = map[row];
    if (out < 0) continue;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int col = n0 + tx * 4 + j;
      if (col >= cout) continue;
      const size_t at = (size_t)out * ldy + yoff + col;
      float v = acc[i][j] + b[col];
      if (flags & flag_bn) v = (v - post[col]) * post[ldw + col] + post[2 * ldw + col];
      if (flags & flag_relu) {
        if constexpr (CTL == CTL_CONV_RELU_MAX) {
          v = fmaxf(v, 0.0f);
        } else {
          v = v < 0.0f ? 0.0f : v;
        }
      }
      if (flags & flag_gelu) v = (v * 0.5f) * (1.0f + erff(v * 0.70710677f));
      if (flags & flag_add) v = r[at] + v;
      y[at] = v;
    }
  }
}

// Row r at channel c: mode 0 row r, 1 row map[r], 2 PatchMerging's [x0 | x1 | x2 | x3] at r's 2 x 2 patch.
__device__ __forceinline__ float norm_at(const float* x, const int* map, int width, int mode, int src_w, int r, int c) {
  if (mode == 0) return x[(size_t)r * width + c];
  if (mode == 1) return x[(size_t)map[r] * width + c];
  const int q = width / 4;
  const int part = c / q;
  const int half_w = src_w / 2;
  const int sy = (r / half_w) * 2 + (part & 1);
  const int sx = (r % half_w) * 2 + (part >> 1);
  return x[(size_t)(sy * src_w + sx) * q + c % q];
}

// y = LayerNorm(row) g + beta, two fp32 passes; a map's negative rows are zeros (torch pads after the norm).
template <int CTL>
__device__ __forceinline__ void norm(const float* __restrict__ x, const int* __restrict__ map, const float* __restrict__ g,
                                     const float* __restrict__ beta, float* __restrict__ y, int width, int mode,
                                     int src_w, int ldy, int yoff, float eps) {
  __shared__ float part[8];
  const int row = blockIdx.x, t = threadIdx.x;
  float* out = y + (size_t)row * ldy + yoff;
  if (mode == 1 && map[row] < 0) {
    for (int c = t; c < width; c += 256) out[c] = 0.0f;
    return;
  }
  float s = 0.0f;
  for (int c = t; c < width; c += 256) {
    if constexpr (CTL == CTL_NORM_HALF_SUM) {
      s = rh(rh(s) + rh(norm_at(x, map, width, mode, src_w, row, c)));
    } else {
      s += norm_at(x, map, width, mode, src_w, row, c);
    }
  }
  const float mean = block_sum(s, part) / (float)width;
  float d2 = 0.0f;
  for (int c = t; c < width; c += 256) {
    const float d = norm_at(x, map, width, mode, src_w, row, c) - mean;
    d2 += d * d;
  }
  const float rstd = 1.0f / sqrtf(block_sum(d2, part) / (float)width + eps);
  for (int c = t; c < width; c += 256) out[c] = (norm_at(x, map, width, mode, src_w, row, c) - mean) * rstd * g[c] + beta[c];
}

// br_window's tile: a 12 x 12 window (144 tokens), one head of 32, 24 queries a block of 192 threads.
constexpr int WN = 144, WD = 32, WQ = 24, WKS = WD + 1, WSS = WN + 1;

// Token i's region (0 to 8) in the shifted grid: rows and columns split at size - 12 and size - shift.
__device__ __forceinline__ int region(int i, int wy, int wx, int rows, int cols, int shift) {
  const int yy = wy * 12 + i / 12;
  const int xx = wx * 12 + i % 12;
  const int ry = yy < rows - 12 ? 0 : (yy < rows - shift ? 1 : 2);
  const int rx = xx < cols - 12 ? 0 : (xx < cols - shift ? 1 : 2);
  return ry * 3 + rx;
}

// softmax(q scale k^T + bias (+ -100 across regions)) v of one window's head: fp32 scores, softmax and P V sums.
template <int CTL>
__device__ __forceinline__ void window(const float* __restrict__ qkv, const float* __restrict__ table,
                                       float* __restrict__ o, int width, int heads, int rows, int cols, int shift,
                                       float scale) {
  __shared__ float Ks[WN * WKS];
  __shared__ float Qs[WQ * WD];
  __shared__ float S[WQ * WSS];
  const int t = threadIdx.x;
  const int h = blockIdx.y, win = blockIdx.z;
  const int ld = 3 * width;
  const int base = win * WN;
  const int first = blockIdx.x * WQ;
  const int qcol = h * WD;
  for (int i = t; i < WN * WD; i += 192) Ks[(i / WD) * WKS + i % WD] = qkv[(size_t)(base + i / WD) * ld + width + qcol + i % WD];
  for (int i = t; i < WQ * WD; i += 192) Qs[i] = qkv[(size_t)(base + first + i / WD) * ld + qcol + i % WD] * scale;
  __syncthreads();
  const int wy = win / (cols / 12), wx = win % (cols / 12);
  for (int i = t; i < WQ * WN; i += 192) {
    const int row = i / WN, j = i % WN;
    const int qi = first + row;
    float s = 0.0f;
#pragma unroll
    for (int d = 0; d < WD; ++d) s = __fmaf_rn(Qs[row * WD + d], Ks[j * WKS + d], s);
    float v = s + table[((qi / 12 + 11 - j / 12) * 23 + (qi % 12 + 11 - j % 12)) * heads + h];
    if (shift) v = v + (region(j, wy, wx, rows, cols, shift) == region(qi, wy, wx, rows, cols, shift) ? 0.0f : -100.0f);
    S[row * WSS + j] = v;
  }
  __syncthreads();
  // eight threads a query row: its max, exp and total, then each P
  {
    const int row = t / 8, sub = t % 8;
    float* sr = S + row * WSS;
    float most = -INFINITY;
    for (int j = sub; j < WN; j += 8) most = fmaxf(most, sr[j]);
#pragma unroll
    for (int m = 1; m < 8; m <<= 1) most = fmaxf(most, __shfl_xor_sync(0xffffffffu, most, m));
    float total = 0.0f;
    for (int j = sub; j < WN; j += 8) {
      const float e = expf(sr[j] - most);
      sr[j] = e;
      total += e;
    }
#pragma unroll
    for (int m = 1; m < 8; m <<= 1) total += __shfl_xor_sync(0xffffffffu, total, m);
    for (int j = sub; j < WN; j += 8) {
      if constexpr (CTL == CTL_WINDOW_HALF_P) {
        sr[j] = rh(sr[j] / total);
      } else {
        sr[j] = sr[j] / total;
      }
    }
  }
  __syncthreads();
  for (int i = t; i < WQ * WD; i += 192) {
    const int row = i / WD, d = i % WD;
    const float* vp = qkv + (size_t)base * ld + 2 * width + qcol + d;
    float acc = 0.0f;
    for (int j = 0; j < WN; ++j) acc = __fmaf_rn(S[row * WSS + j], vp[(size_t)j * ld], acc);
    o[(size_t)(base + first + row) * width + qcol + d] = acc;
  }
}

// y = F.interpolate(x, bilinear, align_corners=True): x [in_h * in_w, ldx] from xoff, y [out_h * out_w, ldy] from yoff.
template <int CTL>
__device__ __forceinline__ void resize(const float* __restrict__ x, float* __restrict__ y, int in_h, int in_w, int out_h,
                                       int out_w, int channels, int ldx, int xoff, int ldy, int yoff, int pix, int ch) {
  if (ch >= channels || pix >= out_h * out_w) return;
  const int oy = pix / out_w, ox = pix % out_w;
  const float sh = out_h > 1 ? (float)(in_h - 1) / (float)(out_h - 1) : 0.0f;
  const float sw = out_w > 1 ? (float)(in_w - 1) / (float)(out_w - 1) : 0.0f;
  const float fy = sh * (float)oy, fx = sw * (float)ox;
  const int y0 = min((int)fy, in_h - 1), x0 = min((int)fx, in_w - 1);
  const int y1 = min(y0 + 1, in_h - 1), x1 = min(x0 + 1, in_w - 1);
  const float ly = fminf(fmaxf(fy - (float)y0, 0.0f), 1.0f);
  const float lx = fminf(fmaxf(fx - (float)x0, 0.0f), 1.0f);
  const float* src = x + xoff + ch;
  float v00 = src[(size_t)(y0 * in_w + x0) * ldx], v01 = src[(size_t)(y0 * in_w + x1) * ldx];
  float v10 = src[(size_t)(y1 * in_w + x0) * ldx], v11 = src[(size_t)(y1 * in_w + x1) * ldx];
  if constexpr (CTL == CTL_RESIZE_HALF) {
    v00 = rh(v00);
    v01 = rh(v01);
    v10 = rh(v10);
    v11 = rh(v11);
  }
  const float top = (1.0f - lx) * v00 + lx * v01;
  const float bottom = (1.0f - lx) * v10 + lx * v11;
  y[(size_t)pix * ldy + yoff + ch] = (1.0f - ly) * top + ly * bottom;
}

// torchvision's modulated deform_conv2d columns [count, k * k * 64]: 2 sigmoid(m) times the moved bilinear sample.
template <int CTL>
__device__ __forceinline__ void deform(const float* __restrict__ x, const float* __restrict__ offs, float* __restrict__ col,
                                       int rows, int cols, int k, int first, int count, int ldo, int r, int tap) {
  const int taps = k * k;
  if (tap >= taps || r >= count) return;
  const int pix = first + r;
  const int oy = pix / cols, ox = pix % cols;
  const int pad = k / 2;
  const float* off = offs + (size_t)pix * ldo;
  const float mask = 2.0f * sigmoid(off[2 * taps + tap]);
  const float h = (float)(oy - pad + tap / k) + off[2 * tap];
  const float w = (float)(ox - pad + tap % k) + off[2 * tap + 1];
  float4* out = reinterpret_cast<float4*>(col + (size_t)r * taps * 64 + tap * 64);
  if (!(h > -1.0f && h < (float)rows && w > -1.0f && w < (float)cols)) {
#pragma unroll
    for (int c = 0; c < 16; ++c) out[c] = make_float4(mask * 0.0f, mask * 0.0f, mask * 0.0f, mask * 0.0f);
    return;
  }
  const int hl = (int)floorf(h), wl = (int)floorf(w);
  const float lh = h - (float)hl, lw = w - (float)wl;
  const float hh = 1.0f - lh, hw = 1.0f - lw;
  const float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  const bool top = hl >= 0, bottom = hl + 1 <= rows - 1, left = wl >= 0, right = wl + 1 <= cols - 1;
  const float4* r1 = reinterpret_cast<const float4*>(x + (size_t)(max(hl, 0) * cols + max(wl, 0)) * 64);
  const float4* r2 = reinterpret_cast<const float4*>(x + (size_t)(max(hl, 0) * cols + min(wl + 1, cols - 1)) * 64);
  const float4* r3 = reinterpret_cast<const float4*>(x + (size_t)(min(hl + 1, rows - 1) * cols + max(wl, 0)) * 64);
  const float4* r4 = reinterpret_cast<const float4*>(x + (size_t)(min(hl + 1, rows - 1) * cols + min(wl + 1, cols - 1)) * 64);
  const float4 z = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll 4
  for (int c = 0; c < 16; ++c) {
    const float4 v1 = top && left ? r1[c] : z;
    const float4 v2 = top && right ? r2[c] : z;
    const float4 v3 = bottom && left ? r3[c] : z;
    const float4 v4 = bottom && right ? r4[c] : z;
    float4 s;
    s.x = mask * (((w1 * v1.x + w2 * v2.x) + w3 * v3.x) + w4 * v4.x);
    s.y = mask * (((w1 * v1.y + w2 * v2.y) + w3 * v3.y) + w4 * v4.y);
    s.z = mask * (((w1 * v1.z + w2 * v2.z) + w3 * v3.z) + w4 * v4.z);
    s.w = mask * (((w1 * v1.w + w2 * v2.w) + w3 * v3.w) + w4 * v4.w);
    if constexpr (CTL == CTL_DEFORM_HALF_COL) s = make_float4(rh(s.x), rh(s.y), rh(s.z), rh(s.w));
    out[c] = s;
  }
}

// mean [channels] of x [pixels, channels] over its pixels: a block of 256 a channel.
template <int CTL>
__device__ __forceinline__ void pool(const float* __restrict__ x, float* __restrict__ mean, int pixels, int channels) {
  __shared__ float part[8];
  const int ch = blockIdx.x;
  float s = 0.0f;
  for (int i = threadIdx.x; i < pixels; i += 256) {
    if constexpr (CTL == CTL_POOL_HALF_SUM) {
      s = rh(rh(s) + rh(x[(size_t)i * channels + ch]));
    } else {
      s += x[(size_t)i * channels + ch];
    }
  }
  const float all = block_sum(s, part);
  if (threadIdx.x == 0) mean[ch] = all / (float)pixels;
}

}  // namespace tf_br
