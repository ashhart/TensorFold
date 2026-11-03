// Unlimited-OCR-2 on CUDA: uocr/vision.metal's and uocr/decoder.metal's steps op for op, rounded where the
// publisher's torch code rounds; the projections run on cuBLASLt (cuda/vit.zig's Gemms) beside these.
#include <cuda_bf16.h>
#include <stdint.h>
#include "common/reduce.cuh"

namespace tf_uocr {

typedef __nv_bfloat16 bf16;

#define UD_NEG -1.0e9f
#define UD_ENTMAX_NEG -1.0e4f
#define UD_CLAMP 1.0e4f

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float ld(const bf16* p) { return __bfloat162float(*p); }
__device__ __forceinline__ bf16 to(float x) { return __float2bfloat16_rn(x); }
__device__ __forceinline__ float clampf(float x, float lo, float hi) { return fminf(fmaxf(x, lo), hi); }

using tf_common::warp_sum;

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

__device__ __forceinline__ int warp_min_int(int v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = min(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

// A block's sum of every thread's `v` (blockDim.x a multiple of 32, at most 1,024), returned to all of them.
__device__ __forceinline__ float block_sum(float v, float* part) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, n = (blockDim.x + 31) >> 5;
  v = warp_sum(v);
  if (lane == 0) part[w] = v;
  __syncthreads();
  if (w == 0) {
    float s = lane < n ? part[lane] : 0.0f;
    s = warp_sum(s);
    if (lane == 0) part[32] = s;
  }
  __syncthreads();
  const float out = part[32];
  __syncthreads();
  return out;
}

// 8 bf16 at p (16-byte aligned) widened.
__device__ __forceinline__ void load8(const bf16* p, float* out) {
  const uint4 u = *reinterpret_cast<const uint4*>(p);
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 f = __bfloat1622float2(h[i]);
    out[2 * i] = f.x;
    out[2 * i + 1] = f.y;
  }
}

// erf to under one ulp (Juffa's faithfully rounded erff, as uv_erf).
__device__ __forceinline__ float erf_(float x) {
  const float t = fabsf(x), s = x * x;
  float r;
  if (t > 0.927734375f) {
    r = fmaf(-1.72853470e-5f, t, 3.83197126e-4f);
    const float u = fmaf(-3.88396438e-3f, t, 2.42546219e-2f);
    r = fmaf(r, s, u);
    r = fmaf(r, t, -1.06777877e-1f);
    r = fmaf(r, t, -6.34846687e-1f);
    r = fmaf(r, t, -1.28717512e-1f);
    r = fmaf(r, t, -t);
    r = copysignf(1.0f - expf(r), x);
  } else {
    r = -5.96761703e-4f;
    r = fmaf(r, s, 4.99119423e-3f);
    r = fmaf(r, s, -2.67681349e-2f);
    r = fmaf(r, s, 1.12819925e-1f);
    r = fmaf(r, s, -3.76125336e-1f);
    r = fmaf(r, s, 1.28379166e-1f);
    r = fmaf(r, x, x);
  }
  return r;
}

}  // namespace tf_uocr

using namespace tf_uocr;

// ---- the image encoder ----

// Convolution patches: row (oy, ox) of [out_h * out_w, channels * k * k] in the weight's (c, ky, kx) order, zero
// outside the input. The input is [channels, h, w] (pixels, chw = 1) or [h, w, channels] (feature maps).
// Rows on grid x, 256-column strips on grid y.
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

// SAM's decomposed relative positions: rel_h[h, g, q, kh] = q . Rh[qh - kh + side - 1] and rel_w likewise over kw,
// q unscaled from the qkv rows [groups x side^2, 3 x width], each sum rounded to bf16. Grid: (rows, heads); a
// thread per one of the 2 x side outputs.
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

// o [groups x len, width] = softmax(q k^T scale + bias) v within each group, heads of 64: fp32 scores, the bias
// bf16(rel_h + rel_w) as the module's bf16 mask (side 0: none), exp weights rounded to bf16 before p v, the sum of
// the unrounded ones dividing at the end (uv_attention's arithmetic). A warp a query row, 8 rows a block; keys in
// tiles of 64 through shared memory, twice: once for the row's maximum, once for the weights and p v.
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

// ---- the decoder ----

// Token rows: x[i] = embed[ids[i]] (1280 wide), image tokens' rows left for their pictures. Grid (rows, 5).
extern "C" __global__ void __launch_bounds__(256) tf_uocr_embed(const unsigned* __restrict__ ids, const bf16* __restrict__ embed,
                                                                bf16* __restrict__ x, unsigned image_token) {
  const int c = blockIdx.y * 256 + threadIdx.x;
  const unsigned id = ids[blockIdx.x];
  if (c >= 1280 || id == image_token) return;
  x[(size_t)blockIdx.x * 1280 + c] = embed[(size_t)id * 1280 + c];
}

// DeepseekV2RMSNorm on bf16 rows: x * rsqrt(mean(x^2) + eps) in fp32, rounded, then times the weight in bf16.
extern "C" __global__ void __launch_bounds__(256) tf_uocr_rms_norm(const bf16* __restrict__ x, const bf16* __restrict__ w,
                                                                   bf16* __restrict__ y, int width, float eps) {
  __shared__ float part[33];
  const bf16* xr = x + (size_t)blockIdx.x * width;
  float s = 0.0f;
  for (int i = threadIdx.x; i < width; i += blockDim.x) {
    const float v = ld(xr + i);
    s += v * v;
  }
  s = block_sum(s, part);
  const float r = 1.0f / sqrtf(s / (float)width + eps);
  bf16* yr = y + (size_t)blockIdx.x * width;
  for (int i = threadIdx.x; i < width; i += blockDim.x) yr[i] = to(ld(w + i) * bf(ld(xr + i) * r));
}

// The page2d angles for one position: 63 token frequencies, then the page parity (page mod 2) at frequency 1.
__device__ __forceinline__ float angle(const float* inv_tok, unsigned tok, unsigned page, int i) {
  const int j = i % 64;
  return j < 63 ? (float)tok * inv_tok[j] : (float)(page % 2);
}

// The main branch's page2d rotary in place on rows of 128-wide heads: row r at (tok[r], page[r]); x * cos +
// rotate_half(x) * sin, every product and the sum rounded to bf16, cos and sin bf16. Grid (rows, heads), 128 threads.
extern "C" __global__ void __launch_bounds__(128) tf_uocr_rope(bf16* __restrict__ x, const unsigned* __restrict__ tok,
                                                               const unsigned* __restrict__ page,
                                                               const float* __restrict__ inv_tok, int stride) {
  const int d = threadIdx.x, row = blockIdx.x, head = blockIdx.y;
  bf16* h = x + (size_t)row * stride + head * 128;
  const float ang = angle(inv_tok, tok[row], page[row], d);
  const float c = bf(cosf(ang)), s = bf(sinf(ang));
  const float rh = d < 64 ? -ld(h + d + 64) : ld(h + d - 64);
  const bf16 out = to(bf(ld(h + d) * c) + bf(rh * s));
  __syncthreads();
  h[d] = out;
}

// The router's index q or k: Gemma RMSNorm (fp32, clamped, times 1 + w, clamped, bf16), the same rotary in bf16,
// then fp32. A 128-thread block a row; x rows are 128 bf16, y rows 128 floats.
extern "C" __global__ void __launch_bounds__(128) tf_uocr_router_norm(const bf16* __restrict__ x, const bf16* __restrict__ w,
                                                                      const unsigned* __restrict__ tok,
                                                                      const unsigned* __restrict__ page,
                                                                      const float* __restrict__ inv_tok, float* __restrict__ y) {
  __shared__ float part[33];
  __shared__ bf16 normed[128];
  const int row = blockIdx.x, d = threadIdx.x;
  float v = ld(x + (size_t)row * 128 + d);
  if (!isfinite(v)) v = 0.0f;
  const float s = block_sum(v * v, part);
  float n = clampf(v * (1.0f / sqrtf(s / 128.0f + 1e-6f)), -UD_CLAMP, UD_CLAMP);
  n = clampf(n * (1.0f + ld(w + d)), -UD_CLAMP, UD_CLAMP);
  normed[d] = to(n);
  __syncthreads();
  const float ang = angle(inv_tok, tok[row], page[row], d);
  const float c = bf(cosf(ang)), sn = bf(sinf(ang));
  const float rh = d < 64 ? -ld(normed + d + 64) : ld(normed + d - 64);
  float out = bf(bf(ld(normed + d) * c) + bf(rh * sn));
  if (!isfinite(out)) out = 0.0f;
  y[(size_t)row * 128 + d] = out;
}

// A complete block's summary key: softmax over its 32 keys of clamp(k . head_cls / sqrt(128)), then the weighted
// sum of the keys, in fp32 (one warp a block, one key a lane).
extern "C" __global__ void __launch_bounds__(32) tf_uocr_block_keys(const float* __restrict__ idx_k, const bf16* __restrict__ head_cls,
                                                                    float* __restrict__ kbar, int first, float scale) {
  __shared__ float wts[32];
  const int c = first + blockIdx.x, j = threadIdx.x;
  const float* k = idx_k + ((size_t)c * 32 + j) * 128;
  float a = 0.0f;
  for (int d = 0; d < 128; d++) a += k[d] * ld(head_cls + d);
  a = clampf(a * scale, -UD_CLAMP, UD_CLAMP);
  const float m = warp_max(a);
  const float e = expf(a - m);
  const float total = fmaxf(warp_sum(e), 1e-6f);
  wts[j] = e / total;
  __syncwarp();
  for (int d = j; d < 128; d += 32) {
    float s = 0.0f;
    for (int t = 0; t < 32; t++) s += wts[t] * idx_k[((size_t)c * 32 + t) * 128 + d];
    kbar[(size_t)c * 128 + d] = isfinite(s) ? s : 0.0f;
  }
}

// One query's remote reads (a warp a query): block scores against the summary keys, the page-aware candidate
// blocks, the top 16 by score, entmax-1.5 over them, and the first 12 with bias log(max(prior, floor)) - their mean.
// sel[q][s] = -1 marks an unread slot (ud_route's arithmetic). At most 4,096 blocks.
extern "C" __global__ void __launch_bounds__(32) tf_uocr_route(const float* __restrict__ idx_q, const float* __restrict__ kbar,
                                                               const int* __restrict__ bmin, const int* __restrict__ bmax,
                                                               const int* __restrict__ qpage, const unsigned* __restrict__ qpos,
                                                               int* __restrict__ sel, float* __restrict__ bias, int nblk,
                                                               int complete, float scale, float idx_scale, float floor_,
                                                               int round_bias) {
  __shared__ float x[4096];
  __shared__ unsigned char cand[4096];
  const int q = blockIdx.x, lane = threadIdx.x;
  const float* iq = idx_q + (size_t)q * 128;
  const int qp = qpage[q];
  const int last_remote = (int)(qpos[q] / 32) - 4;
  bool any_page = false;
  if (qp >= 0) {
    for (int c = lane; c < nblk; c += 32) any_page = any_page || (bmin[c] >= 0 && bmin[c] <= qp && qp <= bmax[c]);
    any_page = __any_sync(0xffffffffu, any_page);
  }
  bool any_cand = false;
  for (int c = lane; c < nblk; c += 32) {
    const bool remote = c <= last_remote && c < complete;
    const bool allow = !any_page || (bmin[c] >= 0 && bmin[c] <= qp && qp <= bmax[c]);
    float z = UD_NEG;
    if (remote) {
      const float* kb = kbar + (size_t)c * 128;
      float s = 0.0f;
      for (int d = 0; d < 128; d++) s += iq[d] * kb[d];
      z = clampf(s * scale, -UD_CLAMP, UD_CLAMP);
    }
    x[c] = z;
    cand[c] = (unsigned char)(remote ? (allow ? 2 : 1) : 0);
    any_cand = any_cand || (remote && allow);
  }
  any_cand = __any_sync(0xffffffffu, any_cand);
  __syncwarp();
  for (int c = lane; c < nblk; c += 32) {
    const bool is = any_cand ? cand[c] == 2 : cand[c] >= 1;
    cand[c] = (unsigned char)is;
    float v = is ? x[c] * idx_scale : UD_ENTMAX_NEG;
    if (!isfinite(v)) v = UD_ENTMAX_NEG;
    x[c] = v;
  }
  __syncwarp();
  // the top M by value, ties to the lower block
  const int M = max(1, min(16, nblk));
  float xv[16];
  int xi[16];
  for (int r = 0; r < M; r++) {
    float best = -INFINITY;
    int at = 0x7fffffff;
    for (int c = lane; c < nblk; c += 32) {
      bool taken = false;
      for (int t = 0; t < r; t++) taken = taken || xi[t] == c;
      if (!taken && (x[c] > best || (x[c] == best && c < at))) {
        best = x[c];
        at = c;
      }
    }
    const float top = warp_max(best);
    const int idx = warp_min_int(best == top ? at : 0x7fffffff);
    xv[r] = top;
    xi[r] = idx;
  }
  if (lane != 0) return;
  // entmax-1.5 over the M values (already in descending order)
  const float mx = xv[0];
  float X[16], tau[16];
  float cs = 0.0f, cs2 = 0.0f;
  int support = 0;
  for (int k = 0; k < M; k++) {
    X[k] = (xv[k] - mx) / 2.0f;
    cs += X[k];
    cs2 += X[k] * X[k];
    const float rho = (float)(k + 1);
    const float mean = cs / rho, mean_sq = cs2 / rho;
    const float ss = rho * (mean_sq - mean * mean);
    const float delta = fmaxf((1.0f - ss) / rho, 0.0f);
    tau[k] = mean - sqrtf(delta);
    support += tau[k] <= X[k] ? 1 : 0;
  }
  const float tau_star = tau[max(support, 1) - 1];
  const int K = max(1, min(12, M));
  float logp[12];
  float total = 0.0f;
  int count = 0;
  for (int s = 0; s < K; s++) {
    const float d = fmaxf(X[s] - tau_star, 0.0f);
    logp[s] = logf(fmaxf(d * d, floor_));
    if (cand[xi[s]]) {
      total += logp[s];
      count++;
    }
  }
  const float mu = total / (float)max(count, 1);
  for (int s = 0; s < 12; s++) {
    const bool valid = s < K && cand[xi[s]];
    sel[q * 12 + s] = valid ? xi[s] : -1;
    const float b = valid ? logp[s] - mu : 0.0f;
    bias[q * 12 + s] = round_bias ? bf(b) : b;
  }
}

// One query row and head (8 warps): softmax over the local keys p-127..p (no bias) and the selected blocks' keys
// before the window (their bias), fp32 scores and weights, out = sum(p v) / sum(p) rounded to bf16 (ud_attend's
// arithmetic). Grid (rows, heads); 256 threads.
extern "C" __global__ void __launch_bounds__(256) tf_uocr_attend(const bf16* __restrict__ q, const bf16* __restrict__ kc,
                                                                 const bf16* __restrict__ vc, const int* __restrict__ sel,
                                                                 const float* __restrict__ bias, const unsigned* __restrict__ qpos,
                                                                 bf16* __restrict__ o, int width, float scale, int local) {
  __shared__ float score[512];
  __shared__ unsigned key[512];
  __shared__ unsigned seg_first[13], seg_start[14];
  __shared__ float seg_bias[13];
  __shared__ float part_max[8], part_sum[8];
  __shared__ float4 part_acc[8][32];
  const int row = blockIdx.x, h = blockIdx.y;
  const int t = threadIdx.x, sg = t >> 5, lane = t & 31;
  const unsigned pos = qpos[row];
  const unsigned lo = pos + 1 > (unsigned)local ? pos + 1 - local : 0;
  if (t == 0) {
    unsigned n = 0;
    for (int s = 0; s <= 12; s++) {
      unsigned first = 0, last = 0;
      float b = 0.0f;
      if (s < 12) {
        const int c = sel[row * 12 + s];
        if (c >= 0) {
          first = (unsigned)c * 32;
          last = max(first, min(first + 32, lo));
          b = bias[row * 12 + s];
        }
      } else {
        first = lo;
        last = pos + 1;
      }
      seg_first[s] = first;
      seg_start[s] = n;
      seg_bias[s] = b;
      n += last - first;
    }
    seg_start[13] = n;
  }
  __syncthreads();
  const unsigned n = seg_start[13];
  const bf16* qr = q + (size_t)row * width + h * 128 + lane * 4;
  const float4 qv = make_float4(ld(qr), ld(qr + 1), ld(qr + 2), ld(qr + 3));
  float m = -INFINITY;
  unsigned s = 0;
  for (unsigned i = sg; i < n; i += 8) {
    while (i >= seg_start[s + 1]) s++;
    const unsigned j = seg_first[s] + (i - seg_start[s]);
    const bf16* kr = kc + (size_t)j * width + h * 128 + lane * 4;
    const float dot = qv.x * ld(kr) + qv.y * ld(kr + 1) + qv.z * ld(kr + 2) + qv.w * ld(kr + 3);
    const float sc = warp_sum(dot) * scale + seg_bias[s];
    if (lane == 0) {
      score[i] = sc;
      key[i] = j;
    }
    m = fmaxf(m, sc);
  }
  if (lane == 0) part_max[sg] = m;
  __syncthreads();
  m = part_max[0];
  for (int g = 1; g < 8; g++) m = fmaxf(m, part_max[g]);
  float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  float total = 0.0f;
  for (unsigned i = sg; i < n; i += 8) {
    const float e = expf(score[i] - m);
    total += e;
    const bf16* vr = vc + (size_t)key[i] * width + h * 128 + lane * 4;
    acc.x += e * ld(vr);
    acc.y += e * ld(vr + 1);
    acc.z += e * ld(vr + 2);
    acc.w += e * ld(vr + 3);
  }
  part_acc[sg][lane] = acc;
  if (lane == 0) part_sum[sg] = total;
  __syncthreads();
  if (sg != 0) return;
  float4 sum = part_acc[0][lane];
  float tot = part_sum[0];
  for (int g = 1; g < 8; g++) {
    sum.x += part_acc[g][lane].x;
    sum.y += part_acc[g][lane].y;
    sum.z += part_acc[g][lane].z;
    sum.w += part_acc[g][lane].w;
    tot += part_sum[g];
  }
  bf16* out = o + (size_t)row * width + h * 128 + lane * 4;
  out[0] = to(sum.x / tot);
  out[1] = to(sum.y / tot);
  out[2] = to(sum.z / tot);
  out[3] = to(sum.w / tot);
}

// y = bf16(bf16(silu(g)) * u): the MLP's act_fn(gate) * up in bf16.
extern "C" __global__ void __launch_bounds__(256) tf_uocr_silu_mul(const bf16* __restrict__ g, const bf16* __restrict__ u,
                                                                   bf16* __restrict__ y, long long n) {
  const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
  if (i >= n) return;
  const float v = ld(g + i);
  y[i] = to(bf(v / (1.0f + expf(-v))) * ld(u + i));
}

// The MoE gate for one row (a 64-thread block): fp32 logits of x against the gate's 64 rows, softmax, and the 6
// largest scores (descending; ties to the lower expert) as ids and weights, not renormalized.
extern "C" __global__ void __launch_bounds__(64) tf_uocr_moe_gate(const bf16* __restrict__ x, const bf16* __restrict__ w,
                                                                  unsigned* __restrict__ ids, float* __restrict__ wts) {
  __shared__ float part[33];
  __shared__ float score[64];
  const int row = blockIdx.x, e = threadIdx.x;
  const bf16* xr = x + (size_t)row * 1280;
  const bf16* wr = w + (size_t)e * 1280;
  float s = 0.0f;
  for (int d = 0; d < 1280; d++) s += ld(xr + d) * ld(wr + d);
  float m = warp_max(s);
  if ((e & 31) == 0) part[e >> 5] = m;
  __syncthreads();
  m = fmaxf(part[0], part[1]);
  __syncthreads();
  const float ex = expf(s - m);
  const float total = block_sum(ex, part);
  score[e] = ex / total;
  __syncthreads();
  if (e != 0) return;
  bool taken[64];
  for (int i = 0; i < 64; i++) taken[i] = false;
  for (int k = 0; k < 6; k++) {
    int best = 0;
    float bv = -INFINITY;
    for (int i = 0; i < 64; i++) {
      if (!taken[i] && score[i] > bv) {
        bv = score[i];
        best = i;
      }
    }
    taken[best] = true;
    ids[row * 6 + k] = best;
    wts[row * 6 + k] = bv;
  }
}

// Gathered rows: y[i] = x[rows[i]] (1280 wide). Grid (count, 5).
extern "C" __global__ void __launch_bounds__(256) tf_uocr_gather(const bf16* __restrict__ x, const unsigned* __restrict__ rows,
                                                                 bf16* __restrict__ y) {
  const int c = blockIdx.y * 256 + threadIdx.x;
  if (c >= 1280) return;
  y[(size_t)blockIdx.x * 1280 + c] = x[(size_t)rows[blockIdx.x] * 1280 + c];
}

// An expert's rows back to their slots: slots[dest[i]] = y[i]. Grid (count, 5).
extern "C" __global__ void __launch_bounds__(256) tf_uocr_scatter(const bf16* __restrict__ y, const unsigned* __restrict__ dest,
                                                                  bf16* __restrict__ slots) {
  const int c = blockIdx.y * 256 + threadIdx.x;
  if (c >= 1280) return;
  slots[(size_t)dest[blockIdx.x] * 1280 + c] = y[(size_t)blockIdx.x * 1280 + c];
}

// The routed sum and the shared expert: y = bf16(bf16(sum_s w_s * slot_s in fp32) + shared), slots in gate order.
// Grid (rows, 5).
extern "C" __global__ void __launch_bounds__(256) tf_uocr_moe_combine(const bf16* __restrict__ slots, const float* __restrict__ wts,
                                                                      const bf16* __restrict__ shared, bf16* __restrict__ y) {
  const int c = blockIdx.y * 256 + threadIdx.x, row = blockIdx.x;
  if (c >= 1280) return;
  float s = 0.0f;
  for (int k = 0; k < 6; k++) s += ld(slots + ((size_t)row * 6 + k) * 1280 + c) * wts[row * 6 + k];
  const size_t i = (size_t)row * 1280 + c;
  y[i] = to(bf(s) + ld(shared + i));
}

// One token's projection y[n] = bf16(W[n, :] . x) with fp32 sums: 4 rows a warp, 8 warps a block; k a multiple of 8.
extern "C" __global__ void __launch_bounds__(256) tf_uocr_matvec(const bf16* __restrict__ x, const bf16* __restrict__ W,
                                                                 bf16* __restrict__ y, int n, int k) {
  const int sg = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row0 = (blockIdx.x * 8 + sg) * 4;
  if (row0 >= n) return;
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int i = lane * 8; i < k; i += 256) {
    float xv[8];
    load8(x + i, xv);
#pragma unroll
    for (int r = 0; r < 4; r++) {
      if (row0 + r >= n) break;
      float wv[8];
      load8(W + (size_t)(row0 + r) * k + i, wv);
      float d0 = 0.0f, d1 = 0.0f;
#pragma unroll
      for (int j = 0; j < 4; j++) d0 += xv[j] * wv[j];
#pragma unroll
      for (int j = 4; j < 8; j++) d1 += xv[j] * wv[j];
      acc[r] += d0 + d1;
    }
  }
#pragma unroll
  for (int r = 0; r < 4; r++) {
    const float s = warp_sum(acc[r]);
    if (lane == 0 && row0 + r < n) y[row0 + r] = to(s);
  }
}

// An expert's three projections, [F, 1280], [F, 1280] and [1280, F] as stored.
struct UdExpert {
  const bf16* gate;
  const bf16* up;
  const bf16* down;
};

// One token's routed experts, first half: for slot s (grid y) and its expert ids[s],
// act[s][j] = bf16(bf16(silu(bf16(gate_j . x))) * bf16(up_j . x)), 4 rows a warp.
extern "C" __global__ void __launch_bounds__(256) tf_uocr_expert_gu(const bf16* __restrict__ x, const UdExpert* __restrict__ tab,
                                                                    const unsigned* __restrict__ ids, bf16* __restrict__ act,
                                                                    int ffn) {
  const int s = blockIdx.y, sg = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const UdExpert e = tab[ids[s]];
  const int row0 = (blockIdx.x * 8 + sg) * 4;
  if (row0 >= ffn) return;
  float g[4] = {0.0f, 0.0f, 0.0f, 0.0f}, u[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int i = lane * 8; i < 1280; i += 256) {
    float xv[8];
    load8(x + i, xv);
#pragma unroll
    for (int r = 0; r < 4; r++) {
      float wg[8], wu[8];
      load8(e.gate + (size_t)(row0 + r) * 1280 + i, wg);
      load8(e.up + (size_t)(row0 + r) * 1280 + i, wu);
      float g0 = 0.0f, g1 = 0.0f, u0 = 0.0f, u1 = 0.0f;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        g0 += xv[j] * wg[j];
        u0 += xv[j] * wu[j];
      }
#pragma unroll
      for (int j = 4; j < 8; j++) {
        g1 += xv[j] * wg[j];
        u1 += xv[j] * wu[j];
      }
      g[r] += g0 + g1;
      u[r] += u0 + u1;
    }
  }
#pragma unroll
  for (int r = 0; r < 4; r++) {
    const float gs = bf(warp_sum(g[r])), us = bf(warp_sum(u[r]));
    if (lane == 0) act[(size_t)s * ffn + row0 + r] = to(bf(gs / (1.0f + expf(-gs))) * us);
  }
}

// The second half: slots[s][i] = bf16(down_i . act[s]) for slot s's expert, 4 rows a warp.
extern "C" __global__ void __launch_bounds__(256) tf_uocr_expert_down(const bf16* __restrict__ act, const UdExpert* __restrict__ tab,
                                                                      const unsigned* __restrict__ ids, bf16* __restrict__ slots,
                                                                      int ffn) {
  const int s = blockIdx.y, sg = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const UdExpert e = tab[ids[s]];
  const int row0 = (blockIdx.x * 8 + sg) * 4;
  if (row0 >= 1280) return;
  const bf16* a = act + (size_t)s * ffn;
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int i = lane * 8; i < ffn; i += 256) {
    float av[8];
    load8(a + i, av);
#pragma unroll
    for (int r = 0; r < 4; r++) {
      float wv[8];
      load8(e.down + (size_t)(row0 + r) * ffn + i, wv);
      float d0 = 0.0f, d1 = 0.0f;
#pragma unroll
      for (int j = 0; j < 4; j++) d0 += av[j] * wv[j];
#pragma unroll
      for (int j = 4; j < 8; j++) d1 += av[j] * wv[j];
      acc[r] += d0 + d1;
    }
  }
#pragma unroll
  for (int r = 0; r < 4; r++) {
    const float v = warp_sum(acc[r]);
    if (lane == 0) slots[(size_t)s * 1280 + row0 + r] = to(v);
  }
}

// Greedy pick under the page guard on device: the first maximum of bf16 logits, EOS (id 1) skipped when `bar_eos`.
// One block of 1,024; out[0] = the id.
extern "C" __global__ void __launch_bounds__(1024) tf_uocr_argmax(const bf16* __restrict__ logits, int n, int bar_eos,
                                                                  unsigned* __restrict__ out) {
  __shared__ float bv[32];
  __shared__ int bi[32];
  float best = -INFINITY;
  int at = 0x7fffffff;
  for (int i = threadIdx.x; i < n; i += 1024) {
    if (bar_eos && i == 1) continue;
    const float v = ld(logits + i);
    if (v > best) {
      best = v;
      at = i;
    }
  }
  // the warp's best, ties to the lower id
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, best, o);
    const int oi = __shfl_xor_sync(0xffffffffu, at, o);
    if (ov > best || (ov == best && oi < at)) {
      best = ov;
      at = oi;
    }
  }
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  if (lane == 0) {
    bv[w] = best;
    bi[w] = at;
  }
  __syncthreads();
  if (w != 0) return;
  best = bv[lane];
  at = bi[lane];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, best, o);
    const int oi = __shfl_xor_sync(0xffffffffu, at, o);
    if (ov > best || (ov == best && oi < at)) {
      best = ov;
      at = oi;
    }
  }
  if (lane == 0) out[0] = (unsigned)(at == 0x7fffffff ? 0 : at);
}
