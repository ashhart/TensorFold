// Unlimited-OCR-2's glue and row kernels (uocr_vision.cuh, uocr_decoder.cuh), each copied with one planted fault.
#pragma once
#include "uocr_common.cuh"

// Gate 3: tf_uocr_add subtracting its rows.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_add_subtracts(const bf16* a, const bf16* b, bf16* y, long long n) {
  const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
  if (i < n) y[i] = to(ld(a + i) - ld(b + i));
}

// Gate 2: tf_uocr_add also writing each sum over its first input.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_add_writes_input(const bf16* a, const bf16* b, bf16* y,
                                                                               long long n) {
  const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
  if (i >= n) return;
  y[i] = to(ld(a + i) + ld(b + i));
  const_cast<bf16*>(a)[i] = y[i];
}

// Gate 3: tf_uocr_layer_norm's mean divided by the threads, not the width.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_layer_norm_threads(const bf16* x, const bf16* w, const bf16* b,
                                                                                 bf16* y, int width, float eps) {
  __shared__ float part[33];
  const bf16* xr = x + (size_t)blockIdx.x * width;
  bf16* yr = y + (size_t)blockIdx.x * width;
  float s = 0.0f;
  for (int i = threadIdx.x; i < width; i += blockDim.x) s += ld(xr + i);
  const float mean = tf_common::block_sum_tree(s, part, blockDim.x >> 5) / (float)blockDim.x;
  float v = 0.0f;
  for (int i = threadIdx.x; i < width; i += blockDim.x) {
    const float d = ld(xr + i) - mean;
    v += d * d;
  }
  const float inv = 1.0f / sqrtf(tf_common::block_sum_tree(v, part, blockDim.x >> 5) / (float)width + eps);
  for (int i = threadIdx.x; i < width; i += blockDim.x) yr[i] = to((ld(xr + i) - mean) * inv * ld(w + i) + ld(b + i));
}

// Gate 3: tf_uocr_gelu with erf's argument x times 0.75, not 1 / sqrt 2.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_gelu_scale(bf16* x, long long n) {
  const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
  if (i >= n) return;
  const float v = ld(x + i);
  x[i] = to(0.5f * v * (1.0f + erf_(v * 0.75f)));
}

// Gate 3: tf_uocr_quick_gelu without the sigmoid's bf16 rounding.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_quick_gelu_unrounded(bf16* x, long long n) {
  const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
  if (i >= n) return;
  const float v = ld(x + i);
  const float z = bf(1.702f * v);
  x[i] = to(v * (1.0f / (1.0f + expf(-z))));
}

// Gate 3: tf_uocr_concat repeating the left rows in the right half.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_concat_left_twice(const bf16* a, const bf16* b, bf16* y, int rows,
                                                                                int left, int right) {
  const int width = left + right;
  const int c = blockIdx.y * 256 + threadIdx.x, r = blockIdx.x;
  if (c >= width || r >= rows) return;
  y[(size_t)r * width + c] = c < left ? a[(size_t)r * left + c] : a[(size_t)r * left + (c - left) % left];
}

// Gate 3: tf_uocr_clip_embed adding SAM's row r - 2, not r - 1.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_clip_embed_row(const bf16* cls, const bf16* sam, const bf16* pos,
                                                                             bf16* y, int width) {
  const int c = blockIdx.y * 256 + threadIdx.x, r = blockIdx.x;
  if (c >= width) return;
  const float v = r == 0 ? ld(cls + c) : ld(sam + (size_t)(r > 1 ? r - 2 : 0) * width + c);
  y[(size_t)r * width + c] = to(v + ld(pos + (size_t)r * width + c));
}

// Gate 3: tf_uocr_embed reading the next id's row.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_embed_next(const unsigned* ids, const bf16* embed, bf16* x,
                                                                         unsigned image_token) {
  const int c = blockIdx.y * 256 + threadIdx.x;
  const unsigned id = ids[blockIdx.x];
  if (c >= 1280 || id == image_token) return;
  x[(size_t)blockIdx.x * 1280 + c] = embed[(size_t)(id + 1) * 1280 + c];
}

// Gate 3: tf_uocr_scatter writing the previous row to each slot.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_scatter_prev(const bf16* y, const unsigned* dest, bf16* slots) {
  const int c = blockIdx.y * 256 + threadIdx.x;
  if (c >= 1280) return;
  slots[(size_t)dest[blockIdx.x] * 1280 + c] = y[(size_t)(blockIdx.x > 0 ? blockIdx.x - 1 : 0) * 1280 + c];
}

// Gate 3: tf_uocr_argmax breaking ties to the higher id.
extern "C" __global__ void __launch_bounds__(1024) tf_ctl_uocr_argmax_high(const bf16* logits, int n, int bar_eos,
                                                                           unsigned* out) {
  __shared__ float bv[32];
  __shared__ int bi[32];
  float best = -INFINITY;
  int at = -1;
  for (int i = threadIdx.x; i < n; i += 1024) {
    if (bar_eos && i == 1) continue;
    const float v = ld(logits + i);
    if (v >= best) {
      best = v;
      at = i;
    }
  }
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, best, o);
    const int oi = __shfl_xor_sync(0xffffffffu, at, o);
    if (ov > best || (ov == best && oi > at)) {
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
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, best, o);
    const int oi = __shfl_xor_sync(0xffffffffu, at, o);
    if (ov > best || (ov == best && oi > at)) {
      best = ov;
      at = oi;
    }
  }
  if (lane == 0) out[0] = (unsigned)(at < 0 ? 0 : at);
}
