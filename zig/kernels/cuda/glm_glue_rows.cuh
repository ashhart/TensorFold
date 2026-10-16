// GLM's element-wise glue by hand: each output value alone, in the Triton kernel's order of fp32 ops.
#pragma once

#include "glm_triton.cuh"

namespace glm_glue {
using namespace glm_triton;

// Column c of a 2-D grid's row blockIdx.x, 256 threads a block along y.
__device__ __forceinline__ int64_t col() { return (int64_t)blockIdx.y * blockDim.x + threadIdx.x; }

// The first of a thread's 8 contiguous columns: Triton's BLOCK 1024 over 4 warps, 128 threads a block along y.
__device__ __forceinline__ int64_t col8() { return ((int64_t)blockIdx.y * blockDim.x + threadIdx.x) * 8; }

// 8 contiguous values in one 16-byte load (bf16) or two (fp32), and 8 bf16 out in one 16-byte store.
__device__ __forceinline__ void ld8(const __nv_bfloat16* p, float v[8]) {
  const uint4 u = *reinterpret_cast<const uint4*>(p);
  const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&u);
  for (int j = 0; j < 8; ++j) v[j] = bf(h[j]);
}
__device__ __forceinline__ void ld8(const float* p, float v[8]) {
  const float4 a = *reinterpret_cast<const float4*>(p), b = *reinterpret_cast<const float4*>(p + 4);
  v[0] = a.x, v[1] = a.y, v[2] = a.z, v[3] = a.w, v[4] = b.x, v[5] = b.y, v[6] = b.z, v[7] = b.w;
}
__device__ __forceinline__ void st8(__nv_bfloat16* p, const float v[8]) {
  uint4 u;
  __nv_bfloat16* h = reinterpret_cast<__nv_bfloat16*>(&u);
  for (int j = 0; j < 8; ++j) h[j] = __float2bfloat16_rn(v[j]);
  *reinterpret_cast<uint4*>(p) = u;
}

// The ranks' fp32 partials at i .. i + 7, rank 0 first and `rs` apart.
__device__ __forceinline__ void ranks8(const float* g, int64_t rs, int64_t i, int world, float a[8]) {
  ld8(g + i, a);
  for (int k = 1; k < world; ++k) {
    float b[8];
    ld8(g + k * rs + i, b);
    for (int j = 0; j < 8; ++j) a[j] = a[j] + b[j];
  }
}

}  // namespace glm_glue

using namespace glm_triton;

// glue._residual_add: xout = bf16(x + bf16(the partials summed rank 0 first)), 8 columns a thread.
extern "C" __global__ void tf_glm_residual_add(const __nv_bfloat16* x, __nv_bfloat16* xout, const float* g, int64_t rs,
                                               int64_t d, int world) {
  const int64_t c = glm_glue::col8();
  if (c >= d) return;
  const int64_t i = blockIdx.x * d + c;
  float a[8], v[8];
  glm_glue::ranks8(g, rs, i, world, a);
  glm_glue::ld8(x + i, v);
  for (int j = 0; j < 8; ++j) v[j] = v[j] + rbf(a[j]);
  glm_glue::st8(xout + i, v);
}

// glue._stream_mean: bf16 of the s streams' sum over s (div.full, as Triton divides), 8 columns a thread.
extern "C" __global__ void tf_glm_stream_mean(const __nv_bfloat16* x, __nv_bfloat16* out, int64_t d, int s) {
  const int64_t c = glm_glue::col8();
  if (c >= d) return;
  const __nv_bfloat16* row = x + blockIdx.x * s * d + c;
  float a[8];
  glm_glue::ld8(row, a);
  for (int k = 1; k < s; ++k) {
    float b[8];
    glm_glue::ld8(row + k * d, b);
    for (int j = 0; j < 8; ++j) a[j] = a[j] + b[j];
  }
  for (int j = 0; j < 8; ++j) a[j] = div_full(a[j], (float)s);
  glm_glue::st8(out + blockIdx.x * d + c, a);
}

// glue._rank_sum: bf16 of the partials summed rank 0 first, `me` from `own`, others from `peers`, 8 columns a thread.
extern "C" __global__ void tf_glm_rank_sum(const float* own, const float* peers, __nv_bfloat16* out, int64_t rs,
                                           int64_t d, int world, int me) {
  const int64_t c = glm_glue::col8();
  if (c >= d) return;
  const int64_t i = blockIdx.x * d + c;
  float a[8];
  glm_glue::ld8(me == 0 ? own + i : peers + i, a);
  for (int k = 1; k < world; ++k) {
    float b[8];
    glm_glue::ld8(k == me ? own + i : peers + (k > me ? k - 1 : k) * rs + i, b);
    for (int j = 0; j < 8; ++j) a[j] = a[j] + b[j];
  }
  glm_glue::st8(out + i, a);
}

// glue._hc_post: stream s = bf16(post[s] branch + sum_t comb[t][s] x[t]) in LLVM's fma order; x may be xout.
extern "C" __global__ void tf_glm_hc_post(const __nv_bfloat16* x, __nv_bfloat16* xout, const float* g,
                                          const float* post, const float* comb, int64_t rs, int64_t d, int world) {
  const int64_t c = glm_glue::col8();
  if (c >= d) return;
  const int64_t r = blockIdx.x;
  float branch[8];
  glm_glue::ranks8(g, rs, r * d + c, world, branch);
  float xv[4][8];
  for (int t = 0; t < 4; ++t) glm_glue::ld8(x + r * 4 * d + t * d + c, xv[t]);
  for (int s = 0; s < 4; ++s) {
    const float* cm = comb + r * 16 + s;
    const float c0 = cm[0], c1 = cm[4], c2 = cm[8], c3 = cm[12], p = post[r * 4 + s];
    float o[8];
    for (int j = 0; j < 8; ++j) {
      float m = __fmul_rn(xv[1][j], c1);
      m = __fmaf_rn(xv[0][j], c0, m);
      m = __fmaf_rn(xv[2][j], c2, m);
      m = __fmaf_rn(xv[3][j], c3, m);
      o[j] = __fmaf_rn(rbf(branch[j]), p, m);
    }
    glm_glue::st8(xout + r * 4 * d + s * d + c, o);
  }
}

// glue._combine: fp32 sum over slots in order of y[r][k] wts[r][k] (fma from 0), y fp32 or bf16, 8 columns a thread.
extern "C" __global__ void tf_glm_combine(
        const void* y, int y_f32, const float* wts, float* out, int64_t d, int slots) {
  const int64_t c = glm_glue::col8();
  if (c >= d) return;
  const int64_t r = blockIdx.x;
  float a[8] = {};
  for (int k = 0; k < slots; ++k) {
    const int64_t i = (r * slots + k) * d + c;
    float v[8];
    if (y_f32)
      glm_glue::ld8(static_cast<const float*>(y) + i, v);
    else
      glm_glue::ld8(static_cast<const __nv_bfloat16*>(y) + i, v);
    const float w = wts[r * slots + k];
    for (int j = 0; j < 8; ++j) a[j] = __fmaf_rn(v[j], w, a[j]);
  }
  float* o = out + r * d + c;
  *reinterpret_cast<float4*>(o) = make_float4(a[0], a[1], a[2], a[3]);
  *reinterpret_cast<float4*>(o + 4) = make_float4(a[4], a[5], a[6], a[7]);
}

// glue._router_sum and qmm._reduce: ks slices of `total` fp32 added in order (fp32 or bf16 out), 8 values a thread.
extern "C" __global__ void tf_glm_slice_sum(const float* part, void* out, int64_t total, int ks, int f32) {
  const int64_t i0 = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (i0 >= total) return;
  const bool aligned = ((reinterpret_cast<uintptr_t>(out) | reinterpret_cast<uintptr_t>(part)) & 15) == 0;
  if (i0 + 8 <= total && (total & 3) == 0 && aligned) {
    float a[8];
    glm_glue::ld8(part + i0, a);
    for (int s = 1; s < ks; ++s) {
      float b[8];
      glm_glue::ld8(part + s * total + i0, b);
      for (int j = 0; j < 8; ++j) a[j] = a[j] + b[j];
    }
    if (f32) {
      float* o = static_cast<float*>(out) + i0;
      *reinterpret_cast<float4*>(o) = make_float4(a[0], a[1], a[2], a[3]);
      *reinterpret_cast<float4*>(o + 4) = make_float4(a[4], a[5], a[6], a[7]);
    } else {
      glm_glue::st8(static_cast<__nv_bfloat16*>(out) + i0, a);
    }
    return;
  }
  for (int64_t i = i0; i < i0 + 8 && i < total; ++i) {
    float a = part[i];
    for (int s = 1; s < ks; ++s) a = a + part[s * total + i];
    if (f32)
      static_cast<float*>(out)[i] = a;
    else
      static_cast<__nv_bfloat16*>(out)[i] = __float2bfloat16_rn(a);
  }
}

// glue._embed_b16: token ids[r]'s bf16 row into `copies` streams of d.
extern "C" __global__ void tf_glm_embed_b16(const int* ids, const __nv_bfloat16* w, __nv_bfloat16* out, int64_t d,
                                            int copies) {
  const int64_t c = glm_glue::col();
  if (c >= d) return;
  const int64_t r = blockIdx.x;
  const __nv_bfloat16 v = w[(int64_t)ids[r] * d + c];
  for (int k = 0; k < copies; ++k) out[(r * copies + k) * d + c] = v;
}

// forward._conv_shift: layer blockIdx.x's 3-row window takes rows keep .. keep + 2 of [window; projection rows].
extern "C" __global__ void tf_glm_conv_shift(__nv_bfloat16* conv, const __nv_bfloat16* proj, int64_t keep,
                                             int64_t conv_layer, int64_t proj_layer, int64_t proj_row, int64_t c) {
  const int64_t k = glm_glue::col();
  if (k >= c) return;
  const int64_t l = blockIdx.x;
  __nv_bfloat16 v[3];
  for (int j = 0; j < 3; ++j) {
    const int64_t src = keep + j;
    v[j] = src < 3 ? conv[l * conv_layer + src * c + k] : proj[l * proj_layer + (src - 3) * proj_row + k];
  }
  for (int j = 0; j < 3; ++j) conv[l * conv_layer + j * c + k] = v[j];
}

// decode.sample_rows' greedy pick, as the host scan: first max of n bf16 logits, NaN never wins; out fp32 bits, lo+id.
extern "C" __global__ void __launch_bounds__(1024) tf_glm_top1(const unsigned short* logits, int64_t n, unsigned lo,
                                                                unsigned* out) {
  __shared__ float sv[1024];
  __shared__ unsigned si[1024];
  const int t = threadIdx.x;
  float top = -__int_as_float(0x7f800000);
  unsigned best = 0xffffffffu;
  for (int64_t i = t; i < n; i += 1024) {
    const float v = __int_as_float(static_cast<int>(static_cast<unsigned>(logits[i]) << 16));
    if (v > top) top = v, best = static_cast<unsigned>(i);
  }
  sv[t] = top, si[t] = best;
  __syncthreads();
  for (int h = 512; h > 0; h >>= 1) {
    if (t < h) {
      const float a = sv[t], b = sv[t + h];
      const unsigned ia = si[t], ib = si[t + h];
      if (b > a || (b == a && ib < ia)) sv[t] = b, si[t] = ib;
    }
    __syncthreads();
  }
  if (t == 0) {
    out[0] = __float_as_uint(sv[0]);
    out[1] = lo + (si[0] == 0xffffffffu ? 0u : si[0]);
  }
}

// The leader's step for the other ranks, written to the link's buffer from launch arguments (cuda_follow's Link.send).
struct tf_glm_words {
  unsigned w[96];
};
extern "C" __global__ void tf_glm_put_words(unsigned* dst, unsigned n, tf_glm_words v) {
  for (unsigned i = threadIdx.x; i < n; i += blockDim.x) dst[i] = v.w[i];
}
