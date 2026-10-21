// latent.py's BF16 absorb, expand and cache write by hand (the MLA layers' 512-wide latent, 256-wide heads).
#pragma once

#include "glm_triton.cuh"

using namespace glm_triton;

// latent._absorb_q(_rows), block (head, 32 latent columns, row block) of 128 threads; rows per block change no bits.
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_absorb_q(const __nv_bfloat16* q, const __nv_bfloat16* wk, __nv_bfloat16* qa, int64_t rows, int64_t heads,
                    int64_t rb) {
  constexpr int D = 256, LW = 512;
  __shared__ float smem[128];
  const int t = threadIdx.x, k0 = t >> 2, nq = t & 3, warp = t >> 5;
  const int64_t h = blockIdx.x, n0 = (int64_t)blockIdx.y * 32 + nq * 8, r0 = blockIdx.z * rb;
  float w[8][8];
  for (int rep = 0; rep < 8; ++rep)
    for (int j = 0; j < 8; ++j) w[rep][j] = bf(wk[(h * D + k0 + 32 * rep) * LW + n0 + j]);
  for (int64_t r = r0; r < r0 + rb && r < rows; ++r) {
    float qv[8];
    for (int rep = 0; rep < 8; ++rep) qv[rep] = bf(q[(r * heads + h) * D + k0 + 32 * rep]);
    for (int j = 0; j < 8; ++j) {
      float a = fmul(w[1][j], qv[1]);
      a = __fmaf_rn(w[0][j], qv[0], a);
      for (int rep = 2; rep < 8; ++rep) a = __fmaf_rn(w[rep][j], qv[rep], a);
      for (int o = 16; o >= 4; o >>= 1) a = a + __shfl_xor_sync(0xffffffffu, a, o);
      if ((t & 28) == 0) smem[(nq * 8 + j) * 4 + warp] = a;
    }
    __syncthreads();
    float v = smem[t];
    v = v + __shfl_xor_sync(0xffffffffu, v, 2);
    v = v + __shfl_xor_sync(0xffffffffu, v, 1);
    if ((t & 3) == 0) qa[(r * heads + h) * LW + (int64_t)blockIdx.y * 32 + t / 4] = __float2bfloat16_rn(v);
    __syncthreads();
  }
}

// latent._expand_v(_rows), block (head, 16 outputs, row block) of 128 threads: o . wv by fma, lanes xor 16 .. 1.
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_expand_v(const __nv_bfloat16* ol, const __nv_bfloat16* wv, __nv_bfloat16* out, int64_t rows,
                    int64_t heads, int64_t rb) {
  constexpr int DV = 256, LW = 512;
  __shared__ float smem[32];
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5, kb = (warp & 1) * 256 + lane * 8, nb = warp >> 1;
  const int64_t h = blockIdx.x, n0 = (int64_t)blockIdx.y * 16, r0 = blockIdx.z * rb;
  float w[8][8];
  for (int rep = 0; rep < 8; ++rep)
    for (int j = 0; j < 8; ++j) w[rep][j] = bf(wv[(h * DV + n0 + nb + 2 * rep) * LW + kb + j]);
  for (int64_t r = r0; r < r0 + rb && r < rows; ++r) {
    float o[8];
    for (int j = 0; j < 8; ++j) o[j] = bf(ol[(r * heads + h) * LW + kb + j]);
    for (int rep = 0; rep < 8; ++rep) {
      float a = fmul(w[rep][1], o[1]);
      a = __fmaf_rn(w[rep][0], o[0], a);
      for (int j = 2; j < 8; ++j) a = __fmaf_rn(w[rep][j], o[j], a);
      for (int s = 16; s > 0; s >>= 1) a = a + __shfl_xor_sync(0xffffffffu, a, s);
      if (lane == 0) smem[(nb + 2 * rep) * 2 + (warp & 1)] = a;
    }
    __syncthreads();
    if (t < 32) {
      float v = smem[t];
      v = v + __shfl_xor_sync(0xffffffffu, v, 1);
      if ((t & 1) == 0) out[(r * heads + h) * DV + n0 + t / 2] = __float2bfloat16_rn(v);
    }
    __syncthreads();
  }
}

namespace glm_pages {

// Whether a loop reads through a stream's table, known when it compiles (each form unrolls on its own).
template <bool B>
struct Paged {
  static constexpr bool value = B;
};

// Row r's first element in a cache of `width`-element rows: in order (pt null), or row r % P of page pt[r / P].
template <int P>
__device__ __forceinline__ int64_t at(const int* pt, int64_t page_elems, int64_t r, int64_t width) {
  return pt ? (int64_t)pt[r / P] * page_elems + (r % P) * width : r * width;
}

}  // namespace glm_pages

// latent._lat_write (BF16 cache): row r of `lat` into cache slot pos + r (on pages, through the stream's table).
extern "C" __global__ void tf_glm_lat_write(const __nv_bfloat16* lat, int64_t lat_stride, __nv_bfloat16* cache,
                                            const int* pos, int64_t lw, const int* pt, int64_t ps) {
  const int64_t r = blockIdx.x, at = glm_pages::at<512>(pt, ps / 2, (int64_t)pos[0] + r, lw);
  for (int64_t k = threadIdx.x; k < lw; k += blockDim.x) cache[at + k] = lat[r * lat_stride + k];
}

#include "kv_fp8.cuh"

// One 512-wide latent row as E4M3 and its FP32 scale, with the same row-wise rounding as other FP8 caches.
extern "C" __global__ void __launch_bounds__(512)
    tf_glm_lat_write_fp8(const __nv_bfloat16* lat, int64_t stride, unsigned char* cache, float* scales, const int* pos) {
  const int64_t r = blockIdx.x, at = (int64_t)pos[0] + r;
  const int d = threadIdx.x;
  __shared__ float red[16];
  tf_kv_fp8::quant_row<512, true>(__bfloat162float(lat[r * stride + d]), d, cache + at * 512, scales + at, red);
}
