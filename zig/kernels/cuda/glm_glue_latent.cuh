// latent.py's BF16 absorb, expand and cache write by hand (the MLA layers' 512-wide latent, 256-wide heads).
#pragma once

#include "glm_triton.cuh"

using namespace glm_triton;

namespace glm_latent_rows {

// q rows r .. r + R - 1 of head h (zero past end) into shared memory, 16 bytes a thread.
template <int R, int W>
__device__ __forceinline__ void stage(__nv_bfloat16 (*dst)[W], const __nv_bfloat16* src, int64_t r, int64_t end,
                                      int64_t heads, int64_t h) {
  for (int u = threadIdx.x; u < R * W / 8; u += 128) {
    const int i = u / (W / 8), c = u - i * (W / 8);
    uint4 v = make_uint4(0, 0, 0, 0);
    if (r + i < end) v = *reinterpret_cast<const uint4*>(src + ((r + i) * heads + h) * W + c * 8);
    reinterpret_cast<uint4*>(dst[i])[c] = v;
  }
}

// latent._absorb_q's sums, R rows a pass: thread (n, g) forms partials k0 = 8g .. 8g + 7 itself (8-fma chains).
template <int R>
__device__ __forceinline__ void absorb(const __nv_bfloat16* q, const float (&w)[8][8], __nv_bfloat16* qa,
                                       __nv_bfloat16 (*qs)[256], int64_t r0, int64_t end, int64_t heads, int64_t h,
                                       int64_t col) {
  constexpr int LW = 512;
  const int g = threadIdx.x & 3;
  for (int64_t r = r0; r < end; r += R) {
    __syncthreads();
    stage<R, 256>(qs, q, r, end, heads, h);
    __syncthreads();
#pragma unroll
    for (int ri = 0; ri < R; ++ri) {
      float a[8];
#pragma unroll
      for (int rep = 0; rep < 8; ++rep) {
        const uint4 u = reinterpret_cast<const uint4*>(qs[ri])[g + 4 * rep];
        const __nv_bfloat16* hq = reinterpret_cast<const __nv_bfloat16*>(&u);
#pragma unroll
        for (int i = 0; i < 8; ++i) {  // main's chain: rep 1's product, rep 0's fma, then reps 2 .. 7
          const float qv = bf(hq[i]);
          if (rep == 0) a[i] = qv;
          else if (rep == 1) a[i] = __fmaf_rn(w[i][0], a[i], fmul(w[i][1], qv));
          else a[i] = __fmaf_rn(w[i][rep], qv, a[i]);
        }
      }
      float v = ((a[0] + a[4]) + (a[2] + a[6])) + ((a[1] + a[5]) + (a[3] + a[7]));  // main's lane tree xor 16, 8, 4
      v = v + __shfl_xor_sync(0xffffffffu, v, 2);  // main's warp sum ((W0 + W2) + (W1 + W3))
      v = v + __shfl_xor_sync(0xffffffffu, v, 1);
      if (g == 0 && r + ri < end) qa[((r + ri) * heads + h) * LW + col] = __float2bfloat16_rn(v);
    }
  }
}

// latent._expand_v's sums, R rows a pass: thread (c, half, n) forms lane partials 4m + c (8-fma chains) itself.
template <int R>
__device__ __forceinline__ void expand(const __nv_bfloat16* ol, const float (&w)[8][8], __nv_bfloat16* out,
                                       __nv_bfloat16 (*os)[512], int64_t r0, int64_t end, int64_t heads, int64_t h,
                                       int64_t n) {
  constexpr int DV = 256;
  const int t = threadIdx.x, c = t & 3, half = (t >> 2) & 1;
  for (int64_t r = r0; r < end; r += R) {
    __syncthreads();
    stage<R, 512>(os, ol, r, end, heads, h);
    __syncthreads();
#pragma unroll
    for (int ri = 0; ri < R; ++ri) {
      float p[8];
#pragma unroll
      for (int m = 0; m < 8; ++m) {
        const uint4 u = reinterpret_cast<const uint4*>(os[ri])[half * 32 + 4 * m + c];
        const __nv_bfloat16* ho = reinterpret_cast<const __nv_bfloat16*>(&u);
        float a = fmul(w[m][1], bf(ho[1]));
        a = __fmaf_rn(w[m][0], bf(ho[0]), a);
#pragma unroll
        for (int j = 2; j < 8; ++j) a = __fmaf_rn(w[m][j], bf(ho[j]), a);
        p[m] = a;
      }
      float v = ((p[0] + p[4]) + (p[2] + p[6])) + ((p[1] + p[5]) + (p[3] + p[7]));  // main's xor 16, 8, 4
      v = v + __shfl_xor_sync(0xffffffffu, v, 2);  // main's xor 2, 1
      v = v + __shfl_xor_sync(0xffffffffu, v, 1);
      v = v + __shfl_xor_sync(0xffffffffu, v, 4);  // main's half 0 + half 1
      if ((t & 7) == 0 && r + ri < end) out[((r + ri) * heads + h) * DV + n] = __float2bfloat16_rn(v);
    }
  }
}

// main's absorb loop, for decode windows: thread (k0, nq) holds 8 columns' weights for 8 inputs, 16-byte loads.
__device__ __forceinline__ void absorb_main(const __nv_bfloat16* q, const __nv_bfloat16* wk, __nv_bfloat16* qa,
                                            int64_t rows, int64_t heads, int64_t rb) {
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

// main's expand loop, for decode windows: thread (lane, warp) holds 8 inputs' weights of 8 outputs, lanes xor 16 .. 1.
__device__ __forceinline__ void expand_main(const __nv_bfloat16* ol, const __nv_bfloat16* wv, __nv_bfloat16* out,
                                            int64_t rows, int64_t heads, int64_t rb) {
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

}  // namespace glm_latent_rows

// latent._absorb_q(_rows), block (head, 32 latent columns, row block) of 128 threads; rows per block change no bits.
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_absorb_q(const __nv_bfloat16* q, const __nv_bfloat16* wk, __nv_bfloat16* qa, int64_t rows, int64_t heads,
                    int64_t rb) {
  constexpr int D = 256, LW = 512;
  __shared__ __align__(16) __nv_bfloat16 qs[4][D];
  const int t = threadIdx.x, g = t & 3;
  const int64_t h = blockIdx.x, col = (int64_t)blockIdx.y * 32 + (t >> 2), r0 = blockIdx.z * rb;
  const int64_t end = min(r0 + rb, rows);
  if (rb < 64) return glm_latent_rows::absorb_main(q, wk, qa, rows, heads, rb);  // decode: weights from DRAM
  float w[8][8];  // [i][rep]: input 8g + i + 32 rep of output col
  for (int i = 0; i < 8; ++i)
    for (int rep = 0; rep < 8; ++rep) w[i][rep] = bf(wk[(h * D + 8 * g + i + 32 * rep) * LW + col]);
  glm_latent_rows::absorb<4>(q, w, qa, qs, r0, end, heads, h, col);
}

// latent._expand_v(_rows), block (head, 16 outputs, row block) of 128 threads: o . wv by fma, main's lane tree.
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_expand_v(const __nv_bfloat16* ol, const __nv_bfloat16* wv, __nv_bfloat16* out, int64_t rows,
                    int64_t heads, int64_t rb) {
  constexpr int DV = 256, LW = 512;
  __shared__ __align__(16) __nv_bfloat16 os[4][LW];
  const int t = threadIdx.x, c = t & 3, half = (t >> 2) & 1;
  const int64_t h = blockIdx.x, n = (int64_t)blockIdx.y * 16 + (t >> 3), r0 = blockIdx.z * rb;
  const int64_t end = min(r0 + rb, rows);
  if (rb < 64) return glm_latent_rows::expand_main(ol, wv, out, rows, heads, rb);  // decode: weights from DRAM
  float w[8][8];  // [m][j]: input half * 256 + (4m + c) * 8 + j of output n
  for (int m = 0; m < 8; ++m) {
    const uint4 u = __ldg(reinterpret_cast<const uint4*>(wv + (h * DV + n) * LW + half * 256 + (4 * m + c) * 8));
    const __nv_bfloat16* hw = reinterpret_cast<const __nv_bfloat16*>(&u);
    for (int j = 0; j < 8; ++j) w[m][j] = bf(hw[j]);
  }
  glm_latent_rows::expand<4>(ol, w, out, os, r0, end, heads, h, n);
}

namespace glm_pages {

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
