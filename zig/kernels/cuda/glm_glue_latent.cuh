// latent.py's BF16 absorb, expand and cache write by hand (512-wide latent; GLM's 256-wide heads, Ling's 128).
#pragma once

#include "glm_triton.cuh"

using namespace glm_triton;

// latent._absorb_q(_rows), block (head, 32 latent columns, row block) of 128 threads; rows per block change no bits.
template <int D>  // q_nope D wide (GLM 256, Ling 128), q rows qs apart per head, qa rows qaw apart
__device__ __forceinline__ void absorb_q(const __nv_bfloat16* q, const __nv_bfloat16* wk, __nv_bfloat16* qa,
                                         int64_t rows, int64_t heads, int64_t rb, int64_t qs, int64_t qaw) {
  constexpr int LW = 512, R = D / 32;
  __shared__ float smem[128];
  const int t = threadIdx.x, k0 = t >> 2, nq = t & 3, warp = t >> 5;
  const int64_t h = blockIdx.x, n0 = (int64_t)blockIdx.y * 32 + nq * 8, r0 = blockIdx.z * rb;
  float w[R][8];
  for (int rep = 0; rep < R; ++rep)
    for (int j = 0; j < 8; ++j) w[rep][j] = bf(wk[(h * D + k0 + 32 * rep) * LW + n0 + j]);
  for (int64_t r = r0; r < r0 + rb && r < rows; ++r) {
    float qv[R];
    for (int rep = 0; rep < R; ++rep) qv[rep] = bf(q[(r * heads + h) * qs + k0 + 32 * rep]);
    for (int j = 0; j < 8; ++j) {
      float a = fmul(w[1][j], qv[1]);
      a = __fmaf_rn(w[0][j], qv[0], a);
      for (int rep = 2; rep < R; ++rep) a = __fmaf_rn(w[rep][j], qv[rep], a);
      for (int o = 16; o >= 4; o >>= 1) a = a + __shfl_xor_sync(0xffffffffu, a, o);
      if ((t & 28) == 0) smem[(nq * 8 + j) * 4 + warp] = a;
    }
    __syncthreads();
    float v = smem[t];
    v = v + __shfl_xor_sync(0xffffffffu, v, 2);
    v = v + __shfl_xor_sync(0xffffffffu, v, 1);
    if ((t & 3) == 0) qa[(r * heads + h) * qaw + (int64_t)blockIdx.y * 32 + t / 4] = __float2bfloat16_rn(v);
    __syncthreads();
  }
}

extern "C" __global__ void __launch_bounds__(128)
    tf_glm_absorb_q(const __nv_bfloat16* q, const __nv_bfloat16* wk, __nv_bfloat16* qa, int64_t rows, int64_t heads,
                    int64_t rb) {
  absorb_q<256>(q, wk, qa, rows, heads, rb, 256, 512);
}

// Ling: q_nope the first 128 of each head's qs columns, qa rows qaw (576: the latent, then the roped q_rope).
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_absorb_q_128(const __nv_bfloat16* q, const __nv_bfloat16* wk, __nv_bfloat16* qa, int64_t rows,
                        int64_t heads, int64_t rb, int64_t qs, int64_t qaw) {
  absorb_q<128>(q, wk, qa, rows, heads, rb, qs, qaw);
}

// latent._expand_v(_rows), block (head, 16 outputs, row block) of 128 threads: o . wv by fma, lanes xor 16 .. 1.
template <int DV>
__device__ __forceinline__ void expand_v(const __nv_bfloat16* ol, const __nv_bfloat16* wv, __nv_bfloat16* out,
                                         int64_t rows, int64_t heads, int64_t rb) {
  constexpr int LW = 512;
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

extern "C" __global__ void __launch_bounds__(128)
    tf_glm_expand_v(const __nv_bfloat16* ol, const __nv_bfloat16* wv, __nv_bfloat16* out, int64_t rows,
                    int64_t heads, int64_t rb) {
  expand_v<256>(ol, wv, out, rows, heads, rb);
}

// Ling's 128-wide v heads.
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_expand_v_128(const __nv_bfloat16* ol, const __nv_bfloat16* wv, __nv_bfloat16* out, int64_t rows,
                        int64_t heads, int64_t rb) {
  expand_v<128>(ol, wv, out, rows, heads, rb);
}

// latent._lat_write (BF16 cache): row r of `lat` into cache slot pos + r.
extern "C" __global__ void tf_glm_lat_write(const __nv_bfloat16* lat, int64_t lat_stride, __nv_bfloat16* cache,
                                            const int* pos, int64_t lw) {
  const int64_t r = blockIdx.x, at = (int64_t)pos[0] + r;
  for (int64_t k = threadIdx.x; k < lw; k += blockDim.x) cache[at * lw + k] = lat[r * lat_stride + k];
}

// Ling's rope (exl3's GPT-J pairs), block (row, head) of 32: x + r xs + h xh into out + r os + h oh, in place ok.
extern "C" __global__ void __launch_bounds__(32)
    tf_glm_rope(const __nv_bfloat16* x, int64_t xs, int64_t xh, __nv_bfloat16* out, int64_t os, int64_t oh,
                const int* pos, const float* inv_freq) {
  const int64_t r = blockIdx.x, h = blockIdx.y;
  const int i = threadIdx.x;  // pair i at angle fp32 inv_freq[i] (pos + r); explicit rn ops, no fma contraction
  const __nv_bfloat16* src = x + r * xs + h * xh + 2 * i;
  const float x0 = bf(src[0]), x1 = bf(src[1]);
  float s, c;
  sincosf(__fmul_rn(inv_freq[i], (float)((int64_t)pos[0] + r)), &s, &c);
  __nv_bfloat16* dst = out + r * os + h * oh + 2 * i;
  dst[0] = __float2bfloat16_rn(__fsub_rn(__fmul_rn(x0, c), __fmul_rn(x1, s)));
  dst[1] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(x1, c), __fmul_rn(x0, s)));
}

// Ling's head-wise output gate, block (row, head) of dv threads: o *= sigmoid(g[r gs + h]) in fp32, rounded to bf16.
extern "C" __global__ void __launch_bounds__(256)
    tf_glm_head_gate(__nv_bfloat16* o, int64_t heads, const float* g, int64_t gs) {
  const int64_t r = blockIdx.x, h = blockIdx.y, dv = blockDim.x;
  const float s = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-g[r * gs + h])));
  __nv_bfloat16* p = o + (r * heads + h) * dv + threadIdx.x;
  *p = __float2bfloat16_rn(__fmul_rn(bf(*p), s));
}
