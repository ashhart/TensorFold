// glue._hc_partial and glue._hc_finish by hand (hyper-connections' pre side, 4 streams of 4096).
#pragma once

#include "glm_triton.cuh"

using namespace glm_triton;

// glue._hc_partial, block (4 rows, K block b) of 128 threads: a row's 24 dots and squares in Triton's fma order.
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_hc_partial(
            const __nv_bfloat16* x, const __nv_bfloat16* fn, float* part, int64_t wide, int nb, int64_t rows) {
  constexpr int R = 4;
  __shared__ float smem[R][4];
  const int64_t r0 = (int64_t)blockIdx.x * R, b = blockIdx.y, kb = wide / nb;
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5, c = t & 15, m0 = (t >> 4) & 7;
  float acc[R][3] = {}, ss[R] = {};
  for (int64_t step = 0; step < kb / 128; ++step) {
    const int64_t base = b * kb + step * 128;
    float xv[R][8];
    for (int q = 0; q < R; ++q) {  // the 8 inputs in one 16-byte load (wide and kb are multiples of 128: aligned)
      uint4 u = make_uint4(0, 0, 0, 0);
      if (r0 + q < rows) u = *reinterpret_cast<const uint4*>(x + (r0 + q) * wide + base + c * 8);
      const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&u);
      for (int j = 0; j < 8; ++j) xv[q][j] = bf(h[j]);
    }
    for (int i = 0; i < 3; ++i) {
      const uint4 u = __ldg(reinterpret_cast<const uint4*>(fn + (int64_t)(m0 + 8 * i) * wide + base + c * 8));
      const __nv_bfloat16* hw = reinterpret_cast<const __nv_bfloat16*>(&u);
      float w[8];
      for (int j = 0; j < 8; ++j) w[j] = bf(hw[j]);
      for (int q = 0; q < R; ++q) {
        float d = __fmul_rn(xv[q][1], w[1]);
        d = __fmaf_rn(xv[q][0], w[0], d);
        for (int j = 2; j < 8; ++j) d = __fmaf_rn(xv[q][j], w[j], d);
        for (int o = 8; o > 0; o >>= 1) d = d + __shfl_xor_sync(0xffffffffu, d, o);
        acc[q][i] = acc[q][i] + d;
      }
    }
    for (int q = 0; q < R; ++q) {
      const float xt = r0 + q < rows ? bf(x[(r0 + q) * wide + base + t]) : 0.0f;
      ss[q] = __fmaf_rn(xt, xt, ss[q]);
    }
  }
  for (int q = 0; q < R; ++q) {
    for (int o = 16; o > 0; o >>= 1) ss[q] += __shfl_xor_sync(0xffffffffu, ss[q], o);
    if (lane == 0) smem[q][warp] = ss[q];
  }
  __syncthreads();
  for (int q = 0; q < R; ++q) {
    if (r0 + q >= rows) break;
    float* out = part + ((r0 + q) * nb + b) * 32;
    if (c == 0)
      for (int i = 0; i < 3; ++i) out[m0 + 8 * i] = acc[q][i];
    if (t == 0) out[24] = (smem[q][0] + smem[q][2]) + (smem[q][1] + smem[q][3]);
  }
}

// The same for decode windows (16 blocks a row): 8 steps of 128 at once, a 128-thread group each; same ops, same order.
extern "C" __global__ void __launch_bounds__(1024)
    tf_glm_hc_partial8(const __nv_bfloat16* x, const __nv_bfloat16* fn, float* part, int64_t wide, int nb) {
  __shared__ float dots[8][24];
  __shared__ float smem[4];
  const int64_t r = blockIdx.x, b = blockIdx.y, kb = wide / nb;
  const int grp = threadIdx.x >> 7, t = threadIdx.x & 127, c = t & 15, m0 = (t >> 4) & 7;
  const int steps = static_cast<int>(kb / 128);
  if (grp < steps) {
    const int64_t base = b * kb + grp * 128;
    float xv[8];
    {
      const uint4 u = *reinterpret_cast<const uint4*>(x + r * wide + base + c * 8);
      const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&u);
      for (int j = 0; j < 8; ++j) xv[j] = bf(h[j]);
    }
    for (int i = 0; i < 3; ++i) {
      const uint4 u = __ldg(reinterpret_cast<const uint4*>(fn + (int64_t)(m0 + 8 * i) * wide + base + c * 8));
      const __nv_bfloat16* w = reinterpret_cast<const __nv_bfloat16*>(&u);
      float d = __fmul_rn(xv[1], bf(w[1]));
      d = __fmaf_rn(xv[0], bf(w[0]), d);
      for (int j = 2; j < 8; ++j) d = __fmaf_rn(xv[j], bf(w[j]), d);
      for (int o = 8; o > 0; o >>= 1) d = d + __shfl_xor_sync(0xffffffffu, d, o);
      if (c == 0) dots[grp][m0 + 8 * i] = d;
    }
  }
  if (grp == 0) {
    float ss = 0.0f;
    for (int s = 0; s < steps; ++s) {
      const float xt = bf(x[r * wide + b * kb + s * 128 + t]);
      ss = __fmaf_rn(xt, xt, ss);
    }
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((t & 31) == 0) smem[t >> 5] = ss;
  }
  __syncthreads();
  float* out = part + (r * nb + b) * 32;
  if (threadIdx.x < 24) {
    float acc = 0.0f;
    for (int s = 0; s < steps; ++s) acc = acc + dots[s][threadIdx.x];
    out[threadIdx.x] = acc;
  } else if (threadIdx.x == 32) {
    out[24] = (smem[0] + smem[2]) + (smem[1] + smem[3]);
  }
}

namespace glm_hc {

// The Sinkhorn sums of 4: lanes (i, j) = (lane / 4 % 4, lane % 4), rows over j (xor 2, 1), columns over i (xor 8, 4).
__device__ __forceinline__ float row4(float v) {
  v = v + __shfl_xor_sync(0xffffffffu, v, 2);
  return v + __shfl_xor_sync(0xffffffffu, v, 1);
}
__device__ __forceinline__ float col4(float v) {
  v = v + __shfl_xor_sync(0xffffffffu, v, 8);
  return v + __shfl_xor_sync(0xffffffffu, v, 4);
}
__device__ __forceinline__ float sigmoid(float v) { return div_full(1.0f, exp_approx(0.0f - v) + 1.0f); }

}  // namespace glm_hc

// glue._hc_finish, 256 threads a row (D 4096, 4 streams): warp 0 sums K blocks and Sinkhorn, then collapse, RMS-norm.
extern "C" __global__ void __launch_bounds__(256)
    tf_glm_hc_finish(const __nv_bfloat16* x, const float* part, const float* base, const float* scale,
                     const __nv_bfloat16* nw, __nv_bfloat16* out, float* xs, float* post, float* comb, float eps,
                     float hc_eps, int nb, int iters) {
  constexpr int D = 4096, S = 4;
  __shared__ float smem[8];
  __shared__ float pre[4];
  const int64_t r = blockIdx.x;
  const int t = threadIdx.x, lane = t & 31;
  // A thread's 8 inputs of each half are contiguous: one 16-byte load per stream and half, issued before Sinkhorn.
  const __nv_bfloat16* row = x + r * S * D;
  uint4 xu[2][S], wu[2];
  for (int rep = 0; rep < 2; ++rep) {
    for (int st = 0; st < S; ++st) xu[rep][st] = *reinterpret_cast<const uint4*>(row + st * D + rep * 2048 + t * 8);
    wu[rep] = __ldg(reinterpret_cast<const uint4*>(nw + rep * 2048 + t * 8));
  }
  if (t < 32) {
    const float* p = part + r * nb * 32;
    float mix = 0.0f, ss = 0.0f;
    for (int b = 0; b < nb; ++b) {
      mix = mix + p[b * 32 + lane];
      ss = ss + p[b * 32 + 24];
    }
    const float rinv = div_full(1.0f, sqrt_approx(eps + div_full(ss, (float)(S * D))));
    mix = __fmul_rn(mix, rinv);
    const int k = lane & 3, idx = lane & 15;
    const float bpre = base[k], bpost = base[4 + k], bcomb = base[8 + idx];
    const float mpre = __shfl_sync(0xffffffffu, mix, k), mpost = __shfl_sync(0xffffffffu, mix, 4 + k);
    const float mcomb = __shfl_sync(0xffffffffu, mix, 8 + idx);
    const float pr = hc_eps + glm_hc::sigmoid(__fmaf_rn(mpre, scale[0], bpre));
    const float sp = glm_hc::sigmoid(__fmaf_rn(mpost, scale[1], bpost));
    const float cl = __fmaf_rn(mcomb, scale[2], bcomb);
    float cmax = fmaxf(cl, __shfl_xor_sync(0xffffffffu, cl, 2));
    cmax = fmaxf(cmax, __shfl_xor_sync(0xffffffffu, cmax, 1));
    const float ce = exp_approx(cl - cmax);
    float cm = fadd(hc_eps, div_full(ce, glm_hc::row4(ce)));  // ptxas: fma(rcp(sum), ce, hc_eps), as for Triton
    cm = div_full(cm, hc_eps + glm_hc::col4(cm));
    for (int it = 1; it < iters; ++it) {
      cm = div_full(cm, hc_eps + glm_hc::row4(cm));
      cm = div_full(cm, hc_eps + glm_hc::col4(cm));
    }
    if (lane < 4) {
      post[r * S + lane] = sp + sp;
      pre[lane] = pr;
    }
    if (lane < 16) comb[r * 16 + lane] = cm;
  }
  __syncthreads();
  const float p0 = pre[0], p1 = pre[1], p2 = pre[2], p3 = pre[3];
  float c[16];
  for (int k = 0; k < 16; ++k) {
    const __nv_bfloat16* h0 = reinterpret_cast<const __nv_bfloat16*>(&xu[k / 8][0]);
    const __nv_bfloat16* h1 = reinterpret_cast<const __nv_bfloat16*>(&xu[k / 8][1]);
    const __nv_bfloat16* h2 = reinterpret_cast<const __nv_bfloat16*>(&xu[k / 8][2]);
    const __nv_bfloat16* h3 = reinterpret_cast<const __nv_bfloat16*>(&xu[k / 8][3]);
    const float x0 = bf(h0[k % 8]), x1 = bf(h1[k % 8]), x2 = bf(h2[k % 8]), x3 = bf(h3[k % 8]);
    float m = k % 2 == 0 ? __fmaf_rn(p1, x1, __fmul_rn(p0, x0)) : __fmaf_rn(p0, x0, __fmul_rn(p1, x1));
    m = __fmaf_rn(p2, x2, m);
    c[k] = rbf(__fmaf_rn(p3, x3, m));
  }
  float acc = __fmaf_rn(c[0], c[0], __fmul_rn(c[1], c[1]));
  for (int k = 2; k < 16; ++k) acc = k % 2 == 0 ? __fmaf_rn(c[k], c[k], acc) : fadd(fmul(c[k], c[k]), acc);
  const float total = block_sum<8>(acc, smem);
  const float rinv = div_full(1.0f, sqrt_approx(eps + div_full(total, (float)D)));
  for (int rep = 0; rep < 2; ++rep) {
    float y[8];
    const __nv_bfloat16* wh = reinterpret_cast<const __nv_bfloat16*>(&wu[rep]);
    uint4 ou;
    __nv_bfloat16* oh = reinterpret_cast<__nv_bfloat16*>(&ou);
    for (int j = 0; j < 8; ++j) {
      const __nv_bfloat16 o = __float2bfloat16_rn(__fmul_rn(bf(wh[j]), rbf(__fmul_rn(rinv, c[rep * 8 + j]))));
      oh[j] = o;
      y[j] = bf(o);
    }
    *reinterpret_cast<uint4*>(out + r * D + rep * 2048 + t * 8) = ou;
    float g = y[0] + y[1];
    for (int j = 2; j < 8; ++j) g = y[j] + g;
    for (int o = 4; o > 0; o >>= 1) g = g + __shfl_xor_sync(0xffffffffu, g, o);
    if ((t & 7) == 0) xs[r * (D / 64) + rep * 32 + t / 8] = g;
  }
}
