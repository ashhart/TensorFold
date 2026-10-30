// Frozen main 958efa3bf BF16 norm implementations for replacement equality; only the contract controls include this.

// glue._rmsnorm by hand (tf_ctl_glm_old_norm.cu's first group): see glm_triton.cuh for the Triton ops it keeps.
#pragma once

#include "glm_triton.cuh"

namespace tf_ctl_glm_old_norm {
using namespace glm_triton;

// S consecutive bf16 (S 4 or 8) moved in one load or store of 2 S bytes when `vec`, else two bytes at a time.
template <int S>
struct Pack;
template <>
struct Pack<4> {
  using T = uint2;
};
template <>
struct Pack<8> {
  using T = uint4;
};
template <int S>
__device__ __forceinline__ bool packed(const void* p, int64_t stride) {
  return ((reinterpret_cast<uintptr_t>(p) | static_cast<uintptr_t>(stride * 2)) & (2 * S - 1)) == 0;
}
template <int S>
__device__ __forceinline__ void load_s(const __nv_bfloat16* p, float* v, bool vec) {
  if (vec) {
    const typename Pack<S>::T u = *reinterpret_cast<const typename Pack<S>::T*>(p);
    const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&u);
#pragma unroll
    for (int j = 0; j < S; ++j) v[j] = bf(h[j]);
  } else {
#pragma unroll
    for (int j = 0; j < S; ++j) v[j] = bf(p[j]);
  }
}
template <int S>
__device__ __forceinline__ void store_s(__nv_bfloat16* p, const __nv_bfloat16* o, bool vec) {
  if (vec) {
    typename Pack<S>::T u;
    __nv_bfloat16* h = reinterpret_cast<__nv_bfloat16*>(&u);
#pragma unroll
    for (int j = 0; j < S; ++j) h[j] = o[j];
    *reinterpret_cast<typename Pack<S>::T*>(p) = u;
  } else {
#pragma unroll
    for (int j = 0; j < S; ++j) p[j] = o[j];
  }
}

// glue._rmsnorm, BLOCK = D to a power of two, NW warps, S values a thread: bf16(w * bf16(x / rms)), `sums` per 64.
template <int D, int BLOCK, int NW, int S>
__device__ void rmsnorm(const __nv_bfloat16* x, int64_t x_stride, const __nv_bfloat16* w, __nv_bfloat16* out,
                        int64_t o_stride, float* xs, float eps, int sums) {
  constexpr int T = NW * 32, TILE = S * T, REPS = BLOCK / TILE, N = REPS * S, G = 64 / S;
  static_assert(D % S == 0, "a thread's values all inside D or all past it");
  __shared__ float smem[NW];
  const int64_t r = blockIdx.x;
  const int t = threadIdx.x;
  const bool xv = packed<S>(x, x_stride), wv = packed<S>(w, 0), ov = packed<S>(out, o_stride);
  float v[N];
#pragma unroll
  for (int rep = 0; rep < REPS; ++rep) {
    const int d = rep * TILE + t * S;
    if (d < D) {
      load_s<S>(x + r * x_stride + d, v + rep * S, xv);
    } else {
#pragma unroll
      for (int j = 0; j < S; ++j) v[rep * S + j] = 0.0f;
    }
  }
  float acc = __fmul_rn(v[1], v[1]);
  acc = __fmaf_rn(v[0], v[0], acc);
#pragma unroll
  for (int k = 2; k < N; ++k) acc = __fmaf_rn(v[k], v[k], acc);
  const float total = block_sum<NW>(acc, smem);
  const float rinv = div_full(1.0f, sqrt_approx(eps + div_full(total, (float)D)));
  float y[N];
#pragma unroll
  for (int rep = 0; rep < REPS; ++rep) {
    const int d = rep * TILE + t * S;
    if (d >= D) {
#pragma unroll
      for (int j = 0; j < S; ++j) y[rep * S + j] = 0.0f;
      continue;
    }
    float wf[S];
    __nv_bfloat16 o[S];
    load_s<S>(w + d, wf, wv);
#pragma unroll
    for (int j = 0; j < S; ++j) {
      o[j] = __float2bfloat16_rn(__fmul_rn(wf[j], rbf(__fmul_rn(rinv, v[rep * S + j]))));
      y[rep * S + j] = bf(o[j]);
    }
    store_s<S>(out + r * o_stride + d, o, ov);
  }
  if (!sums) return;
#pragma unroll
  for (int rep = 0; rep < REPS; ++rep) {
    const float* q = y + rep * S;
    float g = q[1];
    g = q[0] + g;
#pragma unroll
    for (int j = 2; j < S; ++j) g = q[j] + g;
    for (int o = G / 2; o > 0; o >>= 1) g += __shfl_xor_sync(0xffffffffu, g, o);
    const int group = (rep * TILE + t * S) / 64;
    if (t % G == 0 && group < D / 64) xs[r * (D / 64) + group] = g;
  }
}

}  // namespace tf_ctl_glm_old_norm

#define RMSNORM(D, BLOCK, NW, S)                                                                                   \
  extern "C" __global__ void __launch_bounds__(NW * 32)                                                            \
      tf_ctl_glm_old_rmsnorm_##D(const __nv_bfloat16* x, int64_t x_stride, const __nv_bfloat16* w, __nv_bfloat16* out,    \
                         int64_t o_stride, float* xs, float eps, int sums) {                                       \
    tf_ctl_glm_old_norm::rmsnorm<D, BLOCK, NW, S>(x, x_stride, w, out, o_stride, xs, eps, sums);                              \
  }

// The model's norms: hidden 4096 (8 warps), q_lora 1536 and kv_lora 512 (4 warps), as glue.rmsnorm launches them.
RMSNORM(4096, 4096, 8, 8)
RMSNORM(1536, 2048, 4, 8)
RMSNORM(512, 512, 4, 4)

#undef RMSNORM

using namespace glm_triton;

namespace tf_ctl_glm_old_hc {

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

}  // namespace tf_ctl_glm_old_hc

// glue._hc_finish, 256 threads a row (D 4096, 4 streams): warp 0 sums K blocks and Sinkhorn, then collapse, RMS-norm.
extern "C" __global__ void __launch_bounds__(256)
    tf_ctl_glm_old_hc_finish(const __nv_bfloat16* x, const float* part, const float* base, const float* scale,
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
    const float pr = hc_eps + tf_ctl_glm_old_hc::sigmoid(__fmaf_rn(mpre, scale[0], bpre));
    const float sp = tf_ctl_glm_old_hc::sigmoid(__fmaf_rn(mpost, scale[1], bpost));
    const float cl = __fmaf_rn(mcomb, scale[2], bcomb);
    float cmax = fmaxf(cl, __shfl_xor_sync(0xffffffffu, cl, 2));
    cmax = fmaxf(cmax, __shfl_xor_sync(0xffffffffu, cmax, 1));
    const float ce = exp_approx(cl - cmax);
    float cm = fadd(hc_eps, div_full(ce, tf_ctl_glm_old_hc::row4(ce)));  // ptxas: fma(rcp(sum), ce, hc_eps), as for Triton
    cm = div_full(cm, hc_eps + tf_ctl_glm_old_hc::col4(cm));
    for (int it = 1; it < iters; ++it) {
      cm = div_full(cm, hc_eps + tf_ctl_glm_old_hc::row4(cm));
      cm = div_full(cm, hc_eps + tf_ctl_glm_old_hc::col4(cm));
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
