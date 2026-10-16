// glue._rmsnorm by hand (glm_glue.cu's first group): see glm_triton.cuh for the Triton ops it keeps.
#pragma once

#include "glm_triton.cuh"

namespace glm_glue {
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

}  // namespace glm_glue

#define RMSNORM(D, BLOCK, NW, S)                                                                                   \
  extern "C" __global__ void __launch_bounds__(NW * 32)                                                            \
      tf_glm_rmsnorm_##D(const __nv_bfloat16* x, int64_t x_stride, const __nv_bfloat16* w, __nv_bfloat16* out,    \
                         int64_t o_stride, float* xs, float eps, int sums) {                                       \
    glm_glue::rmsnorm<D, BLOCK, NW, S>(x, x_stride, w, out, o_stride, xs, eps, sums);                              \
  }

// The model's norms: hidden 4096 (8 warps), q_lora 1536 and kv_lora 512 (4 warps), as glue.rmsnorm launches them.
RMSNORM(4096, 4096, 8, 8)
RMSNORM(1536, 2048, 4, 8)
RMSNORM(512, 512, 4, 4)
