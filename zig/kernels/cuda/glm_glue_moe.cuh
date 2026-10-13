// GLM's MLP and MoE glue by hand: glue._swiglu, glue._topk and qmm._group_sums (the router's dot is glm_glue_router).
#pragma once

#include "glm_triton.cuh"

using namespace glm_triton;

namespace glm_moe {

// A thread's S consecutive values summed from the second, then the butterfly over the 64 / S lanes of their group.
template <int S>
__device__ __forceinline__ float group64(const float* q) {
  float g = q[0] + q[1];
  for (int j = 2; j < S; ++j) g = q[j] + g;
  for (int o = 32 / S; o > 0; o >>= 1) g = g + __shfl_xor_sync(0xffffffffu, g, o);
  return g;
}

__device__ __forceinline__ float swiglu1(float g, float u, float limit) {
  g = fminf(g, limit);
  u = fminf(fmaxf(u, 0.0f - limit), limit);
  const float s = rbf(div_full(g, exp_approx(0.0f - g) + 1.0f));
  return rbf(__fmul_rn(u, s));
}

}  // namespace glm_moe

// glue._swiglu with BLOCK 512 (128 threads, 4 values each, 64-group sums over 16 lanes), column block blockIdx.y.
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_swiglu_512(const __nv_bfloat16* gu, __nv_bfloat16* out, float* xs, float limit, int64_t w) {
  const int64_t r = blockIdx.x, d0 = blockIdx.y * 512 + threadIdx.x * 4;
  const uint2 gv = *reinterpret_cast<const uint2*>(gu + r * 2 * w + d0);
  const uint2 uv = *reinterpret_cast<const uint2*>(gu + r * 2 * w + w + d0);
  const __nv_bfloat16* gh = reinterpret_cast<const __nv_bfloat16*>(&gv);
  const __nv_bfloat16* uh = reinterpret_cast<const __nv_bfloat16*>(&uv);
  float a[4];
  uint2 ov;
  __nv_bfloat16* oh = reinterpret_cast<__nv_bfloat16*>(&ov);
  for (int j = 0; j < 4; ++j) {
    a[j] = glm_moe::swiglu1(bf(gh[j]), bf(uh[j]), limit);
    oh[j] = __float2bfloat16_rn(a[j]);
  }
  *reinterpret_cast<uint2*>(out + r * w + d0) = ov;
  const float g = glm_moe::group64<4>(a);
  if ((threadIdx.x & 15) == 0) xs[r * (w / 64) + d0 / 64] = g;
}

// glue._swiglu with BLOCK 64 (the TP=3 shared expert's 704): one value a thread, the warp's butterfly, then 2 warps.
extern "C" __global__ void __launch_bounds__(64)
    tf_glm_swiglu_64(const __nv_bfloat16* gu, __nv_bfloat16* out, float* xs, float limit, int64_t w) {
  __shared__ float smem[2];
  const int64_t r = blockIdx.x, d = blockIdx.y * 64 + threadIdx.x;
  const float a = glm_moe::swiglu1(bf(gu[r * 2 * w + d]), bf(gu[r * 2 * w + w + d]), limit);
  out[r * w + d] = __float2bfloat16_rn(a);
  const float g = block_sum<2>(a, smem);
  if (threadIdx.x == 0) xs[r * (w / 64) + blockIdx.y] = g;
}

// qmm._group_sums: row blockIdx.x's 64-input sums, 8 lanes a group of 8 consecutive inputs each, 16 groups a block.
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_group_sums(const __nv_bfloat16* x, float* xs, int64_t x_stride, int64_t kg) {
  const int64_t m = blockIdx.x, g = blockIdx.y * 16 + threadIdx.x / 8;
  float q[8];
  for (int j = 0; j < 8; ++j) q[j] = g < kg ? bf(x[m * x_stride + g * 64 + (threadIdx.x & 7) * 8 + j]) : 0.0f;
  const float s = glm_moe::group64<8>(q);
  if ((threadIdx.x & 7) == 0 && g < kg) xs[m * kg + g] = s;
}

// glue._topk, a warp a row: top_k by sigmoid + bias (ties to lower id), normed if `norm`, x scale; then shared expert.
extern "C" __global__ void __launch_bounds__(32)
    tf_glm_topk(const float* logits, const float* bias, int* pick, float* wts, float scale, int ne, int top_k,
                int norm) {
  constexpr int MAXE = 16;  // 512 experts a row
  const int64_t r = blockIdx.x;
  const int lane = threadIdx.x;
  float score[MAXE], choice[MAXE];
  for (int i = 0; i < MAXE; ++i) {
    const int e = i * 32 + lane;
    score[i] = e < ne ? div_full(1.0f, exp_approx(0.0f - logits[r * ne + e]) + 1.0f) : 0.0f;
    choice[i] = e < ne ? score[i] + bias[e] : -INFINITY;
  }
  float total = 0.0f, mine = 0.0f;
  int mine_id = 0;
  for (int k = 0; k < top_k; ++k) {
    float mx = choice[0];
    for (int i = 1; i < MAXE; ++i) mx = fmaxf(mx, choice[i]);
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    int id = 1 << 30;
    for (int i = MAXE - 1; i >= 0; --i)
      if (choice[i] == mx) id = i * 32 + lane;
    for (int o = 16; o > 0; o >>= 1) id = min(id, __shfl_xor_sync(0xffffffffu, id, o));
    float sk = 0.0f;
    for (int i = 0; i < MAXE; ++i)
      if (i * 32 + lane == id) {
        sk = score[i];
        choice[i] = -INFINITY;
      }
    sk = __shfl_sync(0xffffffffu, sk, id & 31);
    total = total + sk;
    if (lane == k) {
      mine = sk;
      mine_id = id;
    }
  }
  if (lane < top_k) {
    const float v = norm ? div_full(mine, total + 1e-20f) : mine;
    pick[r * (top_k + 1) + lane] = mine_id;
    wts[r * (top_k + 1) + lane] = __fmul_rn(v, scale);
  } else if (lane == top_k) {
    pick[r * (top_k + 1) + lane] = ne;
    wts[r * (top_k + 1) + lane] = 1.0f;
  }
}
