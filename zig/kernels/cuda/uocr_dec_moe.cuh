// Unlimited-OCR-2's expert gate, SwiGLU, experts and combine on CUDA as decoder.metal's; PLANT > 0: a control.
#pragma once
#include "uocr_dec_rows.cuh"

namespace tf_uocr {

// x . w over 4 values, left to right.
__device__ __forceinline__ float ud_dot4(const float4 x, const float4 w) {
  return x.x * w.x + x.y * w.y + x.z * w.z + x.w * w.w;
}

__device__ __forceinline__ float4 ud_widen4(const bf16* p) {
  const uint2 u = *reinterpret_cast<const uint2*>(p);
  const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
  const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
  return make_float4(a.x, a.y, b.x, b.y);
}

// A warp's extreme of an int: the least (or with MOST the greatest).
template <bool MOST>
__device__ __forceinline__ int ud_warp_pick(int v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const int w = __shfl_xor_sync(0xffffffffu, v, o);
    v = MOST ? max(v, w) : min(v, w);
  }
  return v;
}

// A row a block of 1,024: 2 experts' logits a warp, warp 0's softmax and 6 largest (ties low; PLANT 1 high, 2 no mass).
template <int PLANT>
__device__ __forceinline__ void ud_moe_gate(const bf16* X, const bf16* G, unsigned* IDS, float* WTS) {
  __shared__ float logit[UD_EXPERTS];
  const int row = blockIdx.x, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const bf16* x = X + (size_t)row * UD_H;
  float acc[2] = {0.0f, 0.0f};
  for (int k = lane * 8; k < UD_H; k += 256) {
    const float4 x0 = ud_widen4(x + k), x1 = ud_widen4(x + k + 4);
#pragma unroll
    for (int r = 0; r < 2; r++) {
      const bf16* g = G + (size_t)(w * 2 + r) * UD_H + k;
      acc[r] += ud_dot4(x0, ud_widen4(g)) + ud_dot4(x1, ud_widen4(g + 4));
    }
  }
#pragma unroll
  for (int r = 0; r < 2; r++) {
    const float s = warp_sum(acc[r]);
    if (lane == 0) logit[w * 2 + r] = s;
  }
  __syncthreads();
  if (w != 0) return;
  float v[2] = {logit[lane], logit[lane + 32]};
  const float top = warp_max(fmaxf(v[0], v[1]));
  v[0] = expf(v[0] - top);
  v[1] = expf(v[1] - top);
  const float mass = PLANT == 2 ? 1.0f : warp_sum(v[0] + v[1]);
  v[0] = v[0] / mass;
  v[1] = v[1] / mass;
  bool taken[2] = {false, false};
  for (int k = 0; k < UD_PICK; k++) {
    const float a = taken[0] ? -2.0f : (isnan(v[0]) ? -1.0f : v[0]);
    const float b = taken[1] ? -2.0f : (isnan(v[1]) ? -1.0f : v[1]);
    const float best = warp_max(fmaxf(a, b));
    const int low = a == best ? lane : (b == best ? lane + 32 : UD_EXPERTS);
    const int high = b == best ? lane + 32 : (a == best ? lane : -1);
    const int mine = PLANT == 1 ? high : low;
    const int e = ud_warp_pick<PLANT == 1>(mine);
    const float wt = __shfl_sync(0xffffffffu, e < 32 ? v[0] : v[1], e % 32);
    if (e == lane) taken[0] = true;
    if (e == lane + 32) taken[1] = true;
    if (lane == 0) {
      IDS[(size_t)row * UD_PICK + k] = e;
      WTS[(size_t)row * UD_PICK + k] = wt;
    }
  }
}

// SwiGLU at the release's bf16 points: Y = bf16(bf16(silu(g)) * u); PLANT 1 drops bf16(silu(g)), 2 reads g from U.
template <int PLANT>
__device__ __forceinline__ void ud_silu_mul(const bf16* G, const bf16* U, bf16* Y, int width, int gstride, int ustride) {
  const int x = blockIdx.y * 256 + threadIdx.x, row = blockIdx.x;
  if (x >= width) return;
  const float g = PLANT == 2 ? ld(U + (size_t)row * ustride + x) : ld(G + (size_t)row * gstride + x);
  const float s = g / (1.0f + expf(-g));
  const float a = PLANT == 1 ? s : bf(s);
  Y[(size_t)row * width + x] = to(a * ld(U + (size_t)row * ustride + x));
}

// One token's experts' GU[j] = [gate . x | up . x] by E's addresses; PLANT 1 sums in bf16, 2 reads expert j.
template <int PLANT>
__device__ __forceinline__ void ud_expert_gu(const bf16* x, const unsigned long long* E, const unsigned* ids, bf16* GU,
                                             int width) {
  const int sg = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row0 = (blockIdx.x * 8 + sg) * 4;
  if (row0 >= UD_PICK * 2 * width) return;
  const int j = row0 / (2 * width), r0 = row0 % (2 * width);
  const unsigned long long* ex = E + (size_t)(PLANT == 2 ? j : ids[j]) * 3;
  const bf16* gate = reinterpret_cast<const bf16*>(ex[0]);
  const bf16* up = reinterpret_cast<const bf16*>(ex[1]);
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int k = lane * 8; k < UD_H; k += 256) {
    const float4 x0 = ud_widen4(x + k), x1 = ud_widen4(x + k + 4);
#pragma unroll
    for (int r = 0; r < 4; r++) {
      const int rr = r0 + r;
      const bf16* w = (rr < width ? gate + (size_t)rr * UD_H : up + (size_t)(rr - width) * UD_H) + k;
      const float s = ud_dot4(x0, ud_widen4(w)) + ud_dot4(x1, ud_widen4(w + 4));
      acc[r] = PLANT == 1 ? bf(acc[r] + s) : acc[r] + s;
    }
  }
#pragma unroll
  for (int r = 0; r < 4; r++) {
    const float s = warp_sum(acc[r]);
    if (lane == 0) GU[(size_t)row0 + r] = to(s);
  }
}

// One token's experts' D[j] = down_j . act_j, 4 rows a warp; PLANT 1 skips each row's last 8 columns, 2 reads expert j.
template <int PLANT>
__device__ __forceinline__ void ud_expert_down(const bf16* A, const unsigned long long* E, const unsigned* ids, bf16* D,
                                               int width) {
  const int sg = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row0 = (blockIdx.x * 8 + sg) * 4;
  if (row0 >= UD_PICK * UD_H) return;
  const int j = row0 / UD_H, r0 = row0 % UD_H;
  const bf16* x = A + (size_t)j * width;
  const bf16* W = reinterpret_cast<const bf16*>(E[(size_t)(PLANT == 2 ? j : ids[j]) * 3 + 2]);
  const int end = PLANT == 1 ? width - 8 : width;
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int k = lane * 8; k < end; k += 256) {
    const float4 x0 = ud_widen4(x + k), x1 = ud_widen4(x + k + 4);
#pragma unroll
    for (int r = 0; r < 4; r++) {
      const bf16* w = W + (size_t)(r0 + r) * width + k;
      acc[r] += ud_dot4(x0, ud_widen4(w)) + ud_dot4(x1, ud_widen4(w + 4));
    }
  }
#pragma unroll
  for (int r = 0; r < 4; r++) {
    const float s = warp_sum(acc[r]);
    if (lane == 0) D[(size_t)row0 + r] = to(s);
  }
}

// A row's bf16(bf16(sum_j D[slot[r, j]] w[r, j]) + S), the sum in fp32; PLANT 1 drops the inner bf16, 2 the S.
template <int PLANT>
__device__ __forceinline__ void ud_combine(const bf16* D, const unsigned* slot, const float* WTS, const bf16* S, bf16* Y) {
  const int x = blockIdx.y * 256 + threadIdx.x, row = blockIdx.x;
  if (x >= UD_H) return;
  float acc = 0.0f;
  for (int j = 0; j < UD_PICK; j++) {
    const size_t i = (size_t)row * UD_PICK + j;
    acc += ld(D + (size_t)slot[i] * UD_H + x) * WTS[i];
  }
  const size_t o = (size_t)row * UD_H + x;
  Y[o] = PLANT == 2 ? to(bf(acc)) : to((PLANT == 1 ? acc : bf(acc)) + ld(S + o));
}

}  // namespace tf_uocr
