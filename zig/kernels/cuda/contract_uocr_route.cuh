// Gate 3 controls for uocr.cu's route group: tf_uocr_route, tf_uocr_moe_gate and tf_uocr_argmax, each with one break.
#pragma once
#include <cuda_bf16.h>
#include <math.h>
#include "common/reduce.cuh"

namespace tf_ctl_uocr_route {

typedef __nv_bfloat16 bf16;
constexpr float kNeg = -1.0e9f;
constexpr float kEntmaxNeg = -1.0e4f;
constexpr float kClamp = 1.0e4f;
constexpr int kThreads = 256;

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float ld(const bf16* p) { return __bfloat162float(*p); }
__device__ __forceinline__ float clampf(float x, float lo, float hi) { return fminf(fmaxf(x, lo), hi); }

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

__device__ __forceinline__ float block_sum(float v, float* part) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, n = (blockDim.x + 31) >> 5;
  v = tf_common::warp_sum(v);
  if (lane == 0) part[w] = v;
  __syncthreads();
  if (w == 0) {
    float s = lane < n ? part[lane] : 0.0f;
    s = tf_common::warp_sum(s);
    if (lane == 0) part[32] = s;
  }
  __syncthreads();
  const float out = part[32];
  __syncthreads();
  return out;
}

// Whether (v, i) ranks before (bv, bi): the higher value, then the lower id (HI: the higher id).
template <bool HI>
__device__ __forceinline__ bool before(float v, int i, float bv, int bi) {
  return v > bv || (v == bv && (HI ? i > bi : i < bi));
}

// tf_uocr_route with one break. MODE 1: ties to the higher block; 2: the gap one block short; 3: a last page left out.
template <int MODE>
__device__ void route(const float* __restrict__ idx_q, const float* __restrict__ kbar, const int* __restrict__ bmin,
                      const int* __restrict__ bmax, const int* __restrict__ qpage, const unsigned* __restrict__ qpos,
                      int* __restrict__ sel, float* __restrict__ bias, int nblk, int complete, float scale,
                      float idx_scale, float floor_, int round_bias) {
  constexpr bool HI = MODE == 1;
  constexpr int NONE = HI ? -1 : 0x7fffffff;
  __shared__ float x[4096];
  __shared__ unsigned char cand[4096];
  __shared__ float iq[128];
  __shared__ float xv[16];
  __shared__ int xi[16];
  __shared__ float lv[kThreads / 32][16];
  __shared__ int li[kThreads / 32][16];
  const int q = blockIdx.x, t = threadIdx.x, lane = t & 31, w = t >> 5;
  if (t < 128) iq[t] = idx_q[(size_t)q * 128 + t];
  const int qp = qpage[q];
  const int last_remote = (int)(qpos[q] / 32) - (MODE == 2 ? 3 : 4);
  bool any_page = false;
  if (qp >= 0)
    for (int c = t; c < nblk; c += kThreads)
      any_page = any_page || (bmin[c] >= 0 && bmin[c] <= qp && (MODE == 3 ? qp < bmax[c] : qp <= bmax[c]));
  any_page = __syncthreads_or(any_page);
  bool any_cand = false;
  for (int c = t; c < nblk; c += kThreads) {
    const bool remote = c <= last_remote && c < complete;
    const bool allow = !any_page || (bmin[c] >= 0 && bmin[c] <= qp && (MODE == 3 ? qp < bmax[c] : qp <= bmax[c]));
    float z = kNeg;
    if (remote) {
      const float4* kb = reinterpret_cast<const float4*>(kbar + (size_t)c * 128);
      float s = 0.0f;
      for (int d = 0; d < 32; d++) {
        const float4 k4 = kb[d];
        s += iq[4 * d] * k4.x;
        s += iq[4 * d + 1] * k4.y;
        s += iq[4 * d + 2] * k4.z;
        s += iq[4 * d + 3] * k4.w;
      }
      z = clampf(s * scale, -kClamp, kClamp);
    }
    any_cand = any_cand || (remote && allow);
    cand[c] = (unsigned char)(remote ? (allow ? 2 : 1) : 0);
    x[c] = z;
  }
  any_cand = __syncthreads_or(any_cand);
  for (int c = t; c < nblk; c += kThreads) {
    const bool is = any_cand ? cand[c] == 2 : cand[c] >= 1;
    cand[c] = (unsigned char)is;
    float v = is ? x[c] * idx_scale : kEntmaxNeg;
    if (!isfinite(v)) v = kEntmaxNeg;
    x[c] = v;
  }
  __syncthreads();
  const int M = max(1, min(16, nblk));
  for (int r = 0; r < M; r++) {
    float best = -INFINITY;
    int at = NONE;
    for (int c = t; c < nblk; c += kThreads)
      if (x[c] != -INFINITY && before<HI>(x[c], c, best, at)) {
        best = x[c];
        at = c;
      }
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, best, o);
      const int oi = __shfl_xor_sync(0xffffffffu, at, o);
      if (before<HI>(ov, oi, best, at)) {
        best = ov;
        at = oi;
      }
    }
    if (at != NONE && (at & 31) == lane) x[at] = -INFINITY;
    if (lane == 0) {
      lv[w][r] = best;
      li[w][r] = at;
    }
    __syncwarp();
  }
  __syncthreads();
  if (w != 0) return;
  for (int r = 0; r < M; r++) {
    float best = -INFINITY;
    int at = NONE, slot = -1;
    for (int e = lane; e < (kThreads / 32) * M; e += 32) {
      const float v = lv[e / M][e % M];
      const int i = li[e / M][e % M];
      if (i != NONE && before<HI>(v, i, best, at)) {
        best = v;
        at = i;
        slot = e;
      }
    }
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, best, o);
      const int oi = __shfl_xor_sync(0xffffffffu, at, o);
      const int os = __shfl_xor_sync(0xffffffffu, slot, o);
      if (before<HI>(ov, oi, best, at)) {
        best = ov;
        at = oi;
        slot = os;
      }
    }
    if (lane == 0) {
      xv[r] = best;
      xi[r] = at;
      li[slot / M][slot % M] = NONE;
    }
    __syncwarp();
  }
  if (t != 0) return;
  const float mx = xv[0];
  float X[16], tau[16];
  float cs = 0.0f, cs2 = 0.0f;
  int support = 0;
  for (int k = 0; k < M; k++) {
    X[k] = (xv[k] - mx) / 2.0f;
    cs += X[k];
    cs2 += X[k] * X[k];
    const float rho = (float)(k + 1);
    const float mean = cs / rho, mean_sq = cs2 / rho;
    const float ss = rho * (mean_sq - mean * mean);
    const float delta = fmaxf((1.0f - ss) / rho, 0.0f);
    tau[k] = mean - sqrtf(delta);
    support += tau[k] <= X[k] ? 1 : 0;
  }
  const float tau_star = tau[max(support, 1) - 1];
  const int K = max(1, min(12, M));
  float logp[12];
  float total = 0.0f;
  int count = 0;
  for (int s = 0; s < K; s++) {
    const float d = fmaxf(X[s] - tau_star, 0.0f);
    logp[s] = logf(fmaxf(d * d, floor_));
    if (cand[xi[s]]) {
      total += logp[s];
      count++;
    }
  }
  const float mu = total / (float)max(count, 1);
  for (int s = 0; s < 12; s++) {
    const bool valid = s < K && cand[xi[s]];
    sel[q * 12 + s] = valid ? xi[s] : -1;
    const float b = valid ? logp[s] - mu : 0.0f;
    bias[q * 12 + s] = round_bias ? bf(b) : b;
  }
}

}  // namespace tf_ctl_uocr_route

#define TF_CTL_UOCR_ROUTE(NAME, MODE)                                                                                  \
  extern "C" __global__ void __launch_bounds__(256)                                                                    \
      NAME(const float* idx_q, const float* kbar, const int* bmin, const int* bmax, const int* qpage,                  \
           const unsigned* qpos, int* sel, float* bias, int nblk, int complete, float scale, float idx_scale,           \
           float floor_, int round_bias) {                                                                              \
    tf_ctl_uocr_route::route<MODE>(idx_q, kbar, bmin, bmax, qpage, qpos, sel, bias, nblk, complete, scale, idx_scale,   \
                                   floor_, round_bias);                                                                 \
  }
TF_CTL_UOCR_ROUTE(tf_ctl_uocr_route_tie_high, 1)
TF_CTL_UOCR_ROUTE(tf_ctl_uocr_route_gap_short, 2)
TF_CTL_UOCR_ROUTE(tf_ctl_uocr_route_last_page, 3)
#undef TF_CTL_UOCR_ROUTE

// tf_uocr_moe_gate with ties among the scores going to the higher expert.
extern "C" __global__ void __launch_bounds__(64) tf_ctl_uocr_moe_gate_tie_high(const __nv_bfloat16* __restrict__ x,
                                                                              const __nv_bfloat16* __restrict__ w,
                                                                              unsigned* __restrict__ ids,
                                                                              float* __restrict__ wts) {
  using namespace tf_ctl_uocr_route;
  __shared__ float part[33];
  __shared__ float score[64];
  const int row = blockIdx.x, e = threadIdx.x;
  const bf16* xr = x + (size_t)row * 1280;
  const bf16* wr = w + (size_t)e * 1280;
  float s = 0.0f;
  for (int d = 0; d < 1280; d++) s += ld(xr + d) * ld(wr + d);
  float m = warp_max(s);
  if ((e & 31) == 0) part[e >> 5] = m;
  __syncthreads();
  m = fmaxf(part[0], part[1]);
  __syncthreads();
  const float ex = expf(s - m);
  const float total = block_sum(ex, part);
  score[e] = ex / total;
  __syncthreads();
  if (e != 0) return;
  bool taken[64];
  for (int i = 0; i < 64; i++) taken[i] = false;
  for (int k = 0; k < 6; k++) {
    int best = 0;
    float bv = -INFINITY;
    for (int i = 0; i < 64; i++) {
      if (!taken[i] && score[i] >= bv) {
        bv = score[i];
        best = i;
      }
    }
    taken[best] = true;
    ids[row * 6 + k] = best;
    wts[row * 6 + k] = bv;
  }
}

// tf_uocr_argmax with ties going to the higher id.
extern "C" __global__ void __launch_bounds__(1024) tf_ctl_uocr_argmax_tie_high(const __nv_bfloat16* __restrict__ logits,
                                                                               int n, int bar_eos,
                                                                               unsigned* __restrict__ out) {
  using namespace tf_ctl_uocr_route;
  __shared__ float bv[32];
  __shared__ int bi[32];
  float best = -INFINITY;
  int at = -1;
  for (int i = threadIdx.x; i < n; i += 1024) {
    if (bar_eos && i == 1) continue;
    const float v = ld(logits + i);
    if (v >= best && !isnan(v)) {
      best = v;
      at = i;
    }
  }
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, best, o);
    const int oi = __shfl_xor_sync(0xffffffffu, at, o);
    if (before<true>(ov, oi, best, at)) {
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
    if (before<true>(ov, oi, best, at)) {
      best = ov;
      at = oi;
    }
  }
  if (lane == 0) out[0] = (unsigned)(at < 0 ? 0 : at);
}
