// Unlimited-OCR-2's decoder norms, rotary and block keys on CUDA as decoder.metal's; PLANT > 0: a planted control.
#pragma once
#include <cuda_fp16.h>
#include "uocr_common.cuh"

namespace tf_uocr {

constexpr int UD_H = 1280, UD_HD = 128, UD_HEADS = 10, UD_BLK = 32, UD_LOCAL = 128, UD_RD = 128;
constexpr int UD_READ = 12, UD_TOP = 16, UD_EXPERTS = 64, UD_PICK = 6;
// The router's last block a query may read sits this many blocks before its own (the local window's blocks).
constexpr int UD_LOCAL_BLOCKS = 4;
// The release's floor under a block's entmax weight.
constexpr float UD_FLOOR = 1.0e-4f;
constexpr unsigned UD_NONE = 0xffffffffu;
// 1 / sqrt(128) and 128 / sqrt(128) rounded to fp32: the scores' scale and the router's.
constexpr float UD_INV_SQRT_HD = 0.0883883476483184405f, UD_ROUTER_SCALE = 11.3137084989847604f;

// The release's clamp at +-1e4: NaN stays NaN (KEEP_NAN false: a planted control's fminf/fmaxf, NaN reads -1e4).
template <bool KEEP_NAN = true>
__device__ __forceinline__ float ud_clamp(float x) {
  return KEEP_NAN && isnan(x) ? x : clampf(x, -UD_CLAMP, UD_CLAMP);
}

// A row a block of 256: R = bf16(R + A) if `add`, Y = bf16(w bf16(r inv)); PLANT 1 no bf16, 2 mean over 256, 3 R - A.
template <int PLANT>
__device__ __forceinline__ void ud_norm(bf16* R, const bf16* A, const bf16* w, bf16* Y, int add, float eps) {
  __shared__ float part[33];
  const int row = blockIdx.x, t = threadIdx.x;
  bf16* r = R + (size_t)row * UD_H;
  float v[5];
  float ss = 0.0f;
#pragma unroll
  for (int i = 0; i < 5; i++) {
    const int c = t + 256 * i;
    float x = ld(r + c);
    if (add != 0) {
      x = bf(PLANT == 3 ? x - ld(A + (size_t)row * UD_H + c) : x + ld(A + (size_t)row * UD_H + c));
      r[c] = to(x);
    }
    v[i] = x;
    ss += x * x;
  }
  ss = tf_common::block_sum_tree(ss, part, blockDim.x >> 5);
  const float inv = 1.0f / sqrtf(ss / (PLANT == 2 ? 256.0f : (float)UD_H) + eps);
#pragma unroll
  for (int i = 0; i < 5; i++) {
    const int c = t + 256 * i;
    const float n = PLANT == 1 ? v[i] * inv : bf(v[i] * inv);
    Y[(size_t)row * UD_H + c] = to(ld(w + c) * n);
  }
}

// Rotary on one value of a 128-wide head: bf16(bf16(x c) + bf16(rot(x) s)); PLANT 1 drops bf16(x c), 2 turns back.
template <int PLANT>
__device__ __forceinline__ float ud_rope(float x, float partner, int d, const bf16* cs, const bf16* sn) {
  const float c = ld(cs + (d & 63)), s = ld(sn + (d & 63));
  const float rot = (d < 64) != (PLANT == 2) ? -partner : partner;
  return bf((PLANT == 1 ? x * c : bf(x * c)) + bf(rot * s));
}

// The router's norm of a 128-wide row in fp32, (1 + w) scaled, clamped at +-1e4 before and after the scale; bf16 out.
template <bool KEEP_NAN>
__device__ __forceinline__ float ud_router_norm(float x, const bf16* w, int d, float eps, float* part) {
  const float ss = tf_common::block_sum_tree(x * x, part, blockDim.x >> 5);
  const float n = ud_clamp<KEEP_NAN>(x * (1.0f / sqrtf(ss / (float)UD_RD + eps)));
  return bf(ud_clamp<KEEP_NAN>(n * (1.0f + ld(w + d))));
}

// A row a block of 128: P (q | k | v), RP (router q | k) rotated into q, the caches and router q; PLANT 3: NaN clamped.
template <int PLANT>
__device__ __forceinline__ void ud_qk(const bf16* P, const bf16* RP, const bf16* COS, const bf16* SIN, const bf16* qn,
                                      const bf16* kn, bf16* Q, bf16* K, bf16* V, bf16* RQ, bf16* RK, int pos0, float eps) {
  __shared__ float part[33];
  __shared__ float rr[2][UD_RD];
  const int row = blockIdx.x, d = threadIdx.x;
  const bf16* x = P + (size_t)row * 3 * UD_H;
  const bf16* rp = RP + (size_t)row * 2 * UD_RD;
  const bf16* cs = COS + (size_t)row * 64;
  const bf16* sn = SIN + (size_t)row * 64;
  const size_t at = (size_t)(pos0 + row) * UD_H;
  const int pd = d ^ 64;
  for (int h = 0; h < UD_HEADS; h++) {
    const int o = h * UD_HD;
    Q[(size_t)row * UD_H + o + d] = to(ud_rope<PLANT>(ld(x + o + d), ld(x + o + pd), d, cs, sn));
    K[at + o + d] = to(ud_rope<PLANT>(ld(x + UD_H + o + d), ld(x + UD_H + o + pd), d, cs, sn));
    V[at + o + d] = x[2 * UD_H + o + d];
  }
  rr[0][d] = ud_router_norm<PLANT != 3>(ld(rp + d), qn, d, eps, part);
  rr[1][d] = ud_router_norm<PLANT != 3>(ld(rp + UD_RD + d), kn, d, eps, part);
  __syncthreads();
  RQ[(size_t)row * UD_RD + d] = to(ud_rope<PLANT>(rr[0][d], rr[0][pd], d, cs, sn));
  RK[(size_t)(pos0 + row) * UD_RD + d] = to(ud_rope<PLANT>(rr[1][d], rr[1][pd], d, cs, sn));
}

// A warp a block: block c0 + b's fp32 key by softmax(clamp(k . hc / sqrt(128))); PLANT 1 no scale, 2 no division.
template <int PLANT>
__device__ __forceinline__ void ud_block_keys(const bf16* RK, const bf16* hc, float* KB, int c0) {
  const size_t c = (size_t)c0 + blockIdx.x;
  const int j = threadIdx.x;
  const bf16* k = RK + (c * UD_BLK + j) * UD_RD;
  float aw = 0.0f;
  for (int d = 0; d < UD_RD; d++) aw += ld(k + d) * ld(hc + d);
  aw = ud_clamp(PLANT == 1 ? aw : aw * UD_INV_SQRT_HD);
  const float e = expf(aw - warp_max(aw));
  const float w = PLANT == 2 ? e : e / fmaxf(warp_sum(e), 1.0e-6f);
  for (int d = j; d < UD_RD; d += 32) {
    float acc = 0.0f;
    for (int i = 0; i < UD_BLK; i++) acc += __shfl_sync(0xffffffffu, w, i) * ld(RK + (c * UD_BLK + i) * UD_RD + d);
    KB[c * UD_RD + d] = acc;
  }
}

// Gathered rows y[i] = x[rows[i]] (1280 wide), grid (count, 5); PLANT 1 reads the next entry's row.
template <int PLANT>
__device__ __forceinline__ void ud_gather(const bf16* x, const unsigned* rows, bf16* y) {
  const int c = blockIdx.y * 256 + threadIdx.x;
  if (c >= UD_H) return;
  const size_t from = rows[PLANT == 1 ? (blockIdx.x + 1) % gridDim.x : blockIdx.x];
  y[(size_t)blockIdx.x * UD_H + c] = x[from * UD_H + c];
}

// One token's y[n] = bf16(W[n, :] . x), fp32 sums, 4 rows a warp; PLANT 1 drops the last 8 columns, 2 sums in fp16.
template <int PLANT>
__device__ __forceinline__ void ud_matvec(const bf16* x, const bf16* W, bf16* y, int n, int k) {
  const int sg = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row0 = (blockIdx.x * 8 + sg) * 4;
  if (row0 >= n) return;
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int i = lane * 8; PLANT == 1 ? i + 8 < k : i < k; i += 256) {
    float xv[8];
    load8(x + i, xv);
#pragma unroll
    for (int r = 0; r < 4; r++) {
      if (row0 + r >= n) break;
      float wv[8];
      load8(W + (size_t)(row0 + r) * k + i, wv);
      float d0 = 0.0f, d1 = 0.0f;
#pragma unroll
      for (int j = 0; j < 4; j++) d0 += xv[j] * wv[j];
#pragma unroll
      for (int j = 4; j < 8; j++) d1 += xv[j] * wv[j];
      acc[r] += d0 + d1;
      if (PLANT == 2) acc[r] = __half2float(__float2half_rn(acc[r]));
    }
  }
#pragma unroll
  for (int r = 0; r < 4; r++) {
    const float s = warp_sum(acc[r]);
    if (lane == 0 && row0 + r < n) y[row0 + r] = to(s);
  }
}

}  // namespace tf_uocr
