// Ripple's attention prep (Metal ricursive/glue.metal's rc_prep) step for step; `keep`: its bf16 rounding points.
#pragma once

#include <cuda_bf16.h>
#include <stddef.h>
#include "common/reduce.cuh"

// tf_rcp, not tf_rc_prep: the kernel's extern "C" symbol takes that name.
namespace tf_rcp {

typedef __nv_bfloat16 bf16;

// Head dim 128, a lane's dims lane + 32 j; a warp a (head, row), 8 warps a block.
constexpr int D = 128, HALF = 64, PER = D / 32, WARPS = 8, NT = 32 * WARPS;

// rc_prep's rounding points: x r, w times that, cos, sin, and the stored output (dropped: truncated, not rounded).
enum : int { NORM_ROW = 1, NORM_OUT = 2, COS = 4, SIN = 8, OUT = 16 };
constexpr int ALL = 31;

__device__ __forceinline__ float bf(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

// v rounded to bf16 where `keep` holds the point p (a constant in the served kernel, so the test folds away).
__device__ __forceinline__ float point(float v, int keep, int p) { return (keep & p) != 0 ? bf(v) : v; }

// The output as stored: to nearest even, or its fp32 bits' top half where `keep` drops OUT.
__device__ __forceinline__ bf16 store(float v, int keep) {
  return (keep & OUT) != 0 ? __float2bfloat16_rn(v) : __float2bfloat16_rz(v);
}

// One projection's output rows: its first row and the elements between rows (V is its own projection).
struct Rows {
  const bf16* x;
  int stride;
  __device__ __forceinline__ const bf16* at(int row, int head) const { return x + (size_t)row * stride + (size_t)head * D; }
};

// A warp's unit of row `row` at position p: query head `unit` into Q, else key head unit - heads and its value at `at`.
__device__ __forceinline__ void unit_at(const Rows XQ, const Rows XK, const Rows XV, const float* __restrict__ inv,
                                        const bf16* __restrict__ qw, const bf16* __restrict__ kw, bf16* __restrict__ Q,
                                        bf16* __restrict__ KC, bf16* __restrict__ VC, int heads, int unit, int lane, int row,
                                        int p, size_t at, int rope, float eps, int keep) {
  const bool q = unit < heads;
  const int head = q ? unit : unit - heads;
  const bf16* src = q ? XQ.at(row, head) : XK.at(row, head);
  const bf16* w = q ? qw : kw;
  bf16* y = q ? Q + ((size_t)row * heads + head) * D : KC + at + (size_t)head * D;
  float x[PER];
  float sum = 0.0f;  // bf16(w bf16(x r)), r = rsqrt(sum x^2 / 128 + eps); intrinsics and fma: the same under any flags
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    x[j] = __bfloat162float(src[lane + 32 * j]);
    sum = fmaf(x[j], x[j], sum);
  }
  const float r = rsqrtf(__fadd_rn(__fdiv_rn(tf_common::warp_sum(sum), (float)D), eps));
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const float wv = __bfloat162float(w[lane + 32 * j]);
    x[j] = point(__fmul_rn(wv, point(__fmul_rn(x[j], r), keep, NORM_ROW)), keep, NORM_OUT);
  }
  const float position = (float)p;
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const int d = lane + 32 * j;
    float v = x[j];
    if (rope != 0) {
      const float angle = __fmul_rn(position, inv[d % HALF]);
      const float c = point(cosf(angle), keep, COS), s = point(sinf(angle), keep, SIN);
      const float other = j < 2 ? -x[j ^ 2] : x[j ^ 2];  // d < 64 pairs with d + 64 (negated), else with d - 64
      v = __fadd_rn(__fmul_rn(x[j], c), __fmul_rn(other, s));
    }
    y[d] = store(v, keep);
  }
  if (q) return;
  const bf16* vs = XV.at(row, head);
  bf16* vd = VC + at + (size_t)head * D;
#pragma unroll
  for (int j = 0; j < PER; ++j) vd[lane + 32 * j] = vs[lane + 32 * j];  // the value as it is
}

// One stream's rows at pos + row: keys and values at ring slot (pos + row) % ring, or row p % tok of page tab[p / tok].
__device__ __forceinline__ void prep(const Rows XQ, const Rows XK, const Rows XV, const float* __restrict__ inv,
                                     const bf16* __restrict__ qw, const bf16* __restrict__ kw, bf16* __restrict__ Q,
                                     bf16* __restrict__ KC, bf16* __restrict__ VC, int rows, int heads, int kv_heads,
                                     int pos, int ring, int rope, float eps, const int* __restrict__ tab, long long page,
                                     int tok, int keep) {
  const int unit = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.y;
  if (unit >= heads + kv_heads || row >= rows) return;  // whole warps: unit_at's shuffles stay converged
  const size_t kvw = (size_t)kv_heads * D;
  const int p = pos + row;
  // the key's cache row: its ring slot, or its page's row (an address only: the numerics are the ring's)
  const size_t at = tab != nullptr ? (size_t)tab[p / tok] * (size_t)page + (size_t)(p % tok) * kvw
                                    : (size_t)(p % ring) * kvw;
  unit_at(XQ, XK, XV, inv, qw, kw, Q, KC, VC, heads, unit, lane, row, p, at, rope, eps, keep);
}

// Row r at pos[r] of stream slot[r] (null: stream 0): ring row slot * ring + p % ring, or tab[slot * stride + p / tok].
__device__ __forceinline__ void prep_rows(const Rows XQ, const Rows XK, const Rows XV, const float* __restrict__ inv,
                                          const bf16* __restrict__ qw, const bf16* __restrict__ kw, bf16* __restrict__ Q,
                                          bf16* __restrict__ KC, bf16* __restrict__ VC, int rows, int heads, int kv_heads,
                                          const int* __restrict__ pos, const int* __restrict__ slot, int ring, int rope,
                                          float eps, const int* __restrict__ tab, long long page, int tok, int stride,
                                          int keep) {
  const int unit = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.y;
  if (unit >= heads + kv_heads || row >= rows) return;  // whole warps: unit_at's shuffles stay converged
  const size_t kvw = (size_t)kv_heads * D;
  const int p = pos[row];
  const size_t s = slot != nullptr ? (size_t)slot[row] : 0;
  // the key's cache row: its stream's ring slot, or its page's row by the stream's row of the table
  const size_t at = tab != nullptr
                        ? (size_t)tab[s * (size_t)stride + (size_t)(p / tok)] * (size_t)page + (size_t)(p % tok) * kvw
                        : (s * (size_t)ring + (size_t)(p % ring)) * kvw;
  unit_at(XQ, XK, XV, inv, qw, kw, Q, KC, VC, heads, unit, lane, row, p, at, rope, eps, keep);
}

}  // namespace tf_rcp
