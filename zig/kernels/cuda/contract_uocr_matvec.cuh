// Controls for uocr.cu's matvec group: each kernel with one break its gate must catch.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace tf_ctl_uocr_matvec {

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ __forceinline__ void load8(const bf16* p, float* out) {
  const uint4 u = *reinterpret_cast<const uint4*>(p);
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 f = __bfloat1622float2(h[i]);
    out[2 * i] = f.x;
    out[2 * i + 1] = f.y;
  }
}

// tf_uocr_matvec; SHORT: the last 8 inputs never read, HALF: the running sums kept in half.
template <bool SHORT, bool HALF>
__device__ __forceinline__ void matvec(const bf16* x, const bf16* W, bf16* y, int n, int k) {
  const int sg = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row0 = (blockIdx.x * 8 + sg) * 4;
  if (row0 >= n) return;
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  __half hacc[4] = {__float2half(0.0f), __float2half(0.0f), __float2half(0.0f), __float2half(0.0f)};
  for (int i = lane * 8; SHORT ? i + 8 < k : i < k; i += 256) {
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
      if (HALF)
        hacc[r] = __float2half(__half2float(hacc[r]) + (d0 + d1));
      else
        acc[r] += d0 + d1;
    }
  }
#pragma unroll
  for (int r = 0; r < 4; r++) {
    const float s = warp_sum(HALF ? __half2float(hacc[r]) : acc[r]);
    if (lane == 0 && row0 + r < n) y[row0 + r] = __float2bfloat16_rn(s);
  }
}

struct UdExpert {
  const bf16* gate;
  const bf16* up;
  const bf16* down;
};

// tf_uocr_expert_gu; SLOT0: every slot reads slot 0's expert, NO_POINT: silu left in fp32.
template <bool SLOT0, bool NO_POINT>
__device__ __forceinline__ void expert_gu(const bf16* x, const UdExpert* tab, const unsigned* ids, bf16* act, int ffn) {
  const int s = blockIdx.y, sg = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const UdExpert e = tab[ids[SLOT0 ? 0 : s]];
  const int row0 = (blockIdx.x * 8 + sg) * 4;
  if (row0 >= ffn) return;
  float g[4] = {0.0f, 0.0f, 0.0f, 0.0f}, u[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int i = lane * 8; i < 1280; i += 256) {
    float xv[8];
    load8(x + i, xv);
#pragma unroll
    for (int r = 0; r < 4; r++) {
      float wg[8], wu[8];
      load8(e.gate + (size_t)(row0 + r) * 1280 + i, wg);
      load8(e.up + (size_t)(row0 + r) * 1280 + i, wu);
      float g0 = 0.0f, g1 = 0.0f, u0 = 0.0f, u1 = 0.0f;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        g0 += xv[j] * wg[j];
        u0 += xv[j] * wu[j];
      }
#pragma unroll
      for (int j = 4; j < 8; j++) {
        g1 += xv[j] * wg[j];
        u1 += xv[j] * wu[j];
      }
      g[r] += g0 + g1;
      u[r] += u0 + u1;
    }
  }
#pragma unroll
  for (int r = 0; r < 4; r++) {
    const float gs = bf(warp_sum(g[r])), us = bf(warp_sum(u[r]));
    const float sl = gs / (1.0f + expf(-gs));
    if (lane == 0) act[(size_t)s * ffn + row0 + r] = __float2bfloat16_rn((NO_POINT ? sl : bf(sl)) * us);
  }
}

// tf_uocr_expert_down with every slot on slot 0's expert.
__device__ __forceinline__ void expert_down_slot0(const bf16* act, const UdExpert* tab, const unsigned* ids, bf16* slots, int ffn) {
  const int s = blockIdx.y, sg = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const UdExpert e = tab[ids[0]];
  const int row0 = (blockIdx.x * 8 + sg) * 4;
  if (row0 >= 1280) return;
  const bf16* a = act + (size_t)s * ffn;
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int i = lane * 8; i < ffn; i += 256) {
    float av[8];
    load8(a + i, av);
#pragma unroll
    for (int r = 0; r < 4; r++) {
      float wv[8];
      load8(e.down + (size_t)(row0 + r) * ffn + i, wv);
      float d0 = 0.0f, d1 = 0.0f;
#pragma unroll
      for (int j = 0; j < 4; j++) d0 += av[j] * wv[j];
#pragma unroll
      for (int j = 4; j < 8; j++) d1 += av[j] * wv[j];
      acc[r] += d0 + d1;
    }
  }
#pragma unroll
  for (int r = 0; r < 4; r++) {
    const float v = warp_sum(acc[r]);
    if (lane == 0) slots[(size_t)s * 1280 + row0 + r] = __float2bfloat16_rn(v);
  }
}

}  // namespace tf_ctl_uocr_matvec

extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_matvec_short_k(const __nv_bfloat16* x, const __nv_bfloat16* W,
                                                                             __nv_bfloat16* y, int n, int k) {
  tf_ctl_uocr_matvec::matvec<true, false>(x, W, y, n, k);
}

extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_matvec_half_sums(const __nv_bfloat16* x, const __nv_bfloat16* W,
                                                                               __nv_bfloat16* y, int n, int k) {
  tf_ctl_uocr_matvec::matvec<false, true>(x, W, y, n, k);
}

extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_expert_gu_slot0(const __nv_bfloat16* x, const tf_ctl_uocr_matvec::UdExpert* tab,
                                                                              const unsigned* ids, __nv_bfloat16* act, int ffn) {
  tf_ctl_uocr_matvec::expert_gu<true, false>(x, tab, ids, act, ffn);
}

extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_expert_gu_no_point(const __nv_bfloat16* x, const tf_ctl_uocr_matvec::UdExpert* tab,
                                                                                 const unsigned* ids, __nv_bfloat16* act, int ffn) {
  tf_ctl_uocr_matvec::expert_gu<false, true>(x, tab, ids, act, ffn);
}

extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_expert_down_slot0(const __nv_bfloat16* act, const tf_ctl_uocr_matvec::UdExpert* tab,
                                                                                const unsigned* ids, __nv_bfloat16* slots, int ffn) {
  tf_ctl_uocr_matvec::expert_down_slot0(act, tab, ids, slots, ffn);
}
