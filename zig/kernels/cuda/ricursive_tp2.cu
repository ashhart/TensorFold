// Ripple's own glue at two ranks: embedding scale, value residual, gate, clamped SwiGLU, post-norm add, rank sum, taps.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include "common/reduce.cuh"

using tf_common::warp_sum;

// x [rows, width] fp32 = bf16(e * scale) of each id's bf16 row (the release scales its bf16 rows); seed = fp16(x).
extern "C" __global__ void __launch_bounds__(256) tf_rtp2_embed(const int* __restrict__ ids, const __nv_bfloat16* __restrict__ table,
                                                                  float* __restrict__ x, __half* __restrict__ seed, int width,
                                                                  float scale) {
  const int64_t r = blockIdx.x;
  const __nv_bfloat16* e = table + (int64_t)ids[r] * width;
  for (int i = threadIdx.x; i < width; i += 256) {
    const float v = __bfloat162float(__float2bfloat16_rn(__fmul_rn(__bfloat162float(e[i]), scale)));
    x[r * width + i] = v;
    seed[r * width + i] = __float2half_rn(v);
  }
}

// out = lerp(seed, y, w) as torch.lerp rounds it (fp32 opmath, the end-anchored form past w = 0.5), fp16 in and out.
extern "C" __global__ void __launch_bounds__(256) tf_rtp2_lerp(const __half* __restrict__ seed, const __half* __restrict__ y,
                                                                 __half* __restrict__ out, int64_t n, float w) {
  const int64_t i = (int64_t)blockIdx.x * 256 + threadIdx.x;
  if (i >= n) return;
  const float s = __half2float(seed[i]), e = __half2float(y[i]);
  const float d = __fsub_rn(e, s);
  const float v = w < 0.5f ? __fadd_rn(s, __fmul_rn(w, d)) : __fsub_rn(e, __fmul_rn(d, __fsub_rn(1.0f, w)));
  out[i] = __float2half_rn(v);
}

// out = fp16(o * sigmoid(g)), the full gate in fp32 over the attention's bf16 output and g_proj's fp16 rows.
extern "C" __global__ void __launch_bounds__(256) tf_rtp2_gate(const __nv_bfloat16* __restrict__ o, const __half* __restrict__ g,
                                                                 __half* __restrict__ out, int64_t n) {
  const int64_t i = (int64_t)blockIdx.x * 256 + threadIdx.x;
  if (i >= n) return;
  const float gv = __half2float(g[i]);
  const float sg = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-gv)));
  out[i] = __float2half_rn(__fmul_rn(__bfloat162float(o[i]), sg));
}

// out = fp16(silu(min(g, limit)) * clamp(u, -limit, limit)) in fp32; limit 0: no clamp.
extern "C" __global__ void __launch_bounds__(256) tf_rtp2_swiglu(const __half* __restrict__ g, const __half* __restrict__ u,
                                                                   __half* __restrict__ out, int64_t n, float limit) {
  const int64_t i = (int64_t)blockIdx.x * 256 + threadIdx.x;
  if (i >= n) return;
  float gv = __half2float(g[i]), uv = __half2float(u[i]);
  if (limit > 0.0f) gv = fminf(gv, limit), uv = fminf(fmaxf(uv, -limit), limit);
  const float s = __fmul_rn(gv, __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-gv))));
  out[i] = __float2half_rn(__fmul_rn(s, uv));
}

// x [rows, width] += rms(a) * w, all fp32 (PeriNorm's post-attention and post-FFN norms into the residual).
extern "C" __global__ void __launch_bounds__(256) tf_rtp2_norm_add(const float* __restrict__ A, const float* __restrict__ W,
                                                                     float* __restrict__ X, int width, float eps) {
  __shared__ float part[8];
  const float* a = A + (int64_t)blockIdx.x * width;
  float* x = X + (int64_t)blockIdx.x * width;
  float s = 0.0f;
  for (int i = threadIdx.x; i < width; i += 256) s = __fadd_rn(s, __fmul_rn(a[i], a[i]));
  s = warp_sum(s);
  if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = s;
  __syncthreads();
  float total = 0.0f;
#pragma unroll
  for (int k = 0; k < 8; ++k) total = __fadd_rn(total, part[k]);
  const float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(total, (float)width), eps)));
  for (int i = threadIdx.x; i < width; i += 256) x[i] = __fadd_rn(x[i], __fmul_rn(__fmul_rn(a[i], inv), W[i]));
}

// out [n] = parts[0][i] + parts[1][i] + ... in rank order: two ranks' fp32 o_proj or down_proj partials, added once.
extern "C" __global__ void __launch_bounds__(256) tf_rtp2_rank_sum(const float* __restrict__ parts, float* __restrict__ out,
                                                                     int64_t n, int world) {
  const int64_t i = (int64_t)blockIdx.x * 256 + threadIdx.x;
  if (i >= n) return;
  float v = parts[i];
  for (int r = 1; r < world; ++r) v = __fadd_rn(v, parts[r * n + i]);
  out[i] = v;
}

// dst [groups, gd, 128] bf16 = the first g heads of each src [groups, gs, 128] group (16-byte moves; pads kept).
extern "C" __global__ void __launch_bounds__(256) tf_rtp2_heads(const uint4* __restrict__ src, uint4* __restrict__ dst, int64_t n,
                                                                  int g, int gs, int gd) {
  const int64_t i = (int64_t)blockIdx.x * 256 + threadIdx.x;
  if (i >= n) return;
  const int64_t c = i % 16, rest = i / 16, j = rest % g, grp = rest / g;
  dst[(grp * gd + j) * 16 + c] = src[(grp * gs + j) * 16 + c];
}

// Row r's fp32 residual into RING[slot[r]][pos[r] % TR][tap] as bf16 (rows of ntaps * H): a drafter's taps. Grid (rows).
extern "C" __global__ void __launch_bounds__(256) tf_rtp2_taps(const float* __restrict__ X, const int* __restrict__ POS, const int* __restrict__ SLOT,
                                                             __nv_bfloat16* __restrict__ RING, int TR, int ntaps, int tap, int H) {
  const int r = blockIdx.x;
  const long long row = (long long)SLOT[r] * TR + POS[r] % TR;
  __nv_bfloat16* dst = RING + (row * ntaps + tap) * H;
  const float* x = X + (long long)r * H;
  for (int i = threadIdx.x; i < H; i += blockDim.x) dst[i] = __float2bfloat16_rn(x[i]);
}
