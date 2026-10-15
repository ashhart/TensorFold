// Ling-3.1-flash's bf16 glue at the HF model's rounding points: the residual add, the head-wise gate, the MoE sum.
#include <cuda_bf16.h>
#include <stdint.h>

namespace ling_glue {
__device__ __forceinline__ float bf(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float rbf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
}  // namespace ling_glue

// torch's bf16 x + y: the fp32 sum rounded once; out may alias x or y.
extern "C" __global__ void tf_ling_add(const __nv_bfloat16* x, const __nv_bfloat16* y, __nv_bfloat16* out, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __float2bfloat16_rn(__fadd_rn(ling_glue::bf(x[i]), ling_glue::bf(y[i])));
}

// The head-wise gate: o [rows][heads][dv] times bf16(sigmoid(float(g [rows][gs], column h))), rounded to bf16.
extern "C" __global__ void tf_ling_gate(__nv_bfloat16* o, int64_t heads, const __nv_bfloat16* g, int64_t gs) {
  const int64_t r = blockIdx.x, h = blockIdx.y, dv = blockDim.x;
  const float z = ling_glue::bf(g[r * gs + h]);
  const float s = ling_glue::rbf(__fdiv_rn(1.0f, __fadd_rn(1.0f, expf(0.0f - z))));
  __nv_bfloat16* p = o + (r * heads + h) * dv + threadIdx.x;
  *p = __float2bfloat16_rn(__fmul_rn(ling_glue::bf(*p), s));
}

// moe_infer's sum, bf16(sum of float(y) w over k < used) + shared, y [rows][slots][d] or by_pick [experts][rows][d].
extern "C" __global__ void tf_ling_combine(const __nv_bfloat16* y, const int* pick, const float* wts,
                                           const __nv_bfloat16* shared, __nv_bfloat16* out, int64_t rows, int64_t d,
                                           int slots, int used, int by_pick) {
  const int64_t r = blockIdx.x, c = static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
  if (c >= d) return;
  float a = 0.0f;
  for (int k = 0; k < used; ++k) {
    const int64_t at = by_pick ? (static_cast<int64_t>(pick[r * slots + k]) * rows + r) * d + c : (r * slots + k) * d + c;
    a = __fadd_rn(a, __fmul_rn(ling_glue::bf(y[at]), wts[r * slots + k]));
  }
  float v = ling_glue::rbf(a);
  if (shared != nullptr) v = __fadd_rn(v, ling_glue::bf(shared[r * d + c]));
  out[r * d + c] = __float2bfloat16_rn(v);
}
