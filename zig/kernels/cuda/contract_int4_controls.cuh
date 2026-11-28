// Gate 4's control for fn_int4.cu: the routed down with each group's sum rounded to bf16 before its scale.
#pragma once

extern "C" __global__ void tf_ctl_int4_bf16_sums(const __nv_bfloat16* X, int x_stride, int slots, const uint32_t* W,
                                                const __half* S, int K, int N, const int* items,
                                                const int* counts, const int* members, int rows, float* out,
                                                int out_stride, float limit, int skip) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= N) return;
  const int KG = K / 128, nt = col / 8, lane = 4 * (col % 8);
  const size_t mat_words = (size_t)N * K / 8, mat_scales = (size_t)N * KG;
  for (int it = 0; it < __ldg(counts); ++it) {
    const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    if (e == skip) continue;
    const uint32_t* w = W + (size_t)e * mat_words;
    const __half* s = S + (size_t)e * mat_scales;
    for (int i = 0; i < cnt; ++i) {
      const int p = __ldg(members + first + i);
      const int xr = slots ? p / slots : p;
      float acc = 0.0f;
      for (int g = 0; g < KG; ++g) {
        float sum = 0.0f;
        for (int b = 0; b < 4; ++b)
          for (int t = 0; t < 4; ++t) {
            const uint32_t word = w[((size_t)(nt * KG + g) * 32 + lane + t) * 4 + b];
            for (int j = 0; j < 8; ++j) {
              const int nib = (j % 2 == 0) ? j / 2 : j / 2 + 4;
              const int code = (word >> (4 * nib)) & 0xF;
              const float x = tf_ctl_x(X, (int64_t)xr * x_stride + g * 128 + b * 32 + 8 * t + j);
              sum += (float)(code - 8) * x;
            }
          }
        const float sc = __half2float(s[(size_t)(nt * KG + g) * 8 + (col % 8)]);
        acc = fmaf(__bfloat162float(__float2bfloat16_rn(sum)), sc, acc);
      }
      out[(size_t)p * out_stride + col] = acc;
    }
  }
}

__device__ __forceinline__ float tf_ctl_int4_rn(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__device__ __forceinline__ float tf_ctl_int4_rz(float x) { return __bfloat162float(__float2bfloat16_rz(x)); }

// One matrix's column `col` for the input row at `x`: each group's 128 products in input order, then its scale.
__device__ __forceinline__ float tf_ctl_int4_sum(const __nv_bfloat16* X, int64_t x, const uint32_t* w, const __half* s,
                                                   int col, int KG, int fault) {
  const int nt = col / 8, lane = 4 * (col % 8);
  float acc = 0.0f;
  for (int g = 0; g < KG; ++g) {
    float sum = 0.0f;
    for (int b = 0; b < 4; ++b)
      for (int t = 0; t < 4; ++t) {
        const uint32_t word = w[((size_t)(nt * KG + g) * 32 + lane + t) * 4 + b];
        for (int j = 0; j < 8; ++j) {
          const int nib = (j % 2 == 0) ? j / 2 : j / 2 + 4;
          const int code = (word >> (4 * nib)) & 0xF;
          const float xv = tf_ctl_x(X, x + g * 128 + b * 32 + 8 * t + j);
          sum += (float)(code - 8) * xv;
        }
      }
    const float sc = __half2float(s[(size_t)(nt * KG + g) * 8 + (col % 8)]);
    acc = fmaf(fault == 3 ? tf_ctl_int4_rn(sum) : sum, sc, acc);
  }
  return acc;
}

// The routed gate/up as one thread an output and a block row a pair: fn_int4.cu's swiglu at limit 0 on two fp32 sums.

// fault 1: silu of the up sum times the gate sum; 2: gate, up and silu cut toward zero; 3: each group's sum in bf16.
extern "C" __global__ void tf_ctl_int4_swiglu(const __nv_bfloat16* X, int x_stride, int slots, const uint32_t* W,
                                               const __half* S, int K, int N, const int* items,
                                               const int* counts, const int* members, __nv_bfloat16* out,
                                               int out_stride, int fault) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x, p = blockIdx.y;
  if (col >= N) return;
  int e = -1, row = -1;
  for (int it = 0; it < __ldg(counts); ++it) {
    const int first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    if (p >= first && p < first + cnt) {
      e = __ldg(items + 3 * it);
      row = __ldg(members + p);
    }
  }
  if (e < 0) return;
  const int KG = K / 128;
  const size_t mat_words = (size_t)N * K / 8, mat_scales = (size_t)N * KG;
  const int64_t x = (int64_t)(slots ? row / slots : row) * x_stride;
  const int gate_matrix = fault == 1 ? 1 : 0;
  const float gate = tf_ctl_int4_sum(X, x, W + ((size_t)2 * e + gate_matrix) * mat_words,
                                     S + ((size_t)2 * e + gate_matrix) * mat_scales, col, KG, fault);
  const float up = tf_ctl_int4_sum(X, x, W + ((size_t)2 * e + 1 - gate_matrix) * mat_words,
                                   S + ((size_t)2 * e + 1 - gate_matrix) * mat_scales, col, KG, fault);
  const float gv = fault == 2 ? tf_ctl_int4_rz(gate) : tf_ctl_int4_rn(gate);
  const float uv = fault == 2 ? tf_ctl_int4_rz(up) : tf_ctl_int4_rn(up);
  const float act = gv / (1.0f + expf(-gv));
  out[(size_t)row * out_stride + col] =
      __float2bfloat16_rn((fault == 2 ? tf_ctl_int4_rz(act) : tf_ctl_int4_rn(act)) * uv);
}
