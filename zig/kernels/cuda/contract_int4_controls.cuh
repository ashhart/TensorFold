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
