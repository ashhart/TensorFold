// Decode HC tiles retain their weights while visiting every row; each row keeps partial8's operations and order.
#pragma once

namespace glm_hc {

template <bool WrongRow = false>
__device__ __forceinline__ void partial_rows(const __nv_bfloat16* x, const __nv_bfloat16* fn, float* part,
                                            int64_t wide, int nb, int64_t rows) {
  __shared__ float dots[8][24];
  __shared__ float smem[4];
  const int64_t b = blockIdx.y, kb = wide / nb;
  const int grp = threadIdx.x >> 7, t = threadIdx.x & 127, c = t & 15, m0 = (t >> 4) & 7;
  const int steps = static_cast<int>(kb / 128);
  const int64_t base = b * kb + grp * 128;
  uint4 weights[3];
  if (grp < steps) {
#pragma unroll
    for (int i = 0; i < 3; ++i)
      weights[i] = __ldg(reinterpret_cast<const uint4*>(fn + (int64_t)(m0 + 8 * i) * wide + base + c * 8));
  }
  for (int64_t q = 0; q < rows; ++q) {
    const int64_t r = q + (WrongRow ? 1 : 0);
    if (grp < steps) {
      float xv[8];
      {
        const uint4 u = *reinterpret_cast<const uint4*>(x + r * wide + base + c * 8);
        const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&u);
        for (int j = 0; j < 8; ++j) xv[j] = glm_triton::bf(h[j]);
      }
#pragma unroll
      for (int i = 0; i < 3; ++i) {
        const __nv_bfloat16* w = reinterpret_cast<const __nv_bfloat16*>(&weights[i]);
        float d = __fmul_rn(xv[1], glm_triton::bf(w[1]));
        d = __fmaf_rn(xv[0], glm_triton::bf(w[0]), d);
        for (int j = 2; j < 8; ++j) d = __fmaf_rn(xv[j], glm_triton::bf(w[j]), d);
        for (int o = 8; o > 0; o >>= 1) d = d + __shfl_xor_sync(0xffffffffu, d, o);
        if (c == 0) dots[grp][m0 + 8 * i] = d;
      }
    }
    if (grp == 0) {
      float ss = 0.0f;
      for (int s = 0; s < steps; ++s) {
        const float xt = glm_triton::bf(x[r * wide + b * kb + s * 128 + t]);
        ss = __fmaf_rn(xt, xt, ss);
      }
      for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
      if ((t & 31) == 0) smem[t >> 5] = ss;
    }
    __syncthreads();
    float* out = part + (q * nb + b) * 32;
    if (threadIdx.x < 24) {
      float acc = 0.0f;
      for (int s = 0; s < steps; ++s) acc = acc + dots[s][threadIdx.x];
      out[threadIdx.x] = acc;
    } else if (threadIdx.x == 32) {
      out[24] = (smem[0] + smem[2]) + (smem[1] + smem[3]);
    }
    __syncthreads();  // all readers finish before the next row reuses the shared partials
  }
}

}  // namespace glm_hc

extern "C" __global__ void __launch_bounds__(1024)
    tf_glm_hc_partial8_rows(const __nv_bfloat16* x, const __nv_bfloat16* fn, float* part,
                            int64_t wide, int nb, int64_t rows) {
  glm_pdl::enter();
  glm_hc::partial_rows(x, fn, part, wide, nb, rows);
}

// Contract control: only the input row changes. The fixture supplies one extra guarded row.
extern "C" __global__ void __launch_bounds__(1024)
    tf_glm_hc_partial8_rows_control(const __nv_bfloat16* x, const __nv_bfloat16* fn, float* part,
                                    int64_t wide, int nb, int64_t rows) {
  glm_pdl::enter();
  glm_hc::partial_rows<true>(x, fn, part, wide, nb, rows);
}
