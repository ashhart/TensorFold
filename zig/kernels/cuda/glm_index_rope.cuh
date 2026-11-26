// The DSA indexer's rope (indexer_rope_interleave): dims (2i, 2i + 1) below `rope` turned by pos inv[i], in place.
#pragma once

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

namespace glm_rope {

// Sine and cosine of pair i's angle pos inv[i], out of line: one copy of sincosf's large-angle path, not one a site.
static __device__ __noinline__ void turn(int64_t pos, const float* inv, int i, float* s, float* c) {
  sincosf(__fmul_rn((float)pos, inv[i]), s, c);
}

// A pair turned by (s, c): (x0 c - x1 s, x1 c + x0 s).
__device__ __forceinline__ void apply(float& x0, float& x1, float s, float c) {
  const float y0 = __fmaf_rn(x0, c, -__fmul_rn(x1, s)), y1 = __fmaf_rn(x1, c, __fmul_rn(x0, s));
  x0 = y0;
  x1 = y1;
}

// Pair i of a key or query at position pos.
__device__ __forceinline__ void pair(float& x0, float& x1, int64_t pos, const float* inv, int i) {
  float s, c;
  turn(pos, inv, i, &s, &c);
  apply(x0, x1, s, c);
}

// A query row's m16n8k16 A fragments (KQ k16 steps, two 16-head tiles), dims below `rope` roped at pos, back to bf16.
template <int KQ>
__device__ __forceinline__ void frags(uint32_t (&aq)[KQ][2][4], int q, int64_t pos, int rope, const float* inv) {
#pragma unroll
  for (int kq = 0; kq < KQ; ++kq)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int d = kq * 16 + h * 8 + 2 * q;
      if (d >= rope) continue;
      float s, c;
      turn(pos, inv, d / 2, &s, &c);
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        uint32_t& w = aq[kq][k >> 1][2 * h + (k & 1)];
        __nv_bfloat16 b[2];
        memcpy(b, &w, 4);
        float x0 = __bfloat162float(b[0]), x1 = __bfloat162float(b[1]);
        apply(x0, x1, s, c);
        b[0] = __float2bfloat16_rn(x0);
        b[1] = __float2bfloat16_rn(x1);
        memcpy(&w, b, 4);
      }
    }
}

}  // namespace glm_rope
