// Ours: grouped MXFP4 experts read as stored, each weight decoded to bf16 in registers, then experts_bf16's chain.

#include <cuda_bf16.h>
#include <stdint.h>

#include "common/fp_pairs.cuh"
#include "experts.cuh"

namespace tf_mxfp4_experts {

// A row's weights at columns c..c+3 and c+16..c+19 (c = 32g + 4t), as experts_bf16's pack lays them, exact in bf16.
__device__ __forceinline__ uint4 weights(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales, int g, int t) {
  const uint32_t lo = __ldg(reinterpret_cast<const uint16_t*>(codes + 16 * g + 2 * t));
  const uint32_t hi = __ldg(reinterpret_cast<const uint16_t*>(codes + 16 * g + 8 + 2 * t));
  const uint32_t s = __ldg(scales + g), sb = s ? s << 7 : 0x40u;  // bf16 2^(s - 127); 0 is 2^-127
  const uint32_t scale = sb | sb << 16;
  const uint32_t a = lo | lo << 12, b = hi | hi << 12;  // nibbles 0, 2 in the low half, 1, 3 in the high half
  uint4 w;
  w.x = tf_common::mul2(tf_common::fp4pair(a, 0), scale);
  w.y = tf_common::mul2(tf_common::fp4pair(a, 8), scale);
  w.z = tf_common::mul2(tf_common::fp4pair(b, 0), scale);
  w.w = tf_common::mul2(tf_common::fp4pair(b, 8), scale);
  return w;
}

// experts_bf16's body with W read as stored: codes [E][M][N][K/2] bytes, scales [E][M][N][K/32] E8M0 (M 2: gate, up).
template <int EPI, int M, int RT, int WARPS>
__device__ __forceinline__ void body(const __nv_bfloat16* __restrict__ X, int x_stride, int slots,
                                     const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales, int KG,
                                     int NB, const int* __restrict__ items, const int* __restrict__ counts,
                                     const int* __restrict__ members, void* __restrict__ out, int N, int skip,
                                     float beta, float lin) {
  const int lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3;
  const int units = __ldg(counts) * NB;
  for (int unit = blockIdx.x * WARPS + (threadIdx.x >> 5); unit < units; unit += gridDim.x * WARPS) {
    const int it = unit / NB, cb = unit - it * NB;
    const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    if (e == skip) continue;
    const uint8_t* wc[M][NTW];
    const uint8_t* ws[M][NTW];
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
      for (int j = 0; j < NTW; ++j) {
        const size_t row = ((size_t)e * M + m) * N + cb * COLS + 8 * j + gq;
        wc[m][j] = codes + row * (size_t)KG * 16;
        ws[m][j] = scales + row * (size_t)KG;
      }
    for (int r0 = 0; r0 < cnt; r0 += 16 * RT) {
      bool v[RT][2];
      int pr[RT][2];
      const __nv_bfloat16* x[RT][2];
#pragma unroll
      for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int i = r0 + 16 * r + gq + 8 * h;
          v[r][h] = i < cnt;
          pr[r][h] = v[r][h] ? __ldg(members + first + i) : 0;
          x[r][h] = X + (size_t)(slots ? pr[r][h] / slots : pr[r][h]) * x_stride + 4 * t;
        }
      float acc[M][RT][NTW][4];
#pragma unroll
      for (int m = 0; m < M; ++m)
#pragma unroll
        for (int r = 0; r < RT; ++r)
#pragma unroll
          for (int j = 0; j < NTW; ++j) acc[m][r][j][0] = acc[m][r][j][1] = acc[m][r][j][2] = acc[m][r][j][3] = 0.f;
      const uint2 zero = make_uint2(0u, 0u);
      for (int g = 0; g < KG; ++g) {
        uint4 w[M][NTW];
#pragma unroll
        for (int m = 0; m < M; ++m)
#pragma unroll
          for (int j = 0; j < NTW; ++j) w[m][j] = weights(wc[m][j], ws[m][j], g, t);
#pragma unroll
        for (int r = 0; r < RT; ++r) {
          uint2 xa[2], xb[2];
#pragma unroll
          for (int q = 0; q < 2; ++q) {
            xa[q] = v[r][0] ? __ldg(reinterpret_cast<const uint2*>(x[r][0] + 32 * g + 16 * q)) : zero;
            xb[q] = v[r][1] ? __ldg(reinterpret_cast<const uint2*>(x[r][1] + 32 * g + 16 * q)) : zero;
          }
#pragma unroll
          for (int q = 0; q < 2; ++q)
#pragma unroll
            for (int m = 0; m < M; ++m)
#pragma unroll
              for (int j = 0; j < NTW; ++j)
                mma(acc[m][r][j], xa[q].x, xb[q].x, xa[q].y, xb[q].y, q ? w[m][j].z : w[m][j].x,
                    q ? w[m][j].w : w[m][j].y);
        }
      }
#pragma unroll
      for (int r = 0; r < RT; ++r) {
        if constexpr (M == 2)
          epilogue_situ<RT>(acc, r, out, N, cb * COLS + 2 * t, pr[r][0], pr[r][1], v[r][0], v[r][1], beta, lin);
        else
          epilogue<EPI, 1, RT>(acc, r, out, N, cb * COLS + 2 * t, pr[r][0], pr[r][1], v[r][0], v[r][1], 0.f);
      }
    }
  }
}

}  // namespace tf_mxfp4_experts

// The two downs' decode (items of up to 16 pairs) and prompt (up to 64) entries: experts_bf16's, codes and scales.
#define TF_MXFP4_EXPERTS(NAME, EPI, RT, WARPS)                                                                      \
  extern "C" __global__ void __launch_bounds__(WARPS * 32)                                                         \
      NAME(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint8_t* __restrict__ codes,        \
           const uint8_t* __restrict__ scales, int KG, int NB, const int* __restrict__ items,                      \
           const int* __restrict__ counts, const int* __restrict__ members, void* __restrict__ out, int N,         \
           int skip) {                                                                                              \
    tf_mxfp4_experts::body<EPI, 1, RT, WARPS>(X, x_stride, slots, codes, scales, KG, NB, items, counts, members,    \
                                              out, N, skip, 0.f, 0.f);                                              \
  }
TF_MXFP4_EXPERTS(tf_mxfp4_experts_down_f32, 0, 1, 4)
TF_MXFP4_EXPERTS(tf_mxfp4_experts_down_bf16, 3, 1, 4)
TF_MXFP4_EXPERTS(tf_mxfp4_experts_prompt_down_f32, 0, 4, 4)
TF_MXFP4_EXPERTS(tf_mxfp4_experts_prompt_down_bf16, 3, 4, 4)

// The gated up's entries: gate and up as one stored pair, SiTU into bf16 rows; prompt tiles of 32 (two sums a column).
#define TF_MXFP4_EXPERTS_UP(NAME, RT, WARPS)                                                                        \
  extern "C" __global__ void __launch_bounds__(WARPS * 32)                                                          \
      NAME(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint8_t* __restrict__ codes,         \
           const uint8_t* __restrict__ scales, int KG, int NB, const int* __restrict__ items,                       \
           const int* __restrict__ counts, const int* __restrict__ members, void* __restrict__ out, int N,          \
           int skip, float beta, float lin) {                                                                       \
    tf_mxfp4_experts::body<4, 2, RT, WARPS>(X, x_stride, slots, codes, scales, KG, NB, items, counts, members,      \
                                            out, N, skip, beta, lin);                                               \
  }
TF_MXFP4_EXPERTS_UP(tf_mxfp4_experts_up_situ, 1, 4)
TF_MXFP4_EXPERTS_UP(tf_mxfp4_experts_prompt_up_situ, 2, 4)
