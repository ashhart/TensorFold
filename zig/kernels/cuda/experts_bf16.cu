// Ours: grouped bf16 experts on mma-fragment blocks; a row's sums chain in input order, whatever its tile or form.

#include <cuda_bf16.h>
#include <stdint.h>

#include "experts.cuh"

namespace tf_bf16_experts {

constexpr int WV = 4;  // uint4 a lane a block

__device__ __forceinline__ uint4 ld_nc(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p));
  return r;
}

// RT row tiles of 16 pairs share each block's fragments; a unit is one item's 32-column block.
template <int EPI, int RT, int WARPS>
__device__ __forceinline__ void body(const __nv_bfloat16* __restrict__ X, int x_stride, int slots,
                                     const uint4* __restrict__ W, int KG, int NB, const int* __restrict__ items,
                                     const int* __restrict__ counts, const int* __restrict__ members,
                                     void* __restrict__ out, int N, int skip) {
  const int lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3;
  const int units = __ldg(counts) * NB;
  for (int unit = blockIdx.x * WARPS + (threadIdx.x >> 5); unit < units; unit += gridDim.x * WARPS) {
    const int it = unit / NB, cb = unit - it * NB;
    const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    if (e == skip) continue;
    const uint4* blk = W + ((size_t)e * NB + cb) * (size_t)KG * (32 * WV) + lane;
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
      float acc[1][RT][NTW][4];
#pragma unroll
      for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int j = 0; j < NTW; ++j) acc[0][r][j][0] = acc[0][r][j][1] = acc[0][r][j][2] = acc[0][r][j][3] = 0.f;
      const uint2 zero = make_uint2(0u, 0u);
      for (int g = 0; g < KG; ++g) {
        uint4 w[NTW];
#pragma unroll
        for (int j = 0; j < NTW; ++j) w[j] = ld_nc(blk + (size_t)g * (32 * WV) + 32 * j);
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
            for (int j = 0; j < NTW; ++j)
              mma(acc[0][r][j], xa[q].x, xb[q].x, xa[q].y, xb[q].y, q ? w[j].z : w[j].x, q ? w[j].w : w[j].y);
        }
      }
#pragma unroll
      for (int r = 0; r < RT; ++r)
        epilogue<EPI, 1, RT>(acc, r, out, N, cb * COLS + 2 * t, pr[r][0], pr[r][1], v[r][0], v[r][1], 0.f);
    }
  }
}

}  // namespace tf_bf16_experts

// Decode (items of up to 16 pairs) and prompt (up to 64) entries per epilogue: the NVFP4 entries' arguments, no scales.
#define TF_BF16_EXPERTS(NAME, EPI, RT, WARPS)                                                                         \
  extern "C" __global__ void __launch_bounds__(WARPS * 32)                                                           \
      NAME(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W, int KG, int NB, \
           const int* __restrict__ items, const int* __restrict__ counts, const int* __restrict__ members,           \
           void* __restrict__ out, int N, int skip) {                                                                \
    tf_bf16_experts::body<EPI, RT, WARPS>(X, x_stride, slots, W, KG, NB, items, counts, members, out, N, skip);     \
  }
TF_BF16_EXPERTS(tf_bf16_experts_up, 1, 1, 4)
TF_BF16_EXPERTS(tf_bf16_experts_down_f32, 0, 1, 4)
TF_BF16_EXPERTS(tf_bf16_experts_down_bf16, 3, 1, 4)
TF_BF16_EXPERTS(tf_bf16_experts_prompt_up, 1, 4, 4)
TF_BF16_EXPERTS(tf_bf16_experts_prompt_down_f32, 0, 4, 4)
TF_BF16_EXPERTS(tf_bf16_experts_prompt_down_bf16, 3, 4, 4)

// The repack, one thread a uint4: matrix e's [k][n] (in_out, x @ W) or [n][k] (nn.Linear) rows into its blocks.
extern "C" __global__ void tf_bf16_experts_pack(const __nv_bfloat16* __restrict__ src, long long e_stride, int ld,
                                                int n0, int k0, int in_out, int n, int k, uint4* __restrict__ dst) {
  const int e = blockIdx.y;
  const long long per = (long long)(n / 32) * (k / 32) * 32 * tf_bf16_experts::WV;
  const long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= per) return;
  const int lane = (int)(q % 32), j = (int)(q / 32 % tf_bf16_experts::WV);
  const long long b = q / (32 * tf_bf16_experts::WV);
  const int kg = k / 32, g = (int)(b % kg), cb = (int)(b / kg);
  const int row = n0 + cb * 32 + 8 * j + (lane >> 2), col = k0 + g * 32 + 4 * (lane & 3);
  const __nv_bfloat16* s = src + e * e_stride;
  uint16_t h[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int c = col + (i & 3) + 16 * (i >> 2);
    const __nv_bfloat16 v = in_out ? s[(size_t)c * ld + row] : s[(size_t)row * ld + c];
    h[i] = __bfloat16_as_ushort(v);
  }
  uint4 o;
  o.x = h[0] | (uint32_t)h[1] << 16;
  o.y = h[2] | (uint32_t)h[3] << 16;
  o.z = h[4] | (uint32_t)h[5] << 16;
  o.w = h[6] | (uint32_t)h[7] << 16;
  dst[e * per + q] = o;
}
