// experts.cu's decode expert kernel over up to 16 * RT pair rows a warp; a row's bits do not depend on its window.

#include <cuda_bf16.h>
#include <stdint.h>

#include "experts.cuh"

namespace tf_experts_rows {

__device__ __forceinline__ uint4 ld_w(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p));
  return r;
}

__device__ __forceinline__ uint4 ld_x(const __nv_bfloat16* p) { return __ldg(reinterpret_cast<const uint4*>(p)); }

// experts.cu's group_sum: a lane's 8 inputs a row in order, then the quad's partners.
template <int XV>
__device__ __forceinline__ float group_sum(const uint4 (&v)[XV]) {
  float s = 0.f;
#pragma unroll
  for (int c = 0; c < XV; ++c)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint32_t u = comp(v[c], i);
      s += lo_f(u);
      s += hi_f(u);
    }
  s += __shfl_xor_sync(0xffffffffu, s, 1);
  s += __shfl_xor_sync(0xffffffffu, s, 2);
  return s;
}

template <int GS, int M, int EPI, int RT, int WARPS>
__global__ void __launch_bounds__(WARPS * 32)
    expert_rows(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W, int KG,
                int NB, const int* __restrict__ items, const int* __restrict__ counts,
                const int* __restrict__ members, void* __restrict__ out, int N, float limit) {
  using G = Geo<GS>;
  const int lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3;
  const int units = __ldg(counts) * NB;
  for (int unit = blockIdx.x * WARPS + (threadIdx.x >> 5); unit < units; unit += gridDim.x * WARPS) {
    const int it = unit / NB, cb = unit - it * NB;
    const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    bool v0[RT], v1[RT];
    int pr0[RT], pr1[RT];
    const __nv_bfloat16* x0[RT];
    const __nv_bfloat16* x1[RT];
    int nt = 0;                                  // row tiles of 16 this item fills
#pragma unroll
    for (int r = 0; r < RT; ++r) {
      v0[r] = 16 * r + gq < cnt;
      v1[r] = 16 * r + gq + 8 < cnt;
      pr0[r] = v0[r] ? __ldg(members + first + 16 * r + gq) : 0;
      pr1[r] = v1[r] ? __ldg(members + first + 16 * r + gq + 8) : 0;
      const int r0 = slots ? pr0[r] / slots : pr0[r], r1 = slots ? pr1[r] / slots : pr1[r];
      x0[r] = X + (size_t)r0 * x_stride + t * (GS / 4);
      x1[r] = X + (size_t)r1 * x_stride + t * (GS / 4);
      if (16 * r < cnt) nt = r + 1;
    }
    const uint4* blk = W + ((size_t)e * NB + cb) * (size_t)KG * (M * G::BLOCK);
    float acc[M][RT][NTW][4];
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
      for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int j = 0; j < NTW; ++j) acc[m][r][j][0] = acc[m][r][j][1] = acc[m][r][j][2] = acc[m][r][j][3] = 0.f;
    const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
    for (int g = 0; g < KG; ++g) {
      const uint4* b = blk + (size_t)g * (M * G::BLOCK);
      uint4 w[M][G::WV], sb[M][G::SBV];
#pragma unroll
      for (int m = 0; m < M; ++m) {
#pragma unroll
        for (int c = 0; c < G::WV; ++c) w[m][c] = ld_w(b + m * G::BLOCK + c * 32 + lane);
#pragma unroll
        for (int c = 0; c < G::SBV; ++c) sb[m][c] = ld_w(b + m * G::BLOCK + 32 * G::WV + t * G::SBV + c);
      }
      const int k0 = g * GS;
#pragma unroll
      for (int r = 0; r < RT; ++r) {
        if (r >= nt) break;
        uint4 xa[G::XV], xb[G::XV];
#pragma unroll
        for (int c = 0; c < G::XV; ++c) {
          xa[c] = v0[r] ? ld_x(x0[r] + k0 + 8 * c) : zero;
          xb[c] = v1[r] ? ld_x(x1[r] + k0 + 8 * c) : zero;
        }
        const float xsa = group_sum<G::XV>(xa);
        const float xsb = group_sum<G::XV>(xb);
#pragma unroll
        for (int m = 0; m < M; ++m) {
          float p[NTW][4];
#pragma unroll
          for (int j = 0; j < NTW; ++j) p[j][0] = p[j][1] = p[j][2] = p[j][3] = 0.f;
#pragma unroll
          for (int ks = 0; ks < G::KS; ++ks) {
            const uint32_t a0 = comp(xa[ks >> 1], 2 * (ks & 1));
            const uint32_t a2 = comp(xa[ks >> 1], 2 * (ks & 1) + 1);
            const uint32_t a1 = comp(xb[ks >> 1], 2 * (ks & 1));
            const uint32_t a3 = comp(xb[ks >> 1], 2 * (ks & 1) + 1);
            const int sh = (ks & 1) ? 8 : 0;
#pragma unroll
            for (int j = 0; j < NTW; ++j) {
              const int wi = j * (GS / 32) + (ks >> 1);
              const uint32_t word = comp(w[m][wi >> 2], wi & 3);
              mma(p[j], a0, a1, a2, a3, nib2(word, sh), nib2(word, sh + 4));
            }
          }
#pragma unroll
          for (int j = 0; j < NTW; ++j) {
            const uint32_t sp = comp(sb[m][j >> 2], j & 3);
            const uint32_t bp = comp(sb[m][(NTW + j) >> 2], (NTW + j) & 3);
            const float s0 = lo_f(sp), s1 = hi_f(sp), b0 = lo_f(bp), b1 = hi_f(bp);
            float(&a)[4] = acc[m][r][j];
            a[0] = fmaf(xsa, b0, fmaf(p[j][0], s0, a[0]));
            a[1] = fmaf(xsa, b1, fmaf(p[j][1], s1, a[1]));
            if (v1[r]) {
              a[2] = fmaf(xsb, b0, fmaf(p[j][2], s0, a[2]));
              a[3] = fmaf(xsb, b1, fmaf(p[j][3], s1, a[3]));
            }
          }
        }
      }
    }
#pragma unroll
    for (int r = 0; r < RT; ++r) {
      if (r >= nt) break;
      epilogue<EPI, M, RT>(acc, r, out, N, cb * COLS + 2 * t, pr0[r], pr1[r], v0[r], v1[r], limit);
    }
  }
}

}  // namespace tf_experts_rows

// Groups of 32, two row tiles (32 rows) a warp: SwiGLU gate and up together (epilogue 2), down to fp32 (epilogue 0).
template __global__ void tf_experts_rows::expert_rows<32, 2, 2, 2, 4>(const __nv_bfloat16*, int, int, const uint4*,
    int, int, const int*, const int*, const int*, void*, int, float);
template __global__ void tf_experts_rows::expert_rows<32, 1, 0, 2, 4>(const __nv_bfloat16*, int, int, const uint4*,
    int, int, const int*, const int*, const int*, void*, int, float);
