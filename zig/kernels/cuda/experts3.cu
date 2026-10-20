// 3-bit experts, decode form: experts.cu's steps on nibble words rebuilt from bit planes, so a code's products match.

#include <cuda_bf16.h>
#include <stdint.h>

#include "experts3.cuh"

namespace tf_experts3 {

__device__ __forceinline__ uint4 ld_w(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p));
  return r;
}

__device__ __forceinline__ uint2 ld_w2(const uint2* p) {
  uint2 r;
  asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0, %1}, [%2];\n" : "=r"(r.x), "=r"(r.y) : "l"(p));
  return r;
}

__device__ __forceinline__ uint4 ld_x(const __nv_bfloat16* p) { return __ldg(reinterpret_cast<const uint4*>(p)); }

constexpr int XV = Geo<64>::XV;
constexpr int SBV = Geo<64>::SBV;

template <int XN>
__device__ __forceinline__ float group_sum(const uint4 (&v)[XN]) {
  float s = 0.f;
#pragma unroll
  for (int c = 0; c < XN; ++c)
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

template <int M, int BITS>
struct Stage {
  uint32_t w[M][Lane<BITS>::WORDS];
  uint4 sb[M][SBV];
  uint4 xa[XV];
  uint4 xb[XV];
};

template <int M, int BITS>
__device__ __forceinline__ void load_stage(Stage<M, BITS>& st, const uint4* blk, int g, int lane, int t,
                                           const __nv_bfloat16* x0, const __nv_bfloat16* x1, bool v0, bool v1) {
  using L = Lane<BITS>;
  const uint4* b = blk + (size_t)g * (M * L::BLOCK);
#pragma unroll
  for (int m = 0; m < M; ++m) {
    lane_words<BITS>(st.w[m], b + m * L::BLOCK, lane, [](const uint4* q) { return ld_w(q); },
                     [](const uint2* q) { return ld_w2(q); });
#pragma unroll
    for (int c = 0; c < SBV; ++c) st.sb[m][c] = ld_w(b + m * L::BLOCK + L::WU + t * SBV + c);
  }
  const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
  const int k0 = g * 64;
#pragma unroll
  for (int c = 0; c < XV; ++c) {
    st.xa[c] = v0 ? ld_x(x0 + k0 + 8 * c) : zero;
    st.xb[c] = v1 ? ld_x(x1 + k0 + 8 * c) : zero;
  }
}

template <int M, int BITS>
__device__ __forceinline__ void compute_stage(float (&acc)[M][1][NTW][4], const Stage<M, BITS>& st, bool hi) {
  const float xsa = group_sum<XV>(st.xa);
  const float xsb = group_sum<XV>(st.xb);
#pragma unroll
  for (int m = 0; m < M; ++m) {
    float p[NTW][4];
#pragma unroll
    for (int j = 0; j < NTW; ++j) p[j][0] = p[j][1] = p[j][2] = p[j][3] = 0.f;
#pragma unroll
    for (int ks = 0; ks < Geo<64>::KS; ++ks) {
      const uint32_t a0 = comp(st.xa[ks >> 1], 2 * (ks & 1));
      const uint32_t a2 = comp(st.xa[ks >> 1], 2 * (ks & 1) + 1);
      const uint32_t a1 = comp(st.xb[ks >> 1], 2 * (ks & 1));
      const uint32_t a3 = comp(st.xb[ks >> 1], 2 * (ks & 1) + 1);
      const int sh = (ks & 1) ? 8 : 0;
#pragma unroll
      for (int j = 0; j < NTW; ++j) {
        const uint32_t word = nibword<BITS>(st.w[m], j * 2 + (ks >> 1));
        mma(p[j], a0, a1, a2, a3, nib2(word, sh), nib2(word, sh + 4));
      }
    }
#pragma unroll
    for (int j = 0; j < NTW; ++j) {
      const uint32_t sp = comp(st.sb[m][j >> 2], j & 3);
      const uint32_t bp = comp(st.sb[m][(NTW + j) >> 2], (NTW + j) & 3);
      const float s0 = lo_f(sp), s1 = hi_f(sp), b0 = lo_f(bp), b1 = hi_f(bp);
      float(&a)[4] = acc[m][0][j];
      a[0] = fmaf(xsa, b0, fmaf(p[j][0], s0, a[0]));
      a[1] = fmaf(xsa, b1, fmaf(p[j][1], s1, a[1]));
      if (hi) {
        a[2] = fmaf(xsb, b0, fmaf(p[j][2], s0, a[2]));
        a[3] = fmaf(xsb, b1, fmaf(p[j][3], s1, a[3]));
      }
    }
  }
}

template <int M, int D, int BITS>
__device__ __forceinline__ void k_loop(float (&acc)[M][1][NTW][4], const uint4* blk, int KG, int lane, int t,
                                       const __nv_bfloat16* x0, const __nv_bfloat16* x1, bool v0, bool v1) {
  Stage<M, BITS> st[D];
#pragma unroll
  for (int d = 0; d < D; ++d)
    if (d < KG) load_stage<M, BITS>(st[d], blk, d, lane, t, x0, x1, v0, v1);
  for (int g0 = 0; g0 < KG; g0 += D) {
#pragma unroll
    for (int d = 0; d < D; ++d) {
      const int g = g0 + d;
      if (g < KG) {
        compute_stage<M, BITS>(acc, st[d], v1);
        if (g + D < KG) load_stage<M, BITS>(st[d], blk, g + D, lane, t, x0, x1, v0, v1);
      }
    }
  }
}

// experts.cu's expert_kernel on 3-bit blocks in W; expert e4 (the shared one, -1 for none) reads 4-bit blocks in W4.
template <int M, int EPI, int D, int WARPS>
__global__ void __launch_bounds__(WARPS * 32)
    expert_kernel(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W, int KG,
                  int NB, const int* __restrict__ items, const int* __restrict__ counts,
                  const int* __restrict__ members, void* __restrict__ out, int N, float limit,
                  const uint4* __restrict__ W4, int e4) {
  const int lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3;
  const int units = __ldg(counts) * NB;
  for (int unit = blockIdx.x * WARPS + (threadIdx.x >> 5); unit < units; unit += gridDim.x * WARPS) {
    const int it = unit / NB, cb = unit - it * NB;
    const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    const bool v0 = gq < cnt, v1 = gq + 8 < cnt;
    const int pr0 = v0 ? __ldg(members + first + gq) : 0;
    const int pr1 = v1 ? __ldg(members + first + gq + 8) : 0;
    const int r0 = slots ? pr0 / slots : pr0, r1 = slots ? pr1 / slots : pr1;
    const __nv_bfloat16* x0 = X + (size_t)r0 * x_stride + t * 16;
    const __nv_bfloat16* x1 = X + (size_t)r1 * x_stride + t * 16;
    float acc[M][1][NTW][4];
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
      for (int j = 0; j < NTW; ++j) acc[m][0][j][0] = acc[m][0][j][1] = acc[m][0][j][2] = acc[m][0][j][3] = 0.f;
    if (e == e4) {  // warp-uniform: a unit is one expert's column block
      k_loop<M, D, 4>(acc, W4 + (size_t)cb * KG * (M * Lane<4>::BLOCK), KG, lane, t, x0, x1, v0, v1);
    } else {
      const uint4* blk = W + ((size_t)e * NB + cb) * (size_t)KG * (M * Lane<3>::BLOCK);
      k_loop<M, D, 3>(acc, blk, KG, lane, t, x0, x1, v0, v1);
    }
    epilogue<EPI, M, 1>(acc, 0, out, N, cb * COLS + 2 * t, pr0, pr1, v0, v1, limit);
  }
}

// One group's terms for the chain: a lane's 4 mma steps a tile, its scale and bias words, its rows' input sums.
template <int M>
struct Part {
  float p[M][NTW][4][32];
  uint32_t sb[M][NTW][2][32];
  float xs[2][32];
};

template <int M, int BITS>
__device__ __forceinline__ void part_stage(Part<M>& out, const Stage<M, BITS>& st, int lane) {
  out.xs[0][lane] = group_sum<XV>(st.xa);
  out.xs[1][lane] = group_sum<XV>(st.xb);
#pragma unroll
  for (int m = 0; m < M; ++m) {
    float p[NTW][4];
#pragma unroll
    for (int j = 0; j < NTW; ++j) p[j][0] = p[j][1] = p[j][2] = p[j][3] = 0.f;
#pragma unroll
    for (int ks = 0; ks < Geo<64>::KS; ++ks) {
      const uint32_t a0 = comp(st.xa[ks >> 1], 2 * (ks & 1));
      const uint32_t a2 = comp(st.xa[ks >> 1], 2 * (ks & 1) + 1);
      const uint32_t a1 = comp(st.xb[ks >> 1], 2 * (ks & 1));
      const uint32_t a3 = comp(st.xb[ks >> 1], 2 * (ks & 1) + 1);
      const int sh = (ks & 1) ? 8 : 0;
#pragma unroll
      for (int j = 0; j < NTW; ++j) {
        const uint32_t word = nibword<BITS>(st.w[m], j * 2 + (ks >> 1));
        mma(p[j], a0, a1, a2, a3, nib2(word, sh), nib2(word, sh + 4));
      }
    }
#pragma unroll
    for (int j = 0; j < NTW; ++j) {
#pragma unroll
      for (int r = 0; r < 4; ++r) out.p[m][j][r][lane] = p[j][r];
      out.sb[m][j][0][lane] = comp(st.sb[m][j >> 2], j & 3);
      out.sb[m][j][1][lane] = comp(st.sb[m][(NTW + j) >> 2], (NTW + j) & 3);
    }
  }
}

// Tile j's chain over one group's terms, as compute_stage runs it.
template <int M>
__device__ __forceinline__ void chain(float (&acc)[M][4], const Part<M>& in, int j, int lane, bool hi) {
  const float xsa = in.xs[0][lane], xsb = in.xs[1][lane];
#pragma unroll
  for (int m = 0; m < M; ++m) {
    const uint32_t sp = in.sb[m][j][0][lane], bp = in.sb[m][j][1][lane];
    const float s0 = lo_f(sp), s1 = hi_f(sp), b0 = lo_f(bp), b1 = hi_f(bp);
    float(&a)[4] = acc[m];
    a[0] = fmaf(xsa, b0, fmaf(in.p[m][j][0][lane], s0, a[0]));
    a[1] = fmaf(xsa, b1, fmaf(in.p[m][j][1][lane], s1, a[1]));
    if (hi) {
      a[2] = fmaf(xsb, b0, fmaf(in.p[m][j][2][lane], s0, a[2]));
      a[3] = fmaf(xsb, b1, fmaf(in.p[m][j][3][lane], s1, a[3]));
    }
  }
}

// Warp w takes groups w, w + W, ...; after each round of W groups, warp j chains tile j over them in group order.
template <int M, int D, int W, int BITS>
__device__ __forceinline__ void split_loop(float (&acc)[M][4], Part<M> (&parts)[W], const uint4* blk, int KG, int w,
                                           int lane, int t, const __nv_bfloat16* x0, const __nv_bfloat16* x1, bool v0,
                                           bool v1) {
  Stage<M, BITS> st[D];
#pragma unroll
  for (int d = 0; d < D; ++d)
    if (d * W + w < KG) load_stage<M, BITS>(st[d], blk, d * W + w, lane, t, x0, x1, v0, v1);
  const int rounds = (KG + W - 1) / W;
  for (int i0 = 0; i0 < rounds; i0 += D) {
#pragma unroll
    for (int d = 0; d < D; ++d) {
      const int i = i0 + d;
      if (i < rounds) {
        const int g = i * W + w;
        if (g < KG) {
          part_stage<M, BITS>(parts[w], st[d], lane);
          if (g + D * W < KG) load_stage<M, BITS>(st[d], blk, g + D * W, lane, t, x0, x1, v0, v1);
        }
        __syncthreads();
        const int n = min(W, KG - i * W);
        for (int s = 0; s < n; ++s) chain<M>(acc, parts[s], w, lane, v1);
        __syncthreads();
      }
    }
  }
}

// epilogue's SwiGLU stores for tile j alone.
template <int M>
__device__ __forceinline__ void store_tile(const float (&acc)[M][4], __nv_bfloat16* out, int N, int col, int pr0,
                                           int pr1, bool v0, bool v1, float limit) {
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    if (!(h ? v1 : v0)) continue;
    const size_t row = (size_t)(h ? pr1 : pr0) * N;
    const float o0 = swiglu(acc[0][2 * h], acc[M - 1][2 * h], limit);
    const float o1 = swiglu(acc[0][2 * h + 1], acc[M - 1][2 * h + 1], limit);
    *reinterpret_cast<__nv_bfloat162*>(out + row + col) = __floats2bfloat162_rn(o0, o1);
  }
}

// expert_kernel's sums with a column block a CTA, its groups shared by the warps: for few blocks, as at one row.
template <int M, int EPI, int D, int WARPS>
__global__ void __launch_bounds__(WARPS * 32)
    split_kernel(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W, int KG,
                 int NB, const int* __restrict__ items, const int* __restrict__ counts,
                 const int* __restrict__ members, void* __restrict__ out, int N, float limit,
                 const uint4* __restrict__ W4, int e4) {
  static_assert(WARPS == NTW && M == 2 && EPI == 2, "the SwiGLU pair, a warp a tile in the chain");
  __shared__ Part<M> parts[WARPS];
  const int lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3, w = threadIdx.x >> 5;
  const int units = __ldg(counts) * NB;
  for (int unit = blockIdx.x; unit < units; unit += gridDim.x) {
    const int it = unit / NB, cb = unit - it * NB;
    const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    const bool v0 = gq < cnt, v1 = gq + 8 < cnt;
    const int pr0 = v0 ? __ldg(members + first + gq) : 0;
    const int pr1 = v1 ? __ldg(members + first + gq + 8) : 0;
    const int r0 = slots ? pr0 / slots : pr0, r1 = slots ? pr1 / slots : pr1;
    const __nv_bfloat16* x0 = X + (size_t)r0 * x_stride + t * 16;
    const __nv_bfloat16* x1 = X + (size_t)r1 * x_stride + t * 16;
    float acc[M][4];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m][0] = acc[m][1] = acc[m][2] = acc[m][3] = 0.f;
    if (e == e4) {  // block-uniform: a unit is one expert's column block
      const uint4* blk = W4 + (size_t)cb * KG * (M * Lane<4>::BLOCK);
      split_loop<M, D, WARPS, 4>(acc, parts, blk, KG, w, lane, t, x0, x1, v0, v1);
    } else {
      const uint4* blk = W + ((size_t)e * NB + cb) * (size_t)KG * (M * Lane<3>::BLOCK);
      split_loop<M, D, WARPS, 3>(acc, parts, blk, KG, w, lane, t, x0, x1, v0, v1);
    }
    store_tile<M>(acc, reinterpret_cast<__nv_bfloat16*>(out), N, cb * COLS + 2 * t + 8 * w, pr0, pr1, v0, v1, limit);
  }
}

}  // namespace tf_experts3

// MLX 3-bit words (an LSB-first bitstream, 3 bytes a run of 8 codes) to MLX 4-bit words: a thread an output word.
extern "C" __global__ void tf_experts3_widen(const uint8_t* __restrict__ in, uint32_t* __restrict__ out, long words) {
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < words; i += (long)gridDim.x * blockDim.x) {
    const uint32_t v = in[3 * i] | (in[3 * i + 1] << 8) | (in[3 * i + 2] << 16);
    uint32_t o = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) o |= ((v >> (3 * j)) & 7u) << (4 * j);
    out[i] = o;
  }
}

// 4-bit blocks [blocks][288] (codes below 8) to 3-bit blocks of 224 words, block i at out + i * stride + offset.
extern "C" __global__ void tf_experts3_planes(const uint32_t* __restrict__ in, uint32_t* __restrict__ out,
                                              long blocks, int stride, int offset) {
  for (long q = blockIdx.x * (long)blockDim.x + threadIdx.x; q < blocks * 32; q += (long)gridDim.x * blockDim.x) {
    const long i = q / 32;
    const int lane = q % 32;
    const uint32_t* b = in + i * 288;
    uint32_t* o = out + i * stride + offset;
    uint32_t w[8];
#pragma unroll
    for (int c = 0; c < 2; ++c)
#pragma unroll
      for (int j = 0; j < 4; ++j) w[c * 4 + j] = b[(c * 32 + lane) * 4 + j];
#pragma unroll
    for (int k = 0; k < 4; ++k) o[lane * 4 + k] = (w[2 * k] & 0x33333333u) | ((w[2 * k + 1] & 0x33333333u) << 2);
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      uint32_t p = 0;
#pragma unroll
      for (int c = 0; c < 4; ++c) p |= ((w[4 * k + c] >> 2) & 0x11111111u) << c;
      o[128 + lane * 2 + k] = p;
    }
    o[192 + lane] = b[256 + lane];
  }
}

// Decode form for groups of 64: the SwiGLU pair form (M = 2, epilogue 2) and its split form, down to fp32 (epilogue 0).
template __global__ void tf_experts3::expert_kernel<2, 2, 2, 4>(const __nv_bfloat16* __restrict__, int, int,
    const uint4* __restrict__, int, int, const int* __restrict__, const int* __restrict__, const int* __restrict__,
    void* __restrict__, int, float, const uint4* __restrict__, int);
template __global__ void tf_experts3::expert_kernel<1, 0, 2, 4>(const __nv_bfloat16* __restrict__, int, int,
    const uint4* __restrict__, int, int, const int* __restrict__, const int* __restrict__, const int* __restrict__,
    void* __restrict__, int, float, const uint4* __restrict__, int);
template __global__ void tf_experts3::split_kernel<2, 2, 2, 4>(const __nv_bfloat16* __restrict__, int, int,
    const uint4* __restrict__, int, int, const int* __restrict__, const int* __restrict__, const int* __restrict__,
    void* __restrict__, int, float, const uint4* __restrict__, int);
