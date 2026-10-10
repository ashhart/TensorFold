// Device code of nvfp4/experts.cu (lines 14-149, host launchers, comments and ATen dropped) and its instances.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#include "experts.cuh"

namespace tf_nvfp4_experts {

constexpr int BLOCK4 = 36;

__device__ __forceinline__ uint4 ld_nc(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p));
  return r;
}

__device__ __forceinline__ uint32_t fp4pair(uint32_t w, int s) {
  const uint32_t v = w >> s;
  const uint32_t t = ((v & 0x00070007u) << 6) | ((v & 0x00080008u) << 12);
  uint32_t r;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(t), "r"(0x7E807E80u), "r"(0x80008000u));
  return r;
}

__device__ __forceinline__ float e4m3f(uint32_t b) {
  return __half2float(__half(__nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(b), __NV_E4M3)));
}

template <int M>
struct Stage {
  uint4 w[M];
  uint4 s[M];
  uint2 xa[2], xb[2];
};

template <int M>
__device__ __forceinline__ void load_stage(Stage<M>& st, const uint4* blk, int g, int lane, int t,
                                           const __nv_bfloat16* x0, const __nv_bfloat16* x1, bool v0, bool v1) {
  const uint4* b = blk + (size_t)g * (M * BLOCK4);
#pragma unroll
  for (int m = 0; m < M; ++m) {
    st.w[m] = ld_nc(b + m * BLOCK4 + lane);
    st.s[m] = ld_nc(b + m * BLOCK4 + 32 + t);
  }
  const uint2 zero = make_uint2(0u, 0u);
  const int k0 = g * 32 + 4 * t;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    st.xa[h] = v0 ? __ldg(reinterpret_cast<const uint2*>(x0 + k0 + 16 * h)) : zero;
    st.xb[h] = v1 ? __ldg(reinterpret_cast<const uint2*>(x1 + k0 + 16 * h)) : zero;
  }
}

template <int M>
__device__ __forceinline__ void compute_stage(float (&acc)[M][1][NTW][4], const Stage<M>& st, bool hi) {
#pragma unroll
  for (int m = 0; m < M; ++m)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint32_t a0 = st.xa[h].x, a2 = st.xa[h].y, a1 = st.xb[h].x, a3 = st.xb[h].y;
#pragma unroll
      for (int j = 0; j < NTW; ++j) {
        const uint32_t word = comp(st.w[m], j);
        float p[4] = {0.f, 0.f, 0.f, 0.f};
        mma(p, a0, a1, a2, a3, fp4pair(word, 8 * h), fp4pair(word, 8 * h + 4));
        const uint32_t sw = comp(st.s[m], 2 * h + (j >> 1));
        const int sh = (j & 1) * 16;
        const float s0 = e4m3f((sw >> sh) & 0xFFu), s1 = e4m3f((sw >> (sh + 8)) & 0xFFu);
        float(&a)[4] = acc[m][0][j];
        a[0] = fmaf(p[0], s0, a[0]);
        a[1] = fmaf(p[1], s1, a[1]);
        if (hi) {
          a[2] = fmaf(p[2], s0, a[2]);
          a[3] = fmaf(p[3], s1, a[3]);
        }
      }
    }
}

template <int M>
__device__ __forceinline__ void k_loop(float (&acc)[M][1][NTW][4], const uint4* blk, int KG, int lane, int t,
                                       const __nv_bfloat16* x0, const __nv_bfloat16* x1, bool v0, bool v1) {
  constexpr int D = 2;
  Stage<M> st[D];
#pragma unroll
  for (int d = 0; d < D; ++d)
    if (d < KG) load_stage<M>(st[d], blk, d, lane, t, x0, x1, v0, v1);
  for (int g0 = 0; g0 < KG; g0 += D) {
#pragma unroll
    for (int d = 0; d < D; ++d) {
      const int g = g0 + d;
      if (g < KG) {
        compute_stage<M>(acc, st[d], v1);
        if (g + D < KG) load_stage<M>(st[d], blk, g + D, lane, t, x0, x1, v0, v1);
      }
    }
  }
}

template <int M, int EPI, int WARPS>
__global__ void __launch_bounds__(WARPS * 32)
    nvfp4_expert_kernel(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W,
                        const float* __restrict__ scale, int KG, int NB, const int* __restrict__ items,
                        const int* __restrict__ counts, const int* __restrict__ members, void* __restrict__ out, int N,
                        float limit, int skip) {
  const int lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3;
  const int units = __ldg(counts) * NB;
  for (int unit = blockIdx.x * WARPS + (threadIdx.x >> 5); unit < units; unit += gridDim.x * WARPS) {
    const int it = unit / NB, cb = unit - it * NB;
    const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    if (e == skip) continue;
    const uint4* blk = W + ((size_t)e * NB + cb) * (size_t)KG * (M * BLOCK4);
    for (int r0 = 0; r0 < cnt; r0 += 16) {
      const bool v0 = r0 + gq < cnt, v1 = r0 + gq + 8 < cnt;
      const int pr0 = v0 ? __ldg(members + first + r0 + gq) : 0;
      const int pr1 = v1 ? __ldg(members + first + r0 + gq + 8) : 0;
      const int x0r = slots ? pr0 / slots : pr0, x1r = slots ? pr1 / slots : pr1;
      const __nv_bfloat16* x0 = X + (size_t)x0r * x_stride;
      const __nv_bfloat16* x1 = X + (size_t)x1r * x_stride;
      float acc[M][1][NTW][4];
#pragma unroll
      for (int m = 0; m < M; ++m)
#pragma unroll
        for (int j = 0; j < NTW; ++j) acc[m][0][j][0] = acc[m][0][j][1] = acc[m][0][j][2] = acc[m][0][j][3] = 0.f;
      k_loop<M>(acc, blk, KG, lane, t, x0, x1, v0, v1);
#pragma unroll
      for (int m = 0; m < M; ++m) {
        const float g = __ldg(scale + e * M + m);
#pragma unroll
        for (int j = 0; j < NTW; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[m][0][j][q] *= g;
      }
      epilogue<EPI, M, 1>(acc, 0, out, N, cb * COLS + 2 * t, pr0, pr1, v0, v1, limit);
    }
  }
}

// The prompt form's tile: 16 RT rows a warp, WM x WN warps (BM pairs of one item by WN 32-column blocks a CTA).
template <int M, int RT, int WM, int WN>
struct Prompt4 {
  static constexpr int THREADS = WM * WN * 32, BM = 16 * RT * WM, XC = 8, SG = 2, WB = M * BLOCK4;
  static constexpr int XU = BM * XC, WU = WN * SG * WB, SU = XU + WU;  // uint4 a stage: 64 inputs of X, then weights
  static constexpr int XPT = (XU + THREADS - 1) / THREADS;
  static constexpr int STAGES = SU * 16 * 4 <= 49152 ? 4 : SU * 16 * 3 <= 49152 ? 3 : 2;
  // row r's 16-byte chunk c: rows r .. r + 3 put a 16-input block's lanes on distinct banks
  static __device__ __forceinline__ int chunk(int r, int c) { return r * XC + (c ^ ((r & 3) << 1)); }
};

// An e4m3 scale in both bf16 halves (bytes 0 and 2 of x, the others zero), exact.
__device__ __forceinline__ uint32_t e4m3x2(uint32_t x) {
  const uint32_t t = ((x & 0x007F007Fu) << 4) | ((x & 0x00800080u) << 8);
  uint32_t r;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(t), "r"(0x7B807B80u), "r"(0x80008000u));
  return r;
}

// a * b in bf16 pairs; exact for e2m1 codes times e4m3 scales (six significant bits at most).
__device__ __forceinline__ uint32_t mul2(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(a), "r"(b), "r"(0x80008000u));
  return r;
}

// Prompt rows: pairs and weight blocks staged 64 inputs at a time, B = code x block scale in bf16, one fp32 sum over K.
template <int M, int EPI, int RT, int WM, int WN>
__device__ __forceinline__ void prompt_body(const __nv_bfloat16* __restrict__ X, int x_stride, int slots,
                                            const uint4* __restrict__ W, const float* __restrict__ scale, int KG, int NB,
                                            const int* __restrict__ items, const int* __restrict__ counts,
                                            const int* __restrict__ members, void* __restrict__ out, int N, float limit) {
  using P = Prompt4<M, RT, WM, WN>;
  constexpr int STAGES = P::STAGES;
  extern __shared__ uint4 sm[];
  const int nbt = (NB + WN - 1) / WN, it = blockIdx.x / nbt, cbt = blockIdx.x - it * nbt;
  if (it >= __ldg(counts)) return;
  const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp / WN, wn = warp - wm * WN;
  const int t = lane & 3, gq = lane >> 2, cb = cbt * WN + wn, cbs = min(WN, NB - cbt * WN);
  const __nv_bfloat16* xsrc[P::XPT];
  int xdst[P::XPT];
#pragma unroll
  for (int i = 0; i < P::XPT; ++i) {
    const int q = tid + i * P::THREADS, r = q / P::XC, c = q - r * P::XC;
    xdst[i] = P::chunk(r, c);
    xsrc[i] = nullptr;
    if (q < P::XU && r < cnt) {
      const int p = __ldg(members + first + r);
      xsrc[i] = X + (size_t)(slots ? p / slots : p) * x_stride + 8 * c;
    }
  }
  const uint4* wsrc = W + ((size_t)e * NB + cbt * WN) * (size_t)KG * P::WB;
  auto stage = [&](int s, int g0) {  // k32 groups g0 .. g0 + ng - 1 (ng < SG only at an odd end)
    const int ng = min(P::SG, KG - g0);
    uint4* xs = sm + s * P::SU;
#pragma unroll
    for (int i = 0; i < P::XPT; ++i)
      if (xsrc[i] && ((tid + i * P::THREADS) % P::XC) < ng * 4) cp16(xs + xdst[i], xsrc[i] + (size_t)g0 * 32);
    uint4* ws = xs + P::XU;
    for (int q = tid; q < cbs * P::SG * P::WB; q += P::THREADS) {
      const int j = q / (P::SG * P::WB), o = q - j * P::SG * P::WB;
      if (o < ng * P::WB) cp16(ws + q, wsrc + ((size_t)j * KG + g0) * P::WB + o);
    }
  };
  const int base = 16 * RT * wm;
  const int nt = max(0, min(RT, (cnt - base + 15) >> 4));
  const bool active = nt > 0 && cb < NB;
  const uint32_t b = gq & 1;  // byte 2 (j & 1) + (gq & 1) of a scale word holds column j * 8 + gq's scale
  const uint32_t pick[2] = {b | 0x4040u | (b << 8), (b + 2) | 0x4040u | ((b + 2) << 8)};
  float acc[M][RT][NTW][4];
#pragma unroll
  for (int m = 0; m < M; ++m)
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int j = 0; j < NTW; ++j) acc[m][r][j][0] = acc[m][r][j][1] = acc[m][r][j][2] = acc[m][r][j][3] = 0.f;
  const int steps = (KG + P::SG - 1) / P::SG;
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < steps) stage(s, s * P::SG);
    cp_commit();
  }
  for (int k = 0; k < steps; ++k) {
    cp_wait<STAGES - 2>();
    __syncthreads();
    if (k + STAGES - 1 < steps) stage((k + STAGES - 1) % STAGES, (k + STAGES - 1) * P::SG);
    cp_commit();
    if (!active) continue;
    const uint4* xs = sm + (k % STAGES) * P::SU;
#pragma unroll
    for (int gi = 0; gi < P::SG; ++gi) {
      if (k * P::SG + gi >= KG) break;
      const uint4* ws = xs + P::XU + (wn * P::SG + gi) * P::WB;
      uint4 wv[M];
      uint2 sc[M][2];
#pragma unroll
      for (int m = 0; m < M; ++m) {
        wv[m] = ws[m * BLOCK4 + lane];
        // column j * 8 + gq's scales of 16-block h: bytes 2j + (gq & 1) of an 8-byte run
#pragma unroll
        for (int h = 0; h < 2; ++h)
          sc[m][h] = reinterpret_cast<const uint2*>(ws + m * BLOCK4 + 32)[(gq >> 1) * 2 + h];
      }
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        uint2 xa[RT], xb[RT];
        const int u = gi * 8 + h * 4 + t;  // the 8-byte unit holding lane t's inputs 16h + 4t .. + 3 of group gi
#pragma unroll
        for (int r = 0; r < RT; ++r) {
          if (r >= nt) break;
          const int r0 = base + 16 * r + gq;
          xa[r] = reinterpret_cast<const uint2*>(xs + P::chunk(r0, u >> 1))[u & 1];
          xb[r] = reinterpret_cast<const uint2*>(xs + P::chunk(r0 + 8, u >> 1))[u & 1];
        }
#pragma unroll
        for (int m = 0; m < M; ++m)
#pragma unroll
          for (int j = 0; j < NTW; ++j) {
            const uint32_t word = comp(wv[m], j);
            const uint32_t sv = e4m3x2(__byte_perm(j < 2 ? sc[m][h].x : sc[m][h].y, 0u, pick[j & 1]));
            const uint32_t b0 = mul2(fp4pair(word, 8 * h), sv), b1 = mul2(fp4pair(word, 8 * h + 4), sv);
#pragma unroll
            for (int r = 0; r < RT; ++r) {
              if (r >= nt) break;
              mma(acc[m][r][j], xa[r].x, xb[r].x, xa[r].y, xb[r].y, b0, b1);
            }
          }
      }
    }
  }
  if (!active) return;
#pragma unroll
  for (int m = 0; m < M; ++m) {
    const float g = __ldg(scale + e * M + m);
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int j = 0; j < NTW; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[m][r][j][q] *= g;
  }
#pragma unroll
  for (int r = 0; r < RT; ++r) {
    if (r >= nt) break;
    const int m0 = base + 16 * r + gq, m1 = m0 + 8;
    const bool v0 = m0 < cnt, v1 = m1 < cnt;
    const int p0 = v0 ? __ldg(members + first + m0) : 0, p1 = v1 ? __ldg(members + first + m1) : 0;
    epilogue<EPI, M, RT>(acc, r, out, N, cb * COLS + 2 * t, p0, p1, v0, v1, limit);
  }
}

// The instances: SwiGLU gate-up, fp32 and bf16 down, and relu^2 up for experts without a gate (Nemotron).
template __global__ void nvfp4_expert_kernel<2, 2, 4>(const __nv_bfloat16* __restrict__, int, int, const uint4* __restrict__, const float* __restrict__, int, int, const int* __restrict__, const int* __restrict__, const int* __restrict__, void* __restrict__, int, float, int);
template __global__ void nvfp4_expert_kernel<1, 0, 4>(const __nv_bfloat16* __restrict__, int, int, const uint4* __restrict__, const float* __restrict__, int, int, const int* __restrict__, const int* __restrict__, const int* __restrict__, void* __restrict__, int, float, int);
template __global__ void nvfp4_expert_kernel<1, 3, 4>(const __nv_bfloat16* __restrict__, int, int, const uint4* __restrict__, const float* __restrict__, int, int, const int* __restrict__, const int* __restrict__, const int* __restrict__, void* __restrict__, int, float, int);
template __global__ void nvfp4_expert_kernel<1, 1, 4>(const __nv_bfloat16* __restrict__, int, int, const uint4* __restrict__, const float* __restrict__, int, int, const int* __restrict__, const int* __restrict__, const int* __restrict__, void* __restrict__, int, float, int);
}

// The prompt form's entries, 4 row tiles a warp in one row of WN warps: relu^2 up, SwiGLU gate-up, bf16 sums down.
#define TF_NVFP4_PROMPT(NAME, M, EPI, WN)                                                                              \
  extern "C" __global__ void __launch_bounds__(WN * 32) NAME(                                                        \
      const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W,                     \
      const float* __restrict__ scale, int KG, int NB, const int* __restrict__ items, const int* __restrict__ counts, \
      const int* __restrict__ members, void* __restrict__ out, int N, float limit) {                                 \
    tf_nvfp4_experts::prompt_body<M, EPI, 4, 1, WN>(X, x_stride, slots, W, scale, KG, NB, items, counts, members,    \
                                                    out, N, limit);                                                  \
  }
TF_NVFP4_PROMPT(tf_nvfp4_prompt_relu2, 1, 1, 8)
TF_NVFP4_PROMPT(tf_nvfp4_prompt_swiglu, 2, 2, 4)
TF_NVFP4_PROMPT(tf_nvfp4_prompt_down, 1, 3, 8)
