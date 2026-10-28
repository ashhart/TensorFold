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
  // FACT's X: rows r, r + 8 of pair p interleaved by word (unit u = one mma A quad); u ^ 5 on odd p: no bank conflicts
  static __device__ __forceinline__ int pair(int p, int u) { return p * 16 + (u ^ ((p & 1) * 5)); }
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

// mma from zero (C = 0): the same sums as the decode form's zeroed p[4], no register moves.
__device__ __forceinline__ void mma_z(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                      uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%10, %10, %10, %10};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(0.f));
}

// Prompt rows, 64 inputs a stage, NT row tiles a warp. FACT: the decode form's sums (per-16 mma from zero, then fma).
template <int M, int EPI, int RT, int WM, int WN, int NT, bool FACT = false>
__device__ __forceinline__ void prompt_tiles(const __nv_bfloat16* __restrict__ X, int x_stride, int slots,
                                             const uint4* __restrict__ W, const float* __restrict__ scale, int KG,
                                             int NB, const int* __restrict__ members, void* __restrict__ out, int N,
                                             float limit, int cbt, int e, int first, int cnt) {
  using P = Prompt4<M, RT, WM, WN>;
  constexpr int STAGES = P::STAGES;
  // WM 1: the CTA's warps share NT, so they share X's layout. fact_swiglu (M 2) only: the down stays on the pair reads
  constexpr bool XQ = FACT && M == 2 && NT == RT && WM == 1;
  extern __shared__ uint4 sm[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp / WN, wn = warp - wm * WN;
  const int t = lane & 3, gq = lane >> 2, cb = cbt * WN + wn, cbs = min(WN, NB - cbt * WN);
  // XQ (FACT on full 4-tile items): X through registers, chunk tid & 3 of group gi of pair (tid >> 2) % (BM / 2)
  constexpr int XP = XQ ? 4 * P::BM / P::THREADS : 1;
  static_assert(!XQ || (P::THREADS == 128 || P::THREADS == 256) && P::BM == 64 && P::SG == 2, "FACT's pair copy");
  const uint4* xpr[2] = {nullptr, nullptr};
  int xpd = 0;  // the thread's first destination chunk in a stage
  if constexpr (XQ) {
    const int p = (tid >> 2) % (P::BM / 2), c = tid & 3;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int r = (p >> 3) * 16 + (p & 7) + 8 * h;
      if (r < cnt) {
        const int m = __ldg(members + first + r);
        xpr[h] = reinterpret_cast<const uint4*>(X + (size_t)(slots ? m / slots : m) * x_stride) + c;
      }
    }
    xpd = p * 16 + 8 * (tid / (2 * P::BM)) + ((2 * c) ^ ((p & 1) * 5));
  }
  uint4 xg[XP][2];
  auto xload = [&](int g0) {  // the thread's chunks of k32 groups g0 .. g0 + ng - 1 into registers
    const int ng = min(P::SG, KG - g0);
#pragma unroll
    for (int i = 0; i < XP; ++i) {
      const int gi = (tid + i * P::THREADS) / (2 * P::BM);
#pragma unroll
      for (int h = 0; h < 2; ++h)
        xg[i][h] = xpr[h] && gi < ng ? ld_nc(xpr[h] + (size_t)(g0 + gi) * 4) : make_uint4(0u, 0u, 0u, 0u);
    }
  };
  auto xstore = [&](int s) {  // ... interleaved into stage s: units 2c and 2c + 1 (u xor 5 swaps them on odd pairs)
    uint4* xs = sm + s * P::SU;
#pragma unroll
    for (int i = 0; i < XP; ++i) {
      const uint4 a = xg[i][0], b = xg[i][1];
      xs[xpd + 8 * i] = make_uint4(a.x, b.x, a.y, b.y);
      xs[(xpd + 8 * i) ^ 1] = make_uint4(a.z, b.z, a.w, b.w);
    }
  };
  constexpr int XPT = XQ ? 1 : P::XPT;
  const __nv_bfloat16* xsrc[XPT];
  int xdst[XPT];
#pragma unroll
  for (int i = 0; i < XPT && !XQ; ++i) {
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
    if constexpr (!XQ) {
#pragma unroll
    for (int i = 0; i < P::XPT; ++i)
      if (xsrc[i] && ((tid + i * P::THREADS) % P::XC) < ng * 4) cp16(xs + xdst[i], xsrc[i] + (size_t)g0 * 32);
    }
    uint4* ws = xs + P::XU;
    for (int q = tid; q < cbs * P::SG * P::WB; q += P::THREADS) {
      const int j = q / (P::SG * P::WB), o = q - j * P::SG * P::WB;
      if (o < ng * P::WB) cp16(ws + q, wsrc + ((size_t)j * KG + g0) * P::WB + o);
    }
  };
  constexpr int TA = NT > 0 ? NT : 1;  // array extent
  const int base = 16 * RT * wm;
  // XQ: lane quad (h, gi, r) = P::pair((base + 16 r) / 2 + gq, 8 gi + 4 h + t) = xs[xqh[h] + 128 r + 8 gi]
  const int xqh[2] = {(base / 2 + gq) * 16 + 4 * (gq & 1) + (t ^ (gq & 1)),
                      (base / 2 + gq) * 16 + 4 * (~gq & 1) + (t ^ (gq & 1))};
  const bool active = NT > 0 && cb < NB;
  const uint32_t b = gq & 1;  // byte 2 (j & 1) + (gq & 1) of a scale word holds column j * 8 + gq's scale
  const uint32_t pick[2] = {b | 0x4040u | (b << 8), (b + 2) | 0x4040u | ((b + 2) << 8)};
  float acc[M][TA][NTW][4];
#pragma unroll
  for (int m = 0; m < M; ++m)
#pragma unroll
    for (int r = 0; r < NT; ++r)
#pragma unroll
      for (int j = 0; j < NTW; ++j) acc[m][r][j][0] = acc[m][r][j][1] = acc[m][r][j][2] = acc[m][r][j][3] = 0.f;
  const int steps = (KG + P::SG - 1) / P::SG;
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < steps) {
      stage(s, s * P::SG);
      if constexpr (XQ) {
        xload(s * P::SG);
        xstore(s);
      }
    }
    cp_commit();
  }
  if constexpr (XQ)  // X runs a stage ahead of the weights' cp.async: stage s is stored after stage s - 2's math
    if (STAGES - 1 < steps) xload((STAGES - 1) * P::SG);
  for (int k = 0; k < steps; ++k) {
    cp_wait<STAGES - 2>();
    __syncthreads();
    if (k + STAGES - 1 < steps) stage((k + STAGES - 1) % STAGES, (k + STAGES - 1) * P::SG);
    cp_commit();
    if constexpr (XQ) {
      if (!active) {
        if (k + STAGES - 1 < steps) xstore((k + STAGES - 1) % STAGES);
        if (k + STAGES < steps) xload((k + STAGES) * P::SG);
        continue;
      }
    }
    if (!active) continue;
    const uint4* xs = sm + (k % STAGES) * P::SU;
#pragma unroll
    for (int gi = 0; gi < P::SG; ++gi) {
      if (k * P::SG + gi >= KG) break;
      const uint4* ws = xs + P::XU + (wn * P::SG + gi) * P::WB;
      uint4 wv[M];
      uint2 sc[M][2];
      uint4 sd[M];  // FACT: the decode form's scale word of quad t (columns 2t, 2t + 1 of each n8 tile)
#pragma unroll
      for (int m = 0; m < M; ++m) {
        wv[m] = ws[m * BLOCK4 + lane];
        if constexpr (FACT) {
          sd[m] = ws[m * BLOCK4 + 32 + t];
        } else {
        // column j * 8 + gq's scales of 16-block h: bytes 2j + (gq & 1) of an 8-byte run
#pragma unroll
        for (int h = 0; h < 2; ++h)
          sc[m][h] = reinterpret_cast<const uint2*>(ws + m * BLOCK4 + 32)[(gq >> 1) * 2 + h];
        }
      }
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        uint2 xa[TA], xb[TA];
        uint4 xq[TA];  // FACT: the A quad (xa.x, xb.x, xa.y, xb.y) in one read
        const int u = gi * 8 + h * 4 + t;  // the 8-byte unit holding lane t's inputs 16h + 4t .. + 3 of group gi
#pragma unroll
        for (int r = 0; r < NT; ++r) {
          const int r0 = base + 16 * r + gq;
          if constexpr (XQ) {
            xq[r] = xs[xqh[h] + 128 * r + 8 * gi];
          } else {
          xa[r] = reinterpret_cast<const uint2*>(xs + P::chunk(r0, u >> 1))[u & 1];
          xb[r] = reinterpret_cast<const uint2*>(xs + P::chunk(r0 + 8, u >> 1))[u & 1];
          }
        }
#pragma unroll
        for (int m = 0; m < M; ++m)
#pragma unroll
          for (int j = 0; j < NTW; ++j) {
            const uint32_t word = comp(wv[m], j);
            if constexpr (FACT) {
              const uint32_t b0 = fp4pair(word, 8 * h), b1 = fp4pair(word, 8 * h + 4);
              const uint32_t sw = comp(sd[m], 2 * h + (j >> 1));
              const int sh = (j & 1) * 16;
              const float s0 = e4m3f((sw >> sh) & 0xFFu), s1 = e4m3f((sw >> (sh + 8)) & 0xFFu);
#pragma unroll
              for (int r = 0; r < NT; ++r) {
                float p[4];
                if constexpr (XQ)
                  mma_z(p, xq[r].x, xq[r].y, xq[r].z, xq[r].w, b0, b1);
                else
                  mma_z(p, xa[r].x, xb[r].x, xa[r].y, xb[r].y, b0, b1);
                float(&a)[4] = acc[m][r][j];
                a[0] = fmaf(p[0], s0, a[0]);
                a[1] = fmaf(p[1], s1, a[1]);
                a[2] = fmaf(p[2], s0, a[2]);
                a[3] = fmaf(p[3], s1, a[3]);
              }
            } else {
            const uint32_t sv = e4m3x2(__byte_perm(j < 2 ? sc[m][h].x : sc[m][h].y, 0u, pick[j & 1]));
            const uint32_t b0 = mul2(fp4pair(word, 8 * h), sv), b1 = mul2(fp4pair(word, 8 * h + 4), sv);
#pragma unroll
            for (int r = 0; r < NT; ++r) mma(acc[m][r][j], xa[r].x, xb[r].x, xa[r].y, xb[r].y, b0, b1);
            }
          }
      }
    }
    if constexpr (XQ) {  // X of stage k + STAGES - 1, loaded during this stage's math; then the next one's loads
      if (k + STAGES - 1 < steps) xstore((k + STAGES - 1) % STAGES);
      if (k + STAGES < steps) xload((k + STAGES) * P::SG);
    }
  }
  if (!active) return;
  if constexpr (NT > 0) {
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const float g = __ldg(scale + e * M + m);
#pragma unroll
      for (int r = 0; r < NT; ++r)
#pragma unroll
        for (int j = 0; j < NTW; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[m][r][j][q] *= g;
    }
#pragma unroll
    for (int r = 0; r < NT; ++r) {
      const int m0 = base + 16 * r + gq, m1 = m0 + 8;
      const bool v0 = m0 < cnt, v1 = m1 < cnt;
      const int p0 = v0 ? __ldg(members + first + m0) : 0, p1 = v1 ? __ldg(members + first + m1) : 0;
      epilogue<EPI, M, TA>(acc, r, out, N, cb * COLS + 2 * t, p0, p1, v0, v1, limit);
    }
  }
}

template <int M, int EPI, int RT, int WM, int WN, bool FACT = false>
__device__ __forceinline__ void prompt_body(const __nv_bfloat16* __restrict__ X, int x_stride, int slots,
                                            const uint4* __restrict__ W, const float* __restrict__ scale, int KG,
                                            int NB, const int* __restrict__ items, const int* __restrict__ counts,
                                            const int* __restrict__ members, void* __restrict__ out, int N,
                                            float limit, int skip) {
  static_assert(RT == 4, "the dispatch below spells out 0 .. 4 row tiles");
  const int nbt = (NB + WN - 1) / WN, it = blockIdx.x / nbt, cbt = blockIdx.x - it * nbt;
  if (it >= __ldg(counts)) return;
  const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
  if (e == skip) return;  // an expert the family computes itself, left alone as the decode form does
  const int nt = max(0, min(RT, (cnt - 16 * RT * ((threadIdx.x >> 5) / WN) + 15) >> 4));
#define TF_TILES(NT) \
  prompt_tiles<M, EPI, RT, WM, WN, NT, FACT>(X, x_stride, slots, W, scale, KG, NB, members, out, N, limit, cbt, e, first, cnt)
  switch (nt) {
    case 4: TF_TILES(4); break;
    case 3: TF_TILES(3); break;
    case 2: TF_TILES(2); break;
    case 1: TF_TILES(1); break;
    default: TF_TILES(0);
  }
#undef TF_TILES
}

// The instances: SwiGLU gate-up, fp32 and bf16 down, and relu^2 up for experts without a gate (Nemotron).
template __global__ void nvfp4_expert_kernel<2, 2, 4>(const __nv_bfloat16* __restrict__, int, int, const uint4* __restrict__, const float* __restrict__, int, int, const int* __restrict__, const int* __restrict__, const int* __restrict__, void* __restrict__, int, float, int);
template __global__ void nvfp4_expert_kernel<1, 0, 4>(const __nv_bfloat16* __restrict__, int, int, const uint4* __restrict__, const float* __restrict__, int, int, const int* __restrict__, const int* __restrict__, const int* __restrict__, void* __restrict__, int, float, int);
template __global__ void nvfp4_expert_kernel<1, 3, 4>(const __nv_bfloat16* __restrict__, int, int, const uint4* __restrict__, const float* __restrict__, int, int, const int* __restrict__, const int* __restrict__, const int* __restrict__, void* __restrict__, int, float, int);
template __global__ void nvfp4_expert_kernel<1, 1, 4>(const __nv_bfloat16* __restrict__, int, int, const uint4* __restrict__, const float* __restrict__, int, int, const int* __restrict__, const int* __restrict__, const int* __restrict__, void* __restrict__, int, float, int);

// One warp per CTA exposes more independent expert tiles for narrow rank-local projections.
template __global__ void nvfp4_expert_kernel<2, 2, 1>(const __nv_bfloat16* __restrict__, int, int, const uint4* __restrict__, const float* __restrict__, int, int, const int* __restrict__, const int* __restrict__, const int* __restrict__, void* __restrict__, int, float, int);
template __global__ void nvfp4_expert_kernel<1, 0, 1>(const __nv_bfloat16* __restrict__, int, int, const uint4* __restrict__, const float* __restrict__, int, int, const int* __restrict__, const int* __restrict__, const int* __restrict__, void* __restrict__, int, float, int);
template __global__ void nvfp4_expert_kernel<1, 3, 1>(const __nv_bfloat16* __restrict__, int, int, const uint4* __restrict__, const float* __restrict__, int, int, const int* __restrict__, const int* __restrict__, const int* __restrict__, void* __restrict__, int, float, int);
template __global__ void nvfp4_expert_kernel<1, 1, 1>(const __nv_bfloat16* __restrict__, int, int, const uint4* __restrict__, const float* __restrict__, int, int, const int* __restrict__, const int* __restrict__, const int* __restrict__, void* __restrict__, int, float, int);
}

// The prompt form's entries, 4 row tiles a warp in one row of WN warps: relu^2 up, SwiGLU gate-up, bf16 sums down.
#define TF_NVFP4_PROMPT(NAME, M, EPI, WN)                                                                              \
  extern "C" __global__ void __launch_bounds__(WN * 32) NAME(                                                        \
      const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W,                     \
      const float* __restrict__ scale, int KG, int NB, const int* __restrict__ items, const int* __restrict__ counts, \
      const int* __restrict__ members, void* __restrict__ out, int N, float limit, int skip) {                       \
    tf_nvfp4_experts::prompt_body<M, EPI, 4, 1, WN>(X, x_stride, slots, W, scale, KG, NB, items, counts, members,    \
                                                    out, N, limit, skip);                                            \
  }
TF_NVFP4_PROMPT(tf_nvfp4_prompt_relu2, 1, 1, 8)
TF_NVFP4_PROMPT(tf_nvfp4_prompt_swiglu, 2, 2, 4)
TF_NVFP4_PROMPT(tf_nvfp4_prompt_down, 1, 3, 8)

// The decode form's factored sums on the prompt tiles, bit for bit: SwiGLU gate-up to bf16, down to fp32 slot rows.
#define TF_NVFP4_FACT(NAME, M, EPI, WN)                                                                                \
  extern "C" __global__ void __launch_bounds__(WN * 32) NAME(                                                        \
      const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W,                     \
      const float* __restrict__ scale, int KG, int NB, const int* __restrict__ items, const int* __restrict__ counts, \
      const int* __restrict__ members, void* __restrict__ out, int N, float limit, int skip) {                       \
    tf_nvfp4_experts::prompt_body<M, EPI, 4, 1, WN, true>(X, x_stride, slots, W, scale, KG, NB, items, counts,       \
                                                          members, out, N, limit, skip);                             \
  }
TF_NVFP4_FACT(tf_nvfp4_fact_swiglu, 2, 2, 4)
TF_NVFP4_FACT(tf_nvfp4_fact_down_f32, 1, 0, 8)
