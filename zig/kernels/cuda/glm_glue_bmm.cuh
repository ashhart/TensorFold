// qmm._bmm and qmm._bmm_rows by hand (the BF16 matmuls): out [m][n] = x rows . w rows over K slices.
#pragma once

#include "glm_glue_router.cuh"

namespace glm_bmm {

// ROWS rows of one KS-wide K step into smem (pad 8: no bank conflicts), 16-byte loads if aligned, zeros past the end.
template <int ROWS, int KS>
__device__ __forceinline__ void stage(__nv_bfloat16 (*dst)[KS + 8], const __nv_bfloat16* src, int64_t first,
                                      int64_t limit, int64_t stride, int64_t kk, bool wide) {
  for (int idx = threadIdx.x; idx < ROWS * (KS / 8); idx += blockDim.x) {
    const int r = idx / (KS / 8), c = (idx % (KS / 8)) * 8;
    const int64_t row = first + r;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (row < limit) {
      const __nv_bfloat16* p = src + row * stride + kk + c;
      if (wide) {
        v = __ldg(reinterpret_cast<const uint4*>(p));
      } else {
        uint32_t h[4];
        for (int e = 0; e < 4; ++e) h[e] = glm_router::pair(p + 2 * e, true);
        v = make_uint4(h[0], h[1], h[2], h[3]);
      }
    }
    *reinterpret_cast<uint4*>(&dst[r][c]) = v;
  }
}

// mma chain per slice from zero, K ascending, as Triton; decode writes slice z, prompt adds in order; tiles keep bits.
template <int BM, int BN, int KS, int WM, int WN>
__device__ void bmm(const __nv_bfloat16* x, const __nv_bfloat16* w, void* out, float* part, int64_t m, int64_t x_stride,
                    int64_t n, int64_t k, int sk, int rows, int f32) {
  constexpr int MT = BM / WM / 16, NT = BN / WN / 8;
  __shared__ __align__(16) __nv_bfloat16 sa[BM][KS + 8];
  __shared__ __align__(16) __nv_bfloat16 sb[BN][KS + 8];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, q = lane & 3;
  const int64_t bm0 = (int64_t)blockIdx.x * BM, bn0 = (int64_t)blockIdx.y * BN;
  const int wm = (warp / WN) * MT * 16, wn = (warp % WN) * NT * 8;
  const bool xwide = (reinterpret_cast<uintptr_t>(x) & 15) == 0 && (x_stride & 7) == 0;
  const int64_t per = k / sk;
  const int first = rows ? 0 : blockIdx.z, last = rows ? sk : blockIdx.z + 1;
  float total[MT][NT][4];
  for (int s = first; s < last; ++s) {
    float c[MT][NT][4] = {};
    for (int64_t kk = s * per; kk < (s + 1) * per; kk += KS) {
      stage<BM, KS>(sa, x, bm0, m, x_stride, kk, xwide);
      stage<BN, KS>(sb, w, bn0, n, k, kk, true);
      __syncthreads();
#pragma unroll
      for (int k16 = 0; k16 < KS; k16 += 16) {
        uint32_t a[MT][4], b[NT][2];
#pragma unroll
        for (int i = 0; i < MT; ++i) {
          const int r = wm + i * 16 + g;
          a[i][0] = *reinterpret_cast<const uint32_t*>(&sa[r][k16 + q * 2]);
          a[i][1] = *reinterpret_cast<const uint32_t*>(&sa[r + 8][k16 + q * 2]);
          a[i][2] = *reinterpret_cast<const uint32_t*>(&sa[r][k16 + q * 2 + 8]);
          a[i][3] = *reinterpret_cast<const uint32_t*>(&sa[r + 8][k16 + q * 2 + 8]);
        }
#pragma unroll
        for (int j = 0; j < NT; ++j) {
          const int col = wn + j * 8 + g;
          b[j][0] = *reinterpret_cast<const uint32_t*>(&sb[col][k16 + q * 2]);
          b[j][1] = *reinterpret_cast<const uint32_t*>(&sb[col][k16 + q * 2 + 8]);
        }
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
          for (int j = 0; j < NT; ++j) glm_router::mma16816(c[i][j], a[i], b[j]);
      }
      __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) total[i][j][e] = s == first ? c[i][j][e] : total[i][j][e] + c[i][j][e];
  }
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int64_t r = bm0 + wm + i * 16 + g + (e / 2) * 8, col = bn0 + wn + j * 8 + q * 2 + e % 2;
        if (r >= m || col >= n) continue;
        const float v = total[i][j][e];
        if (!rows && sk > 1)
          part[((int64_t)blockIdx.z * m + r) * n + col] = v;
        else if (f32)
          static_cast<float*>(out)[r * n + col] = v;
        else
          static_cast<__nv_bfloat16*>(out)[r * n + col] = __float2bfloat16_rn(v);
      }
}

__device__ __forceinline__ void cp16(void* dst, const void* src, bool ok) {
  const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp16s(unsigned d, const void* src, bool ok) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void ldm4s(uint32_t* r, unsigned a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}
__device__ __forceinline__ void ldm2s(uint32_t* r, unsigned a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
__device__ __forceinline__ void ldm4(uint32_t* r, const void* p) {
  ldm4s(r, static_cast<unsigned>(__cvta_generic_to_shared(p)));
}

// Prompt rows by cp.async (STAGES buffers, swizzled rows, ldmatrix): bmm's rows-form chains. SLICE: decode slice z.
template <int BM, int BN, int KS, int STAGES, int WM, int WN>
struct Tile {
  static constexpr int LD = KS, A = BM * LD, B = BN * LD, STAGED = STAGES * (A + B) * 2, LDO = BN + 4,
                       LDH = BN + 8, THREADS = WM * WN * 32;
  // Dynamic shared bytes: the stages, or the output tile if larger (pitches BN + 4 words, BN + 8 bf16: 16-byte rows).
  static constexpr int smem(bool f32) {
    const int tile = f32 ? BM * LDO * 4 : BM * LDH * 2;
    return tile > STAGED ? tile : STAGED;
  }
  static_assert(KS == 32 || KS == 64, "swizzles for rows of 4 or 8 chunks");
  __device__ static int swz(int r) { return KS == 32 ? (r >> 1) & 3 : r & 7; }
};

template <int BM, int BN, int KS, int STAGES, int WM, int WN, bool SPLIT, bool SLICE = false>
__device__ void bmm_tiles(const __nv_bfloat16* x, const __nv_bfloat16* w, void* out, int64_t m, int64_t x_stride,
                          int64_t n, int64_t k, int sk, int f32) {
  using T = Tile<BM, BN, KS, STAGES, WM, WN>;
  constexpr int MT = BM / WM / 16, NT = BN / WN / 8;
  static_assert(MT >= 1 && (NT == 1 || NT % 2 == 0), "a warp's tile is whole m16 rows and one or pairs of n8 columns");
  extern __shared__ __align__(16) unsigned char smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, q = lane & 3;
  const int64_t bm0 = (int64_t)blockIdx.x * BM, bn0 = (int64_t)blockIdx.y * BN;
  const int wm = (warp / WN) * MT * 16, wn = (warp % WN) * NT * 8;
  const int steps = static_cast<int>((SLICE ? k / sk : k) / KS), per = static_cast<int>(k / sk / KS);
  const int64_t k0 = SLICE ? blockIdx.z * (k / sk) : 0;
  // A thread copies one 16-byte column of rows lr, lr + RPI, ...; a step moves it KS along K (past the matrix: zeros).
  constexpr int CPR = KS / 8, RPI = T::THREADS / CPR, AI = BM / RPI, BI = BN / RPI, STAGE = (T::A + T::B) * 2;
  static_assert(T::THREADS % CPR == 0 && BM % RPI == 0 && BN % RPI == 0 && RPI % 8 == 0, "whole rows a pass");
  const int lr = tid / CPR, lc = (tid % CPR) * 8;
  const unsigned sbase = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  const unsigned soff = (lr * T::LD + ((lc / 8) ^ T::swz(lr)) * 8) * 2;  // rows lr + i RPI swizzle alike (RPI % 8 == 0)
  const __nv_bfloat16* ga[AI];
  const __nv_bfloat16* gb[BI];
  unsigned oka = 0, okb = 0;
#pragma unroll
  for (int i = 0; i < AI; ++i) {
    const int64_t row = bm0 + lr + i * RPI;
    oka |= (row < m ? 1u : 0u) << i;
    ga[i] = x + (row < m ? row : 0) * x_stride + k0 + lc;
  }
#pragma unroll
  for (int i = 0; i < BI; ++i) {
    const int64_t row = bn0 + lr + i * RPI;
    okb |= (row < n ? 1u : 0u) << i;
    gb[i] = w + (row < n ? row : 0) * k + k0 + lc;
  }
  auto load = [&](int st, int kt) {
    const unsigned s = sbase + st * STAGE + soff;
    const int kk = kt * KS;
#pragma unroll
    for (int i = 0; i < AI; ++i) cp16s(s + i * RPI * T::LD * 2, ga[i] + kk, (oka >> i) & 1);
#pragma unroll
    for (int i = 0; i < BI; ++i) cp16s(s + T::A * 2 + i * RPI * T::LD * 2, gb[i] + kk, (okb >> i) & 1);
  };
  // ldmatrix rows in stage 0: the warp's A rows (lane & 15, chunk + 1 for lanes 16 on) and B rows; k16s swizzle as r.
  const int ra = wm + (lane & 15), rb = wn + (NT == 1 ? 0 : (lane >> 4) * 8) + (lane & 7);
  const unsigned fa = sbase + ra * T::LD * 2, fb = sbase + T::A * 2 + rb * T::LD * 2;
  unsigned ca[KS / 16], cb[KS / 16];
#pragma unroll
  for (int s16 = 0; s16 < KS / 16; ++s16) {
    ca[s16] = ((2 * s16 + (lane >> 4)) ^ T::swz(ra)) * 16;
    cb[s16] = ((2 * s16 + ((lane >> 3) & 1)) ^ T::swz(rb)) * 16;
  }
  float c[MT][NT][4] = {}, total[SPLIT ? MT : 1][SPLIT ? NT : 1][4] = {};
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < steps) load(s, s);
    commit();
  }
  int rs = 0, ws = STAGES - 1, left = per;
  for (int kt = 0; kt < steps; ++kt) {
    if constexpr (STAGES == 1) {
      __syncthreads();  // every warp is done with the one buffer's last step
      load(0, kt);
      commit();
      wait<0>();
      __syncthreads();
    } else if constexpr (SLICE) {
      // decode: the next stage's copies go out before this step's wait, as Triton's _bmm issues them (two barriers)
      __syncthreads();  // every warp is done with step kt - 1's buffer (ws)
      if (kt + STAGES - 1 < steps) load(ws, kt + STAGES - 1);
      commit();
      ws = ws + 1 == STAGES ? 0 : ws + 1;
      wait<STAGES - 1>();
      __syncthreads();
    } else {
      wait<STAGES - 2>();
      __syncthreads();
      if (kt + STAGES - 1 < steps) load(ws, kt + STAGES - 1);
      commit();
      ws = ws + 1 == STAGES ? 0 : ws + 1;
    }
    const unsigned sa = fa + rs * STAGE, sb = fb + rs * STAGE;
    rs = rs + 1 == STAGES ? 0 : rs + 1;
#pragma unroll
    for (int k16 = 0; k16 < KS; k16 += 16) {
      uint32_t a[MT][4], b[NT][2];
#pragma unroll
      for (int i = 0; i < MT; ++i) ldm4s(a[i], sa + i * 16 * T::LD * 2 + ca[k16 / 16]);
      if constexpr (NT == 1) {
        ldm2s(b[0], sb + cb[k16 / 16]);
      } else {
#pragma unroll
        for (int j = 0; j < NT; j += 2) {
          uint32_t r[4];
          ldm4s(r, sb + j * 8 * T::LD * 2 + cb[k16 / 16]);
          b[j][0] = r[0], b[j][1] = r[1], b[j + 1][0] = r[2], b[j + 1][1] = r[3];
        }
      }
#pragma unroll
      for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j) glm_router::mma16816(c[i][j], a[i], b[j]);
    }
    if constexpr (SPLIT) {
      if (--left == 0) {  // a slice's last step: its chain into the running sum, the next slice from zero
        const bool first = kt + 1 == per;
        left = per;
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
          for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              total[i][j][e] = first ? c[i][j][e] : total[i][j][e] + c[i][j][e];
              c[i][j][e] = 0.0f;
            }
      }
    }
  }
  wait<0>();
  auto val = [&](int i, int j, int e) -> float {
    if constexpr (SPLIT) return total[i][j][e];
    else return c[i][j][e];
  };
  // The tile into shared memory (Tile's pitches), then rows out 16 bytes a store.
  __syncthreads();
  float* so = reinterpret_cast<float*>(smem);
  __nv_bfloat16* sh = reinterpret_cast<__nv_bfloat16*>(smem);
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int r = wm + i * 16 + g + h * 8, col = wn + j * 8 + q * 2;
        if (f32)
          *reinterpret_cast<float2*>(&so[r * T::LDO + col]) = make_float2(val(i, j, 2 * h), val(i, j, 2 * h + 1));
        else
          *reinterpret_cast<__nv_bfloat162*>(&sh[r * T::LDH + col]) =
              __halves2bfloat162(__float2bfloat16_rn(val(i, j, 2 * h)), __float2bfloat16_rn(val(i, j, 2 * h + 1)));
      }
  __syncthreads();
  const int per_row = f32 ? BN / 4 : BN / 8;
  for (int idx = tid; idx < BM * per_row; idx += T::THREADS) {
    const int r = idx / per_row, v = idx % per_row;
    const int64_t row = bm0 + r, col = bn0 + v * (f32 ? 4 : 8);
    if (row >= m || col >= n) continue;
    if (f32)
      *reinterpret_cast<float4*>(static_cast<float*>(out) + row * n + col) =
          *reinterpret_cast<const float4*>(&so[r * T::LDO + v * 4]);
    else
      *reinterpret_cast<uint4*>(static_cast<__nv_bfloat16*>(out) + row * n + col) =
          *reinterpret_cast<const uint4*>(&sh[r * T::LDH + v * 8]);
  }
}

}  // namespace glm_bmm

// Prompt windows on aligned rows, K sliced or at most 256: 128 x 64 a block of 8 warps, K 64 in 3 cp.async stages.
extern "C" __global__ void __launch_bounds__(256)
    tf_glm_bmm_tile(const __nv_bfloat16* x, const __nv_bfloat16* w, void* out, int64_t m, int64_t x_stride,
                    int64_t n, int64_t k, int sk, int f32) {
  glm_bmm::bmm_tiles<128, 64, 64, 3, 2, 4, true>(x, w, out, m, x_stride, n, k, sk, f32);
}

// Prompt windows on aligned rows, one slice: 128 x 128 a block of 8 warps, K 32 in three stages, two blocks an SM.
extern "C" __global__ void __launch_bounds__(256, 2)
    tf_glm_bmm_tile1(const __nv_bfloat16* x, const __nv_bfloat16* w, void* out, int64_t m, int64_t x_stride,
                     int64_t n, int64_t k, int sk, int f32) {
  glm_bmm::bmm_tiles<128, 128, 32, 3, 2, 4, false>(x, w, out, m, x_stride, n, k, sk, f32);
}

// glue._router_part on aligned rows: part[s][m][e] = x[m] . w[e] over K slice s, tf_glm_router_part's chain, tiled.
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_router_tile(const __nv_bfloat16* x, const __nv_bfloat16* w, float* part, int64_t rows, int64_t x_stride,
                       int64_t d, int64_t ne, int ks) {
  constexpr int BM = 64, BE = 32, KS = 64, LD = KS + 8, A = BM * LD, B = BE * LD, STAGE = (A + B) * 2, RPI = 16;
  __shared__ __align__(16) __nv_bfloat16 sm[2][A + B];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, q = lane & 3;
  const int64_t m0 = (int64_t)blockIdx.x * BM, e0 = (int64_t)blockIdx.y * BE, s = blockIdx.z, span = d / ks;
  const int steps = static_cast<int>(span / KS), lr = tid / 8, lc = (tid % 8) * 8;
  const __nv_bfloat16* ga[BM / RPI];
  const __nv_bfloat16* gb[BE / RPI];
  unsigned oka = 0, okb = 0;
#pragma unroll
  for (int i = 0; i < BM / RPI; ++i) {
    const int64_t row = m0 + lr + i * RPI;
    oka |= (row < rows ? 1u : 0u) << i;
    ga[i] = x + (row < rows ? row : 0) * x_stride + s * span + lc;
  }
#pragma unroll
  for (int i = 0; i < BE / RPI; ++i) {
    const int64_t e = e0 + lr + i * RPI;
    okb |= (e < ne ? 1u : 0u) << i;
    gb[i] = w + (e < ne ? e : 0) * d + s * span + lc;
  }
  const unsigned sbase = static_cast<unsigned>(__cvta_generic_to_shared(sm)), soff = (lr * LD + lc) * 2;
  auto load = [&](int st, int kt) {
    const unsigned a = sbase + st * STAGE + soff;
#pragma unroll
    for (int i = 0; i < BM / RPI; ++i) glm_bmm::cp16s(a + i * RPI * LD * 2, ga[i] + kt * KS, (oka >> i) & 1);
#pragma unroll
    for (int i = 0; i < BE / RPI; ++i) glm_bmm::cp16s(a + A * 2 + i * RPI * LD * 2, gb[i] + kt * KS, (okb >> i) & 1);
  };
  const unsigned fa = sbase + ((warp * 16 + (lane & 15)) * LD + (lane >> 4) * 8) * 2;
  const unsigned fb = sbase + A * 2 + (((lane >> 4) * 8 + (lane & 7)) * LD + ((lane >> 3) & 1) * 8) * 2;
  float c[BE / 8][4] = {};
  load(0, 0);
  glm_bmm::commit();
  for (int kt = 0; kt < steps; ++kt) {
    if (kt + 1 < steps) load((kt + 1) & 1, kt + 1);
    glm_bmm::commit();
    glm_bmm::wait<1>();
    __syncthreads();
    const unsigned st = (kt & 1) * STAGE;
#pragma unroll
    for (int k16 = 0; k16 < KS; k16 += 16) {
      uint32_t a[4], b[BE / 8][2];
      glm_bmm::ldm4s(a, fa + st + k16 * 2);
#pragma unroll
      for (int j = 0; j < BE / 8; j += 2) {
        uint32_t r[4];
        glm_bmm::ldm4s(r, fb + st + (j * 8 * LD + k16) * 2);
        b[j][0] = r[0], b[j][1] = r[1], b[j + 1][0] = r[2], b[j + 1][1] = r[3];
      }
#pragma unroll
      for (int j = 0; j < BE / 8; ++j) glm_router::mma16816(c[j], a, b[j]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int j = 0; j < BE / 8; ++j)
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int jj = 0; jj < 2; ++jj) {
        const int64_t r = m0 + warp * 16 + g + h * 8, col = e0 + j * 8 + q * 2 + jj;
        if (r < rows && col < ne) part[(s * rows + r) * ne + col] = c[j][h * 2 + jj];
      }
}

// Decode windows (under 128 rows): 16 x 64 a block of 4 warps, K 64 at a time, one K slice a block along z.
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_bmm_decode(const __nv_bfloat16* x, const __nv_bfloat16* w, void* out, float* part, int64_t m,
                      int64_t x_stride, int64_t n, int64_t k, int sk, int f32) {
  glm_bmm::bmm<16, 64, 64, 1, 4>(x, w, out, part, m, x_stride, n, k, sk, 0, f32);
}

// Decode windows on aligned rows as Triton's _bmm stages a row (cuda_b16.decodeTile, 4 warps): 16 x BN, K 64, slice z.
namespace glm_bmm {
template <int BN, int ST>
__device__ __forceinline__ void dec_tile(const __nv_bfloat16* x, const __nv_bfloat16* w, void* out, float* part,
                                         int64_t m, int64_t x_stride, int64_t n, int64_t k, int sk, int f32) {
  void* o = sk > 1 ? static_cast<void*>(part + blockIdx.z * m * n) : out;
  bmm_tiles<16, BN, 64, ST, 1, 4, false, true>(x, w, o, m, x_stride, n, k, sk, sk > 1 ? 1 : f32);
}
}  // namespace glm_bmm
#define TF_GLM_DEC_TILE(name, BN, ST)                                                                              \
  extern "C" __global__ void __launch_bounds__(128)                                                                \
      name(const __nv_bfloat16* x, const __nv_bfloat16* w, void* out, float* part, int64_t m, int64_t x_stride,  \
           int64_t n, int64_t k, int sk, int f32) {                                                                \
    glm_bmm::dec_tile<BN, ST>(x, w, out, part, m, x_stride, n, k, sk, f32);                                       \
  }
TF_GLM_DEC_TILE(tf_glm_bmm_d32s1, 32, 1)
TF_GLM_DEC_TILE(tf_glm_bmm_d32s2, 32, 2)
TF_GLM_DEC_TILE(tf_glm_bmm_d32s3, 32, 3)
TF_GLM_DEC_TILE(tf_glm_bmm_d64s2, 64, 2)
#undef TF_GLM_DEC_TILE

// Prompt windows (_bmm_rows): 128 x 64 a block of 4 warps, K 32 at a time, the slices added in order in registers.
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_bmm_rows(const __nv_bfloat16* x, const __nv_bfloat16* w, void* out, int64_t m, int64_t x_stride,
                    int64_t n, int64_t k, int sk, int f32) {
  glm_bmm::bmm<128, 64, 32, 2, 2>(x, w, out, nullptr, m, x_stride, n, k, sk, 1, f32);
}
