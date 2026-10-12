// EXL3 routed experts on CUDA, on grouped.zig's plan: each row's arithmetic is exl3.cu's, with the rotation inline.
#include "exl3.cuh"
#include "experts.cuh"

// A layer's experts: `mats` matrices an expert (gate and up, or down), each trellis `stride` bytes and k + n halves.
struct TfExl3Experts {
  unsigned k, n, bits, tiles_n, mats, epi, book, pad;
  unsigned long long stride;
};

// A warp's two column sums for each of R rows into zs, after its last input block (lanes 4 c + q hold rows 2 q ...).
template <int R>
__device__ __forceinline__ void tf_exl3x_fold(float (&total)[2][R], unsigned sg, unsigned lane, float* zs) {
#pragma unroll
  for (int i = 0; i < R; ++i) {
    for (int c = 0; c < 2; ++c) {
      float v = total[c][i];
      v = __fadd_rn(v, __shfl_xor_sync(TF_FULL, v, 1));
      v = __fadd_rn(v, __shfl_xor_sync(TF_FULL, v, 2));
      if (lane % 4 == 0) zs[i * 128 + 16 * sg + lane / 4 + 8 * c] = v;
    }
  }
}

// Few rows: the shared register path (exl3.cuh), the shared-memory path's sums without its barriers.
template <int K, int R>
__device__ __forceinline__ void tf_exl3x_free(const __nv_bfloat16* X, int x_stride, const unsigned (&xr)[16],
                                              unsigned rows, const uint8_t* T, const __half* suh, const TfExl3Experts g,
                                              unsigned cb, unsigned sg, unsigned lane, float* zs) {
  constexpr unsigned TB = 32 * K;
  const __nv_bfloat16* xrow[R];
#pragma unroll
  for (int i = 0; i < R; ++i) xrow[i] = X + (uint64_t)xr[i] * x_stride;
  float total[2][R];
  // the hot decode loop keeps its codebook a constant: one switch a call
  if (g.book == 2) tf_exl3_free<K, 2, R>(xrow, rows, suh, T + (uint64_t)(8 * cb + sg) * TB, (uint64_t)g.tiles_n * TB, 0, g.k / 128, lane, total);
  else if (g.book == 1) tf_exl3_free<K, 1, R>(xrow, rows, suh, T + (uint64_t)(8 * cb + sg) * TB, (uint64_t)g.tiles_n * TB, 0, g.k / 128, lane, total);
  else tf_exl3_free<K, 0, R>(xrow, rows, suh, T + (uint64_t)(8 * cb + sg) * TB, (uint64_t)g.tiles_n * TB, 0, g.k / 128, lane, total);
  tf_exl3x_fold<R>(total, sg, lane, zs);
}

// More rows: warps rotate the block's rows into shared memory once, then every warp reads them there.
template <int K, int R>
__device__ __forceinline__ void tf_exl3x_shared(const __nv_bfloat16* X, int x_stride, const unsigned (&xr)[16],
                                                unsigned rows, const uint8_t* T, const __half* suh, const TfExl3Experts g,
                                                unsigned cb, unsigned sg, unsigned lane, float* xs, float* zs) {
  constexpr unsigned TB = 32 * K;
  const unsigned r0 = 2 * (lane % 4), kb = g.k / 128;
  const uint64_t row_bytes = (uint64_t)g.tiles_n * TB;
  const uint8_t* col = T + (uint64_t)(8 * cb + sg) * TB;
  float total[2][R], part[2][R];
#pragma unroll
  for (int i = 0; i < R; ++i) total[0][i] = total[1][i] = 0.0f;
  uint4 next = tf_exl3_words<K>(reinterpret_cast<const uint32_t*>(col), lane);
  for (unsigned ib = 0; ib < kb; ++ib) {
    for (unsigned i = sg; i < R; i += 8) {
      float v[4];
      tf_exl3_rotated(X + (uint64_t)xr[i] * x_stride + ib * 128, suh + ib * 128, i < rows, lane, v);
      for (int q = 0; q < 4; ++q) xs[i * 128 + lane + 32 * q] = v[q];
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < R; ++i) part[0][i] = part[1][i] = 0.0f;
    for (unsigned r = 0; r < 8; ++r) {
      const uint4 now = next;
      next = tf_exl3_words<K>(reinterpret_cast<const uint32_t*>(col + (uint64_t)(r < 7 ? 8 * ib + r + 1 : (ib + 1 < kb ? 8 * ib + 8 : 8 * ib + 7)) * row_bytes), lane);
      __half w[8];
      tf_exl3_unpack_any<K>(now, lane, g.book, w);
      tf_exl3_rows<R>(w, xs, r, r0, part);
    }
#pragma unroll
    for (int i = 0; i < R; ++i) {
      total[0][i] = __fadd_rn(total[0][i], part[0][i]);
      total[1][i] = __fadd_rn(total[1][i], part[1][i]);
    }
    __syncthreads();
  }
  tf_exl3x_fold<R>(total, sg, lane, zs);
}

// A row's 128 columns of one matrix out of zs: the transform, then svh times 2^-7, one rounding.
__device__ __forceinline__ void tf_exl3x_out(const float* zs, unsigned i, const __half* svh, unsigned cb, unsigned lane,
                                             float (&v)[4]) {
  for (int q = 0; q < 4; ++q) v[q] = zs[i * 128 + lane + 32 * q];
  tf_exl3_fwht(v, lane);
  for (int q = 0; q < 4; ++q) v[q] = __fmul_rn(v[q], __fmul_rn(__half2float(svh[cb * 128 + lane + 32 * q]), 0.0078125f));
}

// Units (item, 128 columns), 16 pairs at a time; epi 0 fp32, 2 SwiGLU bf16, 3 bf16 (experts.cuh's numbers).
template <int K>
__device__ __forceinline__ void tf_exl3x_units(const __nv_bfloat16* X, int x_stride, int slots, const uint8_t* W,
                                               const __half* H, const int* items, const int* counts, const int* members,
                                               void* out, float limit, int skip, const TfExl3Experts g, float* xs,
                                               float* zs) {
  const unsigned tid = threadIdx.x, sg = tid / 32, lane = tid % 32, nb = g.n / 128;
  const unsigned units = (unsigned)__ldg(counts) * nb;
  __shared__ unsigned xr[16], pr[16];
  for (unsigned unit = blockIdx.x; unit < units; unit += gridDim.x) {
    const unsigned it = unit / nb, cb = unit % nb;
    const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    if (e == skip) continue;
    for (int r0 = 0; r0 < cnt; r0 += 16) {
      const unsigned rows = cnt - r0 < 16 ? cnt - r0 : 16;
      if (tid < 16) {
        const int p = tid < rows ? __ldg(members + first + r0 + tid) : 0;
        pr[tid] = p;
        xr[tid] = slots ? p / slots : p;
      }
      __syncthreads();
      unsigned rx[16];
#pragma unroll
      for (int i = 0; i < 16; ++i) rx[i] = xr[i];
      // the capacity the chunk's rows need: a row's sums are the same at every capacity and on either path
      for (unsigned m = 0; m < g.mats; ++m) {
        const uint64_t at = (uint64_t)e * g.mats + m;
        const uint8_t* t = W + at * g.stride;
        const __half* suh = H + at * (g.k + g.n);
        float* z = zs + m * 2048;
        if (rows <= 1) tf_exl3x_free<K, 1>(X, x_stride, rx, rows, t, suh, g, cb, sg, lane, z);
        else if (rows <= 2) tf_exl3x_free<K, 2>(X, x_stride, rx, rows, t, suh, g, cb, sg, lane, z);
        else tf_exl3x_shared<K, 16>(X, x_stride, rx, rows, t, suh, g, cb, sg, lane, xs, z);
      }
      __syncthreads();
      for (unsigned i = sg; i < rows; i += 8) {
        const uint64_t at = (uint64_t)e * g.mats;
        float a[4], b[4];
        tf_exl3x_out(zs, i, H + at * (g.k + g.n) + g.k, cb, lane, a);
        if (g.epi == 2) tf_exl3x_out(zs + 2048, i, H + (at + 1) * (g.k + g.n) + g.k, cb, lane, b);
        for (int q = 0; q < 4; ++q) {
          const uint64_t o = (uint64_t)pr[i] * g.n + cb * 128 + lane + 32 * q;
          if (g.epi == 0) reinterpret_cast<float*>(out)[o] = a[q];
          else reinterpret_cast<__nv_bfloat16*>(out)[o] = __float2bfloat16_rn(g.epi == 2 ? swiglu(a[q], b[q], limit) : a[q]);
        }
      }
      __syncthreads();
    }
  }
}

__device__ __forceinline__ void tf_exl3x_width(const __nv_bfloat16* X, int x_stride, int slots, const uint8_t* W,
                                               const __half* H, const int* items, const int* counts, const int* members,
                                               void* out, float limit, int skip, const TfExl3Experts g, float* xs,
                                               float* zs) {
  switch (g.bits) {
    case 1: tf_exl3x_units<1>(X, x_stride, slots, W, H, items, counts, members, out, limit, skip, g, xs, zs); break;
    case 2: tf_exl3x_units<2>(X, x_stride, slots, W, H, items, counts, members, out, limit, skip, g, xs, zs); break;
    case 3: tf_exl3x_units<3>(X, x_stride, slots, W, H, items, counts, members, out, limit, skip, g, xs, zs); break;
    case 4: tf_exl3x_units<4>(X, x_stride, slots, W, H, items, counts, members, out, limit, skip, g, xs, zs); break;
    case 5: tf_exl3x_units<5>(X, x_stride, slots, W, H, items, counts, members, out, limit, skip, g, xs, zs); break;
    case 6: tf_exl3x_units<6>(X, x_stride, slots, W, H, items, counts, members, out, limit, skip, g, xs, zs); break;
    case 7: tf_exl3x_units<7>(X, x_stride, slots, W, H, items, counts, members, out, limit, skip, g, xs, zs); break;
    case 8: tf_exl3x_units<8>(X, x_stride, slots, W, H, items, counts, members, out, limit, skip, g, xs, zs); break;
    default: break;
  }
}

// One kernel for every width and codebook; the layer's geometry says which, the epilogue and the matrices.
extern "C" __global__ void __launch_bounds__(256) tf_exl3_experts(const __nv_bfloat16* X, int x_stride, int slots,
                                                                 const uint8_t* W, const __half* H, const int* items,
                                                                 const int* counts, const int* members, void* out,
                                                                 float limit, int skip, const TfExl3Experts g) {
  __shared__ __align__(16) float xs[16 * 128], zs[2 * 16 * 128];
  tf_exl3x_width(X, x_stride, slots, W, H, items, counts, members, out, limit, skip, g, xs, zs);
}

// Prompt rows in exl3.cu's prompt arithmetic: an item's 64 pairs a block, rows rotated inline, gate and up together.
template <int K, int M>
__device__ __forceinline__ void tf_exl3x_prompt_unit(const __nv_bfloat16* X, int x_stride, const unsigned* xr,
                                                     const unsigned* pr, unsigned rows, const uint8_t* W, const __half* H,
                                                     void* out, float limit, unsigned e, unsigned cb,
                                                     const TfExl3Experts g, float* buf) {
  constexpr unsigned TB = 32 * K;
  const unsigned tid = threadIdx.x, sg = tid / 32, lane = tid % 32, g8 = lane / 4, t4 = lane % 4, kb = g.k / 128;
  const unsigned tiles = (rows + 15) / 16;
  __half* xs = reinterpret_cast<__half*>(buf);  // [M][64 rows][128] during the sums, buf [64][128] fp32 after them
  const uint64_t row_bytes = (uint64_t)g.tiles_n * TB;
  const uint8_t* col[M];
  const __half* suh[M];
  uint4 next[M];
  float acc[M][4][2][4];
#pragma unroll
  for (int m = 0; m < M; ++m) {
    const uint64_t at = (uint64_t)e * g.mats + m;
    col[m] = W + at * g.stride + (uint64_t)(8 * cb + sg) * TB;
    suh[m] = H + at * (g.k + g.n);
    next[m] = tf_exl3_words<K>(reinterpret_cast<const uint32_t*>(col[m]), lane);
    for (int i = 0; i < 4; ++i)
      for (int n = 0; n < 2; ++n) acc[m][i][n][0] = acc[m][i][n][1] = acc[m][i][n][2] = acc[m][i][n][3] = 0.0f;
  }
  for (unsigned ib = 0; ib < kb; ++ib) {
#pragma unroll
    for (int m = 0; m < M; ++m)
      for (unsigned i = sg; i < 64; i += 8) {
        float v[4];
        for (int q = 0; q < 4; ++q) {
          const unsigned c = ib * 128 + lane + 32 * q;
          v[q] = i < rows ? __fmul_rn(__bfloat162float(X[(uint64_t)xr[i] * x_stride + c]), __half2float(suh[m][c])) : 0.0f;
        }
        tf_exl3_fwht(v, lane);
        for (int q = 0; q < 4; ++q) xs[(m * 64 + i) * TF_EXL3_XS + lane + 32 * q] = __float2half_rn(__fmul_rn(v[q], 0.0625f));
      }
    __syncthreads();
    for (unsigned r = 0; r < 8; ++r) {
#pragma unroll
      for (int m = 0; m < M; ++m) {
        const uint4 now = next[m];
        next[m] = tf_exl3_words<K>(reinterpret_cast<const uint32_t*>(col[m] + (uint64_t)(r < 7 ? 8 * ib + r + 1 : (ib + 1 < kb ? 8 * ib + 8 : 8 * ib + 7)) * row_bytes), lane);
        __half w[8];
        tf_exl3_unpack_any<K>(now, lane, g.book, w);
        const uint32_t b[2][2] = {{tf_exl3_pair(w[0], w[1]), tf_exl3_pair(w[2], w[3])}, {tf_exl3_pair(w[4], w[5]), tf_exl3_pair(w[6], w[7])}};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          if (i >= tiles) break;
          const __half* x = xs + (m * 64 + 16 * i + g8) * TF_EXL3_XS + 16 * r + 2 * t4;
          const uint32_t a[4] = {*reinterpret_cast<const uint32_t*>(x), *reinterpret_cast<const uint32_t*>(x + 8 * TF_EXL3_XS),
                                 *reinterpret_cast<const uint32_t*>(x + 8), *reinterpret_cast<const uint32_t*>(x + 8 * TF_EXL3_XS + 8)};
          tf_exl3_mma(acc[m][i][0], a, b[0][0], b[0][1]);
          tf_exl3_mma(acc[m][i][1], a, b[1][0], b[1][1]);
        }
      }
    }
    __syncthreads();
  }
  // each matrix in turn: its sums into buf, a warp a row through the transform and svh times 2^-3
  float first[8][4];
#pragma unroll
  for (int m = 0; m < M; ++m) {
#pragma unroll
    for (int i = 0; i < 4; ++i)
      for (int n = 0; n < 2; ++n)
        for (int h = 0; h < 2; ++h) {
          const unsigned c = 16 * sg + 8 * n + 2 * t4, row = 16 * i + g8 + 8 * h;
          buf[row * 128 + c] = acc[m][i][n][2 * h];
          buf[row * 128 + c + 1] = acc[m][i][n][2 * h + 1];
        }
    __syncthreads();
    const __half* svh = H + ((uint64_t)e * g.mats + m) * (g.k + g.n) + g.k;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const unsigned i = sg + 8 * j;
      if (i >= rows) break;
      float v[4];
      for (int q = 0; q < 4; ++q) v[q] = buf[i * 128 + lane + 32 * q];
      tf_exl3_fwht(v, lane);
      for (int q = 0; q < 4; ++q) {
        const unsigned c = cb * 128 + lane + 32 * q;
        v[q] = __fmul_rn(v[q], __fmul_rn(__half2float(svh[c]), 0.125f));
        const uint64_t o = (uint64_t)pr[i] * g.n + c;
        if (m + 1 < M) first[j][q] = v[q];
        else if (g.epi == 0) reinterpret_cast<float*>(out)[o] = v[q];
        else reinterpret_cast<__nv_bfloat16*>(out)[o] = __float2bfloat16_rn(g.epi == 2 ? swiglu(first[j][q], v[q], limit) : v[q]);
      }
    }
    __syncthreads();
  }
}

template <int K>
__device__ __forceinline__ void tf_exl3x_prompt_units(const __nv_bfloat16* X, int x_stride, int slots, const uint8_t* W,
                                                      const __half* H, const int* items, const int* counts,
                                                      const int* members, void* out, float limit, int skip,
                                                      const TfExl3Experts g, float* buf) {
  const unsigned tid = threadIdx.x, nb = g.n / 128;
  const unsigned units = (unsigned)__ldg(counts) * nb;
  __shared__ unsigned xr[64], pr[64];
  for (unsigned unit = blockIdx.x; unit < units; unit += gridDim.x) {
    const unsigned it = unit / nb, cb = unit % nb;
    const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    if (e == skip) continue;
    for (int r0 = 0; r0 < cnt; r0 += 64) {
      const unsigned rows = cnt - r0 < 64 ? cnt - r0 : 64;
      if (tid < 64) {
        const int p = tid < rows ? __ldg(members + first + r0 + tid) : 0;
        pr[tid] = p;
        xr[tid] = slots ? p / slots : p;
      }
      __syncthreads();
      if (g.mats == 2) tf_exl3x_prompt_unit<K, 2>(X, x_stride, xr, pr, rows, W, H, out, limit, e, cb, g, buf);
      else tf_exl3x_prompt_unit<K, 1>(X, x_stride, xr, pr, rows, W, H, out, limit, e, cb, g, buf);
    }
  }
}

#define TF_EXL3X_PROMPT_CASE(B) \
  case B: tf_exl3x_prompt_units<B>(X, x_stride, slots, W, H, items, counts, members, out, limit, skip, g, buf); break;
// One prompt kernel for every width and codebook. buf holds two matrices' fp16 rows, then fp32 sums.
extern "C" __global__ void __launch_bounds__(256) tf_exl3_experts_prompt(const __nv_bfloat16* X, int x_stride, int slots,
                                                                        const uint8_t* W, const __half* H, const int* items,
                                                                        const int* counts, const int* members, void* out,
                                                                        float limit, int skip, const TfExl3Experts g) {
  __shared__ __align__(16) float buf[64 * TF_EXL3_XS];
  switch (g.bits) {
    TF_EXL3X_PROMPT_CASE(1) TF_EXL3X_PROMPT_CASE(2) TF_EXL3X_PROMPT_CASE(3) TF_EXL3X_PROMPT_CASE(4)
    TF_EXL3X_PROMPT_CASE(5) TF_EXL3X_PROMPT_CASE(6) TF_EXL3X_PROMPT_CASE(7) TF_EXL3X_PROMPT_CASE(8)
    default: break;
  }
}
