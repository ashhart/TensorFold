// EXL3 projections on CUDA: trellis tiles as stored ([K/16, N/16, 16 bits] int16), core/exl3_format.zig's codebooks.
#include "exl3_candidate.cuh"

struct TfExl3Rows {
  unsigned rows, x_stride, y_stride, y_f32;
};
struct TfExl3Matrix {
  unsigned k, n, bits, tiles_n, split;
};

// A block's tail from zs (rows x 128 sums): a split's parts summed in order by the last part, transform, scale.
__device__ __forceinline__ void tf_exl3_tail(unsigned rows, const __half* svh, float out_scale, const TfExl3Rows a,
                                             const TfExl3Matrix m, uint8_t* Y, float* ws, unsigned* tickets, unsigned ob,
                                             unsigned sg, unsigned lane, unsigned tid, float* zs) {
  __shared__ unsigned last;
  if (m.split > 1) {
    float* part = ws + ((uint64_t)ob * m.split + blockIdx.y) * (16 * 128);
    for (unsigned i = tid; i < rows * 128; i += 256) part[i] = zs[i];
    __threadfence();
    __syncthreads();
    if (tid == 0) last = atomicAdd(tickets + ob, 1u) == m.split - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    const float* parts = ws + (uint64_t)ob * m.split * (16 * 128);
    for (unsigned i = tid; i < rows * 128; i += 256) {
      float v = __ldcg(parts + i);
      for (unsigned p = 1; p < m.split; ++p) v = __fadd_rn(v, __ldcg(parts + p * (16 * 128) + i));
      zs[i] = v;
    }
    if (tid == 0) tickets[ob] = 0;
    __syncthreads();
  }
  for (unsigned i = sg; i < rows; i += 8) {
    float v[4];
    for (int q = 0; q < 4; ++q) v[q] = zs[i * 128 + lane + 32 * q];
    tf_exl3_fwht(v, lane);
    for (int q = 0; q < 4; ++q) {
      const unsigned c = ob * 128 + lane + 32 * q;
      // svh times a power of two is exact; one rounding remains
      const float scale = __fmul_rn(__half2float(svh[c]), out_scale);
      tf_exl3_store(Y, (uint64_t)i * a.y_stride + c, __fmul_rn(v[q], scale), a.y_f32);
    }
  }
}

// Qualification: every state's weight under codebook c, for the host to compare with exl3_format.codebookValue.
extern "C" __global__ void tf_exl3_codebook(unsigned c, __half* out) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 65536) return;
  out[i] = c == 2 ? tf_exl3_value<2>(i) : c == 1 ? tf_exl3_value<1>(i) : tf_exl3_value<0>(i);
}

template <int K>
__device__ __forceinline__ void tf_exl3_unpack_tile(const uint32_t* tile, unsigned c, unsigned lane, __half (&w)[8]) {
  if (c == 2) tf_exl3_decode<K, 2>(tile, lane, w);
  else if (c == 1) tf_exl3_decode<K, 1>(tile, lane, w);
  else tf_exl3_decode<K, 0>(tile, lane, w);
}

// Qualification: W_q [K, N] fp16 in the rotated domain, a warp per tile, as exl3_format.unpack places it.
extern "C" __global__ void __launch_bounds__(32) tf_exl3_unpack(const uint8_t* T, TfExl3Matrix m, unsigned c, __half* out) {
  const unsigned lane = threadIdx.x, tn = blockIdx.x, tk = blockIdx.y;
  const uint32_t* tile = reinterpret_cast<const uint32_t*>(T + ((uint64_t)tk * m.tiles_n + tn) * 32 * m.bits);
  __half w[8];
  switch (m.bits) {
    case 1: tf_exl3_unpack_tile<1>(tile, c, lane, w); break;
    case 2: tf_exl3_unpack_tile<2>(tile, c, lane, w); break;
    case 3: tf_exl3_unpack_tile<3>(tile, c, lane, w); break;
    case 4: tf_exl3_unpack_tile<4>(tile, c, lane, w); break;
    case 5: tf_exl3_unpack_tile<5>(tile, c, lane, w); break;
    case 6: tf_exl3_unpack_tile<6>(tile, c, lane, w); break;
    case 7: tf_exl3_unpack_tile<7>(tile, c, lane, w); break;
    default: tf_exl3_unpack_tile<8>(tile, c, lane, w); break;
  }
  for (unsigned j = 0; j < 8; ++j) {
    const unsigned row = 16 * tk + 2 * (lane % 4) + (j & 1) + 8 * ((j >> 1) & 1), col = 16 * tn + lane / 4 + 8 * (j >> 2);
    out[(uint64_t)row * m.n + col] = w[j];
  }
}

// Prompt rows (docs/recipes/exl3.md, Prompts): (x * suh) H times 2^-4 rounded once to fp16; 2^-3 goes on the output.
extern "C" __global__ void __launch_bounds__(32) tf_exl3_rotate_h(const __nv_bfloat16* X, unsigned x_stride, const __half* suh,
                                                                  __half* R, unsigned r_stride) {
  const unsigned lane = threadIdx.x, row = blockIdx.y, base = blockIdx.x * 128;
  float v[4];
  for (int q = 0; q < 4; ++q) {
    const unsigned c = base + lane + 32 * q;
    v[q] = __fmul_rn(__bfloat162float(X[(uint64_t)row * x_stride + c]), __half2float(suh[c]));
  }
  tf_exl3_fwht(v, lane);
  for (int q = 0; q < 4; ++q) R[(uint64_t)row * r_stride + base + lane + 32 * q] = __float2half_rn(__fmul_rn(v[q], 0.0625f));
}

// 64 prompt rows x 128 columns a block: each tile decodes once into mma.m16n8k16 B fragments (their value order).
template <int K, int C>
__device__ __forceinline__ void tf_exl3_prompt_job(const __half* Xh, unsigned rows, const uint8_t* T, const __half* svh,
                                                   const TfExl3Matrix m, uint8_t* Y, unsigned y_f32, float* buf) {
  constexpr unsigned TB = 32 * K;
  const unsigned tid = threadIdx.x, sg = tid / 32, lane = tid % 32, g8 = lane / 4, t4 = lane % 4, ob = blockIdx.x;
  const unsigned row0 = blockIdx.y * 64, kb = m.k / 128;
  __half* xs = reinterpret_cast<__half*>(buf);
  const uint64_t row_bytes = (uint64_t)m.tiles_n * TB;
  const uint8_t* col = T + (uint64_t)(8 * ob + sg) * TB;
  float acc[4][2][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
    for (int n = 0; n < 2; ++n) acc[i][n][0] = acc[i][n][1] = acc[i][n][2] = acc[i][n][3] = 0.0f;
  uint4 next = tf_exl3_words<K>(reinterpret_cast<const uint32_t*>(col), lane);
  for (unsigned ib = 0; ib < kb; ++ib) {
    // the block's 64 rows of this 128-input block, fp16, zeros past the last row
    for (unsigned i = tid; i < 64 * 16; i += 256) {
      const unsigned r = i / 16, c = (i % 16) * 8;
      const uint4 v = row0 + r < rows ? *reinterpret_cast<const uint4*>(Xh + (uint64_t)(row0 + r) * m.k + ib * 128 + c) : make_uint4(0, 0, 0, 0);
      *reinterpret_cast<uint4*>(xs + r * TF_EXL3_XS + c) = v;
    }
    __syncthreads();
    for (unsigned r = 0; r < 8; ++r) {
      const uint4 now = next;
      next = tf_exl3_words<K>(reinterpret_cast<const uint32_t*>(col + (uint64_t)(r < 7 ? 8 * ib + r + 1 : (ib + 1 < kb ? 8 * ib + 8 : 8 * ib + 7)) * row_bytes), lane);
      __half w[8];
      tf_exl3_unpack<K, C>(now, lane, w);
      const uint32_t b[2][2] = {{tf_exl3_pair(w[0], w[1]), tf_exl3_pair(w[2], w[3])}, {tf_exl3_pair(w[4], w[5]), tf_exl3_pair(w[6], w[7])}};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const __half* x = xs + (16 * i + g8) * TF_EXL3_XS + 16 * r + 2 * t4;
        const uint32_t a[4] = {*reinterpret_cast<const uint32_t*>(x), *reinterpret_cast<const uint32_t*>(x + 8 * TF_EXL3_XS),
                               *reinterpret_cast<const uint32_t*>(x + 8), *reinterpret_cast<const uint32_t*>(x + 8 * TF_EXL3_XS + 8)};
        tf_exl3_mma(acc[i][0], a, b[0][0], b[0][1]);
        tf_exl3_mma(acc[i][1], a, b[1][0], b[1][1]);
      }
    }
    __syncthreads();
  }
  // the sums by row into shared memory (over the rows' halves), then a warp a row: transform, svh, 2^-3
#pragma unroll
  for (int i = 0; i < 4; ++i)
    for (int n = 0; n < 2; ++n)
      for (int h = 0; h < 2; ++h) {
        const unsigned c = 16 * sg + 8 * n + 2 * t4;
        buf[(16 * i + g8 + 8 * h) * 128 + c] = acc[i][n][2 * h];
        buf[(16 * i + g8 + 8 * h) * 128 + c + 1] = acc[i][n][2 * h + 1];
      }
  __syncthreads();
  for (unsigned i = sg; i < 64 && row0 + i < rows; i += 8) {
    float v[4];
    for (int q = 0; q < 4; ++q) v[q] = buf[i * 128 + lane + 32 * q];
    tf_exl3_fwht(v, lane);
    for (int q = 0; q < 4; ++q) {
      const unsigned c = ob * 128 + lane + 32 * q;
      tf_exl3_store(Y, (uint64_t)(row0 + i) * m.n + c, __fmul_rn(v[q], __fmul_rn(__half2float(svh[c]), 0.125f)), y_f32);
    }
  }
}

// One prompt kernel a codebook; blocks (128-column block, 64-row block).
#define TF_EXL3_PROMPT(C)                                                                                            \
  extern "C" __global__ void __launch_bounds__(256) tf_exl3_prompt_##C(const __half* Xh, unsigned rows,              \
                                                                      const uint8_t* T, const __half* svh,          \
                                                                      const TfExl3Matrix m, uint8_t* Y,              \
                                                                      unsigned y_f32) {                              \
    __shared__ __align__(16) float buf[64 * 128];                                                                    \
    switch (m.bits) {                                                                                                \
      case 1: tf_exl3_prompt_job<1, C>(Xh, rows, T, svh, m, Y, y_f32, buf); break;                                   \
      case 2: tf_exl3_prompt_job<2, C>(Xh, rows, T, svh, m, Y, y_f32, buf); break;                                   \
      case 3: tf_exl3_prompt_job<3, C>(Xh, rows, T, svh, m, Y, y_f32, buf); break;                                   \
      case 4: tf_exl3_prompt_job<4, C>(Xh, rows, T, svh, m, Y, y_f32, buf); break;                                   \
      case 5: tf_exl3_prompt_job<5, C>(Xh, rows, T, svh, m, Y, y_f32, buf); break;                                   \
      case 6: tf_exl3_prompt_job<6, C>(Xh, rows, T, svh, m, Y, y_f32, buf); break;                                   \
      case 7: tf_exl3_prompt_job<7, C>(Xh, rows, T, svh, m, Y, y_f32, buf); break;                                   \
      case 8: tf_exl3_prompt_job<8, C>(Xh, rows, T, svh, m, Y, y_f32, buf); break;                                   \
      default: break;                                                                                                \
    }                                                                                                                \
  }
TF_EXL3_PROMPT(0)
TF_EXL3_PROMPT(1)
TF_EXL3_PROMPT(2)


// Decode rows for the tensor units: tf_exl3_rotate_h's values, each 128-block in fragment order (t4, r, 4 halves).
extern "C" __global__ void __launch_bounds__(32) tf_exl3_rotate_f(const __nv_bfloat16* X, unsigned x_stride, const __half* suh,
                                                                  __half* R, unsigned r_stride) {
  const unsigned lane = threadIdx.x, row = blockIdx.y, base = blockIdx.x * 128;
  float v[4];
  for (int q = 0; q < 4; ++q) {
    const unsigned c = base + lane + 32 * q;
    v[q] = __fmul_rn(__bfloat162float(X[(uint64_t)row * x_stride + c]), __half2float(suh[c]));
  }
  tf_exl3_fwht(v, lane);
  for (int q = 0; q < 4; ++q) {
    const unsigned j = lane + 32 * q, r = j / 16, c = j % 16;
    const unsigned at = ((c % 8) / 2) * 32 + r * 4 + (c % 2) + 2 * (c / 8);
    R[(uint64_t)row * r_stride + base + at] = __float2half_rn(__fmul_rn(v[q], 0.0625f));
  }
}

// Decode rows on the tensor units, 1-16 a launch: tf_exl3_rotate_f's rows as A fragments, the same at every count.
template <int K, int C>
__device__ __forceinline__ void tf_exl3_tc_job(const __half* Xh, const uint8_t* T, const __half* svh, const TfExl3Rows a,
                                               const TfExl3Matrix m, uint8_t* Y, float* ws, unsigned* tickets, float* zs) {
  constexpr unsigned TB = 32 * K;
  const unsigned tid = threadIdx.x, sg = tid / 32, lane = tid % 32, g8 = lane / 4, t4 = lane % 4, ob = blockIdx.x;
  const unsigned rows = a.rows < 16 ? a.rows : 16, kb = m.k / 128, per = (kb + m.split - 1) / m.split;
  const unsigned b0 = blockIdx.y * per, b1 = b0 + per < kb ? b0 + per : kb;
  const uint64_t row_bytes = (uint64_t)m.tiles_n * TB;
  const uint8_t* col = T + (uint64_t)(8 * ob + sg) * TB;
  const bool lo = g8 < rows, hi = g8 + 8 < rows;
  // a lane's A words of a 128-block, rows g8 and g8 + 8 (r 0-7: columns 2t4, 2t4 + 8), loaded a block ahead
  const uint4* x0 = reinterpret_cast<const uint4*>(Xh + (uint64_t)(lo ? g8 : 0) * m.k + 32 * t4);
  const uint4* x1 = reinterpret_cast<const uint4*>(Xh + (uint64_t)(hi ? g8 + 8 : 0) * m.k + 32 * t4);
  float acc[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
  uint4 next = tf_exl3_words<K>(reinterpret_cast<const uint32_t*>(col + (uint64_t)(8 * b0) * row_bytes), lane);
  uint4 an[2][4];
  for (int q = 0; q < 4; ++q) {
    an[0][q] = lo ? __ldg(x0 + b0 * 16 + q) : make_uint4(0, 0, 0, 0);
    an[1][q] = hi ? __ldg(x1 + b0 * 16 + q) : make_uint4(0, 0, 0, 0);
  }
  for (unsigned ib = b0; ib < b1; ++ib) {
    uint32_t au[2][16];
    for (int q = 0; q < 4; ++q) {
      au[0][4 * q] = an[0][q].x, au[0][4 * q + 1] = an[0][q].y, au[0][4 * q + 2] = an[0][q].z, au[0][4 * q + 3] = an[0][q].w;
      au[1][4 * q] = an[1][q].x, au[1][4 * q + 1] = an[1][q].y, au[1][4 * q + 2] = an[1][q].z, au[1][4 * q + 3] = an[1][q].w;
    }
    if (ib + 1 < b1)
      for (int q = 0; q < 4; ++q) {
        an[0][q] = lo ? __ldg(x0 + (ib + 1) * 16 + q) : make_uint4(0, 0, 0, 0);
        an[1][q] = hi ? __ldg(x1 + (ib + 1) * 16 + q) : make_uint4(0, 0, 0, 0);
      }
#pragma unroll
    for (unsigned r = 0; r < 8; ++r) {
      const uint4 now = next;
      next = tf_exl3_words<K>(reinterpret_cast<const uint32_t*>(col + (uint64_t)(r < 7 ? 8 * ib + r + 1 : (ib + 1 < b1 ? 8 * ib + 8 : 8 * ib + 7)) * row_bytes), lane);
      const uint32_t af[4] = {au[0][2 * r], au[1][2 * r], au[0][2 * r + 1], au[1][2 * r + 1]};
      __half w[8];
      tf_exl3_unpack<K, C>(now, lane, w);
      tf_exl3_mma(acc[0], af, tf_exl3_pair(w[0], w[1]), tf_exl3_pair(w[2], w[3]));
      tf_exl3_mma(acc[1], af, tf_exl3_pair(w[4], w[5]), tf_exl3_pair(w[6], w[7]));
    }
  }
#pragma unroll
  for (int n = 0; n < 2; ++n)
    for (int h = 0; h < 2; ++h) {
      const unsigned c = 16 * sg + 8 * n + 2 * t4, row = g8 + 8 * h;
      if (row < rows) {
        zs[row * 128 + c] = acc[n][2 * h];
        zs[row * 128 + c + 1] = acc[n][2 * h + 1];
      }
    }
  __syncthreads();
  tf_exl3_tail(rows, svh, 0.125f, a, m, Y, ws, tickets, ob, sg, lane, tid, zs);
}

#define TF_EXL3_TC_CASE(B, C) \
  case B: tf_exl3_tc_job<B, C>(Xh, T, svh, a, m, Y, ws, tickets, zs); break;
#define TF_EXL3_TC(C)                                                                                                \
  extern "C" __global__ void __launch_bounds__(256) tf_exl3_tc_##C(const __half* Xh, const uint8_t* T, const __half* svh, \
                                                                  const TfExl3Rows a, const TfExl3Matrix m, uint8_t* Y, \
                                                                  float* ws, unsigned* tickets) {                   \
    __shared__ __align__(16) float zs[16 * 128];                                                                     \
    switch (m.bits) {                                                                                                \
      TF_EXL3_TC_CASE(1, C) TF_EXL3_TC_CASE(2, C) TF_EXL3_TC_CASE(3, C) TF_EXL3_TC_CASE(4, C)                        \
      TF_EXL3_TC_CASE(5, C) TF_EXL3_TC_CASE(6, C) TF_EXL3_TC_CASE(7, C) TF_EXL3_TC_CASE(8, C)                        \
      default: break;                                                                                                \
    }                                                                                                                \
  }
TF_EXL3_TC(0)
TF_EXL3_TC(1)
TF_EXL3_TC(2)
