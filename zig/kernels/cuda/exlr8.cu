// EXLR8 lane projections on CUDA: the Metal kernels' arithmetic in the same order, so rows match them bit for bit.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#define TF_NONE 0xFFFFFFFFu
#define TF_FULL 0xFFFFFFFFu

struct TfExlr8Job {
  unsigned k, blocks, base, in_scale, out_scale, out_sign, out_block, x_col, y_col, raw, members, count;
};
struct TfExlr8Rows {
  unsigned rows, x_stride, y_stride, y_f32;
};
struct TfExlr8Rotate {
  unsigned rows, x_stride, x_col, x_f32, r_stride, r_col, sign, scale, slot_cols, scales, row_slots, pad1;
};
struct TfExlr8Combine {
  unsigned rows, slots, z_stride, y_stride, y_f32, sign, z_slot, per_row;
};

// The unnormalized Sylvester transform of 128 values held as lane + 32 q, in one fixed order of fp32 adds.
__device__ __forceinline__ void tf_exlr8_fwht(float (&v)[4], unsigned lane) {
  for (unsigned d = 1; d < 32; d <<= 1) {
    for (int q = 0; q < 4; ++q) {
      const float other = __shfl_xor_sync(TF_FULL, v[q], d);
      v[q] = (lane & d) ? __fsub_rn(other, v[q]) : __fadd_rn(v[q], other);
    }
  }
  const float a0 = __fadd_rn(v[0], v[1]), a1 = __fsub_rn(v[0], v[1]);
  const float a2 = __fadd_rn(v[2], v[3]), a3 = __fsub_rn(v[2], v[3]);
  v[0] = __fadd_rn(a0, a2);
  v[1] = __fadd_rn(a1, a3);
  v[2] = __fsub_rn(a0, a2);
  v[3] = __fsub_rn(a1, a3);
}

// The mcg codebook's fp16 sum times 2^-8 (xor 0x1B60, not 0x3B60: each half 8 binades lower, still normal; exact).
__device__ __forceinline__ unsigned tf_exlr8_sum8(unsigned state) {
  unsigned x = state * 0xCBAC1FEDu;
  x = (x & 0x8FFF8FFFu) ^ 0x1B601B60u;
  return __half_as_ushort(__hadd(__ushort_as_half((unsigned short)(x & 0xFFFFu)), __ushort_as_half((unsigned short)(x >> 16))));
}

// Two states' weights: at 2^-8 E4M3 nearest even is the fp16 pattern's nearest even at bit 7, subnormals included.
__device__ __forceinline__ __half2 tf_exlr8_pair(unsigned s0, unsigned s1) {
  const unsigned b = tf_exlr8_sum8(s0) | (tf_exlr8_sum8(s1) << 16);
  const unsigned r = (b + 0x003F003Fu + ((b >> 7) & 0x00010001u)) & 0xFF80FF80u;
  const __half2 v = __halves2half2(__ushort_as_half((unsigned short)(r & 0xFFFFu)), __ushort_as_half((unsigned short)(r >> 16)));
  return __hmul2(v, __half2half2(__ushort_as_half(0x5C00)));
}

// Lane l's stream words for its eight values 8 l .. 8 l + 7, MSB first per u32 word.
template <int K>
__device__ __forceinline__ uint3 tf_exlr8_words(const uint32_t* tile, unsigned lane) {
  constexpr unsigned ring = 256 * K, words = 8 * K;
  const unsigned i0 = ((8 * lane * K + K + ring - 16) % ring) / 32;
  return make_uint3(tile[i0], tile[(i0 + 1) % words], tile[(i0 + 2) % words]);
}

// Each state is the 16 stream bits that end at its own value: one variable shift a tile, then constant ones.
template <int K>
__device__ __forceinline__ void tf_exlr8_unpack(uint3 t, unsigned lane, __half (&w)[8]) {
  const unsigned off = (8 * lane * K + K + 256 * K - 16) % 32;
  const uint64_t hi = ((uint64_t)t.x << 32) | t.y;
  const uint64_t bits = off ? (hi << off) | (((uint64_t)t.z << off) >> 32) : hi;
#pragma unroll
  for (int n = 0; n < 8; n += 2) {
    const __half2 v = tf_exlr8_pair((unsigned)(bits >> (48 - n * K)) & 0xFFFFu, (unsigned)(bits >> (48 - n * K - K)) & 0xFFFFu);
    w[n] = __low2half(v);
    w[n + 1] = __high2half(v);
  }
}

template <int K>
__device__ __forceinline__ void tf_exlr8_decode(const uint32_t* tile, unsigned lane, __half (&w)[8]) {
  tf_exlr8_unpack<K>(tf_exlr8_words<K>(tile, lane), lane, w);
}

__device__ __forceinline__ void tf_exlr8_store(uint8_t* y, uint64_t at, float v, unsigned f32) {
  if (f32) reinterpret_cast<float*>(y)[at] = v;
  else reinterpret_cast<__nv_bfloat16*>(y)[at] = __float2bfloat16_rn(v);
}

// A warp owns a 16-column tile (lane 4 c + q: columns c, c + 8, rows 2 q, +1, +8, +9); members map rows (x << 16 | y).
template <int K, int R>
__device__ __forceinline__ void tf_exlr8_job(const float* X, const uint8_t* W, const uint64_t* bases, const __half* halves,
                                             const int8_t* signs, const TfExlr8Rows a, const TfExlr8Job j, uint8_t* Y,
                                             const unsigned* members, unsigned sg, unsigned lane, unsigned tid, float* xs, float* zs) {
  constexpr unsigned TB = 32 * K;
  const unsigned want = j.count ? j.count : a.rows;
  const unsigned rows = want < R ? want : R, r0 = 2 * (lane % 4), c0 = lane / 4;
  float total[2][R], part[2][R];
#pragma unroll
  for (int i = 0; i < R; ++i) total[0][i] = total[1][i] = 0.0f;
  // Each tile's words load a tile ahead of its decode.
  const uint8_t* run = W + bases[j.base];
  uint3 next = tf_exlr8_words<K>(reinterpret_cast<const uint32_t*>(run + sg * TB), lane);
  for (unsigned ib = 0; ib < j.blocks; ++ib) {
    for (unsigned i = tid; i < R * 128; i += 256) {
      const unsigned rr = i / 128;
      xs[i] = rr < rows ? X[(uint64_t)(j.count ? members[j.members + rr] >> 16 : rr) * a.x_stride + j.x_col + ib * 128 + i % 128] : 0.0f;
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < R; ++i) part[0][i] = part[1][i] = 0.0f;
    const uint8_t* later = ib + 1 < j.blocks ? W + bases[j.base + ib + 1] : run;
    for (unsigned r = 0; r < 8; ++r) {
      const uint3 now = next;
      next = tf_exlr8_words<K>(reinterpret_cast<const uint32_t*>((r < 7 ? run + (8 * r + 8) * TB : later) + sg * TB), lane);
      __half w[8];
      tf_exlr8_unpack<K>(now, lane, w);
#pragma unroll
      for (int i = 0; i < R; ++i) {
        const float* x = xs + i * 128 + 16 * r + r0;
        const float2 lo = *reinterpret_cast<const float2*>(x);
        const float2 hi = *reinterpret_cast<const float2*>(x + 8);
        part[0][i] = __fmaf_rn(__half2float(w[3]), hi.y, __fmaf_rn(__half2float(w[2]), hi.x, __fmaf_rn(__half2float(w[1]), lo.y, __fmaf_rn(__half2float(w[0]), lo.x, part[0][i]))));
        part[1][i] = __fmaf_rn(__half2float(w[7]), hi.y, __fmaf_rn(__half2float(w[6]), hi.x, __fmaf_rn(__half2float(w[5]), lo.y, __fmaf_rn(__half2float(w[4]), lo.x, part[1][i]))));
      }
    }
    run = later;
    // The input block's scale goes on its partial sums, between 128-wide stages.
    const float s = j.in_scale == TF_NONE ? 1.0f : __half2float(halves[j.in_scale + ib]);
#pragma unroll
    for (int i = 0; i < R; ++i) {
      total[0][i] = __fmaf_rn(s, part[0][i], total[0][i]);
      total[1][i] = __fmaf_rn(s, part[1][i], total[1][i]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < R; ++i) {
    for (int c = 0; c < 2; ++c) {
      float v = total[c][i];
      v = __fadd_rn(v, __shfl_xor_sync(TF_FULL, v, 1));
      v = __fadd_rn(v, __shfl_xor_sync(TF_FULL, v, 2));
      if (lane % 4 == 0) zs[i * 128 + 16 * sg + c0 + 8 * c] = v;
    }
  }
  __syncthreads();
  for (unsigned i = sg; i < rows; i += 8) {
    float v[4];
    for (int q = 0; q < 4; ++q) v[q] = zs[i * 128 + lane + 32 * q];
    const uint64_t row = (uint64_t)(j.count ? members[j.members + i] & 0xFFFFu : i) * a.y_stride + j.y_col;
    if (j.raw) {
      for (int q = 0; q < 4; ++q) reinterpret_cast<float*>(Y)[row + lane + 32 * q] = v[q];
      continue;
    }
    tf_exlr8_fwht(v, lane);
    const float block = j.out_block == TF_NONE ? 1.0f : __half2float(halves[j.out_block]);
    for (int q = 0; q < 4; ++q) {
      const unsigned c = lane + 32 * q;
      // svh, the sign and the block scale times 2^-7 multiply exactly; one rounding remains.
      float scale = __fmul_rn(block, 0.0078125f);
      if (j.out_scale != TF_NONE) scale = __fmul_rn(scale, __half2float(halves[j.out_scale + c]));
      if (j.out_sign != TF_NONE) scale = __fmul_rn(scale, (float)signs[j.out_sign + c]);
      tf_exlr8_store(Y, row + c, __fmul_rn(v[q], scale), a.y_f32);
    }
  }
}

// S sets of 8 warps take a job's blocks round-robin; set 0 folds every block's sums in block order, as one set.
template <int K, int R, int S>
__device__ __forceinline__ void tf_exlr8_split(const float* X, const uint8_t* W, const uint64_t* bases, const __half* halves,
                                               const int8_t* signs, const TfExlr8Rows a, const TfExlr8Job j, uint8_t* Y,
                                               const unsigned* members, unsigned sg, unsigned lane, unsigned tid, float* xs,
                                               float* ps, float* zs) {
  constexpr unsigned TB = 32 * K;
  const unsigned set = sg / 8, ct = sg % 8;
  const unsigned want = j.count ? j.count : a.rows;
  const unsigned rows = want < R ? want : R, r0 = 2 * (lane % 4), c0 = lane / 4;
  float total[2][R], part[2][R];
#pragma unroll
  for (int i = 0; i < R; ++i) total[0][i] = total[1][i] = 0.0f;
  // A set past the last block decodes the last block again and its sums are never folded.
  const uint8_t* run = W + bases[j.base + (set < j.blocks ? set : j.blocks - 1)];
  uint3 next = tf_exlr8_words<K>(reinterpret_cast<const uint32_t*>(run + ct * TB), lane);
  for (unsigned ib0 = 0; ib0 < j.blocks; ib0 += S) {
    for (unsigned i = tid; i < S * R * 128; i += S * 256) {
      const unsigned b0 = ib0 + i / (R * 128), b = b0 < j.blocks ? b0 : j.blocks - 1, rr = (i / 128) % R;
      xs[i] = rr < rows ? X[(uint64_t)(j.count ? members[j.members + rr] >> 16 : rr) * a.x_stride + j.x_col + b * 128 + i % 128] : 0.0f;
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < R; ++i) part[0][i] = part[1][i] = 0.0f;
    const unsigned nb = ib0 + set + S < j.blocks ? ib0 + set + S : j.blocks - 1;
    const uint8_t* later = W + bases[j.base + nb];
    for (unsigned r = 0; r < 8; ++r) {
      const uint3 now = next;
      next = tf_exlr8_words<K>(reinterpret_cast<const uint32_t*>((r < 7 ? run + (8 * r + 8) * TB : later) + ct * TB), lane);
      __half w[8];
      tf_exlr8_unpack<K>(now, lane, w);
#pragma unroll
      for (int i = 0; i < R; ++i) {
        const float* x = xs + (set * R + i) * 128 + 16 * r + r0;
        const float2 lo = *reinterpret_cast<const float2*>(x);
        const float2 hi = *reinterpret_cast<const float2*>(x + 8);
        part[0][i] = __fmaf_rn(__half2float(w[3]), hi.y, __fmaf_rn(__half2float(w[2]), hi.x, __fmaf_rn(__half2float(w[1]), lo.y, __fmaf_rn(__half2float(w[0]), lo.x, part[0][i]))));
        part[1][i] = __fmaf_rn(__half2float(w[7]), hi.y, __fmaf_rn(__half2float(w[6]), hi.x, __fmaf_rn(__half2float(w[5]), lo.y, __fmaf_rn(__half2float(w[4]), lo.x, part[1][i]))));
      }
    }
    run = later;
    if (set > 0) {
#pragma unroll
      for (int i = 0; i < R; ++i)
        for (int c = 0; c < 2; ++c) ps[(((set - 1) * R + i) * 8 + ct) * 64 + 2 * lane + c] = part[c][i];
    }
    __syncthreads();
    if (set == 0) {
      // The input block's scale goes on its partial sums, between 128-wide stages, block after block.
      for (unsigned s = 0; s < S && ib0 + s < j.blocks; ++s) {
        const float sc = j.in_scale == TF_NONE ? 1.0f : __half2float(halves[j.in_scale + ib0 + s]);
#pragma unroll
        for (int i = 0; i < R; ++i)
          for (int c = 0; c < 2; ++c) total[c][i] = __fmaf_rn(sc, s ? ps[(((s - 1) * R + i) * 8 + ct) * 64 + 2 * lane + c] : part[c][i], total[c][i]);
      }
    }
  }
  if (set == 0) {
#pragma unroll
    for (int i = 0; i < R; ++i) {
      for (int c = 0; c < 2; ++c) {
        float v = total[c][i];
        v = __fadd_rn(v, __shfl_xor_sync(TF_FULL, v, 1));
        v = __fadd_rn(v, __shfl_xor_sync(TF_FULL, v, 2));
        if (lane % 4 == 0) zs[i * 128 + 16 * ct + c0 + 8 * c] = v;
      }
    }
  }
  __syncthreads();
  for (unsigned i = sg; i < rows; i += 8 * S) {
    float v[4];
    for (int q = 0; q < 4; ++q) v[q] = zs[i * 128 + lane + 32 * q];
    const uint64_t row = (uint64_t)(j.count ? members[j.members + i] & 0xFFFFu : i) * a.y_stride + j.y_col;
    if (j.raw) {
      for (int q = 0; q < 4; ++q) reinterpret_cast<float*>(Y)[row + lane + 32 * q] = v[q];
      continue;
    }
    tf_exlr8_fwht(v, lane);
    const float block = j.out_block == TF_NONE ? 1.0f : __half2float(halves[j.out_block]);
    for (int q = 0; q < 4; ++q) {
      const unsigned c = lane + 32 * q;
      // svh, the sign and the block scale times 2^-7 multiply exactly; one rounding remains.
      float scale = __fmul_rn(block, 0.0078125f);
      if (j.out_scale != TF_NONE) scale = __fmul_rn(scale, __half2float(halves[j.out_scale + c]));
      if (j.out_sign != TF_NONE) scale = __fmul_rn(scale, (float)signs[j.out_sign + c]);
      tf_exlr8_store(Y, row + c, __fmul_rn(v[q], scale), a.y_f32);
    }
  }
}

// One kernel a row capacity R (1, 2, 4, 8, 16): the accumulators stay in registers; a row's sums never depend on R.
#define TF_EXLR8_PROJECT(R)                                                                                          \
  extern "C" __global__ void __launch_bounds__(256) tf_exlr8_project_##R(                                            \
      const float* X, const uint8_t* W, const uint64_t* bases, const __half* halves, const int8_t* signs,          \
      const TfExlr8Job* jobs, const TfExlr8Rows a, uint8_t* Y, const unsigned* members) {                           \
    __shared__ __align__(16) float xs[R * 128], zs[R * 128];                                                        \
    const TfExlr8Job j = jobs[blockIdx.x];                                                                           \
    const unsigned tid = threadIdx.x, sg = tid / 32, lane = tid % 32;                                                \
    switch (j.k) {                                                                                                   \
      case 2: tf_exlr8_job<2, R>(X, W, bases, halves, signs, a, j, Y, members, sg, lane, tid, xs, zs); break;       \
      case 3: tf_exlr8_job<3, R>(X, W, bases, halves, signs, a, j, Y, members, sg, lane, tid, xs, zs); break;       \
      case 4: tf_exlr8_job<4, R>(X, W, bases, halves, signs, a, j, Y, members, sg, lane, tid, xs, zs); break;       \
      default: break;                                                                                                \
    }                                                                                                                \
  }
TF_EXLR8_PROJECT(1)
TF_EXLR8_PROJECT(2)
TF_EXLR8_PROJECT(4)
TF_EXLR8_PROJECT(8)
TF_EXLR8_PROJECT(16)
// A job on 256 S threads for few long jobs (core/exlr8_job.zig's splits); its bits are the one-set kernel's.
#define TF_EXLR8_SPLIT(R, S)                                                                                         \
  extern "C" __global__ void __launch_bounds__(256 * S) tf_exlr8_split_##R##_##S(                                    \
      const float* X, const uint8_t* W, const uint64_t* bases, const __half* halves, const int8_t* signs,          \
      const TfExlr8Job* jobs, const TfExlr8Rows a, uint8_t* Y, const unsigned* members) {                           \
    __shared__ __align__(16) float xs[S * R * 128], ps[(S - 1) * R * 512], zs[R * 128];                            \
    const TfExlr8Job j = jobs[blockIdx.x];                                                                           \
    const unsigned tid = threadIdx.x, sg = tid / 32, lane = tid % 32;                                                \
    switch (j.k) {                                                                                                   \
      case 2: tf_exlr8_split<2, R, S>(X, W, bases, halves, signs, a, j, Y, members, sg, lane, tid, xs, ps, zs); break; \
      case 3: tf_exlr8_split<3, R, S>(X, W, bases, halves, signs, a, j, Y, members, sg, lane, tid, xs, ps, zs); break; \
      case 4: tf_exlr8_split<4, R, S>(X, W, bases, halves, signs, a, j, Y, members, sg, lane, tid, xs, ps, zs); break; \
      default: break;                                                                                                \
    }                                                                                                                \
  }
TF_EXLR8_SPLIT(1, 2)
TF_EXLR8_SPLIT(1, 4)
TF_EXLR8_SPLIT(2, 2)
TF_EXLR8_SPLIT(2, 4)
TF_EXLR8_SPLIT(4, 2)

// r = (x * s) H0 for one row's 128-block of one slot, s = signs times an optional fp16 per-channel scale.
extern "C" __global__ void __launch_bounds__(32) tf_exlr8_rotate(const uint8_t* X, const int8_t* signs, const __half* halves,
                                                                 const TfExlr8Rotate a, float* R, const unsigned* slot_scale) {
  const unsigned lane = threadIdx.x, row = blockIdx.y, base = blockIdx.x * 128, col = blockIdx.z * a.slot_cols;
  const unsigned scale = a.scales ? slot_scale[a.row_slots ? row * a.row_slots + blockIdx.z : blockIdx.z] : a.scale;
  float v[4];
  for (int q = 0; q < 4; ++q) {
    const unsigned c = base + lane + 32 * q;
    const uint64_t at = (uint64_t)row * a.x_stride + a.x_col + col + c;
    float x = a.x_f32 ? reinterpret_cast<const float*>(X)[at] : __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(X)[at]);
    if (a.sign != TF_NONE) x = __fmul_rn(x, (float)signs[a.sign + c]);
    if (scale != TF_NONE) x = __fmul_rn(x, __half2float(halves[scale + c]));
    v[q] = x;
  }
  tf_exlr8_fwht(v, lane);
  for (int q = 0; q < 4; ++q) R[(uint64_t)row * a.r_stride + a.r_col + col + base + lane + 32 * q] = v[q];
}

// The down projection's fan-in: sum_e w_e sv_e z_e per row and 128-block, then one transform, the signs and 2^-7.
extern "C" __global__ void __launch_bounds__(32) tf_exlr8_combine(const float* Z, const float* weights, const unsigned* sv,
                                                                  const __half* halves, const int8_t* signs,
                                                                  const TfExlr8Combine a, uint8_t* Y) {
  const unsigned lane = threadIdx.x, row = blockIdx.y, b = blockIdx.x;
  float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (unsigned e = 0; e < a.slots; ++e) {
    const float s = __fmul_rn(weights[(uint64_t)row * a.slots + e], __half2float(halves[sv[a.per_row ? row * a.slots + e : e] + b]));
    const float* z = Z + (uint64_t)row * a.z_stride + (uint64_t)e * a.z_slot + b * 128;
    for (int q = 0; q < 4; ++q) v[q] = __fmaf_rn(s, z[lane + 32 * q], v[q]);
  }
  tf_exlr8_fwht(v, lane);
  for (int q = 0; q < 4; ++q) {
    const unsigned c = b * 128 + lane + 32 * q;
    const float scale = a.sign == TF_NONE ? 0.0078125f : __fmul_rn((float)signs[a.sign + c], 0.0078125f);
    tf_exlr8_store(Y, (uint64_t)row * a.y_stride + c, __fmul_rn(v[q], scale), a.y_f32);
  }
}

// Qualification: every state's weight, for the host to compare with its own codebook bit for bit.
extern "C" __global__ void tf_exlr8_weights(float* out) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 65536) return;
  const __half2 v = tf_exlr8_pair(i & ~1u, i | 1u);
  out[i] = __half2float(i & 1 ? __high2half(v) : __low2half(v));
}

// Qualification for trellis matrices: run `run`'s 64 tiles as E4M3 codes, a 128 x 128 row-major block per run.
extern "C" __global__ void __launch_bounds__(32) tf_exlr8_tile_codes(const uint8_t* W, const uint64_t* bases, unsigned k, uint8_t* out) {
  const unsigned lane = threadIdx.x, t = blockIdx.x, run = blockIdx.y;
  const uint32_t* tile = reinterpret_cast<const uint32_t*>(W + bases[run] + (uint64_t)t * 32 * k);
  __half w[8];
  switch (k) {
    case 2: tf_exlr8_decode<2>(tile, lane, w); break;
    case 3: tf_exlr8_decode<3>(tile, lane, w); break;
    default: tf_exlr8_decode<4>(tile, lane, w); break;
  }
  for (unsigned n = 0; n < 8; ++n) {
    // an E4M3 value times 2^-8 is an fp16 whose bits are the code's 7 low bits shifted up by 7
    const float v = __half2float(w[n]);
    const unsigned code = ((unsigned)__half_as_ushort(__float2half_rn(__fmul_rn(fabsf(v), 0.00390625f))) >> 7) | (signbit(v) ? 0x80u : 0u);
    const unsigned row = 16 * (t / 8) + 2 * (lane % 4) + (n & 1) + 8 * ((n >> 1) & 1);
    const unsigned col = 16 * (t % 8) + lane / 4 + 8 * (n >> 2);
    out[(uint64_t)run * 16384 + row * 128 + col] = (uint8_t)code;
  }
}
