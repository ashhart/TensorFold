// EXLR8 dense matrices stored by row on CUDA: the Metal kernels' decode and sums in the same order, bit for bit.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#define TF_FULL 0xFFFFFFFFu
#define TF_INT4 2u
#define TF_MXFP8 3u
#define TF_FP8_BLOCK 4u
#define TF_FP8_ROW 5u
#define TF_BF16 6u
#define TF_INT4_IN 12u

struct TfDenseJob {
  unsigned format, out0, outs, in0, ins, out_total, in_total, x_col, y_col, pad0, pad1, pad2;
  uint64_t w, s;
};
struct TfDenseRows {
  unsigned rows, x_stride, y_stride, y_f32;
};

// An E4M3 code's value: its 7 low bits as fp16 bits are the value times 2^-8 (normal and subnormal alike).
__device__ __forceinline__ float tf_e4m3(unsigned code) {
  const float v = __fmul_rn(__half2float(__ushort_as_half((unsigned short)((code & 0x7Fu) << 7))), 256.0f);
  const float s = (code & 0x80u) ? -v : v;
  return (code & 0x7Fu) == 0x7Fu ? __int_as_float(0x7FC00000) : s;
}

// Lane `lane`'s four weights of matrix row o at chunk inputs 4 lane .. 4 lane + 3, chunk c (128 inputs from in0).
__device__ __forceinline__ void tf_dense_weights(const uint8_t* W, const TfDenseJob& j, unsigned o, unsigned c, unsigned lane, float (&w)[4]) {
  const unsigned i = j.in0 + 128 * c + 4 * lane;
  switch (j.format) {
    case TF_INT4: {
      const unsigned short q = *reinterpret_cast<const unsigned short*>(W + j.w + (uint64_t)o * (j.in_total / 2) + i / 2);
      const float s = __half2float(*reinterpret_cast<const __half*>(W + j.s + 2 * ((uint64_t)o * (j.in_total / 32) + i / 32)));
      for (int n = 0; n < 4; ++n) w[n] = __fmul_rn((float)((int)((q >> (4 * n)) & 15u) - 8), s);
      break;
    }
    case TF_INT4_IN: {
      const uint64_t g = (uint64_t)(i / 128) * j.out_total + o;
      const unsigned short q = *reinterpret_cast<const unsigned short*>(W + j.w + g * 64 + (i % 128) / 2);
      const float s = __half2float(*reinterpret_cast<const __half*>(W + j.s + 2 * (g * 4 + (i % 128) / 32)));
      for (int n = 0; n < 4; ++n) w[n] = __fmul_rn((float)((int)((q >> (4 * n)) & 15u) - 8), s);
      break;
    }
    case TF_MXFP8: {
      const uint32_t q = *reinterpret_cast<const uint32_t*>(W + j.w + (uint64_t)o * j.in_total + i);
      const unsigned e = W[j.s + (uint64_t)o * (j.in_total / 32) + i / 32];
      const float s = __uint_as_float(e << 23);
      for (int n = 0; n < 4; ++n) w[n] = __fmul_rn(tf_e4m3((q >> (8 * n)) & 0xFFu), s);
      break;
    }
    case TF_FP8_BLOCK:
    case TF_FP8_ROW: {
      const uint32_t q = *reinterpret_cast<const uint32_t*>(W + j.w + (uint64_t)o * j.in_total + i);
      const uint64_t at = j.format == TF_FP8_ROW ? (uint64_t)o : (uint64_t)(o / 128) * ((j.in_total + 127) / 128) + i / 128;
      const float s = reinterpret_cast<const float*>(W + j.s)[at];
      // one rounding, as the reference's E4M3 times its fp32 scale
      for (int n = 0; n < 4; ++n) w[n] = __fmul_rn(tf_e4m3((q >> (8 * n)) & 0xFFu), s);
      break;
    }
    case TF_BF16: {
      const uint2 q = *reinterpret_cast<const uint2*>(W + j.w + 2 * ((uint64_t)o * j.in_total + i));
      w[0] = __uint_as_float(q.x << 16);
      w[1] = __uint_as_float(q.x & 0xFFFF0000u);
      w[2] = __uint_as_float(q.y << 16);
      w[3] = __uint_as_float(q.y & 0xFFFF0000u);
      break;
    }
    default: {
      const float4 q = *reinterpret_cast<const float4*>(W + j.w + 4 * ((uint64_t)o * j.in_total + i));
      w[0] = q.x, w[1] = q.y, w[2] = q.z, w[3] = q.w;
      break;
    }
  }
}

// The lane's partial sums, added over the warp in the Metal kernels' fixed butterfly.
__device__ __forceinline__ float tf_dense_lanes(float v) {
  for (unsigned d = 16; d > 0; d >>= 1) v = __fadd_rn(v, __shfl_xor_sync(TF_FULL, v, d));
  return v;
}

// A block is 32 matrix rows of one job (4 a warp); inputs come 128 at a time through shared memory.
template <int R>
__device__ __forceinline__ void tf_dense_job(const float* X, const uint8_t* W, const TfDenseJob j, unsigned block, const TfDenseRows a,
                                             uint8_t* Y, unsigned sg, unsigned lane, unsigned tid, float* xs) {
  const unsigned rows = a.rows < R ? a.rows : R, o0 = 32 * block + 4 * sg;
  float acc[4][R];
#pragma unroll
  for (int q = 0; q < 4; ++q)
#pragma unroll
    for (int r = 0; r < R; ++r) acc[q][r] = 0.0f;
  for (unsigned c = 0; c < j.ins / 128; ++c) {
    for (unsigned i = tid; i < R * 128; i += 256)
      xs[i] = i / 128 < rows ? X[(uint64_t)(i / 128) * a.x_stride + j.x_col + 128 * c + i % 128] : 0.0f;
    __syncthreads();
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      if (o0 + q >= j.outs) break;
      float w[4];
      tf_dense_weights(W, j, j.out0 + o0 + q, c, lane, w);
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const float4 x = *reinterpret_cast<const float4*>(xs + r * 128 + 4 * lane);
        acc[q][r] = __fmaf_rn(w[3], x.w, __fmaf_rn(w[2], x.z, __fmaf_rn(w[1], x.y, __fmaf_rn(w[0], x.x, acc[q][r]))));
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    if (o0 + q >= j.outs) break;
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const float v = tf_dense_lanes(acc[q][r]);
      if (lane != 0 || (unsigned)r >= rows) continue;
      const uint64_t at = (uint64_t)r * a.y_stride + j.y_col + o0 + q;
      if (a.y_f32) reinterpret_cast<float*>(Y)[at] = v;
      else reinterpret_cast<__nv_bfloat16*>(Y)[at] = __float2bfloat16_rn(v);
    }
  }
}

// One kernel a row capacity; groups[g] = (job, 32-row block of that job) for block g.
#define TF_EXLR8_DENSE(R)                                                                                            \
  extern "C" __global__ void __launch_bounds__(256) tf_exlr8_dense_##R(const float* X, const uint8_t* W,           \
                                                                       const TfDenseJob* jobs, const uint2* groups, \
                                                                       const TfDenseRows a, uint8_t* Y) {          \
    __shared__ __align__(16) float xs[R * 128];                                                                     \
    const uint2 at = groups[blockIdx.x];                                                                            \
    const unsigned tid = threadIdx.x;                                                                               \
    tf_dense_job<R>(X, W, jobs[at.x], at.y, a, Y, tid / 32, tid % 32, tid, xs);                                     \
  }
TF_EXLR8_DENSE(1)
TF_EXLR8_DENSE(2)
TF_EXLR8_DENSE(4)
TF_EXLR8_DENSE(8)
TF_EXLR8_DENSE(16)

// Qualification: rows [out0, out0 + outs) of job 0's matrix over its inputs as fp32, row-major, ins wide.
extern "C" __global__ void __launch_bounds__(32) tf_exlr8_dense_decode(const uint8_t* W, const TfDenseJob* jobs, float* out) {
  const TfDenseJob j = jobs[0];
  const unsigned o = blockIdx.y, c = blockIdx.x, lane = threadIdx.x;
  if (o >= j.outs) return;
  float w[4];
  tf_dense_weights(W, j, j.out0 + o, c, lane, w);
  for (int n = 0; n < 4; ++n) out[(uint64_t)o * j.ins + 128 * c + 4 * lane + n] = w[n];
}
