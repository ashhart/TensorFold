// ours: drama3 on CUDA, one fatbin: backbone and tower glue, the GEMV, samplers; decode kernels read pos from device.
#include "drama3_common.cuh"
#include "drama3_dense.cuh"
#include "drama3_attn.cuh"
#include "drama3_moe.cuh"
#include "drama3_sample.cuh"

// Greedy picks (argmax, ties to the lower index): the main token over the candidates, or a code.
extern "C" __global__ void d3_argmax_main(const float* __restrict__ logits, int count, const int* __restrict__ cand_ids,
                                          int* __restrict__ frame, int sem_begin, int cb) {
  float bv = -INFINITY;
  int bi = 0x7FFFFFFF;
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    const float v = logits[i];
    if (v > bv || (v == bv && i < bi)) {
      bv = v;
      bi = i;
    }
  }
  __shared__ float sv[32];
  __shared__ int si[32];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(FULL, bv, o);
    const int oi = __shfl_xor_sync(FULL, bi, o);
    if (ov > bv || (ov == bv && oi < bi)) {
      bv = ov;
      bi = oi;
    }
  }
  if ((threadIdx.x & 31) == 0) {
    sv[threadIdx.x >> 5] = bv;
    si[threadIdx.x >> 5] = bi;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int k = 1; k < (int)(blockDim.x >> 5); ++k)
      if (sv[k] > bv || (sv[k] == bv && si[k] < bi)) {
        bv = sv[k];
        bi = si[k];
      }
    if (cand_ids) {
      const int main = cand_ids[bi];
      frame[0] = main;
      frame[1] = min(max(main - sem_begin, 0), cb - 1);
    } else {
      frame[0] = bi;
    }
  }
}

// After a frame: append it to frames_out, make it the next input, shift the RAS window, advance pos, step and count.
extern "C" __global__ void d3_finish_frame(const int* __restrict__ frame, int* __restrict__ frames_out,
                                           int* __restrict__ next_input, int* __restrict__ window,
                                           int* __restrict__ pos, unsigned long long* __restrict__ step,
                                           int* __restrict__ count) {
  if (threadIdx.x != 0) return;
  const int n = *count;
  for (int i = 0; i < 11; ++i) {
    frames_out[n * 11 + i] = frame[i];
    next_input[i] = frame[i];
  }
  for (int i = 0; i < 15; ++i) window[i] = window[i + 1];
  window[15] = frame[0];
  *pos += 1;
  *step += 1;
  *count = n + 1;
}

// Next backbone row's input once main, code 0 and code 1 are drawn: bb_input = frame[0..3], *pos_bb = *pos + 1.
extern "C" __global__ void d3_next_input(const int* __restrict__ frame, int* __restrict__ bb_input,
                                         const int* __restrict__ pos, int* __restrict__ pos_bb) {
  if (threadIdx.x < 11) bb_input[threadIdx.x] = threadIdx.x < 3 ? frame[threadIdx.x] : 0;
  if (threadIdx.x == 0) *pos_bb = *pos + 1;
}

// Copy n ints (small device-side moves inside a graph).
extern "C" __global__ void d3_copy_i32(const int* __restrict__ src, int* __restrict__ dst, int n) {
  for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i];
}
