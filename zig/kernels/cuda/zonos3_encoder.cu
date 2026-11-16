// ours: ZONOS3's codec encoder front end on CUDA in fp32: the resampler in, causal convolutions, Snake, patches.
#include <math.h>
#include <stdint.h>
#include "common/reduce.cuh"

namespace {

using tf_common::warp_sum;

}  // namespace

// 44100 to 44160 Hz: output m = 736 q + c is sum_j kernel[c][j] x[735 q + j - 17], zero outside [0, total).
extern "C" __global__ void ze_resample_in(const float* __restrict__ x, long long total,
                                          const float* __restrict__ kernel, long long count,
                                          float* __restrict__ out) {
  const long long m = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (m >= count) return;
  const long long q = m / 736;
  const int c = (int)(m % 736);
  float a = 0.f;
  for (int j = 0; j < 769; ++j) {
    const long long p = 735 * q + j - 17;
    if (p >= 0 && p < total) a = fmaf(kernel[c * 769 + j], x[p], a);
  }
  out[m] = a;
}

// out[t, o] = bias[o] + sum_j sum_c W[o][j][c] x[t * stride - pad + j * dilation, c], x zero outside [0, T).
extern "C" __global__ void __launch_bounds__(256) ze_conv(const float* __restrict__ x, int Cin, int T,
                                                          const float* __restrict__ W, const float* __restrict__ bias,
                                                          int K, int dilation, int stride, int pad,
                                                          float* __restrict__ out, int Cout) {
  extern __shared__ float xs[];
  const long long t = blockIdx.x;
  const int n = K * Cin;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const long long p = t * stride - pad + (long long)(i / Cin) * dilation;
    xs[i] = (p >= 0 && p < T) ? x[p * Cin + i % Cin] : 0.f;
  }
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int o = blockIdx.y * 8 + (threadIdx.x >> 5);
  const float* wr = W + (size_t)min(o, Cout - 1) * n;
  float acc = 0.f;
  for (int k = lane; k < n; k += 32) acc = fmaf(wr[k], xs[k], acc);
  acc = warp_sum(acc);
  if (lane == 0 && o < Cout) out[(size_t)t * Cout + o] = acc + bias[o];
}

// Snake: y = x + sin^2(alpha x) / (alpha + 1e-9), alpha a channel; rows of C channels.
extern "C" __global__ void ze_snake(const float* __restrict__ x, const float* __restrict__ alpha, int C,
                                    long long n, float* __restrict__ y) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float a = alpha[i % C];
  const float s = sinf(a * x[i]);
  y[i] = x[i] + s * s / (a + 1e-9f);
}

// x += y over n values.
extern "C" __global__ void ze_add(float* __restrict__ x, const float* __restrict__ y, long long n) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] += y[i];
}

// Patch down by h: in [T * h][C] to out [T][C * h], out[t][c * h + j] = in[t * h + j][c].
extern "C" __global__ void ze_patch_down(const float* __restrict__ in, float* __restrict__ out, int C, int h) {
  const size_t t = blockIdx.y;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < C * h) out[t * C * h + i] = in[(t * h + i % h) * C + i / h];
}
