// music3 DAV kernels (fp32): snake, bias rows, the transposed convs' overlap-add, the halves and tanh.
#pragma once
#include "music3_common.cuh"

// ------------------------------------------------------------------------------------------------ DAV (fp32)

// Snake: y = x + sin^2(alpha x) / (alpha + 1e-9), x [T, C] time-major (C contiguous). y may be x.
extern "C" __global__ void m3_snake(const float* __restrict__ x, const float* __restrict__ alpha, float* __restrict__ y, int C, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const float a = alpha[i % C], v = x[i];
    const float s = sinf(a * v);
    y[i] = v + (1.0f / (a + 1e-9f)) * (s * s);
  }
}

// y[t, c] = bias[c] for T rows (before the taps' GEMMs accumulate).
extern "C" __global__ void m3_bias_rows(float* __restrict__ y, const float* __restrict__ bias, int C, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) y[i] = bias[i % C];
}

// Transposed conv overlap-add: taps z [Tin, 2s, C] into y [Tout, C] + bias, each output its two sources in order.
extern "C" __global__ void m3_overlap_add(const float* __restrict__ z, const float* __restrict__ bias, int Tin, int stride, int pad, int C, int Tout, float* __restrict__ y) {
  const size_t n = (size_t)Tout * C;
  const int K = 2 * stride;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const int t = (int)(i / C), c = (int)(i % C);
    float s = bias[c];
    const int u = t + pad;
    // ti s + j = u, 0 <= j < 2s: ti in [ceil((u - 2s + 1) / s), u / s]
    const int hi = u / stride;
    for (int ti = hi; ti >= 0 && ti > hi - 2; --ti) {
      const int j = u - ti * stride;
      if (j < 0 || j >= K || ti >= Tin) continue;
      s += z[((size_t)ti * K + j) * C + c];
    }
    y[i] = s;
  }
}

// Zero the halo rows of a padded buffer [pad + T + pad, C] (T rows at offset pad).
extern "C" __global__ void m3_zero_rows(float* __restrict__ y, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) y[i] = 0.f;
}

// y[t, c] += bias[c].
extern "C" __global__ void m3_add_bias(float* __restrict__ y, const float* __restrict__ bias, int C, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) y[i] += bias[i % C];
}

// One stereo half of latent rows [a, a + T) of lat [L, 128] (fp32): out [T, 64] = lat[a + t, half 64 + c].
extern "C" __global__ void m3_latent_half(const float* __restrict__ lat, int a, int T, int half, float* __restrict__ out) {
  const size_t n = (size_t)T * 64;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) out[i] = lat[(size_t)(a + i / 64) * 128 + half * 64 + i % 64];
}

// The decoder's tanh, rows [T] of one channel.
extern "C" __global__ void m3_tanh(float* __restrict__ y, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) y[i] = tanhf(y[i]);
}
