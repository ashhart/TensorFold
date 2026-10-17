// music3 DAV glue (fp32): snake, bias rows, the transposed convs' overlap-add, the stereo halves and tanh.
#include "m3_common.h"

struct ChanArgs {
  uint C, n;
};

// Snake: y = x + sin^2(alpha x) / (alpha + 1e-9), x [T, C] time-major (C contiguous). y may be x.
[[max_total_threads_per_threadgroup(256)]] kernel void m3_snake(device const float* x [[buffer(0)]], device const float* alpha [[buffer(1)]],
                     device float* y [[buffer(2)]], constant ChanArgs& a [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  if (i >= a.n) return;
  const float al = alpha[i % a.C], v = x[i];
  const float s = precise::sin(al * v);
  y[i] = v + (1.0f / (al + 1e-9f)) * (s * s);
}

// y[t, c] = bias[c] (set = 1) or y[t, c] += bias[c] (set = 0), over n = T C values.
[[max_total_threads_per_threadgroup(256)]] kernel void m3_bias_rows(device float* y [[buffer(0)]], device const float* bias [[buffer(1)]],
                         constant uint3& a [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i >= a.y) return;
  y[i] = a.z ? bias[i % a.x] : y[i] + bias[i % a.x];
}

struct OverlapArgs {
  uint Tin, stride, pad, C, Tout;
};

// Transposed conv overlap-add: taps z [Tin, 2s, C] into y [Tout, C] + bias, each output its two sources in order.
[[max_total_threads_per_threadgroup(256)]] kernel void m3_overlap_add(device const float* z [[buffer(0)]], device const float* bias [[buffer(1)]],
                           device float* y [[buffer(2)]], constant OverlapArgs& a [[buffer(3)]],
                           uint i [[thread_position_in_grid]]) {
  if (i >= a.Tout * a.C) return;
  const int t = int(i / a.C), c = int(i % a.C), s = int(a.stride), K = 2 * s;
  float sum = bias[c];
  const int u = t + int(a.pad);
  const int hi = u / s;
  for (int ti = hi; ti >= 0 && ti > hi - 2; --ti) {
    const int j = u - ti * s;
    if (j < 0 || j >= K || ti >= int(a.Tin)) continue;
    sum += z[((ulong)ti * K + j) * a.C + c];
  }
  y[i] = sum;
}

// y[0, n) = 0 (the halo rows of a padded buffer).
[[max_total_threads_per_threadgroup(256)]] kernel void m3_zero(device float* y [[buffer(0)]], constant uint& n [[buffer(1)]],
                    uint i [[thread_position_in_grid]]) {
  if (i < n) y[i] = 0.0f;
}

// One stereo half of latent rows [a, a + T) of lat [L, 128] (fp32): out [T, 64] = lat[a + t, half 64 + c].
[[max_total_threads_per_threadgroup(256)]] kernel void m3_latent_half(device const float* lat [[buffer(0)]], device float* out [[buffer(1)]],
                           constant uint3& a [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i >= a.y * 64) return;
  out[i] = lat[(ulong)(a.x + i / 64) * 128 + a.z * 64 + i % 64];
}

// The decoder's tanh, rows [T] of one channel.
[[max_total_threads_per_threadgroup(256)]] kernel void m3_tanh(device float* y [[buffer(0)]], constant uint& n [[buffer(1)]],
                    uint i [[thread_position_in_grid]]) {
  if (i < n) y[i] = precise::tanh(y[i]);
}
