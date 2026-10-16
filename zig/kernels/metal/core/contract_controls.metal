// The kernel contract's planted controls: each breaks one gate on purpose, and that gate must go red.
#include <metal_stdlib>
using namespace metal;

// Gate 1: a row's maximum by fmax, which drops a NaN input, so a row holding NaN reads as its largest number.
[[kernel]] void tf_ctl_fmax_max(const device bfloat* X [[buffer(0)]], constant int& K [[buffer(1)]],
                                device float* Y [[buffer(2)]], uint row [[threadgroup_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]]) {
  float m = -INFINITY;
  for (int i = int(lane); i < K; i += 32) m = fmax(m, float(X[long(row) * K + i]));
  m = simd_max(m);
  if (lane == 0) Y[row] = m;
}

// Gate 2: row sums by an atomic add an input from threads all over the grid, so the order varies run to run.
[[kernel]] void tf_ctl_atomic_sum(const device bfloat* X [[buffer(0)]], constant int& K [[buffer(1)]],
                                  device atomic_float* Y [[buffer(2)]], uint2 at [[thread_position_in_grid]]) {
  const int i = int(at.x), row = int(at.y);
  if (i < K) atomic_fetch_add_explicit(Y + row, float(X[long(row) * K + i]), memory_order_relaxed);
}

// Gate 2 too: a row sum that writes the sum back into its own input, which its spec says it never does.
[[kernel]] void tf_ctl_scribble_sum(device bfloat* X [[buffer(0)]], constant int& K [[buffer(1)]],
                                    device float* Y [[buffer(2)]], uint row [[threadgroup_position_in_grid]],
                                    uint lane [[thread_index_in_simdgroup]]) {
  float s = 0.0f;
  for (int i = int(lane); i < K; i += 32) s += float(X[long(row) * K + i]);
  s = simd_sum(s);
  if (lane == 0) {
    Y[row] = s;
    X[long(row) * K] = bfloat(s);
  }
}

// Gate 3: a row sum whose partial tail loses its last input: right whenever K is a whole number of 32s.
[[kernel]] void tf_ctl_tail_sum(const device bfloat* X [[buffer(0)]], constant int& K [[buffer(1)]],
                                device float* Y [[buffer(2)]], uint row [[threadgroup_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]]) {
  const int whole = K / 32 * 32;
  float s = 0.0f;
  for (int i = int(lane); i < whole; i += 32) s += float(X[long(row) * K + i]);
  if (int(lane) < K - whole - 1) s += float(X[long(row) * K + whole + int(lane)]);
  s = simd_sum(s);
  if (lane == 0) Y[row] = s;
}

// Gate 4: a dot product summed in half, while its spec declares fp32 sums.
[[kernel]] void tf_ctl_half_dot(const device bfloat* X [[buffer(0)]], constant int& K [[buffer(1)]],
                                device float* Y [[buffer(2)]], const device bfloat* W [[buffer(3)]],
                                uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  half s = 0.0h;
  for (int i = int(lane); i < K; i += 32) s += half(float(X[long(row) * K + i]) * float(W[i]));
  s = simd_sum(s);
  if (lane == 0) Y[row] = float(s);
}

#ifdef TF_CTL_BIG_TG
// Gate 5 too: 36 KiB of threadgroup memory, past every Apple family's 32 KiB (compiled apart: it may be refused).
[[kernel]] void tf_ctl_big_tg(const device float* X [[buffer(0)]], device float* Y [[buffer(1)]],
                              uint t [[thread_index_in_threadgroup]]) {
  threadgroup float big[9216];
  for (uint i = t; i < 9216; i += 256) big[i] = X[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  Y[t] = big[9215 - t];
}
#endif
