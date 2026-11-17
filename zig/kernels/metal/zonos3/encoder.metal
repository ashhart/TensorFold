// ours: ZONOS3's codec encoder front end and quantizer search on Metal in fp32, kernel for kernel zonos3_encoder.cu.
#include "z3_common.h"

// 44100 to 44160 Hz: output m = 736 q + c is sum_j kernel[c][j] x[735 q + j - 17], zero outside [0, total).
struct ResampleInArgs { long total; long count; };
kernel void ze_resample_in(device const float* x [[buffer(0)]], device const float* kern [[buffer(1)]],
                           device float* out [[buffer(2)]], constant ResampleInArgs& a [[buffer(3)]],
                           uint gm [[thread_position_in_grid]]) {
  const long m = gm;
  if (m >= a.count) return;
  const long q = m / 736;
  const int c = (int)(m % 736);
  float acc = 0.f;
  for (int j = 0; j < 769; ++j) {
    const long p = 735 * q + j - 17;
    if (p >= 0 && p < a.total) acc = fma(kern[c * 769 + j], x[p], acc);
  }
  out[m] = acc;
}

// out[t, o] = bias[o] + sum_j sum_c W[o][j][c] x[t * stride - pad + j * dilation, c], x zero outside [0, T).
struct ConvArgs { int Cin; int T; int K; int dilation; int stride; int pad; int Cout; };
kernel void ze_conv(device const float* x [[buffer(0)]], device const float* W [[buffer(1)]],
                    device const float* bias [[buffer(2)]], device float* out [[buffer(3)]],
                    constant ConvArgs& a [[buffer(4)]], threadgroup float* xs [[threadgroup(0)]],
                    uint2 gid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  const long t = gid.x;
  // the taps' inputs, staged in K * Cin floats of threadgroup memory
  const int n = a.K * a.Cin;
  for (int i = tid; i < n; i += 256) {
    const long p = t * a.stride - a.pad + (long)(i / a.Cin) * a.dilation;
    xs[i] = (p >= 0 && p < a.T) ? x[p * a.Cin + i % a.Cin] : 0.f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int o = gid.y * 8 + sg;
  device const float* wr = W + (ulong)min(o, a.Cout - 1) * n;
  float acc = 0.f;
  for (int k = lane; k < n; k += 32) acc = fma(wr[k], xs[k], acc);
  acc = warp_sum(acc);
  if (lane == 0 && o < a.Cout) out[(ulong)t * a.Cout + o] = acc + bias[o];
}

// Snake: y = x + sin^2(alpha x) / (alpha + 1e-9), alpha a channel; rows of C channels.
struct SnakeArgs { int C; int pad; long n; };
kernel void ze_snake(device const float* x [[buffer(0)]], device const float* alpha [[buffer(1)]],
                     device float* y [[buffer(2)]], constant SnakeArgs& a [[buffer(3)]],
                     uint gi [[thread_position_in_grid]]) {
  const long i = gi;
  if (i >= a.n) return;
  const float al = alpha[i % a.C];
  const float s = precise::sin(al * x[i]);
  y[i] = x[i] + precise::divide(s * s, al + 1e-9f);
}

// x += y over n values.
struct LongCount { long n; };
kernel void ze_add(device float* x [[buffer(0)]], device const float* y [[buffer(1)]],
                   constant LongCount& a [[buffer(2)]], uint gi [[thread_position_in_grid]]) {
  const long i = gi;
  if (i < a.n) x[i] += y[i];
}

// x -= y over n values.
kernel void ze_sub(device float* x [[buffer(0)]], device const float* y [[buffer(1)]],
                   constant LongCount& a [[buffer(2)]], uint gi [[thread_position_in_grid]]) {
  const long i = gi;
  if (i < a.n) x[i] -= y[i];
}

// Patch down by h: in [T * h][C] to out [T][C * h], out[t][c * h + j] = in[t * h + j][c].
struct PatchArgs { int C; int h; };
kernel void ze_patch_down(device const float* in [[buffer(0)]], device float* out [[buffer(1)]],
                          constant PatchArgs& a [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
  const ulong t = gid.y;
  const int i = gid.x;
  if (i < a.C * a.h) out[t * a.C * a.h + i] = in[(t * a.h + i % a.h) * a.C + i / a.h];
}

// codes[row * 10 + col] = argmax_c e . C[c], C [4096][8]: ties to the lower, NaN never wins, none finite: 0 (as CUDA).
struct NearestArgs { int col; };
kernel void ze_nearest(device const float* e [[buffer(0)]], device const float* C [[buffer(1)]],
                       device int* codes [[buffer(2)]], constant NearestArgs& a [[buffer(3)]],
                       uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float best[256];
  threadgroup int which[256];
  device const float* x = e + (ulong)row * 8;
  float bv = -INFINITY;
  int bi = tid;
  for (int c = tid; c < 4096; c += 256) {
    device const float* r = C + (ulong)c * 8;
    float d = 0.f;
    for (int i = 0; i < 8; ++i) d = fma(x[i], r[i], d);
    if (d > bv) {
      bv = d;
      bi = c;
    }
  }
  best[tid] = bv;
  which[tid] = bi;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid != 0) return;
  for (int t = 1; t < 256; ++t)
    if (best[t] > bv || (best[t] == bv && which[t] < bi)) {
      bv = best[t];
      bi = which[t];
    }
  codes[row * 10 + a.col] = bi;
}

// residual[r] -= table[codes[r][col]] over D values a row.
struct SubCodeArgs { int col; int D; };
kernel void ze_sub_code(device float* residual [[buffer(0)]], device const float* table [[buffer(1)]],
                        device const int* codes [[buffer(2)]], constant SubCodeArgs& a [[buffer(3)]],
                        uint2 gid [[thread_position_in_grid]]) {
  const ulong r = gid.y;
  const int d = gid.x;
  if (d < a.D) residual[r * a.D + d] -= table[(ulong)codes[r * 10 + a.col] * a.D + d];
}
