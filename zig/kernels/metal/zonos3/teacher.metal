// ours: ZONOS3's SSL teacher (Wav2Vec2-BERT) on Metal in fp32, kernel for kernel zonos3_teacher.cu.
#include "z3_common.h"

// Swish in place: x * sigmoid(x).
struct TLongCount { long n; };
kernel void zt_swish(device float* x [[buffer(0)]], constant TLongCount& a [[buffer(1)]],
                     uint gi [[thread_position_in_grid]]) {
  const long i = gi;
  if (i < a.n) x[i] = z3_silu(x[i]);
}

// GLU: out[r, c] = in[r, c] * sigmoid(in[r, C + c]).
struct GluArgs { int C; };
kernel void zt_glu(device const float* in [[buffer(0)]], device float* out [[buffer(1)]],
                   constant GluArgs& a [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
  const ulong r = gid.y;
  const int c = gid.x;
  if (c < a.C) out[r * a.C + c] = in[r * 2 * a.C + c] * z3_sigmoid(in[r * 2 * a.C + a.C + c]);
}

// Depthwise conv of K taps, the newest input last, zeros before row 0: out[t, c] = sum_j w[c][j] x[t - K + 1 + j, c].
struct TDwArgs { int K; int C; };
kernel void zt_dwconv(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                      device float* out [[buffer(2)]], constant TDwArgs& a [[buffer(3)]],
                      uint2 gid [[thread_position_in_grid]]) {
  const long t = gid.y;
  const int c = gid.x;
  if (c >= a.C) return;
  float acc = 0.f;
  for (int j = 0; j < a.K; ++j) {
    const long p = t - a.K + 1 + j;
    if (p >= 0) acc = fma(w[c * a.K + j], x[p * a.C + c], acc);
  }
  out[t * a.C + c] = acc;
}

// Attention of (row l, head h) over all T rows: score(r) = (q . k_r + q . E[clamp(r - l, -64, 8) + 64]) / 8.
struct TAttnArgs { int D; int T; };
[[max_total_threads_per_threadgroup(128)]] kernel void zt_attention(device const float* qkv [[buffer(0)]], device const float* E [[buffer(1)]],
                         device float* out [[buffer(2)]], constant TAttnArgs& a [[buffer(3)]],
                         uint2 gid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[32];
  threadgroup float pe[73];
  threadgroup float acc[128][33];
  threadgroup float total[65];
  const int l = gid.x, h = gid.y, D = a.D;
  float qv[64];
  device const float* qr = qkv + (ulong)l * 3 * D + h * 64;
  for (int i = 0; i < 64; ++i) qv[i] = qr[i];
  if (tid < 73) {
    device const float* e = E + tid * 64;
    float d = 0.f;
    for (int i = 0; i < 64; ++i) d = fma(qv[i], e[i], d);
    pe[tid] = d;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float m = -INFINITY;
  for (int r = tid; r < a.T; r += 128) {
    device const float* k = qkv + (ulong)r * 3 * D + D + h * 64;
    float d = 0.f;
    for (int i = 0; i < 64; ++i) d = fma(qv[i], k[i], d);
    m = fmax(m, (d + pe[min(max(r - l, -64), 8) + 64]) * 0.125f);
  }
  m = block_max_tree(m, part, 4, lane, sg);
  float s_[65];
  for (int i = 0; i < 65; ++i) s_[i] = 0.f;
  for (int r = tid; r < a.T; r += 128) {
    device const float* k = qkv + (ulong)r * 3 * D + D + h * 64;
    device const float* v = k + D;
    float d = 0.f;
    for (int i = 0; i < 64; ++i) d = fma(qv[i], k[i], d);
    const float e = z3_exp((d + pe[min(max(r - l, -64), 8) + 64]) * 0.125f - m);
    s_[64] += e;
    for (int i = 0; i < 64; ++i) s_[i] = fma(e, v[i], s_[i]);
  }
  // a thread's 65 sums folded in two halves (33, then 32), each value's 128 terms in thread order as on CUDA
  for (int half_ = 0; half_ < 2; ++half_) {
    const int first = half_ * 33, count = half_ == 0 ? 33 : 32;
    for (int i = 0; i < count; ++i) acc[tid][i] = s_[first + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((int)tid < count) {
      float s = 0.f;
      for (int w = 0; w < 128; ++w) s += acc[w][tid];
      total[first + tid] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid < 64) out[(ulong)l * D + h * 64 + tid] = precise::divide(total[tid], total[64]);
}
