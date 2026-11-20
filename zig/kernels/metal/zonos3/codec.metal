// ours: ZONOS3's codec decoder on Metal in fp32, zonos3_codec.cu's kernels; an output never depends on stream chunks.
#include "z3_common.h"

// erf in fp32: Taylor to x^15 below 0.5, else 1 - Numerical Recipes' erfcc (under 1.2e-7); about one ulp either way.
inline float z3_erf(float x) {
  const float z = fabs(x);
  if (z < 0.5f) {
    const float s = x * x;
    float p = -1.0f / 75600.0f;
    p = p * s + 1.0f / 9360.0f;
    p = p * s - 1.0f / 1320.0f;
    p = p * s + 1.0f / 216.0f;
    p = p * s - 1.0f / 42.0f;
    p = p * s + 1.0f / 10.0f;
    p = p * s - 1.0f / 3.0f;
    p = p * s + 1.0f;
    return 1.1283791670955126f * x * p;
  }
  const float t = precise::divide(1.0f, 1.0f + 0.5f * z);
  float p = 0.17087277f;
  p = p * t - 0.82215223f;
  p = p * t + 1.48851587f;
  p = p * t - 1.13520398f;
  p = p * t + 0.27886807f;
  p = p * t - 0.18628806f;
  p = p * t + 0.09678418f;
  p = p * t + 0.37409196f;
  p = p * t + 1.00002368f;
  const float erfc = t * precise::exp(-z * z - 1.26551223f + t * p);
  return x < 0.0f ? erfc - 1.0f : 1.0f - erfc;
}

// y[r, n] = sum_k W[n, k] x[r, k] (+ bias[n]): fp32 W [N, K], a simdgroup an output, K <= 8192 (x in threadgroup).
struct ZLinearArgs { int xs; int K; int N; int ys; int has_bias; };
kernel void zc_linear(device const float* W [[buffer(0)]], device const float* x [[buffer(1)]], device float* y [[buffer(2)]],
                      device const float* bias [[buffer(3)]], constant ZLinearArgs& a [[buffer(4)]],
                      threadgroup float* xs [[threadgroup(0)]], uint2 gid [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
  const int r = gid.y;
  device const float* xr = x + (ulong)r * a.xs;
  for (int i = tid; i < a.K; i += 256) xs[i] = xr[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int n = gid.x * 8 + sg;
  device const float* wr = W + (ulong)min(n, a.N - 1) * a.K;
  float acc = 0.f;
  for (int k = lane; k < a.K; k += 32) acc = fma(wr[k], xs[k], acc);
  acc = warp_sum(acc);
  if (lane == 0 && n < a.N) y[(ulong)r * a.ys + n] = a.has_bias ? acc + bias[n] : acc;
}

// LayerNorm over D: y = (x - mean) / sqrt(var + eps) * w + b; a group a row.
struct LayerNormArgs { int D; float eps; };
kernel void zc_layernorm(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                         device const float* b [[buffer(2)]], device float* y [[buffer(3)]],
                         constant LayerNormArgs& a [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[32];
  device const float* xr = x + (ulong)row * a.D;
  device float* yr = y + (ulong)row * a.D;
  float s = 0.f;
  for (int i = tid; i < a.D; i += 256) s += xr[i];
  const float mean = precise::divide(block_sum_tree(s, part, 8, lane, sg), float(a.D));
  float v = 0.f;
  for (int i = tid; i < a.D; i += 256) {
    const float d = xr[i] - mean;
    v = fma(d, d, v);
  }
  const float var = precise::divide(block_sum_tree(v, part, 8, lane, sg), float(a.D));
  const float inv = precise::divide(1.f, precise::sqrt(var + a.eps));
  for (int i = tid; i < a.D; i += 256) yr[i] = (xr[i] - mean) * inv * w[i] + b[i];
}

// Rows [q | k | v] (D = heads * 64) at pos0 + row: q and k rotate in pairs; k and v go to row (pos0 + row) % cap.
struct ZAttnPrepArgs { int D; int pos0; int cap; };
kernel void zc_attn_prep(device float* qkv [[buffer(0)]], device float* kc [[buffer(1)]], device float* vc [[buffer(2)]],
                         constant ZAttnPrepArgs& a [[buffer(3)]], uint2 gid [[threadgroup_position_in_grid]],
                         uint m [[thread_index_in_threadgroup]]) {
  const int t = gid.x, h = gid.y, D = a.D;
  const float freq = precise::exp(float(m) * (-9.210340371976184f * 2.f / 64.f));
  const float phase = float(a.pos0 + t) * freq;
  const float c = precise::cos(phase), s = precise::sin(phase);
  device float* q = qkv + (ulong)t * 3 * D + h * 64 + 2 * m;
  const float qr = q[0], qi = q[1];
  q[0] = qr * c - qi * s;
  q[1] = qr * s + qi * c;
  const float kr = q[D], ki = q[D + 1];
  const ulong at = (ulong)((a.pos0 + t) % a.cap) * D + h * 64 + 2 * m;
  kc[at] = kr * c - ki * s;
  kc[at + 1] = kr * s + ki * c;
  vc[at] = q[2 * D];
  vc[at + 1] = q[2 * D + 1];
}

// Causal local attention of (row, head) at p = pos0 + row over positions p - ctx + 1 .. p; position j is row j % cap.
struct ZAttentionArgs { int D; int pos0; int ctx; int cap; };
[[max_total_threads_per_threadgroup(128)]] kernel void zc_attention(device const float* qkv [[buffer(0)]], device const float* kc [[buffer(1)]],
                         device const float* vc [[buffer(2)]], device float* out [[buffer(3)]],
                         constant ZAttentionArgs& a [[buffer(4)]], uint2 gid [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[32];
  threadgroup float acc[128][33];
  threadgroup float total[65];
  const int t = gid.x, h = gid.y, D = a.D;
  const int p = a.pos0 + t;
  const int lo = max(0, p - a.ctx + 1);
  float qv[64];
  device const float* qr = qkv + (ulong)t * 3 * D + h * 64;
  for (int i = 0; i < 64; ++i) qv[i] = qr[i];
  float m = -INFINITY;
  for (int j = lo + (int)tid; j <= p; j += 128) {
    device const float* k = kc + (ulong)(j % a.cap) * D + h * 64;
    float d = 0.f;
    for (int i = 0; i < 64; ++i) d = fma(qv[i], k[i], d);
    m = fmax(m, d * 0.125f);
  }
  m = block_max_tree(m, part, 4, lane, sg);
  float s_[65];
  for (int i = 0; i < 65; ++i) s_[i] = 0.f;
  for (int j = lo + (int)tid; j <= p; j += 128) {
    device const float* k = kc + (ulong)(j % a.cap) * D + h * 64;
    device const float* v = vc + (ulong)(j % a.cap) * D + h * 64;
    float d = 0.f;
    for (int i = 0; i < 64; ++i) d = fma(qv[i], k[i], d);
    const float e = z3_exp(d * 0.125f - m);
    s_[64] += e;
    for (int i = 0; i < 64; ++i) s_[i] = fma(e, v[i], s_[i]);
  }
  // a thread's 65 sums folded in two halves (33 a thread fit), each in zonos3_codec.cu's order: thread 0 to 127
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
  if (tid < 64) out[(ulong)t * D + h * 64 + tid] = precise::divide(total[tid], total[64]);
}

// x[r, i] += scale[i] * y[r, i] (no scale: a plain residual add).
struct AddScaledArgs { int D; int has_scale; };
kernel void zc_add_scaled(device float* x [[buffer(0)]], device const float* y [[buffer(1)]],
                          device const float* scale [[buffer(2)]], constant AddScaledArgs& a [[buffer(3)]],
                          uint2 gid [[thread_position_in_grid]]) {
  const ulong r = gid.y;
  const int i = gid.x;
  if (i < a.D) x[r * a.D + i] += a.has_scale ? scale[i] * y[r * a.D + i] : y[r * a.D + i];
}

// GELU (erf) in place.
struct CountArgs { uint n; };
kernel void zc_gelu(device float* x [[buffer(0)]], constant CountArgs& a [[buffer(1)]], uint i [[thread_position_in_grid]]) {
  if (i < a.n) x[i] = 0.5f * x[i] * (1.f + z3_erf(x[i] * 0.70710678118654752f));
}

// Patch up by 2: in [T][C] to out [2T][C / 2], out[2t + j][c] = in[t][2c + j].
struct PatchArgs { int C; };
kernel void zc_patch_up(device const float* in [[buffer(0)]], device float* out [[buffer(1)]],
                        constant PatchArgs& a [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
  const ulong r = gid.y;
  const int c = gid.x;
  if (c < a.C / 2) out[r * (a.C / 2) + c] = in[(r >> 1) * a.C + 2 * c + (r & 1)];
}

// Causal depthwise conv (kernel 7) + bias over T rows of C channels; hist [C][6] holds the 6 inputs before them.
struct DwArgs { int T; int C; };
kernel void zc_dwconv(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                      device const float* b [[buffer(2)]], device float* hist [[buffer(3)]], device float* out [[buffer(4)]],
                      constant DwArgs& a [[buffer(5)]], uint c [[thread_position_in_grid]]) {
  if ((int)c >= a.C) return;
  float s[7];
  for (int j = 0; j < 6; ++j) s[j] = hist[c * 6 + j];
  for (int t = 0; t < a.T; ++t) {
    s[6] = x[(ulong)t * a.C + c];
    float acc = b[c];
    for (int j = 0; j < 7; ++j) acc = fma(w[c * 7 + j], s[j], acc);
    out[(ulong)t * a.C + c] = acc;
    for (int j = 0; j < 6; ++j) s[j] = s[j + 1];
  }
  for (int j = 0; j < 6; ++j) hist[c * 6 + j] = s[j];
}

// out[t] = table[codes[t][col]]: codes int32 [T][10], table [4096][D].
struct GatherArgs { int col; int D; };
kernel void zc_gather(device const int* codes [[buffer(0)]], device const float* table [[buffer(1)]],
                      device float* out [[buffer(2)]], constant GatherArgs& a [[buffer(3)]],
                      uint2 gid [[thread_position_in_grid]]) {
  const ulong t = gid.y;
  const int i = gid.x;
  if (i < a.D) out[t * a.D + i] = table[(ulong)codes[t * 10 + a.col] * a.D + i];
}

// z[t] += tables[i][codes[t][1 + i]] for the nine residual codebooks in order: tables [9][4096][D].
struct LatentArgs { int D; };
kernel void zc_latent(device const int* codes [[buffer(0)]], device const float* tables [[buffer(1)]],
                      device float* z [[buffer(2)]], constant LatentArgs& a [[buffer(3)]],
                      uint2 gid [[thread_position_in_grid]]) {
  const ulong t = gid.y;
  const int d = gid.x;
  if (d >= a.D) return;
  float s = z[t * a.D + d];
  for (int i = 0; i < 9; ++i) s += tables[((ulong)i * 4096 + codes[t * 10 + 1 + i]) * a.D + d];
  z[t * a.D + d] = s;
}

// The head's [log-magnitude | phase] rows (each half bins wide) to [real | imaginary] in place, magnitude at most 100.
struct SpecArgs { int half_; };
kernel void zc_spec(device float* y [[buffer(0)]], constant SpecArgs& a [[buffer(1)]], uint2 gid [[thread_position_in_grid]]) {
  device float* r = y + (ulong)gid.y * 2 * a.half_;
  const int i = gid.x;
  if (i >= a.half_) return;
  const float lm = r[i];
  const float mag = precise::exp(lm > 4.605170185988092f ? 4.605170185988092f : lm);  // a comparison: a NaN stays NaN
  const float ph = r[a.half_ + i];
  r[i] = mag * precise::cos(ph);
  r[a.half_ + i] = mag * precise::sin(ph);
}

// Centered overlap-add of windowed frames [F][N] at hop: out[i] is sample n0 + i over the same frames' window envelope.
struct OlaArgs { int F; int N; int hop; int pad; long n0; long count; };
kernel void zc_ola(device const float* frames [[buffer(0)]], device const float* win2 [[buffer(1)]],
                   device float* out [[buffer(2)]], constant OlaArgs& a [[buffer(3)]], uint gi [[thread_position_in_grid]]) {
  const long i = gi;
  if (i >= a.count) return;
  const long p = a.n0 + i + a.N / 2;
  long f0 = (p - a.N) / a.hop + 1;
  if (p - a.N < 0) f0 = 0;
  long f1 = p / a.hop;
  if (f1 > a.F - 1) f1 = a.F - 1;
  float s = 0.f, e = 0.f;
  for (long f = f0; f <= f1; ++f) {
    const int o = (int)(p - f * a.hop);
    s += frames[(ulong)f * a.N + o];
    e += win2[o];
  }
  out[i] = precise::divide(s, e);
}

// The mixer: out[i] = sum_b sum_j w[b][j] s_b[n0 + i + j - 63], zero outside [0, total); s rows are `stride` apart.
struct MixArgs { long stride; long total; long n0; long count; };
kernel void zc_mix(device const float* s [[buffer(0)]], device const float* w [[buffer(1)]], device float* out [[buffer(2)]],
                   constant MixArgs& a [[buffer(3)]], uint gi [[thread_position_in_grid]]) {
  const long i = gi;
  if (i >= a.count) return;
  float acc = 0.f;
  for (int b = 0; b < 2; ++b)
    for (int j = 0; j < 127; ++j) {
      const long p = a.n0 + i + j - 63;
      if (p >= 0 && p < a.total) acc = fma(w[b * 127 + j], s[(ulong)b * a.stride + p], acc);
    }
  out[i] = acc;
}

// 44160 to 44100 Hz: output m0 + i = 735 q + c is sum_j kernel[c][j] x[736 q + j - 17], zero outside [0, total).
struct ResampleArgs { long total; long m0; long count; };
kernel void zc_resample(device const float* x [[buffer(0)]], device const float* kern [[buffer(1)]],
                        device float* out [[buffer(2)]], constant ResampleArgs& a [[buffer(3)]],
                        uint gi [[thread_position_in_grid]]) {
  const long i = gi;
  if (i >= a.count) return;
  const long m = a.m0 + i;
  const long q = m / 735;
  const int c = (int)(m % 735);
  float acc = 0.f;
  for (int j = 0; j < 770; ++j) {
    const long p = 736 * q + j - 17;
    if (p >= 0 && p < a.total) acc = fma(kern[c * 770 + j], x[p], acc);
  }
  out[i] = acc;
}
