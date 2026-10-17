// Drama 3 Gated DeltaNet: the causal conv and SiLU, the gated delta rule's recurrence in fp32, and the gated norm.
#include "d3_common.h"

struct ConvArgs {
  uint ps, T, C;
};

// Depthwise causal conv (kernel 4) + SiLU over the qkv channels of [qkv | z | b | a] rows; state keeps 3 inputs.
[[max_total_threads_per_threadgroup(128)]] kernel void d3_conv_silu(device const bf16* proj [[buffer(0)]], device const bf16* w [[buffer(1)]],
                         device bf16* state [[buffer(2)]], device bf16* out [[buffer(3)]],
                         constant ConvArgs& a [[buffer(4)]], uint c [[thread_position_in_grid]]) {
  if (c >= a.C) return;
  const float w0 = b2f(w[c * 4 + 0]), w1 = b2f(w[c * 4 + 1]), w2 = b2f(w[c * 4 + 2]), w3 = b2f(w[c * 4 + 3]);
  float s0 = b2f(state[c * 3 + 0]), s1 = b2f(state[c * 3 + 1]), s2 = b2f(state[c * 3 + 2]);
  for (uint t = 0; t < a.T; ++t) {
    const float x = b2f(proj[(ulong)t * a.ps + c]);
    float acc = s0 * w0;
    acc = fma(s1, w1, acc);
    acc = fma(s2, w2, acc);
    acc = fma(x, w3, acc);
    out[(ulong)t * a.C + c] = f2b(d3_silu(acc));
    s0 = s1;
    s1 = s2;
    s2 = x;
  }
  state[c * 3 + 0] = f2b(s0);
  state[c * 3 + 1] = f2b(s1);
  state[c * 3 + 2] = f2b(s2);
}

struct GdnArgs {
  uint ps, T;
};

// Gated delta rule in fp32: a group per (value head, 32 columns), a thread holds 32 rows of one state column.
[[max_total_threads_per_threadgroup(128)]] kernel void d3_gdn(device const bf16* conv [[buffer(0)]], device const bf16* proj [[buffer(1)]],
                   device const bf16* A_log [[buffer(2)]], device const bf16* dt_bias [[buffer(3)]],
                   device float* S [[buffer(4)]], device float* o [[buffer(5)]], constant GdnArgs& a [[buffer(6)]],
                   uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                   uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint DK = 128, DV = 128, NV = 32;
  const uint h = tg.x, kh = h >> 1, v = tg.y * 32 + lane, kq = sg;
  threadgroup float qs[DK], ks[DK], part[4][32];
  device float* Sh = S + (ulong)h * DK * DV;
  float s[32];
  for (uint j = 0; j < 32; ++j) s[j] = Sh[(kq * 32 + j) * DV + v];
  const float A = -precise::exp(b2f(A_log[h]));
  const float dtb = b2f(dt_bias[h]);
  const float qscale = rsqrt(float(DK));
  for (uint t = 0; t < a.T; ++t) {
    device const bf16* row = conv + (ulong)t * 8192;
    if (sg == 0) {
      float qv[4], kv[4], qq = 0.0f, kk = 0.0f;
      for (uint j = 0; j < 4; ++j) {
        qv[j] = b2f(row[kh * DK + lane + 32 * j]);
        kv[j] = b2f(row[2048 + kh * DK + lane + 32 * j]);
        qq += qv[j] * qv[j];
        kk += kv[j] * kv[j];
      }
      qq = simd_sum(qq);
      kk = simd_sum(kk);
      const float qi = rsqrt(qq + 1e-6f) * qscale, ki = rsqrt(kk + 1e-6f);
      for (uint j = 0; j < 4; ++j) {
        qs[lane + 32 * j] = qv[j] * qi;
        ks[lane + 32 * j] = kv[j] * ki;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const bf16* pr = proj + (ulong)t * a.ps;
    const float bb = d3_sigmoid(b2f(pr[12288 + h]));
    const float decay = precise::exp(A * d3_softplus(b2f(pr[12288 + NV + h]) + dtb));
    float mem = 0.0f;
    for (uint j = 0; j < 32; ++j) {
      s[j] *= decay;
      mem = fma(s[j], ks[kq * 32 + j], mem);
    }
    part[kq][lane] = mem;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float vv = b2f(row[4096 + h * DV + v]);
    const float delta = (vv - (((part[0][lane] + part[1][lane]) + part[2][lane]) + part[3][lane])) * bb;
    float out = 0.0f;
    for (uint j = 0; j < 32; ++j) {
      s[j] = fma(ks[kq * 32 + j], delta, s[j]);
      out = fma(s[j], qs[kq * 32 + j], out);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    part[kq][lane] = out;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (kq == 0) o[(ulong)t * (NV * DV) + h * DV + v] = ((part[0][lane] + part[1][lane]) + part[2][lane]) + part[3][lane];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint j = 0; j < 32; ++j) Sh[(kq * 32 + j) * DV + v] = s[j];
}

struct GatedNormArgs {
  uint ps;
  float eps;
};

// Gated RMSNorm per 128-wide value head: y = bf16(o * rsqrt(mean(o^2) + eps) * w * silu(z)); a simdgroup a head.
[[max_total_threads_per_threadgroup(256)]] kernel void d3_gated_norm(device const float* o [[buffer(0)]], device const bf16* proj [[buffer(1)]],
                          device const bf16* w [[buffer(2)]], device bf16* y [[buffer(3)]],
                          constant GatedNormArgs& a [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint t = tg.x, h = tg.y * 8 + sg;
  device const float* orow = o + (ulong)t * 4096 + h * 128;
  device const bf16* z = proj + (ulong)t * a.ps + 8192 + h * 128;
  float v[4], ss = 0.0f;
  for (uint j = 0; j < 4; ++j) {
    v[j] = orow[lane + 32 * j];
    ss += v[j] * v[j];
  }
  const float inv = rsqrt(simd_sum(ss) / 128.0f + a.eps);
  for (uint j = 0; j < 4; ++j) {
    const uint i = lane + 32 * j;
    const float n = v[j] * inv * b2f(w[i]);
    y[(ulong)t * 4096 + h * 128 + i] = f2b(n * d3_silu(b2f(z[i])));
  }
}
