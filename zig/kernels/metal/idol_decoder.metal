// Idol's octree and Gaussian decoders: their steps beside the denoiser's (idol.metal), rounded where torch rounds.

#include <metal_stdlib>
#include "../nax.h"
using namespace metal;

// h [rows, 1024] fp32 = in_proj(p) + pe(p): per axis sin then cos of p[a] * freqs[j] * pi (170 each), then 4 zeros.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void dec_embed(
    const device float* p [[buffer(0)]], const device float* w [[buffer(1)]], const device float* b [[buffer(2)]],
    const device float* freqs [[buffer(3)]], device float* h [[buffer(4)]], uint2 at [[thread_position_in_grid]]) {
  const uint c = at.x, r = at.y;
  const float x0 = p[r * 3], x1 = p[r * 3 + 1], x2 = p[r * 3 + 2];
  const float s = fma(x2, w[c * 3 + 2], fma(x1, w[c * 3 + 1], x0 * w[c * 3])) + b[c];
  float pe = 0.0f;
  if (c < 1020) {
    const uint a = c / 340, j = c % 340;
    const float x = a == 0 ? x0 : (a == 1 ? x1 : x2);
    const float ang = (x * freqs[j % 170]) * 3.14159265358979323846f;
    pe = j < 170 ? metal::precise::sin(ang) : metal::precise::cos(ang);
  }
  h[ulong(r) * 1024 + c] = s + pe;
}

// y16 = fp16(y + b) by column: an fp32 Linear's bias, then the cast to the blocks' fp16.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void dec_half(
    const device float* y [[buffer(0)]], const device float* b [[buffer(1)]], device half* o [[buffer(2)]],
    constant uint& width [[buffer(3)]], uint2 at [[thread_position_in_grid]]) {
  const ulong i = ulong(at.y) * width + at.x;
  o[i] = half(y[i] + b[at.x]);
}

// kv [rows, 2048] = fp16(ctx w^T + b): to_kv of the fp16 latent [rows, 16], fp32 sums, fp16 weights and bias.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void dec_kv(
    const device half* ctx [[buffer(0)]], const device half* w [[buffer(1)]], const device half* b [[buffer(2)]],
    device half* kv [[buffer(3)]], uint2 at [[thread_position_in_grid]]) {
  const uint c = at.x, r = at.y;
  float s = 0.0f;
  for (uint k = 0; k < 16; k++) s = fma(float(ctx[r * 16 + k]), float(w[c * 16 + k]), s);
  kv[ulong(r) * 2048 + c] = half(s + float(b[c]));
}

struct DecHeads {
  uint rows;
  uint stride;  // fp16 values from one source row to the next
  uint offset;  // the first column of this tensor's 16 heads in a source row
  uint norm;    // 1: MultiHeadRMSNorm (F.normalize times gamma times 8), 0: a plain copy (values)
};

// One tensor's heads [16, rows, 64] from packed fp16 rows, RMS-normed per head in fp32 as MultiHeadRMSNorm does.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void dec_heads(
    const device half* src [[buffer(0)]], const device float* gamma [[buffer(1)]], device half* dst [[buffer(2)]],
    constant DecHeads& p [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint r = tg.x, h = tg.y * 8 + sg;
  if (h >= 16) return;
  const ulong from = ulong(r) * p.stride + p.offset + h * 64 + lane * 2;
  const ulong to = (ulong(h) * p.rows + r) * 64 + lane * 2;
  const float a0 = float(src[from]), a1 = float(src[from + 1]);
  if (p.norm == 0) {
    dst[to] = half(a0);
    dst[to + 1] = half(a1);
    return;
  }
  const float n = max(metal::precise::sqrt(simd_sum(a0 * a0 + a1 * a1)), 1e-12f);
  const uint g = h * 64 + lane * 2;
  dst[to] = half(a0 / n * gamma[g] * 8.0f);
  dst[to + 1] = half(a1 / n * gamma[g + 1] * 8.0f);
}

struct DecFew {
  uint lq;
  uint lk;  // at most 32 * 256 keys
};

// o [lq, 1024] = softmax(q k^T / 8) v in fp32 for lq <= 8 (torch's MPS short-query kernel); 256 threads a row, head.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void dec_attention_few(
    const device half* Q [[buffer(0)]], const device half* K [[buffer(1)]], const device half* V [[buffer(2)]],
    device half* O [[buffer(3)]], constant DecFew& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
    uint t [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float q[64];
  threadgroup float part[8][64];
  threadgroup float red[9];
  const uint r = tg.x, h = tg.y;
  if (t < 64) q[t] = float(Q[(ulong(h) * p.lq + r) * 64 + t]) * 0.125f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const device half* kh = K + ulong(h) * p.lk * 64;
  const device half* vh = V + ulong(h) * p.lk * 64;
  float s[32];
  float top = -INFINITY;
  for (uint i = 0; i < 32; i++) {
    const uint j = t + i * 256;
    s[i] = -INFINITY;
    if (j >= p.lk) continue;
    const device half4* kr = (const device half4*)(kh + ulong(j) * 64);
    float d = 0.0f;
    for (uint c = 0; c < 16; c++) {
      const float4 kv = float4(kr[c]);
      d = fma(q[4 * c], kv.x, fma(q[4 * c + 1], kv.y, fma(q[4 * c + 2], kv.z, fma(q[4 * c + 3], kv.w, d))));
    }
    s[i] = d;
    top = max(top, d);
  }
  top = simd_max(top);
  if (lane == 0) red[sg] = top;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    float m = red[0];
    for (uint i = 1; i < 8; i++) m = max(m, red[i]);
    red[8] = m;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  top = red[8];
  float o[64];
  for (uint c = 0; c < 64; c++) o[c] = 0.0f;
  float sum = 0.0f;
  for (uint i = 0; i < 32; i++) {
    const uint j = t + i * 256;
    if (j >= p.lk) continue;
    const float e = metal::precise::exp(s[i] - top);
    sum += e;
    const device half4* vr = (const device half4*)(vh + ulong(j) * 64);
    for (uint c = 0; c < 16; c++) {
      const float4 vv = float4(vr[c]);
      o[4 * c] = fma(e, vv.x, o[4 * c]);
      o[4 * c + 1] = fma(e, vv.y, o[4 * c + 1]);
      o[4 * c + 2] = fma(e, vv.z, o[4 * c + 2]);
      o[4 * c + 3] = fma(e, vv.w, o[4 * c + 3]);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  sum = simd_sum(sum);
  if (lane == 0) red[sg] = sum;
  for (uint c = 0; c < 64; c++) {
    const float v = simd_sum(o[c]);
    if (lane == 0) part[sg][c] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < 64) {
    float total = 0.0f, acc = 0.0f;
    for (uint i = 0; i < 8; i++) {
      total += red[i];
      acc += part[i][t];
    }
    O[ulong(r) * 1024 + h * 64 + t] = half(acc / total);
  }
}

// idol_attention.metal's, P V as fp16 P hi + lo: the unit takes fp32 P short (outputs low by ~2^-12, compounding).
constant constexpr float kDecMasked = -3.402823466e+38f;
constant constexpr int kDecAnyCols = 1 << 30;

// Fold one fragment's rows into r[2] (rows home.y, home.y + 8): pairs, then lanes 1 and 8 apart, then into r.
template <bool MAX>
inline void dec_fold(thread const tfp::frag<float>& f, thread float (&r)[2]) {
  TF_UNROLL
  for (short h = 0; h < 2; h++) {
    const short b = 4 * h;
    float t = MAX ? max(max(f[b], f[b + 1]), max(f[b + 2], f[b + 3])) : (f[b] + f[b + 1]) + (f[b + 2] + f[b + 3]);
    const float u = simd_shuffle_xor(t, ushort(1));
    t = MAX ? max(t, u) : t + u;
    const float w = simd_shuffle_xor(t, ushort(8));
    t = MAX ? max(t, w) : t + w;
    r[h] = MAX ? max(r[h], t) : r[h] + t;
  }
}

// One 64-query block of one head over 32-key blocks; P and F as Kernels.attention sets them (kernels.zig).
template <typename T, bool AQ, bool AK>
inline void dec_attention(const device T* Q, const device T* K, const device T* V, device T* O, const device int* P,
                         const device float* F, uint3 tg, uint sg, uint lane) {
  constexpr int BQ = 64, BK = 32, D = 64;
  const int h = int(tg.y), lq = P[1], lk = P[2];
  Q += (long(h) * lq + long(tg.x) * BQ) * D;
  K += long(h) * lk * D;
  V += long(h) * lk * D;
  const int os = P[0] * D;
  O += long(tg.x) * BQ * os + h * D;
  const float scale2 = F[0] * 1.44269504089f;
  const short tm = 16 * short(sg);
  Q += tm * D;
  const short2 home = tfp::frag_home(ushort(lane));

  tfp::frag<float> acc[4];
  TF_UNROLL
  for (short i = 0; i < 4; i++) acc[i] = tfp::frag<float>(0);
  float top[2] = {kDecMasked, kDecMasked}, total[2] = {0.0f, 0.0f};
  const bool last_q = int(tg.x) == P[4];
  const short q_rows = short(P[6] - tm), k_rows = short(P[7]);

  for (int kb = 0; kb < P[3]; kb++) {
    const bool last_k = kb == P[5];
    tfp::frag<float> s[2] = {tfp::frag<float>(0), tfp::frag<float>(0)};
    TF_UNROLL
    for (short d = 0; d < 4; d++) {
      tfp::frag<T> q, k0, k1;
      if (!AQ && last_q) {
        tfp::frag_get_in(q, Q, D, 0, 16 * d, home, q_rows, kDecAnyCols);
      } else {
        tfp::frag_get(q, Q, D, 0, 16 * d, home);
      }
      if (!AK && last_k) {
        tfp::frag_get_t_in(k0, K, D, 0, 16 * d, home, k_rows, kDecAnyCols);
        tfp::frag_get_t_in(k1, K, D, 16, 16 * d, home, k_rows, kDecAnyCols);
      } else {
        tfp::frag_get_t(k0, K, D, 0, 16 * d, home);
        tfp::frag_get_t(k1, K, D, 16, 16 * d, home);
      }
      tfp::mma_16x32<false, true>(s[0], s[1], q, k0, k1);
    }
    TF_UNROLL
    for (short f = 0; f < 2; f++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) s[f][e] *= scale2;
    }
    if (!AK && last_k) {
      TF_UNROLL
      for (short f = 0; f < 2; f++) {
        TF_UNROLL
        for (short e = 0; e < 8; e++) s[f][e] = (16 * f + home.x + TF_COL(e)) < k_rows ? s[f][e] : kDecMasked;
      }
    }
    float top_new[2] = {top[0], top[1]};
    dec_fold<true>(s[0], top_new);
    dec_fold<true>(s[1], top_new);
    TF_UNROLL
    for (short f = 0; f < 2; f++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) s[f][e] = fast::exp2(s[f][e] - top_new[e >> 2]);
    }
    float scale_old[2];
    TF_UNROLL
    for (short r = 0; r < 2; r++) {
      scale_old[r] = fast::exp2(top[r] - top_new[r]);
      top[r] = top_new[r];
      total[r] = total[r] * scale_old[r];
    }
    dec_fold<false>(s[0], total);
    dec_fold<false>(s[1], total);
    TF_UNROLL
    for (short i = 0; i < 4; i++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) acc[i][e] = acc[i][e] * scale_old[e >> 2];
    }
    tfp::frag<half> ph[2], pl[2];
    TF_UNROLL
    for (short k = 0; k < 2; k++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        ph[k][e] = half(s[k][e]);
        pl[k][e] = half(s[k][e] - float(ph[k][e]));
      }
    }
    simdgroup_barrier(mem_flags::mem_none);
    TF_UNROLL
    for (short d = 0; d < 4; d += 2) {
      TF_UNROLL
      for (short k = 0; k < 2; k++) {
        tfp::frag<T> v0, v1;
        if (!AK && last_k) {
          tfp::frag_get_in(v0, V, D, 16 * k, 16 * d, home, k_rows, kDecAnyCols);
          tfp::frag_get_in(v1, V, D, 16 * k, 16 * d + 16, home, k_rows, kDecAnyCols);
        } else {
          tfp::frag_get(v0, V, D, 16 * k, 16 * d, home);
          tfp::frag_get(v1, V, D, 16 * k, 16 * d + 16, home);
        }
        tfp::mma_16x32<false, false>(acc[d], acc[d + 1], ph[k], v0, v1);
        tfp::mma_16x32<false, false>(acc[d], acc[d + 1], pl[k], v0, v1);
      }
    }
    K += BK * D;
    V += BK * D;
  }

  threadgroup_barrier(mem_flags::mem_none);
  float inv[2];
  TF_UNROLL
  for (short r = 0; r < 2; r++) inv[r] = 1.f / total[r];
  TF_UNROLL
  for (short i = 0; i < 4; i++) {
    TF_UNROLL
    for (short e = 0; e < 8; e++) acc[i][e] = acc[i][e] * inv[e >> 2];
  }
  O += tm * os;
  if (!AQ && last_q) {
    if (q_rows <= 0) return;
    TF_UNROLL
    for (short i = 0; i < 4; i++) tfp::frag_put_in(acc[i], O, os, 0, 16 * i, home, q_rows, kDecAnyCols);
  } else {
    TF_UNROLL
    for (short i = 0; i < 4; i++) tfp::frag_put(acc[i], O, os, 0, 16 * i, home);
  }
}

#define DEC_ATTENTION(NAME, T, AQ, AK)                                                                            \
  [[max_total_threads_per_threadgroup(128)]] [[kernel]] void NAME(                                             \
      const device T* Q [[buffer(0)]], const device T* K [[buffer(1)]], const device T* V [[buffer(2)]],        \
      const device int32_t* P [[buffer(3)]], const device float* F [[buffer(4)]], device T* O [[buffer(5)]],     \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],                     \
      uint3 tg [[threadgroup_position_in_grid]]) {                                                             \
    dec_attention<T, AQ, AK>(Q, K, V, O, P, F, tg, sg, lane);                                                   \
  }
DEC_ATTENTION(dec_attention_00, half, false, false)
DEC_ATTENTION(dec_attention_01, half, false, true)
DEC_ATTENTION(dec_attention_10, half, true, false)
DEC_ATTENTION(dec_attention_11, half, true, true)
