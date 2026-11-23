// Inkling 2 Mini M3: packed weight decode in registers, BF16 activations, FP32 reductions.
#include <metal_stdlib>
using namespace metal;
#pragma METAL fp contract(off)

struct Ink2MatrixArgs { uint k, n, topk, experts; };
struct Ink2DecodeArgs { uint rows, k; };

inline float ink2_e2m1(uint code) {
  const float magnitude[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float v = magnitude[code & 7u];
  return (code & 8u) ? -v : v;
}

inline float ink2_e4m3(uint code) {
  const uint exp = (code >> 3u) & 15u, mant = code & 7u;
  if (exp == 15u && mant == 7u) return NAN;
  const float v = exp ? ldexp(1.0f + float(mant) * 0.125f, int(exp) - 7)
                      : float(mant) * (1.0f / 512.0f);
  return (code & 0x80u) ? -v : v;
}

inline float ink2_phi(device const float* amax, device const float* denominator, uint expert) {
  // Loader casts both globals to BF16 before the dequantization operation.
  return precise::divide(float(bfloat(amax[expert])), float(bfloat(denominator[0])));
}

inline float ink2_nv_weight(uchar packed, uint column, uchar scale, float phi) {
  const uint code = (column & 1u) ? uint(packed >> 4u) : uint(packed & 15u);
  const float inner = ink2_e2m1(code) * ink2_e4m3(scale);
  return float(bfloat(inner * phi));
}

inline float ink2_mx_weight(uchar value, uchar scale) {
  const float w = ldexp(ink2_e4m3(value), int(scale) - 127);
  // E4M3 times a power of two is exactly representable in BF16, still round explicitly.
  return float(bfloat(w));
}

// Four packed bytes as one word, at any byte offset.
inline uint ink2_word(device const uchar* p) {
  return as_type<uint>(uchar4(*(device const packed_uchar4*)p));
}

// E2M1 times two is 0, 1, 2, 3, 4, 6, 8 or 12: one nibble of this word a code.
inline float ink2_e2m1_twice(uint code) {
  const float v = float((0xC8643210u >> (4u * (code & 7u))) & 15u);
  return (code & 8u) ? -v : v;
}

// ink2_e4m3 with its finite values built from their bits.
inline float ink2_e4m3_bits(uint code) {
  const uint exp = (code >> 3u) & 15u, mant = code & 7u;
  const float v = exp ? as_type<float>(((exp + 120u) << 23u) | (mant << 20u)) : float(mant) * (1.0f / 512.0f);
  const float s = (code & 0x80u) ? -v : v;
  return (exp == 15u && mant == 7u) ? NAN : s;
}

// One simdgroup a dot: lane l sums blocks l, l + 32, ... in column order, then the 32 lanes are summed.
inline float ink2_dot_nv(device const uchar* w, device const uchar* scales,
                         device const bfloat* x, float phi, uint k, uint lane) {
  float sum = 0.0f;
  for (uint b = lane; b < k / 16u; b += 32u) {
    // ink2_nv_weight's inner product, exact in fp32 either way: (2 * e2m1) * (e4m3 / 2).
    const float half_scale = 0.5f * ink2_e4m3(scales[b]);
    const uint lo = ink2_word(w + 8u * b), hi = ink2_word(w + 8u * b + 4u);
    for (uint j = 0; j < 16u; ++j) {
      const uint code = ((j < 8u ? lo : hi) >> (4u * (j & 7u))) & 15u;
      const float weight = float(bfloat(ink2_e2m1_twice(code) * half_scale * phi));
      sum += weight * float(x[16u * b + j]);
    }
  }
  return simd_sum(sum);
}

// As ink2_dot_nv over 32-column MXFP8 blocks.
inline float ink2_dot_mx(device const uchar* w, device const uchar* scales,
                         device const bfloat* x, uint k, uint lane) {
  float sum = 0.0f;
  for (uint b = lane; b < k / 32u; b += 32u) {
    const int shift = int(scales[b]) - 127;
    for (uint j = 0; j < 32u; ++j) {
      const uint c = 32u * b + j;
      const float weight = float(bfloat(ldexp(ink2_e4m3_bits(w[c]), shift)));
      sum += weight * float(x[c]);
    }
  }
  return simd_sum(sum);
}

inline float ink2_cap(float x, float cap) {
  return cap * precise::tanh(precise::divide(x, cap));
}

inline bfloat ink2_act(float gate_dot, float up_dot) {
  const float gate = float(bfloat(gate_dot)), up = float(bfloat(up_dot));
  const float left = ink2_cap(gate, 4.0f) * precise::divide(1.0f, 1.0f + precise::exp(-gate));
  // Both GEMM results round before the softcapped SwiGLU; sigmoid uses the rounded gate.
  return bfloat(left * ink2_cap(up, 25.0f));
}

// The four products: threadgroup (x, y) is outputs 8x to 8x + 7 of route pair y, one simdgroup an output.
[[kernel, max_total_threads_per_threadgroup(256)]] void tf_ink2_nvfp4_gu(
    device const bfloat* X [[buffer(0)]], device const uchar* W [[buffer(1)]],
    device const uchar* S [[buffer(2)]], device const float* amax [[buffer(3)]],
    device const float* denominator [[buffer(4)]], device const int* ids [[buffer(5)]],
    device bfloat* Y [[buffer(6)]], constant Ink2MatrixArgs& a [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint n = tg.x * 8u + sg, group = tg.y;
  if (n >= a.n) return;
  const uint token = group / a.topk, expert = uint(ids[group]);
  const ulong expert_w = ulong(expert) * 2u * a.n * (a.k / 2u);
  const ulong expert_s = ulong(expert) * 2u * a.n * (a.k / 16u);
  const float phi = ink2_phi(amax, denominator, expert);
  const uint gr = 2u * n, ur = gr + 1u;
  const float g = ink2_dot_nv(W + expert_w + ulong(gr) * (a.k / 2u),
                              S + expert_s + ulong(gr) * (a.k / 16u), X + ulong(token) * a.k, phi, a.k, lane);
  const float u = ink2_dot_nv(W + expert_w + ulong(ur) * (a.k / 2u),
                              S + expert_s + ulong(ur) * (a.k / 16u), X + ulong(token) * a.k, phi, a.k, lane);
  if (lane == 0u) Y[ulong(group) * a.n + n] = ink2_act(g, u);
}

[[kernel, max_total_threads_per_threadgroup(256)]] void tf_ink2_mxfp8_gu(
    device const bfloat* X [[buffer(0)]], device const uchar* W [[buffer(1)]],
    device const uchar* S [[buffer(2)]], device const int* ids [[buffer(3)]],
    device bfloat* Y [[buffer(4)]], constant Ink2MatrixArgs& a [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint n = tg.x * 8u + sg, group = tg.y;
  if (n >= a.n) return;
  const uint token = group / a.topk, expert = uint(ids[group]);
  const ulong expert_w = ulong(expert) * 2u * a.n * a.k;
  const ulong expert_s = ulong(expert) * 2u * a.n * (a.k / 32u);
  const uint gr = 2u * n, ur = gr + 1u;
  const float g = ink2_dot_mx(W + expert_w + ulong(gr) * a.k,
                              S + expert_s + ulong(gr) * (a.k / 32u), X + ulong(token) * a.k, a.k, lane);
  const float u = ink2_dot_mx(W + expert_w + ulong(ur) * a.k,
                              S + expert_s + ulong(ur) * (a.k / 32u), X + ulong(token) * a.k, a.k, lane);
  if (lane == 0u) Y[ulong(group) * a.n + n] = ink2_act(g, u);
}

[[kernel, max_total_threads_per_threadgroup(256)]] void tf_ink2_nvfp4_down(
    device const bfloat* X [[buffer(0)]], device const uchar* W [[buffer(1)]],
    device const uchar* S [[buffer(2)]], device const float* amax [[buffer(3)]],
    device const float* denominator [[buffer(4)]], device const int* ids [[buffer(5)]],
    device bfloat* Y [[buffer(6)]], constant Ink2MatrixArgs& a [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint n = tg.x * 8u + sg, group = tg.y;
  if (n >= a.n) return;
  const uint expert = uint(ids[group]);
  const ulong expert_w = ulong(expert) * a.n * (a.k / 2u);
  const ulong expert_s = ulong(expert) * a.n * (a.k / 16u);
  const float phi = ink2_phi(amax, denominator, expert);
  const float d = ink2_dot_nv(W + expert_w + ulong(n) * (a.k / 2u),
                              S + expert_s + ulong(n) * (a.k / 16u), X + ulong(group) * a.k, phi, a.k, lane);
  if (lane == 0u) Y[ulong(group) * a.n + n] = bfloat(d);
}

[[kernel, max_total_threads_per_threadgroup(256)]] void tf_ink2_mxfp8_down(
    device const bfloat* X [[buffer(0)]], device const uchar* W [[buffer(1)]],
    device const uchar* S [[buffer(2)]], device const int* ids [[buffer(3)]],
    device bfloat* Y [[buffer(4)]], constant Ink2MatrixArgs& a [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint n = tg.x * 8u + sg, group = tg.y;
  if (n >= a.n) return;
  const uint expert = uint(ids[group]);
  const ulong expert_w = ulong(expert) * a.n * a.k;
  const ulong expert_s = ulong(expert) * a.n * (a.k / 32u);
  const float d = ink2_dot_mx(W + expert_w + ulong(n) * a.k,
                              S + expert_s + ulong(n) * (a.k / 32u), X + ulong(group) * a.k, a.k, lane);
  if (lane == 0u) Y[ulong(group) * a.n + n] = bfloat(d);
}

// ---- Grouped by expert: the same sums a row as the kernels above, with each weight block decoded once a pass. ----
struct Ink2GroupArgs { uint pairs, experts, cap; };
struct Ink2GroupedArgs { uint k, n, topk, cap; };
#define INK2_PASS 4u

// Expert e's route pairs in pair order: one simdgroup per expert scans ids in coalesced 32-lane steps.
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_ink2_group(
    device const int* ids [[buffer(0)]], device uint* counts [[buffer(1)]],
    device uint* members [[buffer(2)]], constant Ink2GroupArgs& a [[buffer(3)]],
    uint e [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  if (e >= a.experts) return;
  uint n = 0u;
  for (uint base = 0u; base < a.pairs; base += 32u) {
    const uint p = base + lane;
    const uint hit = (p < a.pairs && uint(ids[p]) == e) ? 1u : 0u;
    const uint off = simd_prefix_exclusive_sum(hit);
    if (hit != 0u && n + off < a.cap) members[e * a.cap + n + off] = p;
    n += simd_sum(hit);
  }
  if (lane == 0u) counts[e] = min(n, a.cap);
}

// Small-batch (pairs <= 64) single-TG group + active-expert compaction: writes dense active[0..pairs-1].
[[kernel, max_total_threads_per_threadgroup(128)]] void tf_ink2_group_compact(
    device const int* ids [[buffer(0)]], device uint* counts [[buffer(1)]],
    device uint* members [[buffer(2)]], device uint* active [[buffer(3)]],
    constant Ink2GroupArgs& a [[buffer(4)]], uint e [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup uint s_ids[64];
  threadgroup uint sg_cnt[4];
  if (e < a.pairs) {
    s_ids[e] = uint(ids[e]);
    active[e] = 0xffffffffu;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint n = 0u;
  for (uint p = 0u; p < a.pairs; ++p) {
    if (s_ids[p] == e) {
      if (n < a.cap) members[e * a.cap + n] = p;
      ++n;
    }
  }
  counts[e] = min(n, a.cap);
  const uint act = n > 0u ? 1u : 0u;
  const uint off_in_sg = simd_prefix_exclusive_sum(act);
  const uint total_sg = simd_sum(act);
  if (lane == 0u) sg_cnt[sg] = total_sg;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (act != 0u) {
    uint base = 0u;
    for (uint s = 0u; s < sg; ++s) base += sg_cnt[s];
    active[base + off_in_sg] = e;
  }
}

// Block b of a packed NVFP4 row as its 16 BF16 weights, each as ink2_dot_nv decodes it.
inline void ink2_nv_block(device const uchar* w, device const uchar* scales, float phi, uint b, thread float* out) {
  const float half_scale = 0.5f * ink2_e4m3(scales[b]);
  const uint lo = ink2_word(w + 8u * b), hi = ink2_word(w + 8u * b + 4u);
  for (uint j = 0; j < 16u; ++j) {
    const uint code = ((j < 8u ? lo : hi) >> (4u * (j & 7u))) & 15u;
    out[j] = float(bfloat(ink2_e2m1_twice(code) * half_scale * phi));
  }
}

// Block b of an MXFP8 row as its 32 BF16 weights, each as ink2_dot_mx decodes it.
inline void ink2_mx_block(device const uchar* w, device const uchar* scales, uint b, thread float* out) {
  const int shift = int(scales[b]) - 127;
  for (uint j = 0; j < 32u; ++j) out[j] = float(bfloat(ldexp(ink2_e4m3_bits(w[32u * b + j]), shift)));
}

// Threadgroup (x, e): outputs 8x to 8x + 7 of expert e for each of its route pairs, four pairs a pass.
[[kernel, max_total_threads_per_threadgroup(256)]] void tf_ink2_nvfp4_gu_grouped(
    device const bfloat* X [[buffer(0)]], device const uchar* W [[buffer(1)]],
    device const uchar* S [[buffer(2)]], device const float* amax [[buffer(3)]],
    device const float* denominator [[buffer(4)]], device const uint* counts [[buffer(5)]],
    device const uint* members [[buffer(6)]], device bfloat* Y [[buffer(7)]],
    constant Ink2GroupedArgs& a [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint n = tg.x * 8u + sg, expert = tg.y;
  if (n >= a.n) return;
  const uint count = counts[expert];
  if (count == 0u) return;
  const ulong expert_w = ulong(expert) * 2u * a.n * (a.k / 2u);
  const ulong expert_s = ulong(expert) * 2u * a.n * (a.k / 16u);
  const float phi = precise::divide(float(bfloat(amax[expert])), float(bfloat(denominator[0])));
  device const uchar* gate_w = W + expert_w + ulong(2u * n) * (a.k / 2u);
  device const uchar* gate_s = S + expert_s + ulong(2u * n) * (a.k / 16u);
  device const uchar* up_w = gate_w + (a.k / 2u);
  device const uchar* up_s = gate_s + (a.k / 16u);
  for (uint first = 0u; first < count; first += INK2_PASS) {
    uint pair[INK2_PASS];
    device const bfloat* x[INK2_PASS];
    float g[INK2_PASS], u[INK2_PASS];
    for (uint j = 0u; j < INK2_PASS; ++j) {
      pair[j] = members[expert * a.cap + min(first + j, count - 1u)];
      x[j] = X + ulong(pair[j] / a.topk) * a.k;
      g[j] = 0.0f;
      u[j] = 0.0f;
    }
    for (uint b = lane; b < a.k / 16u; b += 32u) {
      float gate[16], up[16];
      ink2_nv_block(gate_w, gate_s, phi, b, gate);
      ink2_nv_block(up_w, up_s, phi, b, up);
      for (uint j = 0u; j < INK2_PASS; ++j)
        for (uint i = 0u; i < 16u; ++i) {
          const float v = float(x[j][16u * b + i]);
          g[j] += gate[i] * v;
          u[j] += up[i] * v;
        }
    }
    for (uint j = 0u; j < INK2_PASS; ++j) {
      const float gs = simd_sum(g[j]), us = simd_sum(u[j]);
      if (lane == 0u && first + j < count) Y[ulong(pair[j]) * a.n + n] = ink2_act(gs, us);
    }
  }
}

[[kernel, max_total_threads_per_threadgroup(256)]] void tf_ink2_mxfp8_gu_grouped(
    device const bfloat* X [[buffer(0)]], device const uchar* W [[buffer(1)]],
    device const uchar* S [[buffer(2)]], device const uint* counts [[buffer(3)]],
    device const uint* members [[buffer(4)]], device bfloat* Y [[buffer(5)]],
    constant Ink2GroupedArgs& a [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint n = tg.x * 8u + sg, expert = tg.y;
  if (n >= a.n) return;
  const uint count = counts[expert];
  if (count == 0u) return;
  const ulong expert_w = ulong(expert) * 2u * a.n * a.k;
  const ulong expert_s = ulong(expert) * 2u * a.n * (a.k / 32u);
  device const uchar* gate_w = W + expert_w + ulong(2u * n) * a.k;
  device const uchar* gate_s = S + expert_s + ulong(2u * n) * (a.k / 32u);
  device const uchar* up_w = gate_w + a.k;
  device const uchar* up_s = gate_s + (a.k / 32u);
  for (uint first = 0u; first < count; first += INK2_PASS) {
    uint pair[INK2_PASS];
    device const bfloat* x[INK2_PASS];
    float g[INK2_PASS], u[INK2_PASS];
    for (uint j = 0u; j < INK2_PASS; ++j) {
      pair[j] = members[expert * a.cap + min(first + j, count - 1u)];
      x[j] = X + ulong(pair[j] / a.topk) * a.k;
      g[j] = 0.0f;
      u[j] = 0.0f;
    }
    for (uint b = lane; b < a.k / 32u; b += 32u) {
      float gate[32], up[32];
      ink2_mx_block(gate_w, gate_s, b, gate);
      ink2_mx_block(up_w, up_s, b, up);
      for (uint j = 0u; j < INK2_PASS; ++j)
        for (uint i = 0u; i < 32u; ++i) {
          const float v = float(x[j][32u * b + i]);
          g[j] += gate[i] * v;
          u[j] += up[i] * v;
        }
    }
    for (uint j = 0u; j < INK2_PASS; ++j) {
      const float gs = simd_sum(g[j]), us = simd_sum(u[j]);
      if (lane == 0u && first + j < count) Y[ulong(pair[j]) * a.n + n] = ink2_act(gs, us);
    }
  }
}

[[kernel, max_total_threads_per_threadgroup(256)]] void tf_ink2_nvfp4_down_grouped(
    device const bfloat* X [[buffer(0)]], device const uchar* W [[buffer(1)]],
    device const uchar* S [[buffer(2)]], device const float* amax [[buffer(3)]],
    device const float* denominator [[buffer(4)]], device const uint* counts [[buffer(5)]],
    device const uint* members [[buffer(6)]], device bfloat* Y [[buffer(7)]],
    constant Ink2GroupedArgs& a [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint n = tg.x * 8u + sg, expert = tg.y;
  if (n >= a.n) return;
  const uint count = counts[expert];
  if (count == 0u) return;
  const float phi = precise::divide(float(bfloat(amax[expert])), float(bfloat(denominator[0])));
  device const uchar* row_w = W + ulong(expert) * a.n * (a.k / 2u) + ulong(n) * (a.k / 2u);
  device const uchar* row_s = S + ulong(expert) * a.n * (a.k / 16u) + ulong(n) * (a.k / 16u);
  for (uint first = 0u; first < count; first += INK2_PASS) {
    uint pair[INK2_PASS];
    device const bfloat* x[INK2_PASS];
    float d[INK2_PASS];
    for (uint j = 0u; j < INK2_PASS; ++j) {
      pair[j] = members[expert * a.cap + min(first + j, count - 1u)];
      x[j] = X + ulong(pair[j]) * a.k;
      d[j] = 0.0f;
    }
    for (uint b = lane; b < a.k / 16u; b += 32u) {
      float weight[16];
      ink2_nv_block(row_w, row_s, phi, b, weight);
      for (uint j = 0u; j < INK2_PASS; ++j)
        for (uint i = 0u; i < 16u; ++i) d[j] += weight[i] * float(x[j][16u * b + i]);
    }
    for (uint j = 0u; j < INK2_PASS; ++j) {
      const float ds = simd_sum(d[j]);
      if (lane == 0u && first + j < count) Y[ulong(pair[j]) * a.n + n] = bfloat(ds);
    }
  }
}

[[kernel, max_total_threads_per_threadgroup(256)]] void tf_ink2_mxfp8_down_grouped(
    device const bfloat* X [[buffer(0)]], device const uchar* W [[buffer(1)]],
    device const uchar* S [[buffer(2)]], device const uint* counts [[buffer(3)]],
    device const uint* members [[buffer(4)]], device bfloat* Y [[buffer(5)]],
    constant Ink2GroupedArgs& a [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint n = tg.x * 8u + sg, expert = tg.y;
  if (n >= a.n) return;
  const uint count = counts[expert];
  if (count == 0u) return;
  device const uchar* row_w = W + ulong(expert) * a.n * a.k + ulong(n) * a.k;
  device const uchar* row_s = S + ulong(expert) * a.n * (a.k / 32u) + ulong(n) * (a.k / 32u);
  for (uint first = 0u; first < count; first += INK2_PASS) {
    uint pair[INK2_PASS];
    device const bfloat* x[INK2_PASS];
    float d[INK2_PASS];
    for (uint j = 0u; j < INK2_PASS; ++j) {
      pair[j] = members[expert * a.cap + min(first + j, count - 1u)];
      x[j] = X + ulong(pair[j]) * a.k;
      d[j] = 0.0f;
    }
    for (uint b = lane; b < a.k / 32u; b += 32u) {
      float weight[32];
      ink2_mx_block(row_w, row_s, b, weight);
      for (uint j = 0u; j < INK2_PASS; ++j)
        for (uint i = 0u; i < 32u; ++i) d[j] += weight[i] * float(x[j][32u * b + i]);
    }
    for (uint j = 0u; j < INK2_PASS; ++j) {
      const float ds = simd_sum(d[j]);
      if (lane == 0u && first + j < count) Y[ulong(pair[j]) * a.n + n] = bfloat(ds);
    }
  }
}

// Weight decode helpers unpack checkpoint bytes and never quantize an activation.
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_ink2_nvfp4_quantize(
    device const uchar* W [[buffer(0)]], device const uchar* S [[buffer(1)]],
    device const float* amax [[buffer(2)]], device const float* denominator [[buffer(3)]],
    device bfloat* Y [[buffer(4)]], constant Ink2DecodeArgs& a [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
  const uint at = group * 32u + lane;
  if (at >= a.rows * a.k) return;
  const uint row = at / a.k, col = at % a.k;
  const float phi = ink2_phi(amax, denominator, 0);
  Y[at] = bfloat(ink2_nv_weight(W[row * (a.k / 2u) + col / 2u], col,
                                S[row * (a.k / 16u) + col / 16u], phi));
}

[[kernel, max_total_threads_per_threadgroup(32)]] void tf_ink2_mxfp8_quantize(
    device const uchar* W [[buffer(0)]], device const uchar* S [[buffer(1)]],
    device bfloat* Y [[buffer(2)]], constant Ink2DecodeArgs& a [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
  const uint at = group * 32u + lane;
  if (at >= a.rows * a.k) return;
  const uint row = at / a.k, col = at % a.k;
  Y[at] = bfloat(ink2_mx_weight(W[at], S[row * (a.k / 32u) + col / 32u]));
}

// Threadgroup (x, y) is outputs 256x to 256x + 255 of token y; route slots are sorted once per token.
[[kernel, max_total_threads_per_threadgroup(256)]] void tf_ink2_combine(
    device const bfloat* down [[buffer(0)]], device const int* ids [[buffer(1)]],
    device const float* weights [[buffer(2)]], device bfloat* Y [[buffer(3)]],
    constant Ink2MatrixArgs& a [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
    uint t [[thread_index_in_threadgroup]]) {
  const uint n = tg.x * 256u + t, group = tg.y;
  const uint base = group * a.topk;
  // Outer expert order, not route-slot order, fixes the reference's accumulation order.
  threadgroup uint order[32];
  threadgroup float w_ord[32];
  if (a.topk <= 32u) {
    if (t < a.topk) {
      const int id = ids[base + t];
      uint rank = 0u;
      for (uint o = 0u; o < a.topk; ++o) rank += (ids[base + o] < id || (ids[base + o] == id && o < t)) ? 1u : 0u;
      const uint dst = rank;
      order[dst] = uint(id) < a.experts ? t : 0xffffffffu;
      w_ord[dst] = weights[base + t];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (n >= a.n) return;
    float sum = 0.0f;
    for (uint r = 0u; r < a.topk; ++r) {
      const uint slot = order[r];
      if (slot != 0xffffffffu) sum += w_ord[r] * float(down[ulong(base + slot) * a.n + n]);
    }
    Y[ulong(group) * a.n + n] = bfloat(sum);
    return;
  }
  if (n >= a.n) return;
  float sum = 0.0f;
  for (uint r = 0u; r < a.topk; ++r)
    for (uint slot = 0u; slot < a.topk; ++slot) {
      const int id = ids[base + slot];
      uint rank = 0u;
      for (uint o = 0u; o < a.topk; ++o) rank += (ids[base + o] < id || (ids[base + o] == id && o < slot)) ? 1u : 0u;
      if (rank == r && uint(id) < a.experts) sum += weights[base + slot] * float(down[ulong(base + slot) * a.n + n]);
    }
  Y[ulong(group) * a.n + n] = bfloat(sum);
}
