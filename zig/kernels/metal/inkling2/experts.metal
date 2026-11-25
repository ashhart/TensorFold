// Inkling 2 Mini M3: packed weight decode in registers, BF16 activations, FP32 reductions.
#include <metal_stdlib>
using namespace metal;
#pragma METAL fp contract(off)

struct Ink2MatrixArgs { uint k, n, topk, experts; };
struct Ink2DecodeArgs { uint rows, k; };
struct Ink2GroupArgs { uint rows, topk, experts, tile_m; };
struct Ink2GroupedMatrixArgs { uint rows, k, n, topk, experts; };

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

inline float ink2_nv_weight_scaled(uint code, float scale_f, float phi) {
  return float(bfloat((ink2_e2m1(code) * scale_f) * phi));
}

inline float ink2_e4m3_fast(uint code) {
  const uint exp = (code >> 3u) & 15u, mant = code & 7u;
  if (exp == 15u && mant == 7u) return NAN;
  const uint bits = exp ? (((exp + 120u) << 23u) | (mant << 20u)) : as_type<uint>(float(mant) * (1.0f / 512.0f));
  return as_type<float>(bits | ((code & 0x80u) << 24u));
}

inline uint ink2_pack_item(uint expert, uint offset, uint len) {
  return (expert & 0xffffu) | ((offset & 0xfffu) << 16u) | (((len - 1u) & 0xfu) << 28u);
}

inline float ink2_mx_weight(uchar value, uchar scale) {
  const float w = ldexp(ink2_e4m3(value), int(scale) - 127);
  // E4M3 times a power of two is exactly representable in BF16, still round explicitly.
  return float(bfloat(w));
}

inline float2 ink2_bf2(uint pair) {
  return float2(as_type<float>(pair << 16u), as_type<float>(pair & 0xffff0000u));
}

// A byte of two E2M1 codes as two FP16 values scaled by 2^-14, low nibble first; the byte must stand alone.
inline float2 ink2_e2m1_pair(uint code) {
  return float2(as_type<half2>(((code * 0x00200200u) & 0x0e000e00u) | ((code * 0x01001000u) & 0x80008000u)));
}

// ink2_e4m3(code) * 2^14 from the FP16 bits of the same layout; 0x7f and 0xff stay NaN.
inline float ink2_e4m3_scale(uint code) {
  const float v = float(as_type<half>(ushort(((code & 0x7fu) << 7u) | ((code & 0x80u) << 8u)))) * 4194304.0f;
  return (code & 0x7fu) == 0x7fu ? as_type<float>(((code & 0x80u) << 24u) | 0x7fc00000u) : v;
}

// ink2_nv_weight twice: (e2m1 * 2^-14) * (e4m3 * 2^14) is the same exact product, then phi and one BF16 round.
inline float ink2_nv_byte(float sum, uint code, float scale, float phi, float2 x) {
  const float2 m = ink2_e2m1_pair(code);
  sum += float(bfloat((m.x * scale) * phi)) * x.x;
  return sum + float(bfloat((m.y * scale) * phi)) * x.y;
}

// Columns in order, 16 to a scale block; k is a multiple of 16 and w, x are 16-byte aligned.
inline float ink2_dot_nv(device const uchar* w, device const uchar* scales,
                         device const bfloat* x, float phi, uint k) {
  float sum = 0.0f;
  for (uint b = 0; b < k / 16u; ++b) {
    const uint2 codes = *(device const uint2*)(w + 8u * b);
    const float scale = ink2_e4m3_scale(scales[b]);
    const uint4 xa = *(device const uint4*)(x + 16u * b), xb = *(device const uint4*)(x + 16u * b + 8u);
    sum = ink2_nv_byte(sum, codes.x & 255u, scale, phi, ink2_bf2(xa.x));
    sum = ink2_nv_byte(sum, (codes.x >> 8u) & 255u, scale, phi, ink2_bf2(xa.y));
    sum = ink2_nv_byte(sum, (codes.x >> 16u) & 255u, scale, phi, ink2_bf2(xa.z));
    sum = ink2_nv_byte(sum, codes.x >> 24u, scale, phi, ink2_bf2(xa.w));
    sum = ink2_nv_byte(sum, codes.y & 255u, scale, phi, ink2_bf2(xb.x));
    sum = ink2_nv_byte(sum, (codes.y >> 8u) & 255u, scale, phi, ink2_bf2(xb.y));
    sum = ink2_nv_byte(sum, (codes.y >> 16u) & 255u, scale, phi, ink2_bf2(xb.z));
    sum = ink2_nv_byte(sum, codes.y >> 24u, scale, phi, ink2_bf2(xb.w));
  }
  return sum;
}

// Four E4M3 bytes, columns 4i to 4i + 3, added in order; the FP16 route gives e4m3 / 256, 2^(scale - 119) restores it.
inline float ink2_mx_word(float sum, uint word, float factor, float2 x01, float2 x23) {
  const uint even = word & 0x00ff00ffu, odd = (word >> 8u) & 0x00ff00ffu;
  const float2 w02 = float2(as_type<half2>(((even << 7u) & 0x3f803f80u) | ((even << 8u) & 0x80008000u)));
  const float2 w13 = float2(as_type<half2>(((odd << 7u) & 0x3f803f80u) | ((odd << 8u) & 0x80008000u)));
  sum += (w02.x * factor) * x01.x;
  sum += (w13.x * factor) * x01.y;
  sum += (w02.y * factor) * x23.x;
  return sum + (w13.y * factor) * x23.y;
}

// A block of 32 columns: scales 24 to 230 with no NaN code take the word route, the rest decode as ink2_mx_weight.
inline float ink2_mx_block(float sum, device const uchar* w, uint scale, device const bfloat* x) {
  const uint4 q0 = *(device const uint4*)w, q1 = *(device const uint4*)(w + 16);
  uint nan = 0u;
  for (uint i = 0; i < 4u; ++i) nan |= ((q0[i] & 0x7f7f7f7fu) + 0x01010101u) | ((q1[i] & 0x7f7f7f7fu) + 0x01010101u);
  if (scale - 24u > 206u || (nan & 0x80808080u) != 0u) {
    for (uint j = 0; j < 32u; ++j) sum += ink2_mx_weight(w[j], uchar(scale)) * float(x[j]);
    return sum;
  }
  const float factor = as_type<float>((scale + 8u) << 23u);
  const uint4 xa = *(device const uint4*)x, xb = *(device const uint4*)(x + 8), xc = *(device const uint4*)(x + 16);
  const uint4 xd = *(device const uint4*)(x + 24);
  sum = ink2_mx_word(sum, q0.x, factor, ink2_bf2(xa.x), ink2_bf2(xa.y));
  sum = ink2_mx_word(sum, q0.y, factor, ink2_bf2(xa.z), ink2_bf2(xa.w));
  sum = ink2_mx_word(sum, q0.z, factor, ink2_bf2(xb.x), ink2_bf2(xb.y));
  sum = ink2_mx_word(sum, q0.w, factor, ink2_bf2(xb.z), ink2_bf2(xb.w));
  sum = ink2_mx_word(sum, q1.x, factor, ink2_bf2(xc.x), ink2_bf2(xc.y));
  sum = ink2_mx_word(sum, q1.y, factor, ink2_bf2(xc.z), ink2_bf2(xc.w));
  sum = ink2_mx_word(sum, q1.z, factor, ink2_bf2(xd.x), ink2_bf2(xd.y));
  return ink2_mx_word(sum, q1.w, factor, ink2_bf2(xd.z), ink2_bf2(xd.w));
}

inline float ink2_dot_mx(device const uchar* w, device const uchar* scales, device const bfloat* x, uint k) {
  float sum = 0.0f;
  for (uint b = 0; b < k / 32u; ++b) sum = ink2_mx_block(sum, w + 32u * b, scales[b], x + 32u * b);
  return sum;
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

[[kernel, max_total_threads_per_threadgroup(32)]] void tf_ink2_nvfp4_gu(
    device const bfloat* X [[buffer(0)]], device const uchar* W [[buffer(1)]],
    device const uchar* S [[buffer(2)]], device const float* amax [[buffer(3)]],
    device const float* denominator [[buffer(4)]], device const int* ids [[buffer(5)]],
    device bfloat* Y [[buffer(6)]], constant Ink2MatrixArgs& a [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
  // One thread an output: threadgroup x is 32 outputs, y the route pair; each dot is the serial chain above.
  const uint n = tg.x * 32u + lane, group = tg.y;
  if (n >= a.n) return;
  const uint token = group / a.topk, expert = uint(ids[group]);
  const ulong expert_w = ulong(expert) * 2u * a.n * (a.k / 2u);
  const ulong expert_s = ulong(expert) * 2u * a.n * (a.k / 16u);
  const float phi = ink2_phi(amax, denominator, expert);
  const uint gr = 2u * n, ur = gr + 1u;
  const float g = ink2_dot_nv(W + expert_w + ulong(gr) * (a.k / 2u),
                              S + expert_s + ulong(gr) * (a.k / 16u), X + ulong(token) * a.k, phi, a.k);
  const float u = ink2_dot_nv(W + expert_w + ulong(ur) * (a.k / 2u),
                              S + expert_s + ulong(ur) * (a.k / 16u), X + ulong(token) * a.k, phi, a.k);
  Y[group * a.n + n] = ink2_act(g, u);
}

[[kernel, max_total_threads_per_threadgroup(32)]] void tf_ink2_mxfp8_gu(
    device const bfloat* X [[buffer(0)]], device const uchar* W [[buffer(1)]],
    device const uchar* S [[buffer(2)]], device const int* ids [[buffer(3)]],
    device bfloat* Y [[buffer(4)]], constant Ink2MatrixArgs& a [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
  // One thread an output: threadgroup x is 32 outputs, y the route pair; each dot is the serial chain above.
  const uint n = tg.x * 32u + lane, group = tg.y;
  if (n >= a.n) return;
  const uint token = group / a.topk, expert = uint(ids[group]);
  const ulong expert_w = ulong(expert) * 2u * a.n * a.k;
  const ulong expert_s = ulong(expert) * 2u * a.n * (a.k / 32u);
  const uint gr = 2u * n, ur = gr + 1u;
  const float g = ink2_dot_mx(W + expert_w + ulong(gr) * a.k,
                              S + expert_s + ulong(gr) * (a.k / 32u), X + ulong(token) * a.k, a.k);
  const float u = ink2_dot_mx(W + expert_w + ulong(ur) * a.k,
                              S + expert_s + ulong(ur) * (a.k / 32u), X + ulong(token) * a.k, a.k);
  Y[group * a.n + n] = ink2_act(g, u);
}

[[kernel, max_total_threads_per_threadgroup(32)]] void tf_ink2_nvfp4_down(
    device const bfloat* X [[buffer(0)]], device const uchar* W [[buffer(1)]],
    device const uchar* S [[buffer(2)]], device const float* amax [[buffer(3)]],
    device const float* denominator [[buffer(4)]], device const int* ids [[buffer(5)]],
    device bfloat* Y [[buffer(6)]], constant Ink2MatrixArgs& a [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
  // One thread an output: threadgroup x is 32 outputs, y the route pair; each dot is the serial chain above.
  const uint n = tg.x * 32u + lane, group = tg.y;
  if (n >= a.n) return;
  const uint expert = uint(ids[group]);
  const ulong expert_w = ulong(expert) * a.n * (a.k / 2u);
  const ulong expert_s = ulong(expert) * a.n * (a.k / 16u);
  const float phi = ink2_phi(amax, denominator, expert);
  Y[group * a.n + n] = bfloat(ink2_dot_nv(W + expert_w + ulong(n) * (a.k / 2u),
      S + expert_s + ulong(n) * (a.k / 16u), X + ulong(group) * a.k, phi, a.k));
}

[[kernel, max_total_threads_per_threadgroup(32)]] void tf_ink2_mxfp8_down(
    device const bfloat* X [[buffer(0)]], device const uchar* W [[buffer(1)]],
    device const uchar* S [[buffer(2)]], device const int* ids [[buffer(3)]],
    device bfloat* Y [[buffer(4)]], constant Ink2MatrixArgs& a [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
  // One thread an output: threadgroup x is 32 outputs, y the route pair; each dot is the serial chain above.
  const uint n = tg.x * 32u + lane, group = tg.y;
  if (n >= a.n) return;
  const uint expert = uint(ids[group]);
  const ulong expert_w = ulong(expert) * a.n * a.k;
  const ulong expert_s = ulong(expert) * a.n * (a.k / 32u);
  Y[group * a.n + n] = bfloat(ink2_dot_mx(W + expert_w + ulong(n) * a.k,
      S + expert_s + ulong(n) * (a.k / 32u), X + ulong(group) * a.k, a.k));
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

[[kernel, max_total_threads_per_threadgroup(32)]] void tf_ink2_combine(
    device const bfloat* down [[buffer(0)]], device const int* ids [[buffer(1)]],
    device const float* weights [[buffer(2)]], device bfloat* Y [[buffer(3)]],
    constant Ink2MatrixArgs& a [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
  // One thread an output: threadgroup x is 32 outputs, y the token.
  const uint n = tg.x * 32u + lane, group = tg.y;
  if (n >= a.n) return;
  // Outer expert order, not route-slot order, fixes the reference's accumulation order; equal ids keep slot order.
  uint slots[8], used = 0u;
  for (uint slot = 0; slot < a.topk; ++slot) {
    const uint id = uint(ids[group * a.topk + slot]);
    if (id >= a.experts) continue;
    uint at = used++;
    while (at > 0u && uint(ids[group * a.topk + slots[at - 1u]]) > id) {
      slots[at] = slots[at - 1u];
      --at;
    }
    slots[at] = slot;
  }
  float sum = 0.0f;
  for (uint i = 0; i < used; ++i)
    sum += weights[group * a.topk + slots[i]] * float(down[(group * a.topk + slots[i]) * a.n + n]);
  Y[group * a.n + n] = bfloat(sum);
}

// Bucket route pairs by ascending expert ID into <= tile_m slices, preserving pair order within each expert.
[[kernel, max_total_threads_per_threadgroup(128)]] void tf_ink2_expert_group(
    device const int* ids [[buffer(0)]], device uint* group_buf [[buffer(1)]],
    constant Ink2GroupArgs& a [[buffer(2)]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint pairs = a.rows * a.topk;
  if (pairs == 0u || pairs > 512u || a.experts == 0u || a.tile_m == 0u || a.tile_m > 16u) {
    if (tid == 0u) group_buf[0] = 0u;
    return;
  }
  threadgroup int tg_ids[512];
  threadgroup uint tg_scan[8];
  threadgroup uint base_offsets[2];
  for (uint p = tid; p < pairs; p += 128u) tg_ids[p] = ids[p];
  if (tid == 0u) { base_offsets[0] = 0u; base_offsets[1] = 0u; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device uint* work_items = group_buf + 1u;
  device uint* pair_map = group_buf + 1u + pairs;
  for (uint e_base = 0u; e_base < a.experts; e_base += 128u) {
    const uint e = e_base + tid;
    uint cnt = 0u;
    if (e < a.experts)
      for (uint p = 0u; p < pairs; ++p) cnt += (tg_ids[p] == int(e)) ? 1u : 0u;
    const uint tiles = cnt ? ((cnt + a.tile_m - 1u) / a.tile_m) : 0u;
    const uint p_ex = simd_prefix_exclusive_sum(cnt);
    const uint t_ex = simd_prefix_exclusive_sum(tiles);
    if (lane == 31u) {
      tg_scan[sg] = p_ex + cnt;
      tg_scan[4u + sg] = t_ex + tiles;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint p_sg = 0u, t_sg = 0u;
    for (uint s = 0u; s < sg; ++s) { p_sg += tg_scan[s]; t_sg += tg_scan[4u + s]; }
    const uint pair_base = base_offsets[0];
    const uint tile_base = base_offsets[1];
    const uint p_off = pair_base + p_ex + p_sg;
    const uint t_off = tile_base + t_ex + t_sg;
    if (cnt > 0u) {
      for (uint t = 0u; t < tiles; ++t) {
        const uint rem = cnt - t * a.tile_m;
        const uint len = rem < a.tile_m ? rem : a.tile_m;
        work_items[t_off + t] = ink2_pack_item(e, p_off + t * a.tile_m, len);
      }
      uint at = p_off;
      for (uint p = 0u; p < pairs; ++p) if (tg_ids[p] == int(e)) pair_map[at++] = p;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 127u) {
      base_offsets[0] = pair_base + p_ex + p_sg + cnt;
      base_offsets[1] = tile_base + t_ex + t_sg + tiles;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0u) group_buf[0] = base_offsets[1];
}

[[kernel, max_total_threads_per_threadgroup(32)]] void tf_ink2_nvfp4_gu_grouped(
    device const bfloat* X [[buffer(0)]], device const uchar* W [[buffer(1)]],
    device const uchar* S [[buffer(2)]], device const float* amax [[buffer(3)]],
    device const float* denominator [[buffer(4)]], device const uint* group_buf [[buffer(5)]],
    device bfloat* Y [[buffer(6)]], constant Ink2GroupedMatrixArgs& a [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
  const uint n = tg.x * 32u + lane;
  if (n >= a.n) return;
  const uint tile_id = tg.y;
  if (tile_id >= group_buf[0]) return;
  const uint item = group_buf[1u + tile_id];
  const uint expert = item & 0xffffu;
  const uint offset = (item >> 16u) & 0xfffu;
  const uint len = ((item >> 28u) & 0xfu) + 1u;
  const uint pairs = a.rows * a.topk;
  device const uint* pair_map = group_buf + 1u + pairs;
  uint x_off[8];
  for (uint m = 0u; m < len; ++m) x_off[m] = (pair_map[offset + m] / a.topk) * a.k;
  const ulong expert_w = ulong(expert) * 2u * a.n * (a.k / 2u);
  const ulong expert_s = ulong(expert) * 2u * a.n * (a.k / 16u);
  const float phi = ink2_phi(amax, denominator, expert);
  const uint gr = 2u * n, ur = gr + 1u;
  device const uchar* wg = W + expert_w + ulong(gr) * (a.k / 2u);
  device const uchar* sg = S + expert_s + ulong(gr) * (a.k / 16u);
  device const uchar* wu = W + expert_w + ulong(ur) * (a.k / 2u);
  device const uchar* su = S + expert_s + ulong(ur) * (a.k / 16u);
  float g_acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  float u_acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  const uint blocks = a.k / 16u;
  for (uint b = 0u; b < blocks; ++b) {
    const float sg_f = ink2_e4m3_fast(sg[b]);
    const float su_f = ink2_e4m3_fast(su[b]);
    for (uint half_blk = 0u; half_blk < 2u; ++half_blk) {
      const packed_uchar4 pg4 = *(device const packed_uchar4*)(wg + b * 8u + half_blk * 4u);
      const packed_uchar4 pu4 = *(device const packed_uchar4*)(wu + b * 8u + half_blk * 4u);
      const float wg0 = ink2_nv_weight_scaled(uint(pg4.x & 15u), sg_f, phi);
      const float wg1 = ink2_nv_weight_scaled(uint(pg4.x >> 4u), sg_f, phi);
      const float wg2 = ink2_nv_weight_scaled(uint(pg4.y & 15u), sg_f, phi);
      const float wg3 = ink2_nv_weight_scaled(uint(pg4.y >> 4u), sg_f, phi);
      const float wg4 = ink2_nv_weight_scaled(uint(pg4.z & 15u), sg_f, phi);
      const float wg5 = ink2_nv_weight_scaled(uint(pg4.z >> 4u), sg_f, phi);
      const float wg6 = ink2_nv_weight_scaled(uint(pg4.w & 15u), sg_f, phi);
      const float wg7 = ink2_nv_weight_scaled(uint(pg4.w >> 4u), sg_f, phi);
      const float wu0 = ink2_nv_weight_scaled(uint(pu4.x & 15u), su_f, phi);
      const float wu1 = ink2_nv_weight_scaled(uint(pu4.x >> 4u), su_f, phi);
      const float wu2 = ink2_nv_weight_scaled(uint(pu4.y & 15u), su_f, phi);
      const float wu3 = ink2_nv_weight_scaled(uint(pu4.y >> 4u), su_f, phi);
      const float wu4 = ink2_nv_weight_scaled(uint(pu4.z & 15u), su_f, phi);
      const float wu5 = ink2_nv_weight_scaled(uint(pu4.z >> 4u), su_f, phi);
      const float wu6 = ink2_nv_weight_scaled(uint(pu4.w & 15u), su_f, phi);
      const float wu7 = ink2_nv_weight_scaled(uint(pu4.w >> 4u), su_f, phi);
      const uint c0 = b * 16u + half_blk * 8u;
      #pragma unroll
      for (uint m = 0u; m < 8u; ++m) {
        if (m >= len) break;
        const uint xo = x_off[m] + c0;
        const float4 xv0 = float4(*(device const packed_bfloat4*)(X + xo));
        const float4 xv1 = float4(*(device const packed_bfloat4*)(X + xo + 4u));
        g_acc[m] += wg0 * xv0.x; g_acc[m] += wg1 * xv0.y; g_acc[m] += wg2 * xv0.z; g_acc[m] += wg3 * xv0.w;
        g_acc[m] += wg4 * xv1.x; g_acc[m] += wg5 * xv1.y; g_acc[m] += wg6 * xv1.z; g_acc[m] += wg7 * xv1.w;
        u_acc[m] += wu0 * xv0.x; u_acc[m] += wu1 * xv0.y; u_acc[m] += wu2 * xv0.z; u_acc[m] += wu3 * xv0.w;
        u_acc[m] += wu4 * xv1.x; u_acc[m] += wu5 * xv1.y; u_acc[m] += wu6 * xv1.z; u_acc[m] += wu7 * xv1.w;
      }
    }
  }
  for (uint m = 0u; m < len; ++m) Y[ulong(pair_map[offset + m]) * a.n + n] = ink2_act(g_acc[m], u_acc[m]);
}

[[kernel, max_total_threads_per_threadgroup(32)]] void tf_ink2_mxfp8_gu_grouped(
    device const bfloat* X [[buffer(0)]], device const uchar* W [[buffer(1)]],
    device const uchar* S [[buffer(2)]], device const uint* group_buf [[buffer(3)]],
    device bfloat* Y [[buffer(4)]], constant Ink2GroupedMatrixArgs& a [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
  const uint n = tg.x * 32u + lane;
  if (n >= a.n) return;
  const uint tile_id = tg.y;
  if (tile_id >= group_buf[0]) return;
  const uint item = group_buf[1u + tile_id];
  const uint expert = item & 0xffffu;
  const uint offset = (item >> 16u) & 0xfffu;
  const uint len = ((item >> 28u) & 0xfu) + 1u;
  const uint pairs = a.rows * a.topk;
  device const uint* pair_map = group_buf + 1u + pairs;
  uint x_off[8];
  for (uint m = 0u; m < len; ++m) x_off[m] = (pair_map[offset + m] / a.topk) * a.k;
  const ulong expert_w = ulong(expert) * 2u * a.n * a.k;
  const ulong expert_s = ulong(expert) * 2u * a.n * (a.k / 32u);
  const uint gr = 2u * n, ur = gr + 1u;
  device const uchar* wg = W + expert_w + ulong(gr) * a.k;
  device const uchar* sg = S + expert_s + ulong(gr) * (a.k / 32u);
  device const uchar* wu = W + expert_w + ulong(ur) * a.k;
  device const uchar* su = S + expert_s + ulong(ur) * (a.k / 32u);
  float g_acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  float u_acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  const uint blocks = a.k / 32u;
  for (uint b = 0u; b < blocks; ++b) {
    const int exp_g = int(sg[b]) - 127;
    const int exp_u = int(su[b]) - 127;
    const uint cb = b * 32u;
    for (uint j = 0u; j < 32u; j += 4u) {
      const uint c = cb + j;
      const packed_uchar4 wg4 = *(device const packed_uchar4*)(wg + c);
      const packed_uchar4 wu4 = *(device const packed_uchar4*)(wu + c);
      const float wg0 = float(bfloat(ldexp(ink2_e4m3_fast(wg4.x), exp_g)));
      const float wg1 = float(bfloat(ldexp(ink2_e4m3_fast(wg4.y), exp_g)));
      const float wg2 = float(bfloat(ldexp(ink2_e4m3_fast(wg4.z), exp_g)));
      const float wg3 = float(bfloat(ldexp(ink2_e4m3_fast(wg4.w), exp_g)));
      const float wu0 = float(bfloat(ldexp(ink2_e4m3_fast(wu4.x), exp_u)));
      const float wu1 = float(bfloat(ldexp(ink2_e4m3_fast(wu4.y), exp_u)));
      const float wu2 = float(bfloat(ldexp(ink2_e4m3_fast(wu4.z), exp_u)));
      const float wu3 = float(bfloat(ldexp(ink2_e4m3_fast(wu4.w), exp_u)));
      #pragma unroll
      for (uint m = 0u; m < 8u; ++m) {
        if (m >= len) break;
        const float4 xv = float4(*(device const packed_bfloat4*)(X + x_off[m] + c));
        g_acc[m] += wg0 * xv.x; g_acc[m] += wg1 * xv.y; g_acc[m] += wg2 * xv.z; g_acc[m] += wg3 * xv.w;
        u_acc[m] += wu0 * xv.x; u_acc[m] += wu1 * xv.y; u_acc[m] += wu2 * xv.z; u_acc[m] += wu3 * xv.w;
      }
    }
  }
  for (uint m = 0u; m < len; ++m) Y[ulong(pair_map[offset + m]) * a.n + n] = ink2_act(g_acc[m], u_acc[m]);
}

[[kernel, max_total_threads_per_threadgroup(32)]] void tf_ink2_nvfp4_down_grouped(
    device const bfloat* X [[buffer(0)]], device const uchar* W [[buffer(1)]],
    device const uchar* S [[buffer(2)]], device const float* amax [[buffer(3)]],
    device const float* denominator [[buffer(4)]], device const uint* group_buf [[buffer(5)]],
    device bfloat* Y [[buffer(6)]], constant Ink2GroupedMatrixArgs& a [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
  const uint n = tg.x * 32u + lane;
  if (n >= a.n) return;
  const uint tile_id = tg.y;
  if (tile_id >= group_buf[0]) return;
  const uint item = group_buf[1u + tile_id];
  const uint expert = item & 0xffffu;
  const uint offset = (item >> 16u) & 0xfffu;
  const uint len = ((item >> 28u) & 0xfu) + 1u;
  const uint pairs = a.rows * a.topk;
  device const uint* pair_map = group_buf + 1u + pairs;
  uint x_off[8];
  for (uint m = 0u; m < len; ++m) x_off[m] = pair_map[offset + m] * a.k;
  const ulong expert_w = ulong(expert) * a.n * (a.k / 2u);
  const ulong expert_s = ulong(expert) * a.n * (a.k / 16u);
  const float phi = ink2_phi(amax, denominator, expert);
  device const uchar* wr = W + expert_w + ulong(n) * (a.k / 2u);
  device const uchar* sr = S + expert_s + ulong(n) * (a.k / 16u);
  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  const uint blocks = a.k / 16u;
  for (uint b = 0u; b < blocks; ++b) {
    const float s_f = ink2_e4m3_fast(sr[b]);
    for (uint half_blk = 0u; half_blk < 2u; ++half_blk) {
      const packed_uchar4 p4 = *(device const packed_uchar4*)(wr + b * 8u + half_blk * 4u);
      const float w0 = ink2_nv_weight_scaled(uint(p4.x & 15u), s_f, phi);
      const float w1 = ink2_nv_weight_scaled(uint(p4.x >> 4u), s_f, phi);
      const float w2 = ink2_nv_weight_scaled(uint(p4.y & 15u), s_f, phi);
      const float w3 = ink2_nv_weight_scaled(uint(p4.y >> 4u), s_f, phi);
      const float w4 = ink2_nv_weight_scaled(uint(p4.z & 15u), s_f, phi);
      const float w5 = ink2_nv_weight_scaled(uint(p4.z >> 4u), s_f, phi);
      const float w6 = ink2_nv_weight_scaled(uint(p4.w & 15u), s_f, phi);
      const float w7 = ink2_nv_weight_scaled(uint(p4.w >> 4u), s_f, phi);
      const uint c0 = b * 16u + half_blk * 8u;
      #pragma unroll
      for (uint m = 0u; m < 8u; ++m) {
        if (m >= len) break;
        const uint xo = x_off[m] + c0;
        const float4 xv0 = float4(*(device const packed_bfloat4*)(X + xo));
        const float4 xv1 = float4(*(device const packed_bfloat4*)(X + xo + 4u));
        acc[m] += w0 * xv0.x; acc[m] += w1 * xv0.y; acc[m] += w2 * xv0.z; acc[m] += w3 * xv0.w;
        acc[m] += w4 * xv1.x; acc[m] += w5 * xv1.y; acc[m] += w6 * xv1.z; acc[m] += w7 * xv1.w;
      }
    }
  }
  for (uint m = 0u; m < len; ++m) Y[ulong(pair_map[offset + m]) * a.n + n] = bfloat(acc[m]);
}

[[kernel, max_total_threads_per_threadgroup(32)]] void tf_ink2_mxfp8_down_grouped(
    device const bfloat* X [[buffer(0)]], device const uchar* W [[buffer(1)]],
    device const uchar* S [[buffer(2)]], device const uint* group_buf [[buffer(3)]],
    device bfloat* Y [[buffer(4)]], constant Ink2GroupedMatrixArgs& a [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
  const uint n = tg.x * 32u + lane;
  if (n >= a.n) return;
  const uint tile_id = tg.y;
  if (tile_id >= group_buf[0]) return;
  const uint item = group_buf[1u + tile_id];
  const uint expert = item & 0xffffu;
  const uint offset = (item >> 16u) & 0xfffu;
  const uint len = ((item >> 28u) & 0xfu) + 1u;
  const uint pairs = a.rows * a.topk;
  device const uint* pair_map = group_buf + 1u + pairs;
  uint x_off[8];
  for (uint m = 0u; m < len; ++m) x_off[m] = pair_map[offset + m] * a.k;
  const ulong expert_w = ulong(expert) * a.n * a.k;
  const ulong expert_s = ulong(expert) * a.n * (a.k / 32u);
  device const uchar* wr = W + expert_w + ulong(n) * a.k;
  device const uchar* sr = S + expert_s + ulong(n) * (a.k / 32u);
  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  const uint blocks = a.k / 32u;
  for (uint b = 0u; b < blocks; ++b) {
    const int exp_w = int(sr[b]) - 127;
    const uint cb = b * 32u;
    for (uint j = 0u; j < 32u; j += 4u) {
      const uint c = cb + j;
      const packed_uchar4 w4 = *(device const packed_uchar4*)(wr + c);
      const float w0 = float(bfloat(ldexp(ink2_e4m3_fast(w4.x), exp_w)));
      const float w1 = float(bfloat(ldexp(ink2_e4m3_fast(w4.y), exp_w)));
      const float w2 = float(bfloat(ldexp(ink2_e4m3_fast(w4.z), exp_w)));
      const float w3 = float(bfloat(ldexp(ink2_e4m3_fast(w4.w), exp_w)));
      #pragma unroll
      for (uint m = 0u; m < 8u; ++m) {
        if (m >= len) break;
        const float4 xv = float4(*(device const packed_bfloat4*)(X + x_off[m] + c));
        acc[m] += w0 * xv.x; acc[m] += w1 * xv.y; acc[m] += w2 * xv.z; acc[m] += w3 * xv.w;
      }
    }
  }
  for (uint m = 0u; m < len; ++m) Y[ulong(pair_map[offset + m]) * a.n + n] = bfloat(acc[m]);
}
