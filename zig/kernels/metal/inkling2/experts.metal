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

// Exact IEEE-754 bit patterns of all 256 FP8 E4M3 codes (+0, -0, subnormals, normals, NaN at 0x7f/0xff).
constant uint ink2_e4m3_lut[256] = {
  0x00000000u, 0x3b000000u, 0x3b800000u, 0x3bc00000u, 0x3c000000u, 0x3c200000u, 0x3c400000u, 0x3c600000u,
  0x3c800000u, 0x3c900000u, 0x3ca00000u, 0x3cb00000u, 0x3cc00000u, 0x3cd00000u, 0x3ce00000u, 0x3cf00000u,
  0x3d000000u, 0x3d100000u, 0x3d200000u, 0x3d300000u, 0x3d400000u, 0x3d500000u, 0x3d600000u, 0x3d700000u,
  0x3d800000u, 0x3d900000u, 0x3da00000u, 0x3db00000u, 0x3dc00000u, 0x3dd00000u, 0x3de00000u, 0x3df00000u,
  0x3e000000u, 0x3e100000u, 0x3e200000u, 0x3e300000u, 0x3e400000u, 0x3e500000u, 0x3e600000u, 0x3e700000u,
  0x3e800000u, 0x3e900000u, 0x3ea00000u, 0x3eb00000u, 0x3ec00000u, 0x3ed00000u, 0x3ee00000u, 0x3ef00000u,
  0x3f000000u, 0x3f100000u, 0x3f200000u, 0x3f300000u, 0x3f400000u, 0x3f500000u, 0x3f600000u, 0x3f700000u,
  0x3f800000u, 0x3f900000u, 0x3fa00000u, 0x3fb00000u, 0x3fc00000u, 0x3fd00000u, 0x3fe00000u, 0x3ff00000u,
  0x40000000u, 0x40100000u, 0x40200000u, 0x40300000u, 0x40400000u, 0x40500000u, 0x40600000u, 0x40700000u,
  0x40800000u, 0x40900000u, 0x40a00000u, 0x40b00000u, 0x40c00000u, 0x40d00000u, 0x40e00000u, 0x40f00000u,
  0x41000000u, 0x41100000u, 0x41200000u, 0x41300000u, 0x41400000u, 0x41500000u, 0x41600000u, 0x41700000u,
  0x41800000u, 0x41900000u, 0x41a00000u, 0x41b00000u, 0x41c00000u, 0x41d00000u, 0x41e00000u, 0x41f00000u,
  0x42000000u, 0x42100000u, 0x42200000u, 0x42300000u, 0x42400000u, 0x42500000u, 0x42600000u, 0x42700000u,
  0x42800000u, 0x42900000u, 0x42a00000u, 0x42b00000u, 0x42c00000u, 0x42d00000u, 0x42e00000u, 0x42f00000u,
  0x43000000u, 0x43100000u, 0x43200000u, 0x43300000u, 0x43400000u, 0x43500000u, 0x43600000u, 0x43700000u,
  0x43800000u, 0x43900000u, 0x43a00000u, 0x43b00000u, 0x43c00000u, 0x43d00000u, 0x43e00000u, 0x7fc00000u,
  0x80000000u, 0xbb000000u, 0xbb800000u, 0xbbc00000u, 0xbc000000u, 0xbc200000u, 0xbc400000u, 0xbc600000u,
  0xbc800000u, 0xbc900000u, 0xbca00000u, 0xbcb00000u, 0xbcc00000u, 0xbcd00000u, 0xbce00000u, 0xbcf00000u,
  0xbd000000u, 0xbd100000u, 0xbd200000u, 0xbd300000u, 0xbd400000u, 0xbd500000u, 0xbd600000u, 0xbd700000u,
  0xbd800000u, 0xbd900000u, 0xbda00000u, 0xbdb00000u, 0xbdc00000u, 0xbdd00000u, 0xbde00000u, 0xbdf00000u,
  0xbe000000u, 0xbe100000u, 0xbe200000u, 0xbe300000u, 0xbe400000u, 0xbe500000u, 0xbe600000u, 0xbe700000u,
  0xbe800000u, 0xbe900000u, 0xbea00000u, 0xbeb00000u, 0xbec00000u, 0xbed00000u, 0xbee00000u, 0xbef00000u,
  0xbf000000u, 0xbf100000u, 0xbf200000u, 0xbf300000u, 0xbf400000u, 0xbf500000u, 0xbf600000u, 0xbf700000u,
  0xbf800000u, 0xbf900000u, 0xbfa00000u, 0xbfb00000u, 0xbfc00000u, 0xbfd00000u, 0xbfe00000u, 0xbff00000u,
  0xc0000000u, 0xc0100000u, 0xc0200000u, 0xc0300000u, 0xc0400000u, 0xc0500000u, 0xc0600000u, 0xc0700000u,
  0xc0800000u, 0xc0900000u, 0xc0a00000u, 0xc0b00000u, 0xc0c00000u, 0xc0d00000u, 0xc0e00000u, 0xc0f00000u,
  0xc1000000u, 0xc1100000u, 0xc1200000u, 0xc1300000u, 0xc1400000u, 0xc1500000u, 0xc1600000u, 0xc1700000u,
  0xc1800000u, 0xc1900000u, 0xc1a00000u, 0xc1b00000u, 0xc1c00000u, 0xc1d00000u, 0xc1e00000u, 0xc1f00000u,
  0xc2000000u, 0xc2100000u, 0xc2200000u, 0xc2300000u, 0xc2400000u, 0xc2500000u, 0xc2600000u, 0xc2700000u,
  0xc2800000u, 0xc2900000u, 0xc2a00000u, 0xc2b00000u, 0xc2c00000u, 0xc2d00000u, 0xc2e00000u, 0xc2f00000u,
  0xc3000000u, 0xc3100000u, 0xc3200000u, 0xc3300000u, 0xc3400000u, 0xc3500000u, 0xc3600000u, 0xc3700000u,
  0xc3800000u, 0xc3900000u, 0xc3a00000u, 0xc3b00000u, 0xc3c00000u, 0xc3d00000u, 0xc3e00000u, 0x7fc00000u
};

inline float ink2_e4m3_bits(uint code) {
  return as_type<float>(ink2_e4m3_lut[code & 255u]);
}

inline float ink2_e4m3(uint code) {
  return ink2_e4m3_bits(code);
}

// Exact BF16 edge decode for UE8M0 scales < 10 (avoiding FP32 subnormal FTZ) or == 255.
inline bfloat ink2_mx_edge_bf(uint code, uint scale) {
  const float x = ink2_e4m3_bits(code);
  if (scale == 255u) return bfloat((x * 2.0f) * as_type<float>(254u << 23u));
  const uint u = as_type<uint>(x * as_type<float>((scale + 16u) << 23u));
  if ((u & 0x7fffffffu) == 0u || (u & 0x7f800000u) == 0x7f800000u) return as_type<bfloat>(ushort(u >> 16u));
  const uint sign16 = (u >> 16u) & 0x8000u;
  const int e = int((u >> 23u) & 0xffu) - 16;
  if (e >= 1) {
    const uint un = (u & 0x807fffffu) | (uint(e) << 23u);
    return as_type<bfloat>(ushort((un + 0x7fffu + ((un >> 16u) & 1u)) >> 16u));
  }
  const uint shift = uint(17 - e);
  if (shift > 25u) return as_type<bfloat>(ushort(sign16));
  const uint sig = 0x800000u | (u & 0x7fffffu);
  const uint half_bit = 1u << (shift - 1u);
  const uint rem = sig & ((1u << shift) - 1u);
  uint q = sig >> shift;
  if (rem > half_bit || (rem == half_bit && (q & 1u) != 0u)) q += 1u;
  return as_type<bfloat>(ushort(sign16 | q));
}

// Shiftless UE8M0 + E4M3 ALU decode to BF16, exact across all 65,536 (e4m3, ue8m0) pairs.
inline bfloat ink2_mx_bf(uint code, uint scale) {
  if (scale - 10u < 245u) return bfloat(ink2_e4m3_bits(code) * as_type<float>(scale << 23u));
  return ink2_mx_edge_bf(code, scale);
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

inline uint ink2_pack_item(uint expert, uint offset, uint len) {
  return (expert & 0xffffu) | ((offset & 0xfffu) << 16u) | (((len - 1u) & 0xfu) << 28u);
}

inline float ink2_mx_weight(uchar value, uchar scale) {
  return float(ink2_mx_bf(uint(value), uint(scale)));
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
  Y[at] = ink2_mx_bf(W[at], S[row * (a.k / 32u) + col / 32u]);
}

[[kernel, max_total_threads_per_threadgroup(128)]] void tf_ink2_combine(
    device const bfloat* down [[buffer(0)]], device const int* ids [[buffer(1)]],
    device const float* weights [[buffer(2)]], device bfloat* Y [[buffer(3)]],
    constant Ink2MatrixArgs& a [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  // One thread an output: threadgroup x is 128 outputs, y the token; slots sort once per group.
  threadgroup uint tg_slots[8];
  threadgroup float tg_weights[8];
  threadgroup uint tg_used;
  const uint group = tg.y;
  if (tid < 32u) {
    const bool valid = tid < a.topk;
    const uint my_id = valid ? uint(ids[group * a.topk + tid]) : a.experts;
    const bool ok = valid && (my_id < a.experts);
    const float my_w = ok ? weights[group * a.topk + tid] : 0.0f;
    const uint prev_id = simd_shuffle_up(my_id, 1u);
    const bool needs_sort = simd_any(valid && (!ok || (tid > 0u && my_id < prev_id)));
    if (!needs_sort) {
      if (valid) { tg_slots[tid] = tid; tg_weights[tid] = my_w; }
      if (tid == 0u) tg_used = a.topk;
    } else {
      const uint key = ok ? my_id : a.experts;
      uint rank = 0u;
      for (uint s = 0u; s < a.topk; ++s) {
        const uint other = simd_shuffle(key, s);
        rank += (other < my_id || (other == my_id && s < tid)) ? 1u : 0u;
      }
      if (ok) { tg_slots[rank] = tid; tg_weights[rank] = my_w; }
      const uint used = popcount(uint(ulong(simd_vote::vote_t(simd_ballot(ok)))));
      if (tid == 0u) tg_used = used;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint n = tg.x * 128u + tid;
  if (n >= a.n) return;
  const uint used = tg_used;
  float sum = 0.0f;
  for (uint i = 0u; i < used; ++i)
    sum += tg_weights[i] * float(down[ulong(group * a.topk + tg_slots[i]) * a.n + n]);
  Y[ulong(group) * a.n + n] = bfloat(sum);
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
