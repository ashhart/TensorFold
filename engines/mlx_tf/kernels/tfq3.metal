// TensorFold TFQ3 (v3: 3-, 4-, and 5-bit 2D-AffineCoset/half4 + 2D FWHT-128)
// Universal Apple Silicon (M1-M5, Metal 3.1+) Inference Kernels.
//
// Compatible byte-for-byte with cross-platform TFQ3 checkpoints (`pw_v3_tables.bin` + `.tffpw` TFPN/TFPW).
//   - tfq3_fwht_x                  : X_rot = H_128 @ (X * D_R)  (bfloat -> half)
//   - tfq3_fwht_y                  : OUT = D_L @ H_128 @ sum_kp(YR[kp]) (+ optional residual add)
//   - tfq3_dec_fused{1,2,4}_{3,4,5}b : Single-dispatch fused decode + in-TG inverse FWHT-128 (+ add_res)
//   - tfq3_dec_2dsk8_{1,2,4}_{3,4,5}b: Zero-SRAM 2D split-K=8 decode (reduced by tfq3_fwht_y)
//   - tfq3_prefill_{3,4,5}b        : Multi-row simdgroup_matrix prefill + in-TG inverse FWHT-128 (+ add_res)
#include <metal_stdlib>
using namespace metal;

#define TFQ3_UNROLL _Pragma("clang loop unroll(full)")

inline short2 tfq3_frag_home(ushort l) {
  return short2(short((l & 8) + ((l & 1) << 2)), short(((l & 16) >> 2) | ((l >> 1) & 3)));
}

struct Tfq3Dims {
  uint rows;
  uint n;
  uint k;
  uint ldx;
  uint ldo;
  uint sr_split;
  uint ksplit;
  uint add_res;
};

struct Tfq3Tables {
  uint magic;
  uint version;
  float a3, b3, a4, b4, a5, b5;
  uint pad0[8];
  uint lut3[256];
  uint lut4[1024];
  uint lut5[1024];
  half4 c64_3[64];
  half4 c64_4[64];
  half4 c64_5[64];
};

inline float tfq3_sgn(uint seed, uint i) {
  uint h = (seed ^ i) * 2654435761u;
  return ((h >> 31) & 1u) ? -1.0f : 1.0f;
}

inline uint tfq3_sr0(uint r0, uint sr_split) {
  return (sr_split != 0u && r0 >= sr_split) ? (r0 - sr_split) : r0;
}

inline void tfq3_fwht128_simd4(thread float4& v, ushort lane) {
  TFQ3_UNROLL
  for (ushort d = 1; d <= 16; d <<= 1) {
    float4 o = simd_shuffle_xor(v, d);
    v = ((lane & d) == 0u) ? (v + o) : (o - v);
  }
  float a0 = v.x + v.y, b0 = v.x - v.y;
  float a1 = v.z + v.w, b1 = v.z - v.w;
  v = float4(a0 + a1, b0 + b1, a0 - a1, b0 - b1) * 0.08838834764831845f;
}

inline void tfq3_store_out4(device bfloat* dst, ushort lane, float4 v, uint sr0, uint add_res) {
  const float y0 = v.x * tfq3_sgn(0x9E3779B9u, sr0 + lane +  0u);
  const float y1 = v.y * tfq3_sgn(0x9E3779B9u, sr0 + lane + 32u);
  const float y2 = v.z * tfq3_sgn(0x9E3779B9u, sr0 + lane + 64u);
  const float y3 = v.w * tfq3_sgn(0x9E3779B9u, sr0 + lane + 96u);
  if (add_res != 0u) {
    dst[lane +  0u] = bfloat(float(dst[lane +  0u]) + y0);
    dst[lane + 32u] = bfloat(float(dst[lane + 32u]) + y1);
    dst[lane + 64u] = bfloat(float(dst[lane + 64u]) + y2);
    dst[lane + 96u] = bfloat(float(dst[lane + 96u]) + y3);
  } else {
    dst[lane +  0u] = bfloat(y0);
    dst[lane + 32u] = bfloat(y1);
    dst[lane + 64u] = bfloat(y2);
    dst[lane + 96u] = bfloat(y3);
  }
}

kernel void tfq3_fwht_x(
    const device bfloat* X   [[buffer(0)]],
    device half* X_rot       [[buffer(1)]],
    constant Tfq3Dims& d     [[buffer(2)]],
    uint2 gid                [[threadgroup_position_in_grid]],
    ushort lane              [[thread_index_in_simdgroup]]
) {
  uint row = gid.y, c0 = gid.x * 128u;
  const device bfloat* src = X + ulong(row) * ulong(d.ldx) + c0;
  float4 v;
  v.x = float(src[lane +  0u]) * tfq3_sgn(0x12345678u, c0 + lane +  0u);
  v.y = float(src[lane + 32u]) * tfq3_sgn(0x12345678u, c0 + lane + 32u);
  v.z = float(src[lane + 64u]) * tfq3_sgn(0x12345678u, c0 + lane + 64u);
  v.w = float(src[lane + 96u]) * tfq3_sgn(0x12345678u, c0 + lane + 96u);
  tfq3_fwht128_simd4(v, lane);
  device half* dst = X_rot + ulong(row) * ulong(d.k) + c0;
  dst[lane +  0u] = half(v.x); dst[lane + 32u] = half(v.y);
  dst[lane + 64u] = half(v.z); dst[lane + 96u] = half(v.w);
}

kernel void tfq3_fwht_y(
    const device float* YR   [[buffer(0)]],
    device bfloat* OUT       [[buffer(1)]],
    constant Tfq3Dims& d     [[buffer(2)]],
    uint2 gid                [[threadgroup_position_in_grid]],
    ushort lane              [[thread_index_in_simdgroup]]
) {
  uint row = gid.y, r0 = gid.x * 128u;
  float4 v = float4(0.0f);
  uint ks = metal::max(1u, d.ksplit);
  for (uint kp = 0u; kp < ks; ++kp) {
    const device float* src = YR + (ulong(kp) * ulong(d.rows) + ulong(row)) * ulong(d.n) + r0;
    v += float4(src[lane], src[lane + 32u], src[lane + 64u], src[lane + 96u]);
  }
  tfq3_fwht128_simd4(v, lane);
  uint sr0 = tfq3_sr0(r0, d.sr_split);
  device bfloat* dst = OUT + ulong(row) * ulong(d.ldo) + r0;
  tfq3_store_out4(dst, lane, v, sr0, d.add_res);
}

inline void tfq3_q8_4b(thread uint& st, uint wc, constant Tfq3Tables* T, thread half4& q0, thread half4& q1) {
  uint s0 = st;
  uint h0 = s0 * 0x9E3779B1u;
  uint h1 = ((s0 <<  8) | (wc >> 24)) * 0x9E3779B1u;
  uint h2 = ((s0 << 16) | (wc >> 16)) * 0x9E3779B1u;
  uint h3 = ((s0 << 24) | (wc >>  8)) * 0x9E3779B1u;
  st = wc;
  half4 c0 = T->c64_4[(h0 >> 16) & 63u], c2 = T->c64_4[(h2 >> 16) & 63u];
  q0 = fma(half4(as_type<half2>(T->lut4[(((wc >> 24)        << 2) ^ (h0 >> 22)) & 1023u]),
                 as_type<half2>(T->lut4[((((wc >> 16) & 255u) << 2) ^ (h1 >> 22)) & 1023u])), c0.xyxy, c0.zwzw);
  q1 = fma(half4(as_type<half2>(T->lut4[((((wc >>  8) & 255u) << 2) ^ (h2 >> 22)) & 1023u]),
                 as_type<half2>(T->lut4[((( wc        & 255u) << 2) ^ (h3 >> 22)) & 1023u])), c2.xyxy, c2.zwzw);
}

inline void tfq3_q8_3b(thread uint& st, uint m24, constant Tfq3Tables* T, thread half4& q0, thread half4& q1) {
  uint s0 = st;
  uint h0 = s0 * 0x9E3779B1u;
  uint h1 = ((s0 <<  6) | (m24 >> 18)) * 0x9E3779B1u;
  uint h2 = ((s0 << 12) | (m24 >> 12)) * 0x9E3779B1u;
  uint h3 = ((s0 << 18) | (m24 >>  6)) * 0x9E3779B1u;
  st = (s0 << 24) | m24;
  half4 c0 = T->c64_3[(h0 >> 16) & 63u], c2 = T->c64_3[(h2 >> 16) & 63u];
  q0 = fma(half4(as_type<half2>(T->lut3[(((m24 >> 16) & 0xFCu) ^ (h0 >> 24)) & 255u]),
                 as_type<half2>(T->lut3[(((m24 >> 10) & 0xFCu) ^ (h1 >> 24)) & 255u])), c0.xyxy, c0.zwzw);
  q1 = fma(half4(as_type<half2>(T->lut3[(((m24 >>  4) & 0xFCu) ^ (h2 >> 24)) & 255u]),
                 as_type<half2>(T->lut3[(((m24 <<  2) & 0xFCu) ^ (h3 >> 24)) & 255u])), c2.xyxy, c2.zwzw);
}

inline void tfq3_q8_5b(thread uint& st, uint hi8, uint lo32, constant Tfq3Tables* T, thread half4& q0, thread half4& q1) {
  uint s0 = st;
  uint b0 = ((hi8 << 2) | (lo32 >> 30)) & 1023u;
  uint b1 = (lo32 >> 20) & 1023u;
  uint b2 = (lo32 >> 10) & 1023u;
  uint b3 =  lo32        & 1023u;
  uint h0 = s0 * 0x9E3779B1u;
  uint h1 = ((s0 << 10) | b0) * 0x9E3779B1u;
  uint h2 = ((s0 << 20) | (hi8 << 12) | (lo32 >> 20)) * 0x9E3779B1u;
  uint h3 = ((s0 << 30) | (hi8 << 22) | (lo32 >> 10)) * 0x9E3779B1u;
  st = lo32;
  half4 c0 = T->c64_5[(h0 >> 16) & 63u], c2 = T->c64_5[(h2 >> 16) & 63u];
  q0 = fma(half4(as_type<half2>(T->lut5[(b0 ^ (h0 >> 22)) & 1023u]),
                 as_type<half2>(T->lut5[(b1 ^ (h1 >> 22)) & 1023u])), c0.xyxy, c0.zwzw);
  q1 = fma(half4(as_type<half2>(T->lut5[(b2 ^ (h2 >> 22)) & 1023u]),
                 as_type<half2>(T->lut5[(b3 ^ (h3 >> 22)) & 1023u])), c2.xyxy, c2.zwzw);
}

template <uint BITS>
inline uint tfq3_init_st(const device uint* w_base, uint bc0, ushort lane) {
  if (bc0 == 0u) return 0x13579BDFu ^ (uint(lane) * 0x9E3779B1u);
  if constexpr (BITS == 4u) return ((const device uint4*)w_base)[(ulong(bc0) * 2ul - 1ul) * 32ul + lane].w;
  return w_base[(ulong(bc0) * (BITS * 2u) - 1ul) * 32ul + lane];
}

inline float tfq3_h4sum(half4 p) { return (float(p.x) + float(p.y)) + (float(p.z) + float(p.w)); }
struct Tfq3H2 { half4 x, y; };
struct Tfq3H4 { half4 x, y, z, w; };
#define TFQ3_DOT1(Q0, Q1, X, CB, BA0, BA1) { \
  half4 xv = *(const device half4*)((X) + (CB)); \
  (BA0) = fma((Q0), xv, (BA0)); (BA1) = fma((Q1), xv, (BA1)); }
#define TFQ3_DOT2(Q0, Q1, X0, X1, CB, BA0, BA1) { \
  { half4 xv = *(const device half4*)((X0) + (CB)); (BA0).x = fma((Q0), xv, (BA0).x); (BA1).x = fma((Q1), xv, (BA1).x); } \
  { half4 xv = *(const device half4*)((X1) + (CB)); (BA0).y = fma((Q0), xv, (BA0).y); (BA1).y = fma((Q1), xv, (BA1).y); } }
#define TFQ3_DOT4(Q0, Q1, X0, X1, X2, X3, CB, BA0, BA1) { \
  { half4 xv = *(const device half4*)((X0) + (CB)); (BA0).x = fma((Q0), xv, (BA0).x); (BA1).x = fma((Q1), xv, (BA1).x); } \
  { half4 xv = *(const device half4*)((X1) + (CB)); (BA0).y = fma((Q0), xv, (BA0).y); (BA1).y = fma((Q1), xv, (BA1).y); } \
  { half4 xv = *(const device half4*)((X2) + (CB)); (BA0).z = fma((Q0), xv, (BA0).z); (BA1).z = fma((Q1), xv, (BA1).z); } \
  { half4 xv = *(const device half4*)((X3) + (CB)); (BA0).w = fma((Q0), xv, (BA0).w); (BA1).w = fma((Q1), xv, (BA1).w); } }
inline float  tfq3_fin(half4  a) { return tfq3_h4sum(a); }
inline float2 tfq3_fin(Tfq3H2 a) { return float2(tfq3_h4sum(a.x), tfq3_h4sum(a.y)); }
inline float4 tfq3_fin(Tfq3H4 a) { return float4(tfq3_h4sum(a.x), tfq3_h4sum(a.y), tfq3_h4sum(a.z), tfq3_h4sum(a.w)); }

template <uint BITS>
inline void tfq3_mv_range1(
    const device uint* w_base, const device half* sc_base, uint bc0, uint bc1,
    constant Tfq3Tables* T, const device half* X_r, ushort lane, short2 home,
    thread float& acc0, thread float& acc1
) {
  uint st = tfq3_init_st<BITS>(w_base, bc0, lane);
  acc0 = 0.0f; acc1 = 0.0f;
  if constexpr (BITS == 4u) {
    const device uint4* w4 = (const device uint4*)w_base;
    for (uint bc = bc0; bc < bc1; ++bc) {
      half4 ba0 = half4(0.0h), ba1 = half4(0.0h);
      uint4 v = w4[(bc * 2u + 0u) * 32u + lane];
      uint cb = (bc << 7) + uint(home.x);
      half4 q0, q1;
      tfq3_q8_4b(st, v.x, T, q0, q1); TFQ3_DOT1(q0, q1, X_r, cb +   0u, ba0, ba1);
      tfq3_q8_4b(st, v.y, T, q0, q1); TFQ3_DOT1(q0, q1, X_r, cb +  16u, ba0, ba1);
      tfq3_q8_4b(st, v.z, T, q0, q1); TFQ3_DOT1(q0, q1, X_r, cb +  32u, ba0, ba1);
      tfq3_q8_4b(st, v.w, T, q0, q1); TFQ3_DOT1(q0, q1, X_r, cb +  48u, ba0, ba1);
      v = w4[(bc * 2u + 1u) * 32u + lane];
      tfq3_q8_4b(st, v.x, T, q0, q1); TFQ3_DOT1(q0, q1, X_r, cb +  64u, ba0, ba1);
      tfq3_q8_4b(st, v.y, T, q0, q1); TFQ3_DOT1(q0, q1, X_r, cb +  80u, ba0, ba1);
      tfq3_q8_4b(st, v.z, T, q0, q1); TFQ3_DOT1(q0, q1, X_r, cb +  96u, ba0, ba1);
      tfq3_q8_4b(st, v.w, T, q0, q1); TFQ3_DOT1(q0, q1, X_r, cb + 112u, ba0, ba1);
      float sc = float(sc_base[bc]);
      acc0 = fma(tfq3_fin(ba0), sc, acc0); acc1 = fma(tfq3_fin(ba1), sc, acc1);
    }
  } else if constexpr (BITS == 3u) {
    for (uint bc = bc0; bc < bc1; ++bc) {
      half4 ba0 = half4(0.0h), ba1 = half4(0.0h);
      for (uint h = 0u; h < 2u; ++h) {
        const device uint* blk = w_base + (ulong(bc) * 6u + h * 3u) * 32u + lane;
        uint a0 = blk[0], a1 = blk[32], a2 = blk[64];
        uint cb = (bc << 7) + (h << 6) + uint(home.x);
        half4 q0, q1;
        tfq3_q8_3b(st, a0 >> 8,                               T, q0, q1); TFQ3_DOT1(q0, q1, X_r, cb +  0u, ba0, ba1);
        tfq3_q8_3b(st, ((a0 << 16) | (a1 >> 16)) & 0xFFFFFFu, T, q0, q1); TFQ3_DOT1(q0, q1, X_r, cb + 16u, ba0, ba1);
        tfq3_q8_3b(st, ((a1 <<  8) | (a2 >> 24)) & 0xFFFFFFu, T, q0, q1); TFQ3_DOT1(q0, q1, X_r, cb + 32u, ba0, ba1);
        tfq3_q8_3b(st, a2 & 0xFFFFFFu,                        T, q0, q1); TFQ3_DOT1(q0, q1, X_r, cb + 48u, ba0, ba1);
      }
      float sc = float(sc_base[bc]);
      acc0 = fma(tfq3_fin(ba0), sc, acc0); acc1 = fma(tfq3_fin(ba1), sc, acc1);
    }
  } else {
    for (uint bc = bc0; bc < bc1; ++bc) {
      half4 ba0 = half4(0.0h), ba1 = half4(0.0h);
      for (uint h = 0u; h < 2u; ++h) {
        const device uint* blk = w_base + (ulong(bc) * 10u + h * 5u) * 32u + lane;
        uint w0 = blk[0], w1 = blk[32], w2 = blk[64], w3 = blk[96], w4 = blk[128];
        uint cb = (bc << 7) + (h << 6) + uint(home.x);
        half4 q0, q1;
        tfq3_q8_5b(st, w0 >> 24,          (w0 <<  8) | (w1 >> 24), T, q0, q1); TFQ3_DOT1(q0, q1, X_r, cb +  0u, ba0, ba1);
        tfq3_q8_5b(st, (w1 >> 16) & 255u, (w1 << 16) | (w2 >> 16), T, q0, q1); TFQ3_DOT1(q0, q1, X_r, cb + 16u, ba0, ba1);
        tfq3_q8_5b(st, (w2 >>  8) & 255u, (w2 << 24) | (w3 >>  8), T, q0, q1); TFQ3_DOT1(q0, q1, X_r, cb + 32u, ba0, ba1);
        tfq3_q8_5b(st, w3 & 255u,          w4,                     T, q0, q1); TFQ3_DOT1(q0, q1, X_r, cb + 48u, ba0, ba1);
      }
      float sc = float(sc_base[bc]);
      acc0 = fma(tfq3_fin(ba0), sc, acc0); acc1 = fma(tfq3_fin(ba1), sc, acc1);
    }
  }
  acc0 += simd_shuffle_xor(acc0, 1); acc0 += simd_shuffle_xor(acc0, 8);
  acc1 += simd_shuffle_xor(acc1, 1); acc1 += simd_shuffle_xor(acc1, 8);
}

template <uint BITS>
inline void tfq3_mv_range2(
    const device uint* w_base, const device half* sc_base, uint bc0, uint bc1,
    constant Tfq3Tables* T, const device half* x0, const device half* x1,
    ushort lane, short2 home, thread float2& acc0, thread float2& acc1
) {
  uint st = tfq3_init_st<BITS>(w_base, bc0, lane);
  acc0 = float2(0.0f); acc1 = float2(0.0f);
  if constexpr (BITS == 4u) {
    const device uint4* w4 = (const device uint4*)w_base;
    for (uint bc = bc0; bc < bc1; ++bc) {
      Tfq3H2 ba0 = {half4(0.0h), half4(0.0h)}, ba1 = {half4(0.0h), half4(0.0h)};
      uint4 v = w4[(bc * 2u + 0u) * 32u + lane];
      uint cb = (bc << 7) + uint(home.x);
      half4 q0, q1;
      tfq3_q8_4b(st, v.x, T, q0, q1); TFQ3_DOT2(q0, q1, x0, x1, cb +   0u, ba0, ba1);
      tfq3_q8_4b(st, v.y, T, q0, q1); TFQ3_DOT2(q0, q1, x0, x1, cb +  16u, ba0, ba1);
      tfq3_q8_4b(st, v.z, T, q0, q1); TFQ3_DOT2(q0, q1, x0, x1, cb +  32u, ba0, ba1);
      tfq3_q8_4b(st, v.w, T, q0, q1); TFQ3_DOT2(q0, q1, x0, x1, cb +  48u, ba0, ba1);
      v = w4[(bc * 2u + 1u) * 32u + lane];
      tfq3_q8_4b(st, v.x, T, q0, q1); TFQ3_DOT2(q0, q1, x0, x1, cb +  64u, ba0, ba1);
      tfq3_q8_4b(st, v.y, T, q0, q1); TFQ3_DOT2(q0, q1, x0, x1, cb +  80u, ba0, ba1);
      tfq3_q8_4b(st, v.z, T, q0, q1); TFQ3_DOT2(q0, q1, x0, x1, cb +  96u, ba0, ba1);
      tfq3_q8_4b(st, v.w, T, q0, q1); TFQ3_DOT2(q0, q1, x0, x1, cb + 112u, ba0, ba1);
      float sc = float(sc_base[bc]);
      acc0 = fma(tfq3_fin(ba0), sc, acc0); acc1 = fma(tfq3_fin(ba1), sc, acc1);
    }
  } else if constexpr (BITS == 3u) {
    for (uint bc = bc0; bc < bc1; ++bc) {
      Tfq3H2 ba0 = {half4(0.0h), half4(0.0h)}, ba1 = {half4(0.0h), half4(0.0h)};
      for (uint h = 0u; h < 2u; ++h) {
        const device uint* blk = w_base + (ulong(bc) * 6u + h * 3u) * 32u + lane;
        uint a0 = blk[0], a1 = blk[32], a2 = blk[64];
        uint cb = (bc << 7) + (h << 6) + uint(home.x);
        half4 q0, q1;
        tfq3_q8_3b(st, a0 >> 8,                               T, q0, q1); TFQ3_DOT2(q0, q1, x0, x1, cb +  0u, ba0, ba1);
        tfq3_q8_3b(st, ((a0 << 16) | (a1 >> 16)) & 0xFFFFFFu, T, q0, q1); TFQ3_DOT2(q0, q1, x0, x1, cb + 16u, ba0, ba1);
        tfq3_q8_3b(st, ((a1 <<  8) | (a2 >> 24)) & 0xFFFFFFu, T, q0, q1); TFQ3_DOT2(q0, q1, x0, x1, cb + 32u, ba0, ba1);
        tfq3_q8_3b(st, a2 & 0xFFFFFFu,                        T, q0, q1); TFQ3_DOT2(q0, q1, x0, x1, cb + 48u, ba0, ba1);
      }
      float sc = float(sc_base[bc]);
      acc0 = fma(tfq3_fin(ba0), sc, acc0); acc1 = fma(tfq3_fin(ba1), sc, acc1);
    }
  } else {
    for (uint bc = bc0; bc < bc1; ++bc) {
      Tfq3H2 ba0 = {half4(0.0h), half4(0.0h)}, ba1 = {half4(0.0h), half4(0.0h)};
      for (uint h = 0u; h < 2u; ++h) {
        const device uint* blk = w_base + (ulong(bc) * 10u + h * 5u) * 32u + lane;
        uint w0 = blk[0], w1 = blk[32], w2 = blk[64], w3 = blk[96], w4 = blk[128];
        uint cb = (bc << 7) + (h << 6) + uint(home.x);
        half4 q0, q1;
        tfq3_q8_5b(st, w0 >> 24,          (w0 <<  8) | (w1 >> 24), T, q0, q1); TFQ3_DOT2(q0, q1, x0, x1, cb +  0u, ba0, ba1);
        tfq3_q8_5b(st, (w1 >> 16) & 255u, (w1 << 16) | (w2 >> 16), T, q0, q1); TFQ3_DOT2(q0, q1, x0, x1, cb + 16u, ba0, ba1);
        tfq3_q8_5b(st, (w2 >>  8) & 255u, (w2 << 24) | (w3 >>  8), T, q0, q1); TFQ3_DOT2(q0, q1, x0, x1, cb + 32u, ba0, ba1);
        tfq3_q8_5b(st, w3 & 255u,          w4,                     T, q0, q1); TFQ3_DOT2(q0, q1, x0, x1, cb + 48u, ba0, ba1);
      }
      float sc = float(sc_base[bc]);
      acc0 = fma(tfq3_fin(ba0), sc, acc0); acc1 = fma(tfq3_fin(ba1), sc, acc1);
    }
  }
  acc0 += simd_shuffle_xor(acc0, 1); acc0 += simd_shuffle_xor(acc0, 8);
  acc1 += simd_shuffle_xor(acc1, 1); acc1 += simd_shuffle_xor(acc1, 8);
}

template <uint BITS>
inline void tfq3_mv_range4(
    const device uint* w_base, const device half* sc_base, uint bc0, uint bc1,
    constant Tfq3Tables* T,
    const device half* x0, const device half* x1, const device half* x2, const device half* x3,
    ushort lane, short2 home, thread float4& acc0, thread float4& acc1
) {
  uint st = tfq3_init_st<BITS>(w_base, bc0, lane);
  acc0 = float4(0.0f); acc1 = float4(0.0f);
  if constexpr (BITS == 4u) {
    const device uint4* w4 = (const device uint4*)w_base;
    for (uint bc = bc0; bc < bc1; ++bc) {
      Tfq3H4 ba0 = {half4(0.0h), half4(0.0h), half4(0.0h), half4(0.0h)};
      Tfq3H4 ba1 = {half4(0.0h), half4(0.0h), half4(0.0h), half4(0.0h)};
      uint4 v = w4[(bc * 2u + 0u) * 32u + lane];
      uint cb = (bc << 7) + uint(home.x);
      half4 q0, q1;
      tfq3_q8_4b(st, v.x, T, q0, q1); TFQ3_DOT4(q0, q1, x0, x1, x2, x3, cb +   0u, ba0, ba1);
      tfq3_q8_4b(st, v.y, T, q0, q1); TFQ3_DOT4(q0, q1, x0, x1, x2, x3, cb +  16u, ba0, ba1);
      tfq3_q8_4b(st, v.z, T, q0, q1); TFQ3_DOT4(q0, q1, x0, x1, x2, x3, cb +  32u, ba0, ba1);
      tfq3_q8_4b(st, v.w, T, q0, q1); TFQ3_DOT4(q0, q1, x0, x1, x2, x3, cb +  48u, ba0, ba1);
      v = w4[(bc * 2u + 1u) * 32u + lane];
      tfq3_q8_4b(st, v.x, T, q0, q1); TFQ3_DOT4(q0, q1, x0, x1, x2, x3, cb +  64u, ba0, ba1);
      tfq3_q8_4b(st, v.y, T, q0, q1); TFQ3_DOT4(q0, q1, x0, x1, x2, x3, cb +  80u, ba0, ba1);
      tfq3_q8_4b(st, v.z, T, q0, q1); TFQ3_DOT4(q0, q1, x0, x1, x2, x3, cb +  96u, ba0, ba1);
      tfq3_q8_4b(st, v.w, T, q0, q1); TFQ3_DOT4(q0, q1, x0, x1, x2, x3, cb + 112u, ba0, ba1);
      float sc = float(sc_base[bc]);
      acc0 = fma(tfq3_fin(ba0), sc, acc0); acc1 = fma(tfq3_fin(ba1), sc, acc1);
    }
  } else if constexpr (BITS == 3u) {
    for (uint bc = bc0; bc < bc1; ++bc) {
      Tfq3H4 ba0 = {half4(0.0h), half4(0.0h), half4(0.0h), half4(0.0h)};
      Tfq3H4 ba1 = {half4(0.0h), half4(0.0h), half4(0.0h), half4(0.0h)};
      for (uint h = 0u; h < 2u; ++h) {
        const device uint* blk = w_base + (ulong(bc) * 6u + h * 3u) * 32u + lane;
        uint a0 = blk[0], a1 = blk[32], a2 = blk[64];
        uint cb = (bc << 7) + (h << 6) + uint(home.x);
        half4 q0, q1;
        tfq3_q8_3b(st, a0 >> 8,                               T, q0, q1); TFQ3_DOT4(q0, q1, x0, x1, x2, x3, cb +  0u, ba0, ba1);
        tfq3_q8_3b(st, ((a0 << 16) | (a1 >> 16)) & 0xFFFFFFu, T, q0, q1); TFQ3_DOT4(q0, q1, x0, x1, x2, x3, cb + 16u, ba0, ba1);
        tfq3_q8_3b(st, ((a1 <<  8) | (a2 >> 24)) & 0xFFFFFFu, T, q0, q1); TFQ3_DOT4(q0, q1, x0, x1, x2, x3, cb + 32u, ba0, ba1);
        tfq3_q8_3b(st, a2 & 0xFFFFFFu,                        T, q0, q1); TFQ3_DOT4(q0, q1, x0, x1, x2, x3, cb + 48u, ba0, ba1);
      }
      float sc = float(sc_base[bc]);
      acc0 = fma(tfq3_fin(ba0), sc, acc0); acc1 = fma(tfq3_fin(ba1), sc, acc1);
    }
  } else {
    for (uint bc = bc0; bc < bc1; ++bc) {
      Tfq3H4 ba0 = {half4(0.0h), half4(0.0h), half4(0.0h), half4(0.0h)};
      Tfq3H4 ba1 = {half4(0.0h), half4(0.0h), half4(0.0h), half4(0.0h)};
      for (uint h = 0u; h < 2u; ++h) {
        const device uint* blk = w_base + (ulong(bc) * 10u + h * 5u) * 32u + lane;
        uint w0 = blk[0], w1 = blk[32], w2 = blk[64], w3 = blk[96], w4 = blk[128];
        uint cb = (bc << 7) + (h << 6) + uint(home.x);
        half4 q0, q1;
        tfq3_q8_5b(st, w0 >> 24,          (w0 <<  8) | (w1 >> 24), T, q0, q1); TFQ3_DOT4(q0, q1, x0, x1, x2, x3, cb +  0u, ba0, ba1);
        tfq3_q8_5b(st, (w1 >> 16) & 255u, (w1 << 16) | (w2 >> 16), T, q0, q1); TFQ3_DOT4(q0, q1, x0, x1, x2, x3, cb + 16u, ba0, ba1);
        tfq3_q8_5b(st, (w2 >>  8) & 255u, (w2 << 24) | (w3 >>  8), T, q0, q1); TFQ3_DOT4(q0, q1, x0, x1, x2, x3, cb + 32u, ba0, ba1);
        tfq3_q8_5b(st, w3 & 255u,          w4,                     T, q0, q1); TFQ3_DOT4(q0, q1, x0, x1, x2, x3, cb + 48u, ba0, ba1);
      }
      float sc = float(sc_base[bc]);
      acc0 = fma(tfq3_fin(ba0), sc, acc0); acc1 = fma(tfq3_fin(ba1), sc, acc1);
    }
  }
  acc0 += simd_shuffle_xor(acc0, 1); acc0 += simd_shuffle_xor(acc0, 8);
  acc1 += simd_shuffle_xor(acc1, 1); acc1 += simd_shuffle_xor(acc1, 8);
}

// Single-dispatch fused decode (8 simdgroups = 256 threads per 128-row output block).
template <uint BITS>
kernel void tfq3_dec_fused1_t(
    const device uint* W     [[buffer(0)]],
    const device half* scales[[buffer(1)]],
    constant Tfq3Tables* T   [[buffer(2)]],
    const device half* X_rot [[buffer(3)]],
    device bfloat* OUT       [[buffer(4)]],
    constant Tfq3Dims& d     [[buffer(5)]],
    uint gid                 [[threadgroup_position_in_grid]],
    ushort sg                [[simdgroup_index_in_threadgroup]],
    ushort lane              [[thread_index_in_simdgroup]]
) {
  uint br = gid, r0 = br * 128u, strip = br * 8u + uint(sg), n_bc = d.k >> 7;
  constexpr uint wpb = BITS * 2u;
  const device uint* w_base = W + ulong(strip) * ulong(n_bc) * wpb * 32ul;
  const device half* s_base = scales + ulong(br) * ulong(n_bc);
  short2 home = tfq3_frag_home(lane);
  threadgroup float tg_y[128];
  float a0, a1;
  tfq3_mv_range1<BITS>(w_base, s_base, 0u, n_bc, T, X_rot, lane, home, a0, a1);
  if ((lane & 9u) == 0u) {
    uint idx = uint(sg) * 16u + uint(home.y);
    tg_y[idx] = a0; tg_y[idx + 8u] = a1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u) {
    float4 v = float4(tg_y[lane], tg_y[lane + 32u], tg_y[lane + 64u], tg_y[lane + 96u]);
    tfq3_fwht128_simd4(v, lane);
    uint sr0 = tfq3_sr0(r0, d.sr_split);
    tfq3_store_out4(OUT + r0, lane, v, sr0, d.add_res);
  }
}

template <uint BITS>
kernel void tfq3_dec_fused2_t(
    const device uint* W     [[buffer(0)]],
    const device half* scales[[buffer(1)]],
    constant Tfq3Tables* T   [[buffer(2)]],
    const device half* X_rot [[buffer(3)]],
    device bfloat* OUT       [[buffer(4)]],
    constant Tfq3Dims& d     [[buffer(5)]],
    uint gid                 [[threadgroup_position_in_grid]],
    ushort sg                [[simdgroup_index_in_threadgroup]],
    ushort lane              [[thread_index_in_simdgroup]]
) {
  uint br = gid, r0 = br * 128u, strip = br * 8u + uint(sg), n_bc = d.k >> 7;
  constexpr uint wpb = BITS * 2u;
  const device uint* w_base = W + ulong(strip) * ulong(n_bc) * wpb * 32ul;
  const device half* s_base = scales + ulong(br) * ulong(n_bc);
  short2 home = tfq3_frag_home(lane);
  threadgroup float tg_y[2][128];
  float2 a0, a1;
  tfq3_mv_range2<BITS>(w_base, s_base, 0u, n_bc, T, X_rot, X_rot + ulong(d.k), lane, home, a0, a1);
  if ((lane & 9u) == 0u) {
    uint idx = uint(sg) * 16u + uint(home.y);
    tg_y[0][idx] = a0.x; tg_y[0][idx + 8u] = a1.x;
    tg_y[1][idx] = a0.y; tg_y[1][idx + 8u] = a1.y;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg < 2u) {
    float4 v = float4(tg_y[sg][lane], tg_y[sg][lane + 32u], tg_y[sg][lane + 64u], tg_y[sg][lane + 96u]);
    tfq3_fwht128_simd4(v, lane);
    uint sr0 = tfq3_sr0(r0, d.sr_split);
    tfq3_store_out4(OUT + ulong(sg) * ulong(d.ldo) + r0, lane, v, sr0, d.add_res);
  }
}

template <uint BITS>
kernel void tfq3_dec_fused4_t(
    const device uint* W     [[buffer(0)]],
    const device half* scales[[buffer(1)]],
    constant Tfq3Tables* T   [[buffer(2)]],
    const device half* X_rot [[buffer(3)]],
    device bfloat* OUT       [[buffer(4)]],
    constant Tfq3Dims& d     [[buffer(5)]],
    uint gid                 [[threadgroup_position_in_grid]],
    ushort sg                [[simdgroup_index_in_threadgroup]],
    ushort lane              [[thread_index_in_simdgroup]]
) {
  uint br = gid, r0 = br * 128u, strip = br * 8u + uint(sg), n_bc = d.k >> 7;
  constexpr uint wpb = BITS * 2u;
  const device uint* w_base = W + ulong(strip) * ulong(n_bc) * wpb * 32ul;
  const device half* s_base = scales + ulong(br) * ulong(n_bc);
  short2 home = tfq3_frag_home(lane);
  threadgroup float tg_y[4][128];
  for (uint rb = 0u; rb < d.rows; rb += 4u) {
    uint nc = metal::min(4u, d.rows - rb);
    const device half* x0 = X_rot + ulong(rb + 0u) * ulong(d.k);
    const device half* x1 = X_rot + ulong(rb + ((nc > 1u) ? 1u : 0u)) * ulong(d.k);
    const device half* x2 = X_rot + ulong(rb + ((nc > 2u) ? 2u : 0u)) * ulong(d.k);
    const device half* x3 = X_rot + ulong(rb + ((nc > 3u) ? 3u : 0u)) * ulong(d.k);
    float4 a0, a1;
    tfq3_mv_range4<BITS>(w_base, s_base, 0u, n_bc, T, x0, x1, x2, x3, lane, home, a0, a1);
    if ((lane & 9u) == 0u) {
      uint idx = uint(sg) * 16u + uint(home.y);
      tg_y[0][idx] = a0.x; tg_y[0][idx + 8u] = a1.x;
      tg_y[1][idx] = a0.y; tg_y[1][idx + 8u] = a1.y;
      tg_y[2][idx] = a0.z; tg_y[2][idx + 8u] = a1.z;
      tg_y[3][idx] = a0.w; tg_y[3][idx + 8u] = a1.w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (uint(sg) < nc) {
      float4 v = float4(tg_y[sg][lane], tg_y[sg][lane + 32u], tg_y[sg][lane + 64u], tg_y[sg][lane + 96u]);
      tfq3_fwht128_simd4(v, lane);
      uint sr0 = tfq3_sr0(r0, d.sr_split);
      tfq3_store_out4(OUT + ulong(rb + uint(sg)) * ulong(d.ldo) + r0, lane, v, sr0, d.add_res);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// Zero-SRAM 2D split-K=8 decode (grid = (n/32, 8) of 64 threads; reduced in tfq3_fwht_y).
template <uint BITS>
kernel void tfq3_dec_2dsk8_1_t(
    const device uint* W     [[buffer(0)]],
    const device half* scales[[buffer(1)]],
    constant Tfq3Tables* T   [[buffer(2)]],
    const device half* X_rot [[buffer(3)]],
    device float* YR_out     [[buffer(4)]],
    constant Tfq3Dims& d     [[buffer(5)]],
    uint2 gid                [[threadgroup_position_in_grid]],
    ushort sg                [[simdgroup_index_in_threadgroup]],
    ushort lane              [[thread_index_in_simdgroup]]
) {
  uint strip = gid.x * 2u + uint(sg), kp = gid.y, n_bc = d.k >> 7;
  uint bc0 = (kp * n_bc) >> 3, bc1 = ((kp + 1u) * n_bc) >> 3;
  constexpr uint wpb = BITS * 2u;
  const device uint* w_base = W + ulong(strip) * ulong(n_bc) * wpb * 32ul;
  const device half* s_base = scales + ulong(strip >> 3) * ulong(n_bc);
  short2 home = tfq3_frag_home(lane);
  float a0, a1;
  tfq3_mv_range1<BITS>(w_base, s_base, bc0, bc1, T, X_rot, lane, home, a0, a1);
  if ((lane & 9u) == 0u) {
    ulong pbase = ulong(kp) * ulong(d.n);
    YR_out[pbase + strip * 16u + uint(home.y)]      = a0;
    YR_out[pbase + strip * 16u + uint(home.y) + 8u] = a1;
  }
}

template <uint BITS>
kernel void tfq3_dec_2dsk8_2_t(
    const device uint* W     [[buffer(0)]],
    const device half* scales[[buffer(1)]],
    constant Tfq3Tables* T   [[buffer(2)]],
    const device half* X_rot [[buffer(3)]],
    device float* YR_out     [[buffer(4)]],
    constant Tfq3Dims& d     [[buffer(5)]],
    uint2 gid                [[threadgroup_position_in_grid]],
    ushort sg                [[simdgroup_index_in_threadgroup]],
    ushort lane              [[thread_index_in_simdgroup]]
) {
  uint strip = gid.x * 2u + uint(sg), kp = gid.y, n_bc = d.k >> 7;
  uint bc0 = (kp * n_bc) >> 3, bc1 = ((kp + 1u) * n_bc) >> 3;
  constexpr uint wpb = BITS * 2u;
  const device uint* w_base = W + ulong(strip) * ulong(n_bc) * wpb * 32ul;
  const device half* s_base = scales + ulong(strip >> 3) * ulong(n_bc);
  short2 home = tfq3_frag_home(lane);
  float2 a0, a1;
  tfq3_mv_range2<BITS>(w_base, s_base, bc0, bc1, T, X_rot, X_rot + ulong(d.k), lane, home, a0, a1);
  if ((lane & 9u) == 0u) {
    uint r = strip * 16u + uint(home.y);
    ulong pbase = ulong(kp) * 2ul * ulong(d.n);
    YR_out[pbase + r] = a0.x; YR_out[pbase + r + 8u] = a1.x;
    YR_out[pbase + ulong(d.n) + r] = a0.y; YR_out[pbase + ulong(d.n) + r + 8u] = a1.y;
  }
}

template <uint BITS>
kernel void tfq3_dec_2dsk8_4_t(
    const device uint* W     [[buffer(0)]],
    const device half* scales[[buffer(1)]],
    constant Tfq3Tables* T   [[buffer(2)]],
    const device half* X_rot [[buffer(3)]],
    device float* YR_out     [[buffer(4)]],
    constant Tfq3Dims& d     [[buffer(5)]],
    uint2 gid                [[threadgroup_position_in_grid]],
    ushort sg                [[simdgroup_index_in_threadgroup]],
    ushort lane              [[thread_index_in_simdgroup]]
) {
  uint strip = gid.x * 2u + uint(sg), kp = gid.y, n_bc = d.k >> 7;
  uint bc0 = (kp * n_bc) >> 3, bc1 = ((kp + 1u) * n_bc) >> 3;
  constexpr uint wpb = BITS * 2u;
  const device uint* w_base = W + ulong(strip) * ulong(n_bc) * wpb * 32ul;
  const device half* s_base = scales + ulong(strip >> 3) * ulong(n_bc);
  short2 home = tfq3_frag_home(lane);
  ulong pbase = ulong(kp) * ulong(d.rows) * ulong(d.n);
  for (uint rb = 0u; rb < d.rows; rb += 4u) {
    uint nc = metal::min(4u, d.rows - rb);
    const device half* x0 = X_rot + ulong(rb + 0u) * ulong(d.k);
    const device half* x1 = X_rot + ulong(rb + ((nc > 1u) ? 1u : 0u)) * ulong(d.k);
    const device half* x2 = X_rot + ulong(rb + ((nc > 2u) ? 2u : 0u)) * ulong(d.k);
    const device half* x3 = X_rot + ulong(rb + ((nc > 3u) ? 3u : 0u)) * ulong(d.k);
    float4 a0, a1;
    tfq3_mv_range4<BITS>(w_base, s_base, bc0, bc1, T, x0, x1, x2, x3, lane, home, a0, a1);
    if ((lane & 9u) == 0u) {
      uint r = strip * 16u + uint(home.y);
      YR_out[pbase + ulong(rb + 0u) * ulong(d.n) + r] = a0.x; YR_out[pbase + ulong(rb + 0u) * ulong(d.n) + r + 8u] = a1.x;
      if (nc > 1u) { YR_out[pbase + ulong(rb + 1u) * ulong(d.n) + r] = a0.y; YR_out[pbase + ulong(rb + 1u) * ulong(d.n) + r + 8u] = a1.y; }
      if (nc > 2u) { YR_out[pbase + ulong(rb + 2u) * ulong(d.n) + r] = a0.z; YR_out[pbase + ulong(rb + 2u) * ulong(d.n) + r + 8u] = a1.z; }
      if (nc > 3u) { YR_out[pbase + ulong(rb + 3u) * ulong(d.n) + r] = a0.w; YR_out[pbase + ulong(rb + 3u) * ulong(d.n) + r + 8u] = a1.w; }
    }
  }
}

// Universal M1-M5 multi-row TFQ3 prefill using hardware simdgroup_matrix (16 M-rows x 128 N-cols per TG).
// Grid: threadgroups ((rows + 15) / 16, n / 128, 1) of 128 threads (4 simdgroups, 2 strips = 32 cols/SG).
// X-major m-tile ordering shares each 128-row weight block across concurrent M-tiles in L2/SLC!
inline void tfq3_stage_subtile(threadgroup half* dst, half4 q0, half4 q1, short2 home) {
  *(threadgroup half4*)(dst + uint(home.y) * 16u + uint(home.x)) = q0;
  *(threadgroup half4*)(dst + uint(home.y + 8) * 16u + uint(home.x)) = q1;
}

inline void tfq3_mma_step16(
    const threadgroup half* tg_a,
    const threadgroup half* tg_b0,
    const threadgroup half* tg_b1,
    uint kt,
    uint nb,
    thread simdgroup_float8x8 bc_acc[2][4]
) {
  const uint k_off = kt * 16u;
  TFQ3_UNROLL for (uint ki = 0u; ki < 2u; ++ki) {
    simdgroup_half8x8 a0, a1;
    simdgroup_load(a0, tg_a + k_off + ki * 8u, 128u);
    if (nb > 8u) simdgroup_load(a1, tg_a + 8u * 128u + k_off + ki * 8u, 128u);

    simdgroup_half8x8 b[4];
    simdgroup_load(b[0], tg_b0 + 0u * 16u + ki * 8u, 16u, ulong2(0, 0), true);
    simdgroup_load(b[1], tg_b0 + 8u * 16u + ki * 8u, 16u, ulong2(0, 0), true);
    simdgroup_load(b[2], tg_b1 + 0u * 16u + ki * 8u, 16u, ulong2(0, 0), true);
    simdgroup_load(b[3], tg_b1 + 8u * 16u + ki * 8u, 16u, ulong2(0, 0), true);

    TFQ3_UNROLL for (uint ni = 0u; ni < 4u; ++ni) {
      simdgroup_multiply_accumulate(bc_acc[0][ni], a0, b[ni], bc_acc[0][ni]);
      if (nb > 8u) simdgroup_multiply_accumulate(bc_acc[1][ni], a1, b[ni], bc_acc[1][ni]);
    }
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
}

template <uint BITS>
kernel void tfq3_prefill_t(
    const device uint* W     [[buffer(0)]],
    const device half* scales[[buffer(1)]],
    constant Tfq3Tables* T   [[buffer(2)]],
    const device half* X_rot [[buffer(3)]],
    device bfloat* OUT       [[buffer(4)]],
    constant Tfq3Dims& d     [[buffer(5)]],
    uint2 gid                [[threadgroup_position_in_grid]],
    ushort tid               [[thread_index_in_threadgroup]],
    ushort sg                [[simdgroup_index_in_threadgroup]],
    ushort lane              [[thread_index_in_simdgroup]]
) {
  const uint m0 = gid.x * 16u;
  const uint rb = gid.y;
  const uint r0 = rb * 128u;
  if (m0 >= d.rows) return;
  const uint nb = metal::min(16u, d.rows - m0);
  const uint n_bc = d.k >> 7;
  constexpr uint wpb = BITS * 2u;
  const uint strip0 = rb * 8u + uint(sg) * 2u;
  const device uint* w_s0 = W + ulong(strip0 + 0u) * ulong(n_bc) * wpb * 32ul;
  const device uint* w_s1 = W + ulong(strip0 + 1u) * ulong(n_bc) * wpb * 32ul;
  const device half* sc0  = scales + ulong(rb) * ulong(n_bc);

  uint st0 = 0x13579BDFu ^ (uint(lane) * 0x9E3779B1u);
  uint st1 = st0;
  const short2 home = tfq3_frag_home(lane);

  threadgroup float tg_f[16 * 128]; // 8 KiB: holds tg_a (4 KiB half) during K-loop, output during epilogue
  threadgroup half* tg_a = (threadgroup half*)tg_f;
  threadgroup half tg_b[4][512];    // 4 KiB: per-SG 2x(16x16) decoded weight subtiles

  // Final accumulated 16x128 float tile is reduced per-block into tg_f via a separate staging or direct scale:
  // Use tg_accum[16 * 128] or keep per-SG float8x8 accumulators (only 8 matrices = 16 regs/lane!).
  // Wait: simdgroup_float8x8 supports scaling via threadgroup store/add or element loop; keep tg_acc[16*128] (8 KiB):
  threadgroup float tg_acc[16 * 128];
  for (uint i = tid; i < 16u * 128u; i += 128u) tg_acc[i] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint row_l = tid >> 3;           // 0..15 (16 rows)
  const uint col_l = (tid & 7u) * 16u;   // 0, 16, ..., 112 (16 halfs = 4 half4 per thread)

  for (uint bc = 0u; bc < n_bc; ++bc) {
    if (row_l < nb) {
      const device half4* src4 = (const device half4*)(X_rot + ulong(m0 + row_l) * ulong(d.k) + (bc << 7) + col_l);
      threadgroup half4* dst4 = (threadgroup half4*)(tg_a + row_l * 128u + col_l);
      TFQ3_UNROLL for (uint j = 0u; j < 4u; ++j) dst4[j] = src4[j];
    } else {
      threadgroup half4* dst4 = (threadgroup half4*)(tg_a + row_l * 128u + col_l);
      TFQ3_UNROLL for (uint j = 0u; j < 4u; ++j) dst4[j] = half4(0.0h);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 bc_acc[2][4];
    TFQ3_UNROLL for (uint mi = 0u; mi < 2u; ++mi) {
      TFQ3_UNROLL for (uint ni = 0u; ni < 4u; ++ni) {
        bc_acc[mi][ni] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
      }
    }

    if constexpr (BITS == 4u) {
      const device uint4* p0 = (const device uint4*)(w_s0 + ulong(bc) * 8u * 32u);
      const device uint4* p1 = (const device uint4*)(w_s1 + ulong(bc) * 8u * 32u);
      for (uint h = 0u; h < 2u; ++h) {
        uint4 v0 = p0[h * 32u + lane], v1 = p1[h * 32u + lane];
        uint kb = h * 4u;
        half4 q0a, q1a, q0b, q1b;
        tfq3_q8_4b(st0, v0.x, T, q0a, q1a); tfq3_q8_4b(st1, v1.x, T, q0b, q1b);
        tfq3_stage_subtile(tg_b[sg], q0a, q1a, home); tfq3_stage_subtile(tg_b[sg] + 256, q0b, q1b, home);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        tfq3_mma_step16(tg_a, tg_b[sg], tg_b[sg] + 256, kb + 0u, nb, bc_acc);

        tfq3_q8_4b(st0, v0.y, T, q0a, q1a); tfq3_q8_4b(st1, v1.y, T, q0b, q1b);
        tfq3_stage_subtile(tg_b[sg], q0a, q1a, home); tfq3_stage_subtile(tg_b[sg] + 256, q0b, q1b, home);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        tfq3_mma_step16(tg_a, tg_b[sg], tg_b[sg] + 256, kb + 1u, nb, bc_acc);

        tfq3_q8_4b(st0, v0.z, T, q0a, q1a); tfq3_q8_4b(st1, v1.z, T, q0b, q1b);
        tfq3_stage_subtile(tg_b[sg], q0a, q1a, home); tfq3_stage_subtile(tg_b[sg] + 256, q0b, q1b, home);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        tfq3_mma_step16(tg_a, tg_b[sg], tg_b[sg] + 256, kb + 2u, nb, bc_acc);

        tfq3_q8_4b(st0, v0.w, T, q0a, q1a); tfq3_q8_4b(st1, v1.w, T, q0b, q1b);
        tfq3_stage_subtile(tg_b[sg], q0a, q1a, home); tfq3_stage_subtile(tg_b[sg] + 256, q0b, q1b, home);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        tfq3_mma_step16(tg_a, tg_b[sg], tg_b[sg] + 256, kb + 3u, nb, bc_acc);
      }
    } else if constexpr (BITS == 3u) {
      for (uint h = 0u; h < 2u; ++h) {
        const device uint* p0 = w_s0 + (ulong(bc) * 6u + h * 3u) * 32u + lane;
        const device uint* p1 = w_s1 + (ulong(bc) * 6u + h * 3u) * 32u + lane;
        uint a0 = p0[0], a1 = p0[32], a2 = p0[64];
        uint q0 = p1[0], q1 = p1[32], q2 = p1[64];
        uint kb = h * 4u;
        half4 q0a, q1a, q0b, q1b;
        uint m0a = a0 >> 8, m0b = q0 >> 8;
        uint m1a = ((a0 << 16) | (a1 >> 16)) & 0xFFFFFFu, m1b = ((q0 << 16) | (q1 >> 16)) & 0xFFFFFFu;
        uint m2a = ((a1 << 8) | (a2 >> 24)) & 0xFFFFFFu,  m2b = ((q1 << 8) | (q2 >> 24)) & 0xFFFFFFu;
        uint m3a = a2 & 0xFFFFFFu, m3b = q2 & 0xFFFFFFu;

        tfq3_q8_3b(st0, m0a, T, q0a, q1a); tfq3_q8_3b(st1, m0b, T, q0b, q1b);
        tfq3_stage_subtile(tg_b[sg], q0a, q1a, home); tfq3_stage_subtile(tg_b[sg] + 256, q0b, q1b, home);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        tfq3_mma_step16(tg_a, tg_b[sg], tg_b[sg] + 256, kb + 0u, nb, bc_acc);

        tfq3_q8_3b(st0, m1a, T, q0a, q1a); tfq3_q8_3b(st1, m1b, T, q0b, q1b);
        tfq3_stage_subtile(tg_b[sg], q0a, q1a, home); tfq3_stage_subtile(tg_b[sg] + 256, q0b, q1b, home);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        tfq3_mma_step16(tg_a, tg_b[sg], tg_b[sg] + 256, kb + 1u, nb, bc_acc);

        tfq3_q8_3b(st0, m2a, T, q0a, q1a); tfq3_q8_3b(st1, m2b, T, q0b, q1b);
        tfq3_stage_subtile(tg_b[sg], q0a, q1a, home); tfq3_stage_subtile(tg_b[sg] + 256, q0b, q1b, home);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        tfq3_mma_step16(tg_a, tg_b[sg], tg_b[sg] + 256, kb + 2u, nb, bc_acc);

        tfq3_q8_3b(st0, m3a, T, q0a, q1a); tfq3_q8_3b(st1, m3b, T, q0b, q1b);
        tfq3_stage_subtile(tg_b[sg], q0a, q1a, home); tfq3_stage_subtile(tg_b[sg] + 256, q0b, q1b, home);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        tfq3_mma_step16(tg_a, tg_b[sg], tg_b[sg] + 256, kb + 3u, nb, bc_acc);
      }
    } else {
      for (uint h = 0u; h < 2u; ++h) {
        const device uint* p0 = w_s0 + (ulong(bc) * 10u + h * 5u) * 32u + lane;
        const device uint* p1 = w_s1 + (ulong(bc) * 10u + h * 5u) * 32u + lane;
        uint a0 = p0[0], a1 = p0[32], a2 = p0[64], a3 = p0[96], a4 = p0[128];
        uint q0 = p1[0], q1 = p1[32], q2 = p1[64], q3 = p1[96], q4 = p1[128];
        uint kb = h * 4u;
        half4 q0a, q1a, q0b, q1b;

        tfq3_q8_5b(st0, a0 >> 24, (a0 << 8) | (a1 >> 24), T, q0a, q1a);
        tfq3_q8_5b(st1, q0 >> 24, (q0 << 8) | (q1 >> 24), T, q0b, q1b);
        tfq3_stage_subtile(tg_b[sg], q0a, q1a, home); tfq3_stage_subtile(tg_b[sg] + 256, q0b, q1b, home);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        tfq3_mma_step16(tg_a, tg_b[sg], tg_b[sg] + 256, kb + 0u, nb, bc_acc);

        tfq3_q8_5b(st0, (a1 >> 16) & 255u, (a1 << 16) | (a2 >> 16), T, q0a, q1a);
        tfq3_q8_5b(st1, (q1 >> 16) & 255u, (q1 << 16) | (q2 >> 16), T, q0b, q1b);
        tfq3_stage_subtile(tg_b[sg], q0a, q1a, home); tfq3_stage_subtile(tg_b[sg] + 256, q0b, q1b, home);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        tfq3_mma_step16(tg_a, tg_b[sg], tg_b[sg] + 256, kb + 1u, nb, bc_acc);

        tfq3_q8_5b(st0, (a2 >> 8) & 255u, (a2 << 24) | (a3 >> 8), T, q0a, q1a);
        tfq3_q8_5b(st1, (q2 >> 8) & 255u, (q2 << 24) | (q3 >> 8), T, q0b, q1b);
        tfq3_stage_subtile(tg_b[sg], q0a, q1a, home); tfq3_stage_subtile(tg_b[sg] + 256, q0b, q1b, home);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        tfq3_mma_step16(tg_a, tg_b[sg], tg_b[sg] + 256, kb + 2u, nb, bc_acc);

        tfq3_q8_5b(st0, a3 & 255u, a4, T, q0a, q1a);
        tfq3_q8_5b(st1, q3 & 255u, q4, T, q0b, q1b);
        tfq3_stage_subtile(tg_b[sg], q0a, q1a, home); tfq3_stage_subtile(tg_b[sg] + 256, q0b, q1b, home);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        tfq3_mma_step16(tg_a, tg_b[sg], tg_b[sg] + 256, kb + 3u, nb, bc_acc);
      }
    }

    // Store this block's 16x32 partial product into tg_f (safe after barrier since tg_a is done for this bc)
    threadgroup_barrier(mem_flags::mem_threadgroup);
    TFQ3_UNROLL for (uint ni = 0u; ni < 4u; ++ni) {
      simdgroup_store(bc_acc[0][ni], tg_f + 0u * 128u + uint(sg) * 32u + ni * 8u, 128u);
      if (nb > 8u) simdgroup_store(bc_acc[1][ni], tg_f + 8u * 128u + uint(sg) * 32u + ni * 8u, 128u);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const float fsv = float(sc0[bc]);
    for (uint m = 0u; m < nb; ++m) {
      const uint idx = m * 128u + uint(sg) * 32u + uint(lane);
      tg_acc[idx] = fma(tg_f[idx], fsv, tg_acc[idx]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint m = uint(sg); m < nb; m += 4u) {
    const threadgroup float* ys = tg_acc + m * 128u;
    float4 v = float4(ys[lane], ys[lane + 32u], ys[lane + 64u], ys[lane + 96u]);
    tfq3_fwht128_simd4(v, lane);
    uint sr0 = tfq3_sr0(r0, d.sr_split);
    device bfloat* dst = OUT + ulong(m0 + m) * ulong(d.ldo) + r0;
    tfq3_store_out4(dst, lane, v, sr0, d.add_res);
  }
}

template [[host_name("tfq3_dec_fused1_3b")]]  kernel void tfq3_dec_fused1_t<3u>(const device uint*, const device half*, constant Tfq3Tables*, const device half*, device bfloat*, constant Tfq3Dims&, uint, ushort, ushort);
template [[host_name("tfq3_dec_fused1_4b")]]  kernel void tfq3_dec_fused1_t<4u>(const device uint*, const device half*, constant Tfq3Tables*, const device half*, device bfloat*, constant Tfq3Dims&, uint, ushort, ushort);
template [[host_name("tfq3_dec_fused1_5b")]]  kernel void tfq3_dec_fused1_t<5u>(const device uint*, const device half*, constant Tfq3Tables*, const device half*, device bfloat*, constant Tfq3Dims&, uint, ushort, ushort);
template [[host_name("tfq3_dec_fused2_3b")]]  kernel void tfq3_dec_fused2_t<3u>(const device uint*, const device half*, constant Tfq3Tables*, const device half*, device bfloat*, constant Tfq3Dims&, uint, ushort, ushort);
template [[host_name("tfq3_dec_fused2_4b")]]  kernel void tfq3_dec_fused2_t<4u>(const device uint*, const device half*, constant Tfq3Tables*, const device half*, device bfloat*, constant Tfq3Dims&, uint, ushort, ushort);
template [[host_name("tfq3_dec_fused2_5b")]]  kernel void tfq3_dec_fused2_t<5u>(const device uint*, const device half*, constant Tfq3Tables*, const device half*, device bfloat*, constant Tfq3Dims&, uint, ushort, ushort);
template [[host_name("tfq3_dec_fused4_3b")]]  kernel void tfq3_dec_fused4_t<3u>(const device uint*, const device half*, constant Tfq3Tables*, const device half*, device bfloat*, constant Tfq3Dims&, uint, ushort, ushort);
template [[host_name("tfq3_dec_fused4_4b")]]  kernel void tfq3_dec_fused4_t<4u>(const device uint*, const device half*, constant Tfq3Tables*, const device half*, device bfloat*, constant Tfq3Dims&, uint, ushort, ushort);
template [[host_name("tfq3_dec_fused4_5b")]]  kernel void tfq3_dec_fused4_t<5u>(const device uint*, const device half*, constant Tfq3Tables*, const device half*, device bfloat*, constant Tfq3Dims&, uint, ushort, ushort);
template [[host_name("tfq3_dec_2dsk8_1_3b")]] kernel void tfq3_dec_2dsk8_1_t<3u>(const device uint*, const device half*, constant Tfq3Tables*, const device half*, device float*, constant Tfq3Dims&, uint2, ushort, ushort);
template [[host_name("tfq3_dec_2dsk8_1_4b")]] kernel void tfq3_dec_2dsk8_1_t<4u>(const device uint*, const device half*, constant Tfq3Tables*, const device half*, device float*, constant Tfq3Dims&, uint2, ushort, ushort);
template [[host_name("tfq3_dec_2dsk8_1_5b")]] kernel void tfq3_dec_2dsk8_1_t<5u>(const device uint*, const device half*, constant Tfq3Tables*, const device half*, device float*, constant Tfq3Dims&, uint2, ushort, ushort);
template [[host_name("tfq3_dec_2dsk8_2_3b")]] kernel void tfq3_dec_2dsk8_2_t<3u>(const device uint*, const device half*, constant Tfq3Tables*, const device half*, device float*, constant Tfq3Dims&, uint2, ushort, ushort);
template [[host_name("tfq3_dec_2dsk8_2_4b")]] kernel void tfq3_dec_2dsk8_2_t<4u>(const device uint*, const device half*, constant Tfq3Tables*, const device half*, device float*, constant Tfq3Dims&, uint2, ushort, ushort);
template [[host_name("tfq3_dec_2dsk8_2_5b")]] kernel void tfq3_dec_2dsk8_2_t<5u>(const device uint*, const device half*, constant Tfq3Tables*, const device half*, device float*, constant Tfq3Dims&, uint2, ushort, ushort);
template [[host_name("tfq3_dec_2dsk8_4_3b")]] kernel void tfq3_dec_2dsk8_4_t<3u>(const device uint*, const device half*, constant Tfq3Tables*, const device half*, device float*, constant Tfq3Dims&, uint2, ushort, ushort);
template [[host_name("tfq3_dec_2dsk8_4_4b")]] kernel void tfq3_dec_2dsk8_4_t<4u>(const device uint*, const device half*, constant Tfq3Tables*, const device half*, device float*, constant Tfq3Dims&, uint2, ushort, ushort);
template [[host_name("tfq3_dec_2dsk8_4_5b")]] kernel void tfq3_dec_2dsk8_4_t<5u>(const device uint*, const device half*, constant Tfq3Tables*, const device half*, device float*, constant Tfq3Dims&, uint2, ushort, ushort);
template [[host_name("tfq3_prefill_3b")]]     kernel void tfq3_prefill_t<3u>(const device uint*, const device half*, constant Tfq3Tables*, const device half*, device bfloat*, constant Tfq3Dims&, uint2, ushort, ushort, ushort);
template [[host_name("tfq3_prefill_4b")]]     kernel void tfq3_prefill_t<4u>(const device uint*, const device half*, constant Tfq3Tables*, const device half*, device bfloat*, constant Tfq3Dims&, uint2, ushort, ushort, ushort);
template [[host_name("tfq3_prefill_5b")]]     kernel void tfq3_prefill_t<5u>(const device uint*, const device half*, constant Tfq3Tables*, const device half*, device bfloat*, constant Tfq3Dims&, uint2, ushort, ushort, ushort);
