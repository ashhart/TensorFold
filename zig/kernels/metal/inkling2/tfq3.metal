// Inkling 2 Mini TFQ3 (3/4/5-bit 2D-AffineCoset/half4 + 2D FWHT-128) routed MoE kernels.
#include <metal_stdlib>
using namespace metal;

#define PW_UNROLL _Pragma("clang loop unroll(full)")

struct Ink2Tfq3Args {
  uint  rows;
  uint  hidden;
  uint  inter;
  uint  experts;
  uint  topk;
  float gate_cap;
  float up_cap;
  uint  pad;
};

struct PwExpertMeta {
  uint  b_bits;
  uint  heat_rank;
  ulong gate_word_off;
  ulong up_word_off;
  ulong down_word_off;
  uint  gate_sc_off;
  uint  up_sc_off;
  uint  down_sc_off;
  float gate_snr_db;
  float up_snr_db;
  float down_snr_db;
  uint  reserved[2];
};

struct PwV3Tables {
  uint  magic;
  uint  version;
  float a3, b3, a4, b4, a5, b5;
  uint  pad0[8];
  uint  lut3[256];   // off 64   (1 KB)
  uint  lut4[1024];  // off 1088 (4 KB)
  uint  lut5[1024];  // off 5184 (4 KB)
  half4 c64_3[64];   // off 9280 (512 B)
  half4 c64_4[64];   // off 9792 (512 B)
  half4 c64_5[64];   // off 10304 (512 B)
};

inline short2 pw_frag_home(ushort l) {
  return short2(short((l & 8) + ((l & 1) << 2)), short(((l & 16) >> 2) | ((l >> 1) & 3)));
}

inline float pw_sgn(uint seed, uint i) {
  uint h = (seed ^ i) * 2654435761u;
  return ((h >> 31) & 1u) ? -1.0f : 1.0f;
}

inline void pw_fwht128_simd4(thread float4& v, ushort lane) {
  PW_UNROLL
  for (ushort d = 1; d <= 16; d <<= 1) {
    float4 o = simd_shuffle_xor(v, d);
    v = ((lane & d) == 0u) ? (v + o) : (o - v);
  }
  float a0 = v.x + v.y, b0 = v.x - v.y;
  float a1 = v.z + v.w, b1 = v.z - v.w;
  v = float4(a0 + a1, b0 + b1, a0 - a1, b0 - b1) * 0.08838834764831845f;
}

// Inkling 2 softcapped SwiGLU: matches zig/kernels/metal/inkling2/experts.metal:ink2_act() bit-for-bit.
inline float ink2_tfq3_cap(float x, float cap) {
  return cap * metal::precise::tanh(metal::precise::divide(x, cap));
}

inline float ink2_tfq3_glu_softcap(float g, float u, float gate_cap, float up_cap) {
  const float gate = float(bfloat(g));
  const float up = float(bfloat(u));
  const float left = ink2_tfq3_cap(gate, gate_cap) * metal::precise::divide(1.0f, 1.0f + metal::precise::exp(-gate));
  return float(bfloat(left * ink2_tfq3_cap(up, up_cap)));
}

// Core v3 2D-AffineCoset/half4 decoders (bit-exact match to encoder and glm_tfq3/common.metal).
inline void pw_v3_q8_4b(thread uint& st, uint wc, constant PwV3Tables* T, thread half4& q0, thread half4& q1) {
  uint s0 = st;
  uint h0 = s0 * 0x9E3779B1u;
  uint h1 = ((s0 <<  8) | (wc >> 24)) * 0x9E3779B1u;
  uint h2 = ((s0 << 16) | (wc >> 16)) * 0x9E3779B1u;
  uint h3 = ((s0 << 24) | (wc >>  8)) * 0x9E3779B1u;
  st = wc;
  half4 c0 = T->c64_4[(h0 >> 16) & 63u], c2 = T->c64_4[(h2 >> 16) & 63u];
  q0 = fma(half4(as_type<half2>(T->lut4[(((wc >> 24)         << 2) ^ (h0 >> 22)) & 1023u]),
                 as_type<half2>(T->lut4[((((wc >> 16) & 255u) << 2) ^ (h1 >> 22)) & 1023u])), c0.xyxy, c0.zwzw);
  q1 = fma(half4(as_type<half2>(T->lut4[((((wc >>  8) & 255u) << 2) ^ (h2 >> 22)) & 1023u]),
                 as_type<half2>(T->lut4[((( wc        & 255u) << 2) ^ (h3 >> 22)) & 1023u])), c2.xyxy, c2.zwzw);
}

inline void pw_v3_q8_3b(thread uint& st, uint m24, constant PwV3Tables* T, thread half4& q0, thread half4& q1) {
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

inline void pw_v3_q8_5b(thread uint& st, uint hi8, uint lo32, constant PwV3Tables* T, thread half4& q0, thread half4& q1) {
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

inline float pw_fdot(half4 q, half4 x) { return float(dot(q, x)); }

#define PW_MDOT1(Q0, Q1, X, CB, BA0, BA1) { \
  half4 xv = *(const device half4*)((X) + (CB)); \
  (BA0) += pw_fdot((Q0), xv); (BA1) += pw_fdot((Q1), xv); }

template <uint N_BC>
inline void pw_mv_nosplit1_5b(
    const device uint* w_base, const device half* sc_base, constant PwV3Tables* T,
    const device half* X_r, ushort lane, short2 home, thread float& acc0, thread float& acc1
) {
  uint st = 0x13579BDFu ^ (uint(lane) * 0x9E3779B1u);
  acc0 = 0.0f; acc1 = 0.0f;
  for (uint bc = 0u; bc < N_BC; ++bc) {
    float ba0 = 0.0f, ba1 = 0.0f;
    for (uint h = 0u; h < 2u; ++h) {
      const device uint* blk = w_base + (ulong(bc) * 10u + h * 5u) * 32u + lane;
      uint w0 = blk[0], w1 = blk[32], w2 = blk[64], w3 = blk[96], w4 = blk[128];
      uint cb = (bc << 7) + (h << 6) + uint(home.x);
      half4 q0, q1;
      pw_v3_q8_5b(st, w0 >> 24,          (w0 <<  8) | (w1 >> 24), T, q0, q1); PW_MDOT1(q0, q1, X_r, cb +  0u, ba0, ba1);
      pw_v3_q8_5b(st, (w1 >> 16) & 255u, (w1 << 16) | (w2 >> 16), T, q0, q1); PW_MDOT1(q0, q1, X_r, cb + 16u, ba0, ba1);
      pw_v3_q8_5b(st, (w2 >>  8) & 255u, (w2 << 24) | (w3 >>  8), T, q0, q1); PW_MDOT1(q0, q1, X_r, cb + 32u, ba0, ba1);
      pw_v3_q8_5b(st, w3 & 255u,          w4,                     T, q0, q1); PW_MDOT1(q0, q1, X_r, cb + 48u, ba0, ba1);
    }
    float sc = float(sc_base[bc]);
    acc0 = fma(ba0, sc, acc0); acc1 = fma(ba1, sc, acc1);
  }
  acc0 += simd_shuffle_xor(acc0, 1); acc0 += simd_shuffle_xor(acc0, 8);
  acc1 += simd_shuffle_xor(acc1, 1); acc1 += simd_shuffle_xor(acc1, 8);
}

template <uint N_BC, bool HAS_5B>
inline void pw_mv_nosplit1(
    const device uint* w_base, const device half* sc_base, uint b_bits, constant PwV3Tables* T,
    const device half* X_r, ushort lane, short2 home, thread float& acc0, thread float& acc1
) {
  uint st = 0x13579BDFu ^ (uint(lane) * 0x9E3779B1u);
  acc0 = 0.0f; acc1 = 0.0f;
  if (b_bits == 4u) {
    const device uint4* w4 = (const device uint4*)w_base;
    for (uint bc = 0u; bc < N_BC; ++bc) {
      float ba0 = 0.0f, ba1 = 0.0f;
      uint4 v = w4[(bc * 2u + 0u) * 32u + lane];
      uint cb = (bc << 7) + uint(home.x);
      half4 q0, q1;
      pw_v3_q8_4b(st, v.x, T, q0, q1); PW_MDOT1(q0, q1, X_r, cb +   0u, ba0, ba1);
      pw_v3_q8_4b(st, v.y, T, q0, q1); PW_MDOT1(q0, q1, X_r, cb +  16u, ba0, ba1);
      pw_v3_q8_4b(st, v.z, T, q0, q1); PW_MDOT1(q0, q1, X_r, cb +  32u, ba0, ba1);
      pw_v3_q8_4b(st, v.w, T, q0, q1); PW_MDOT1(q0, q1, X_r, cb +  48u, ba0, ba1);
      v = w4[(bc * 2u + 1u) * 32u + lane];
      pw_v3_q8_4b(st, v.x, T, q0, q1); PW_MDOT1(q0, q1, X_r, cb +  64u, ba0, ba1);
      pw_v3_q8_4b(st, v.y, T, q0, q1); PW_MDOT1(q0, q1, X_r, cb +  80u, ba0, ba1);
      pw_v3_q8_4b(st, v.z, T, q0, q1); PW_MDOT1(q0, q1, X_r, cb +  96u, ba0, ba1);
      pw_v3_q8_4b(st, v.w, T, q0, q1); PW_MDOT1(q0, q1, X_r, cb + 112u, ba0, ba1);
      float sc = float(sc_base[bc]);
      acc0 = fma(ba0, sc, acc0); acc1 = fma(ba1, sc, acc1);
    }
  } else if (HAS_5B && b_bits == 5u) {
    pw_mv_nosplit1_5b<N_BC>(w_base, sc_base, T, X_r, lane, home, acc0, acc1);
    return;
  } else {
    for (uint bc = 0u; bc < N_BC; ++bc) {
      float ba0 = 0.0f, ba1 = 0.0f;
      for (uint h = 0u; h < 2u; ++h) {
        const device uint* blk = w_base + (ulong(bc) * 6u + h * 3u) * 32u + lane;
        uint w0 = blk[0], w1 = blk[32], w2 = blk[64];
        uint cb = (bc << 7) + (h << 6) + uint(home.x);
        half4 q0, q1;
        pw_v3_q8_3b(st, w0 >> 8,                               T, q0, q1); PW_MDOT1(q0, q1, X_r, cb +  0u, ba0, ba1);
        pw_v3_q8_3b(st, ((w0 << 16) | (w1 >> 16)) & 0xFFFFFFu, T, q0, q1); PW_MDOT1(q0, q1, X_r, cb + 16u, ba0, ba1);
        pw_v3_q8_3b(st, ((w1 <<  8) | (w2 >> 24)) & 0xFFFFFFu, T, q0, q1); PW_MDOT1(q0, q1, X_r, cb + 32u, ba0, ba1);
        pw_v3_q8_3b(st, w2 & 0xFFFFFFu,                        T, q0, q1); PW_MDOT1(q0, q1, X_r, cb + 48u, ba0, ba1);
      }
      float sc = float(sc_base[bc]);
      acc0 = fma(ba0, sc, acc0); acc1 = fma(ba1, sc, acc1);
    }
  }
  acc0 += simd_shuffle_xor(acc0, 1); acc0 += simd_shuffle_xor(acc0, 8);
  acc1 += simd_shuffle_xor(acc1, 1); acc1 += simd_shuffle_xor(acc1, 8);
}

// Forward SEED_COL + FWHT-128 on BF16 X[rows, 2304] -> FP16 X_rot[rows, 2304].
// Grid: (18, rows, 1), threadgroup: (32, 1, 1).
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_ink2_tfq3_fwht_x(
    const device bfloat* X   [[buffer(0)]],
    device half* X_rot       [[buffer(1)]],
    constant Ink2Tfq3Args& a [[buffer(2)]],
    uint2 gid                [[threadgroup_position_in_grid]],
    ushort lane              [[thread_index_in_simdgroup]]
) {
  const uint row = gid.y, c0 = gid.x * 128u;
  const device bfloat* src = X + ulong(row) * ulong(a.hidden) + c0;
  float4 v;
  v.x = float(src[lane +  0u]) * pw_sgn(0x12345678u, c0 + lane +  0u);
  v.y = float(src[lane + 32u]) * pw_sgn(0x12345678u, c0 + lane + 32u);
  v.z = float(src[lane + 64u]) * pw_sgn(0x12345678u, c0 + lane + 64u);
  v.w = float(src[lane + 96u]) * pw_sgn(0x12345678u, c0 + lane + 96u);
  pw_fwht128_simd4(v, lane);
  device half* dst = X_rot + ulong(row) * ulong(a.hidden) + c0;
  dst[lane +  0u] = half(v.x);
  dst[lane + 32u] = half(v.y);
  dst[lane + 64u] = half(v.z);
  dst[lane + 96u] = half(v.w);
}

// Fused MoE gate+up (N_BC = 18, 3/4/5b) -> inv-FWHT-128 -> Inkling-2 softcapped SwiGLU -> fwd-FWHT-128 -> FP16 act_rot[rows * 8, 1152].
// Grid: (9, rows * 8, 1), threadgroup: (256, 1, 1).
[[kernel, max_total_threads_per_threadgroup(256)]] void tf_ink2_tfq3_moe_gu_softcap(
    const device uint* W           [[buffer(0)]],
    const device half* scales      [[buffer(1)]],
    const device PwExpertMeta* M   [[buffer(2)]],
    constant PwV3Tables* T         [[buffer(3)]],
    const device half* X_rot       [[buffer(4)]],
    const device int* picks        [[buffer(5)]],
    device half* act_rot           [[buffer(6)]],
    constant Ink2Tfq3Args& a       [[buffer(7)]],
    uint2 gid                      [[threadgroup_position_in_grid]],
    ushort sg                      [[simdgroup_index_in_threadgroup]],
    ushort lane                    [[thread_index_in_simdgroup]]
) {
  constexpr uint N_BC = 18u;
  const uint br = gid.x, slot_flat = gid.y;
  const uint row = slot_flat / a.topk;
  const int eid = picks[slot_flat];
  if (eid < 0 || uint(eid) >= a.experts) return;

  const PwExpertMeta em = M[uint(eid)];
  const uint bits = em.b_bits;
  const ulong wpb = ulong(bits) * 2ul;
  const uint strip = br * 8u + uint(sg);
  const uint r0 = br * 128u;
  const short2 home = pw_frag_home(lane);
  const device half* xr = X_rot + ulong(row) * ulong(a.hidden);
  const device uint* wg = W + em.gate_word_off + ulong(strip) * ulong(N_BC) * wpb * 32ul;
  const device uint* wu = W + em.up_word_off   + ulong(strip) * ulong(N_BC) * wpb * 32ul;
  const device half* sg_p = scales + ulong(em.gate_sc_off) + ulong(br) * ulong(N_BC);
  const device half* su_p = scales + ulong(em.up_sc_off)   + ulong(br) * ulong(N_BC);

  threadgroup float tg_g[128], tg_u[128];
  float g0, g1, u0, u1;
  pw_mv_nosplit1<N_BC, true>(wg, sg_p, bits, T, xr, lane, home, g0, g1);
  pw_mv_nosplit1<N_BC, true>(wu, su_p, bits, T, xr, lane, home, u0, u1);
  if ((lane & 9u) == 0u) {
    const uint idx = uint(sg) * 16u + uint(home.y);
    tg_g[idx] = g0; tg_g[idx + 8u] = g1;
    tg_u[idx] = u0; tg_u[idx + 8u] = u1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u) {
    float4 gv = float4(tg_g[lane], tg_g[lane + 32u], tg_g[lane + 64u], tg_g[lane + 96u]);
    float4 uv = float4(tg_u[lane], tg_u[lane + 32u], tg_u[lane + 64u], tg_u[lane + 96u]);
    pw_fwht128_simd4(gv, lane);
    pw_fwht128_simd4(uv, lane);
    float4 z;
    z.x = ink2_tfq3_glu_softcap(gv.x * pw_sgn(0x9E3779B9u, r0 + lane +  0u), uv.x * pw_sgn(0x9E3779B9u, r0 + lane +  0u), a.gate_cap, a.up_cap) * pw_sgn(0x12345678u, r0 + lane +  0u);
    z.y = ink2_tfq3_glu_softcap(gv.y * pw_sgn(0x9E3779B9u, r0 + lane + 32u), uv.y * pw_sgn(0x9E3779B9u, r0 + lane + 32u), a.gate_cap, a.up_cap) * pw_sgn(0x12345678u, r0 + lane + 32u);
    z.z = ink2_tfq3_glu_softcap(gv.z * pw_sgn(0x9E3779B9u, r0 + lane + 64u), uv.z * pw_sgn(0x9E3779B9u, r0 + lane + 64u), a.gate_cap, a.up_cap) * pw_sgn(0x12345678u, r0 + lane + 64u);
    z.w = ink2_tfq3_glu_softcap(gv.w * pw_sgn(0x9E3779B9u, r0 + lane + 96u), uv.w * pw_sgn(0x9E3779B9u, r0 + lane + 96u), a.gate_cap, a.up_cap) * pw_sgn(0x12345678u, r0 + lane + 96u);
    pw_fwht128_simd4(z, lane);
    device half* dst = act_rot + ulong(slot_flat) * ulong(a.inter) + r0;
    dst[lane +  0u] = half(z.x);
    dst[lane + 32u] = half(z.y);
    dst[lane + 64u] = half(z.z);
    dst[lane + 96u] = half(z.w);
  }
}

// Fused MoE down (N_BC = 9, 3/4/5b) -> inv-FWHT-128 + SEED_ROW -> BF16 down_out[rows * 8, 2304].
// Grid: (18, rows * 8, 1), threadgroup: (256, 1, 1).
[[kernel, max_total_threads_per_threadgroup(256)]] void tf_ink2_tfq3_moe_down(
    const device uint* W           [[buffer(0)]],
    const device half* scales      [[buffer(1)]],
    const device PwExpertMeta* M   [[buffer(2)]],
    constant PwV3Tables* T         [[buffer(3)]],
    const device half* act_rot     [[buffer(4)]],
    const device int* picks        [[buffer(5)]],
    device bfloat* down_out        [[buffer(6)]],
    constant Ink2Tfq3Args& a       [[buffer(7)]],
    uint2 gid                      [[threadgroup_position_in_grid]],
    ushort sg                      [[simdgroup_index_in_threadgroup]],
    ushort lane                    [[thread_index_in_simdgroup]]
) {
  constexpr uint N_BC = 9u;
  const uint br = gid.x, slot_flat = gid.y;
  const int eid = picks[slot_flat];
  if (eid < 0 || uint(eid) >= a.experts) return;

  const PwExpertMeta em = M[uint(eid)];
  const uint bits = em.b_bits;
  const ulong wpb = ulong(bits) * 2ul;
  const uint strip = br * 8u + uint(sg);
  const uint r0 = br * 128u;
  const short2 home = pw_frag_home(lane);
  const device half* ar = act_rot + ulong(slot_flat) * ulong(a.inter);
  const device uint* wd = W + em.down_word_off + ulong(strip) * ulong(N_BC) * wpb * 32ul;
  const device half* sd_p = scales + ulong(em.down_sc_off) + ulong(br) * ulong(N_BC);

  threadgroup float tg_y[128];
  float a0, a1;
  pw_mv_nosplit1<N_BC, true>(wd, sd_p, bits, T, ar, lane, home, a0, a1);
  if ((lane & 9u) == 0u) {
    const uint idx = uint(sg) * 16u + uint(home.y);
    tg_y[idx] = a0; tg_y[idx + 8u] = a1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u) {
    float4 v = float4(tg_y[lane], tg_y[lane + 32u], tg_y[lane + 64u], tg_y[lane + 96u]);
    pw_fwht128_simd4(v, lane);
    device bfloat* dst = down_out + ulong(slot_flat) * ulong(a.hidden) + r0;
    dst[lane +  0u] = bfloat(v.x * pw_sgn(0x9E3779B9u, r0 + lane +  0u));
    dst[lane + 32u] = bfloat(v.y * pw_sgn(0x9E3779B9u, r0 + lane + 32u));
    dst[lane + 64u] = bfloat(v.z * pw_sgn(0x9E3779B9u, r0 + lane + 64u));
    dst[lane + 96u] = bfloat(v.w * pw_sgn(0x9E3779B9u, r0 + lane + 96u));
  }
}
