// Inkling 2 Mini TFQ4 (2.001/2.251 bpw 8D E8P) Dense & Hybrid TFPW MoE (2/225/3/4/5b) Metal kernels.

struct Ink2Tfq4DenseArgs {
  uint rows;
  uint n;
  uint k;
  uint ldo;
  uint bits;
  uint table_kind;
  uint row_off;
  uint sr_split;
  uint out_f32;
  uint pad;
};

inline uchar ink2_tfq4_e8p_parity(uint4 u) {
  half4 a = as_type<half4>(u.xy);
  half4 b = as_type<half4>(u.zw);
  uint s = uint(int(floor(float(a.x)))) + uint(int(floor(float(a.y))))
         + uint(int(floor(float(a.z)))) + uint(int(floor(float(a.w))))
         + uint(int(floor(float(b.x)))) + uint(int(floor(float(b.y))))
         + uint(int(floor(float(b.z)))) + uint(int(floor(float(b.w))));
  return uchar(s & 1u);
}

inline float ink2_tfq4_xsum8_qtr(uint4 xv) {
  half4 a = as_type<half4>(xv.xy);
  half4 b = as_type<half4>(xv.zw);
  return (float(a.x) + float(a.y) + float(a.z) + float(a.w)
        + float(b.x) + float(b.y) + float(b.z) + float(b.w)) * 0.25f;
}

inline float ink2_tfq4_dot8(
    uint4 lut_u, uint par, uint code16, uint4 xv, float xsum_qtr, bool is_true_e8p
) {
  uint smask;
  if (is_true_e8p) {
    uint s7 = (code16 >> 8) & 0x7Fu;
    uint s7_bit = (popcount(s7) ^ par) & 1u;
    smask = s7 | (s7_bit << 7);
  } else {
    smask = code16 >> 8;
  }
  half4 la = as_type<half4>(lut_u.xy);
  half4 lb = as_type<half4>(lut_u.zw);
  half4 xa = as_type<half4>(xv.xy);
  half4 xb = as_type<half4>(xv.zw);
  float4 wa = float4(
    (smask &   1u) ? -float(la.x) : float(la.x),
    (smask &   2u) ? -float(la.y) : float(la.y),
    (smask &   4u) ? -float(la.z) : float(la.z),
    (smask &   8u) ? -float(la.w) : float(la.w));
  float4 wb = float4(
    (smask &  16u) ? -float(lb.x) : float(lb.x),
    (smask &  32u) ? -float(lb.y) : float(lb.y),
    (smask &  64u) ? -float(lb.z) : float(lb.z),
    (smask & 128u) ? -float(lb.w) : float(lb.w));
  float d = dot(wa, float4(xa)) + dot(wb, float4(xb));
  if (is_true_e8p) {
    d -= xsum_qtr;
    if ((code16 & 0x8000u) != 0u) d = -d;
  }
  return d;
}

inline float ink2_tfq4_dot16_e8p(
    const device uint* blk, uint sub_w, bool is_225, bool is_true_e8p,
    threadgroup const uint4* s_lut, threadgroup const uchar* s_par,
    uint4 xv0, uint4 xv1, float xs0, float xs1
) {
  uint packed = blk[sub_w];
  uint c0 = packed & 0xFFFFu;
  uint c1 = packed >> 16;
  uint idx0 = c0 & 255u;
  uint idx1 = c1 & 255u;
  if (is_225) {
    uint hi = blk[8];
    idx0 |= ((hi >> (sub_w * 4u)) & 3u) << 8;
    idx1 |= ((hi >> (sub_w * 4u + 2u)) & 3u) << 8;
  }
  uint p0 = is_true_e8p ? uint(s_par[idx0]) : 0u;
  uint p1 = is_true_e8p ? uint(s_par[idx1]) : 0u;
  return ink2_tfq4_dot8(s_lut[idx0], p0, c0, xv0, xs0, is_true_e8p)
       + ink2_tfq4_dot8(s_lut[idx1], p1, c1, xv1, xs1, is_true_e8p);
}

inline float ink2_tfq4_reduce8(float v) {
  v += simd_shuffle_xor(v, 1);
  v += simd_shuffle_xor(v, 2);
  v += simd_shuffle_xor(v, 4);
  return v;
}

inline void ink2_tfq4_mv1_tfq3(
    const device uint* w_base, const device half* sc_base, uint b_bits, uint n_bc,
    constant PwV3Tables* T, const device half* X_r, ushort lane, short2 home,
    thread float& acc0, thread float& acc1
) {
  uint st = 0x13579BDFu ^ (uint(lane) * 0x9E3779B1u);
  acc0 = 0.0f;
  acc1 = 0.0f;
  if (b_bits == 4u) {
    const device uint4* w4 = (const device uint4*)w_base;
    for (uint bc = 0u; bc < n_bc; ++bc) {
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
      acc0 = fma(ba0, sc, acc0);
      acc1 = fma(ba1, sc, acc1);
    }
  } else if (b_bits == 5u) {
    for (uint bc = 0u; bc < n_bc; ++bc) {
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
      acc0 = fma(ba0, sc, acc0);
      acc1 = fma(ba1, sc, acc1);
    }
  } else {
    for (uint bc = 0u; bc < n_bc; ++bc) {
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
      acc0 = fma(ba0, sc, acc0);
      acc1 = fma(ba1, sc, acc1);
    }
  }
  acc0 += simd_shuffle_xor(acc0, 1); acc0 += simd_shuffle_xor(acc0, 8);
  acc1 += simd_shuffle_xor(acc1, 1); acc1 += simd_shuffle_xor(acc1, 8);
}

// Hybrid MoE gate+up (2/225/3/4/5b) -> softcapped SwiGLU -> act_rot; grid (inter/128, rows*topk), tg (256).
[[kernel, max_total_threads_per_threadgroup(256)]] void tf_ink2_tfq4_moe_gu_softcap(
    const device uint* W           [[buffer(0)]],
    const device half* scales      [[buffer(1)]],
    const device PwExpertMeta* M   [[buffer(2)]],
    constant PwV3Tables* T         [[buffer(3)]],
    const device uint4* T4         [[buffer(4)]],
    const device half* X_rot       [[buffer(5)]],
    const device int* picks        [[buffer(6)]],
    device half* act_rot           [[buffer(7)]],
    constant Ink2Tfq3Args& a       [[buffer(8)]],
    uint2 gid                      [[threadgroup_position_in_grid]],
    ushort tid                     [[thread_index_in_threadgroup]],
    ushort sg                      [[simdgroup_index_in_threadgroup]],
    ushort lane                    [[thread_index_in_simdgroup]]
) {
  const uint br = gid.x, slot_flat = gid.y;
  const uint row = slot_flat / a.topk;
  const int eid = picks[slot_flat];
  if (eid < 0 || uint(eid) >= a.experts) return;

  const PwExpertMeta em = M[uint(eid)];
  const uint bits = em.b_bits;
  const uint n_bc = a.hidden >> 7;
  const uint r0 = br * 128u;
  const device half* xr = X_rot + ulong(row) * ulong(a.hidden);
  const device half* sg_p = scales + ulong(em.gate_sc_off) + ulong(br) * ulong(n_bc);
  const device half* su_p = scales + ulong(em.up_sc_off)   + ulong(br) * ulong(n_bc);

  threadgroup uint4 s_lut[1024];
  threadgroup uchar s_par[1024];
  threadgroup uint4 s_x[16];
  threadgroup float s_xsum[16];
  threadgroup float tg_g[128], tg_u[128];

  if (bits == 2u || bits == 225u) {
    const bool is_225 = (bits == 225u);
    const bool is_true_e8p = (a.table4_kind >= 2u);
    const uint k_codes = is_225 ? 1024u : 256u;
    for (uint i = tid; i < k_codes; i += 256u) {
      uint4 u = T4[i];
      s_lut[i] = u;
      if (is_true_e8p) s_par[i] = ink2_tfq4_e8p_parity(u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint sub_r = uint(lane) >> 3;
    const uint sub_w = uint(lane) & 7u;
    const uint wpb_row = is_225 ? 9u : 8u;
    const uint wpr = n_bc * wpb_row;
    float ag[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float au[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint bc = 0u; bc < n_bc; ++bc) {
      if (tid < 16u) {
        uint4 xv = *(const device uint4*)(xr + (bc << 7) + (uint(tid) << 3));
        s_x[tid] = xv;
        if (is_true_e8p) s_xsum[tid] = ink2_tfq4_xsum8_qtr(xv);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const float fsg = float(sg_p[bc]);
      const float fsu = float(su_p[bc]);
      const float xs0 = is_true_e8p ? s_xsum[sub_w * 2u] : 0.0f;
      const float xs1 = is_true_e8p ? s_xsum[sub_w * 2u + 1u] : 0.0f;
      const uint4 xv0 = s_x[sub_w * 2u], xv1 = s_x[sub_w * 2u + 1u];
      PW_UNROLL
      for (uint pass = 0u; pass < 4u; ++pass) {
        const uint global_r = r0 + uint(sg) * 16u + pass * 4u + sub_r;
        const ulong roff = ulong(global_r) * ulong(wpr) + ulong(bc) * ulong(wpb_row);
        float dg = ink2_tfq4_dot16_e8p(W + em.gate_word_off + roff, sub_w, is_225, is_true_e8p, s_lut, s_par, xv0, xv1, xs0, xs1);
        float du = ink2_tfq4_dot16_e8p(W + em.up_word_off   + roff, sub_w, is_225, is_true_e8p, s_lut, s_par, xv0, xv1, xs0, xs1);
        ag[pass] = fma(dg, fsg, ag[pass]);
        au[pass] = fma(du, fsu, au[pass]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    PW_UNROLL
    for (uint pass = 0u; pass < 4u; ++pass) {
      float vg = ink2_tfq4_reduce8(ag[pass]);
      float vu = ink2_tfq4_reduce8(au[pass]);
      if (sub_w == 0u) {
        const uint local_r = uint(sg) * 16u + pass * 4u + sub_r;
        tg_g[local_r] = vg;
        tg_u[local_r] = vu;
      }
    }
  } else {
    const ulong wpb = ulong(bits) * 2ul;
    const uint strip = br * 8u + uint(sg);
    const short2 home = pw_frag_home(lane);
    const device uint* wg = W + em.gate_word_off + ulong(strip) * ulong(n_bc) * wpb * 32ul;
    const device uint* wu = W + em.up_word_off   + ulong(strip) * ulong(n_bc) * wpb * 32ul;
    float g0, g1, u0, u1;
    ink2_tfq4_mv1_tfq3(wg, sg_p, bits, n_bc, T, xr, lane, home, g0, g1);
    ink2_tfq4_mv1_tfq3(wu, su_p, bits, n_bc, T, xr, lane, home, u0, u1);
    if ((lane & 9u) == 0u) {
      const uint idx = uint(sg) * 16u + uint(home.y);
      tg_g[idx] = g0; tg_g[idx + 8u] = g1;
      tg_u[idx] = u0; tg_u[idx + 8u] = u1;
    }
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

// Hybrid MoE down (2/225/3/4/5b) -> inv-FWHT-128 -> BF16 down_out; grid (hidden/128, rows*topk), tg (256).
[[kernel, max_total_threads_per_threadgroup(256)]] void tf_ink2_tfq4_moe_down(
    const device uint* W           [[buffer(0)]],
    const device half* scales      [[buffer(1)]],
    const device PwExpertMeta* M   [[buffer(2)]],
    constant PwV3Tables* T         [[buffer(3)]],
    const device uint4* T4         [[buffer(4)]],
    const device half* act_rot     [[buffer(5)]],
    const device int* picks        [[buffer(6)]],
    device bfloat* down_out        [[buffer(7)]],
    constant Ink2Tfq3Args& a       [[buffer(8)]],
    uint2 gid                      [[threadgroup_position_in_grid]],
    ushort tid                     [[thread_index_in_threadgroup]],
    ushort sg                      [[simdgroup_index_in_threadgroup]],
    ushort lane                    [[thread_index_in_simdgroup]]
) {
  const uint br = gid.x, slot_flat = gid.y;
  const int eid = picks[slot_flat];
  if (eid < 0 || uint(eid) >= a.experts) return;

  const PwExpertMeta em = M[uint(eid)];
  const uint bits = em.b_bits;
  const uint n_bc = a.inter >> 7;
  const uint r0 = br * 128u;
  const device half* ar = act_rot + ulong(slot_flat) * ulong(a.inter);
  const device half* sd_p = scales + ulong(em.down_sc_off) + ulong(br) * ulong(n_bc);

  threadgroup uint4 s_lut[1024];
  threadgroup uchar s_par[1024];
  threadgroup uint4 s_x[16];
  threadgroup float s_xsum[16];
  threadgroup float tg_y[128];

  if (bits == 2u || bits == 225u) {
    const bool is_225 = (bits == 225u);
    const bool is_true_e8p = (a.table4_kind >= 2u);
    const uint k_codes = is_225 ? 1024u : 256u;
    for (uint i = tid; i < k_codes; i += 256u) {
      uint4 u = T4[i];
      s_lut[i] = u;
      if (is_true_e8p) s_par[i] = ink2_tfq4_e8p_parity(u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint sub_r = uint(lane) >> 3;
    const uint sub_w = uint(lane) & 7u;
    const uint wpb_row = is_225 ? 9u : 8u;
    const uint wpr = n_bc * wpb_row;
    float ay[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint bc = 0u; bc < n_bc; ++bc) {
      if (tid < 16u) {
        uint4 xv = *(const device uint4*)(ar + (bc << 7) + (uint(tid) << 3));
        s_x[tid] = xv;
        if (is_true_e8p) s_xsum[tid] = ink2_tfq4_xsum8_qtr(xv);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const float fsd = float(sd_p[bc]);
      const float xs0 = is_true_e8p ? s_xsum[sub_w * 2u] : 0.0f;
      const float xs1 = is_true_e8p ? s_xsum[sub_w * 2u + 1u] : 0.0f;
      const uint4 xv0 = s_x[sub_w * 2u], xv1 = s_x[sub_w * 2u + 1u];
      PW_UNROLL
      for (uint pass = 0u; pass < 4u; ++pass) {
        const uint global_r = r0 + uint(sg) * 16u + pass * 4u + sub_r;
        const ulong roff = ulong(global_r) * ulong(wpr) + ulong(bc) * ulong(wpb_row);
        float dy = ink2_tfq4_dot16_e8p(W + em.down_word_off + roff, sub_w, is_225, is_true_e8p, s_lut, s_par, xv0, xv1, xs0, xs1);
        ay[pass] = fma(dy, fsd, ay[pass]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    PW_UNROLL
    for (uint pass = 0u; pass < 4u; ++pass) {
      float vy = ink2_tfq4_reduce8(ay[pass]);
      if (sub_w == 0u) tg_y[uint(sg) * 16u + pass * 4u + sub_r] = vy;
    }
  } else {
    const ulong wpb = ulong(bits) * 2ul;
    const uint strip = br * 8u + uint(sg);
    const short2 home = pw_frag_home(lane);
    const device uint* wd = W + em.down_word_off + ulong(strip) * ulong(n_bc) * wpb * 32ul;
    float a0, a1;
    ink2_tfq4_mv1_tfq3(wd, sd_p, bits, n_bc, T, ar, lane, home, a0, a1);
    if ((lane & 9u) == 0u) {
      const uint idx = uint(sg) * 16u + uint(home.y);
      tg_y[idx] = a0; tg_y[idx + 8u] = a1;
    }
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

// Standalone TFQ4 dense (b_bits 2/225, table_kind 0..3) -> inv-FWHT-128 -> BF16/FP32; grid (n/128, rows), tg (256).
[[kernel, max_total_threads_per_threadgroup(256)]] void tf_ink2_tfq4_dense(
    const device uint* W           [[buffer(0)]],
    const device half* scales      [[buffer(1)]],
    const device uint4* T4         [[buffer(2)]],
    const device half* X_rot       [[buffer(3)]],
    device uchar* out_raw          [[buffer(4)]],
    constant Ink2Tfq4DenseArgs& a  [[buffer(5)]],
    uint2 gid                      [[threadgroup_position_in_grid]],
    ushort tid                     [[thread_index_in_threadgroup]],
    ushort sg                      [[simdgroup_index_in_threadgroup]],
    ushort lane                    [[thread_index_in_simdgroup]]
) {
  const uint br = gid.x, row = gid.y;
  if (row >= a.rows || br * 128u >= a.n) return;
  const bool is_225 = (a.bits == 225u);
  const bool is_true_e8p = (a.table_kind >= 2u);
  const uint k_codes = is_225 ? 1024u : 256u;
  const uint n_bc = a.k >> 7;
  const uint r0 = br * 128u;

  threadgroup uint4 s_lut[1024];
  threadgroup uchar s_par[1024];
  threadgroup uint4 s_x[16];
  threadgroup float s_xsum[16];
  threadgroup float tg_y[128];

  for (uint i = tid; i < k_codes; i += 256u) {
    uint4 u = T4[i];
    s_lut[i] = u;
    if (is_true_e8p) s_par[i] = ink2_tfq4_e8p_parity(u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint sub_r = uint(lane) >> 3;
  const uint sub_w = uint(lane) & 7u;
  const uint wpb_row = is_225 ? 9u : 8u;
  const uint wpr = n_bc * wpb_row;
  const device half* xr = X_rot + ulong(row) * ulong(a.k);
  const device half* sc_row = scales + ulong(br) * ulong(n_bc);
  float ay[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (uint bc = 0u; bc < n_bc; ++bc) {
    if (tid < 16u) {
      uint4 xv = *(const device uint4*)(xr + (bc << 7) + (uint(tid) << 3));
      s_x[tid] = xv;
      if (is_true_e8p) s_xsum[tid] = ink2_tfq4_xsum8_qtr(xv);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float fsc = float(sc_row[bc]);
    const float xs0 = is_true_e8p ? s_xsum[sub_w * 2u] : 0.0f;
    const float xs1 = is_true_e8p ? s_xsum[sub_w * 2u + 1u] : 0.0f;
    const uint4 xv0 = s_x[sub_w * 2u], xv1 = s_x[sub_w * 2u + 1u];
    PW_UNROLL
    for (uint pass = 0u; pass < 4u; ++pass) {
      const uint global_r = r0 + uint(sg) * 16u + pass * 4u + sub_r;
      const ulong roff = ulong(global_r) * ulong(wpr) + ulong(bc) * ulong(wpb_row);
      float dy = ink2_tfq4_dot16_e8p(W + roff, sub_w, is_225, is_true_e8p, s_lut, s_par, xv0, xv1, xs0, xs1);
      ay[pass] = fma(dy, fsc, ay[pass]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  PW_UNROLL
  for (uint pass = 0u; pass < 4u; ++pass) {
    float vy = ink2_tfq4_reduce8(ay[pass]);
    if (sub_w == 0u) tg_y[uint(sg) * 16u + pass * 4u + sub_r] = vy;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u) {
    float4 v = float4(tg_y[lane], tg_y[lane + 32u], tg_y[lane + 64u], tg_y[lane + 96u]);
    pw_fwht128_simd4(v, lane);
    const uint sr0 = (a.sr_split != 0u) ? ((r0 + a.row_off) % a.sr_split) : (r0 + a.row_off);
    const ulong base = ulong(row) * ulong(a.ldo) + r0;
    const float o0 = v.x * pw_sgn(0x9E3779B9u, sr0 + lane +  0u);
    const float o1 = v.y * pw_sgn(0x9E3779B9u, sr0 + lane + 32u);
    const float o2 = v.z * pw_sgn(0x9E3779B9u, sr0 + lane + 64u);
    const float o3 = v.w * pw_sgn(0x9E3779B9u, sr0 + lane + 96u);
    if (a.out_f32 != 0u) {
      device float* dst = (device float*)out_raw + base;
      dst[lane +  0u] = o0;
      dst[lane + 32u] = o1;
      dst[lane + 64u] = o2;
      dst[lane + 96u] = o3;
    } else {
      device bfloat* dst = (device bfloat*)out_raw + base;
      dst[lane +  0u] = bfloat(o0);
      dst[lane + 32u] = bfloat(o1);
      dst[lane + 64u] = bfloat(o2);
      dst[lane + 96u] = bfloat(o3);
    }
  }
}
