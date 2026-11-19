// Two-stream mHC Metal kernels for SenseNova-Loop: enter, pre_norm, post_merge, post_and_pre_norm.
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

#ifndef TF_CTL
#define TF_CTL 0
#endif

#define SN_TG 256u
#define SN_SG (SN_TG / 32u)

struct SnMhcEnterArgs {
  uint rows;
  uint hidden;
  uint has_pending;
  uint write_h;
};

struct SnMhcPreNormArgs {
  uint rows;
  uint hidden;
  uint iters;
  uint shift_norm;
  uint xs_group;
  uint xs_stride;
  float pre_scale;
  float eps;
};

struct SnMhcPostMergeArgs {
  uint rows;
  uint hidden;
  uint exit;
  uint pad0;
};

inline float sn_nv(float v) {
#if TF_CTL == 1
  return isfinite(v) ? v : 0.0f;
#else
  return v;
#endif
}

inline float sn_bf(ushort bits) {
  return as_type<float>(uint(bits) << 16);
}

inline ushort sn_to_bits(float v) {
  return as_type<ushort>(bfloat(v));
}

inline float sn_acc(float acc, float term) {
#if TF_CTL == 3
  return float(half(acc) + half(term));
#else
  return acc + term;
#endif
}

inline float sn_simd_sum(float v) {
  for (ushort m = 1; m < 32; m *= 2) {
#if TF_CTL == 3
    v = float(half(v) + half(simd_shuffle_xor(v, m)));
#else
    v = v + simd_shuffle_xor(v, m);
#endif
  }
  return v;
}

inline float sn_lse2(float a, float b) {
  if (isnan(a) || isnan(b)) return NAN;
  const float mx = max(a, b);
  if (isinf(mx) && mx < 0.0f) return mx;
  return mx + metal::precise::log(1.0f + metal::precise::exp(-metal::abs(a - b)));
}

inline ushort2 sn_post_pair(float r0, float r1, float yv, float post0, float post1,
                            float m00, float m01, float m10, float m11) {
#if TF_CTL == 3
  const float mixed0 = float(bfloat(float(half(m00 * r0)) + float(half(m01 * r1))));
  const float mixed1 = float(bfloat(float(half(m10 * r0)) + float(half(m11 * r1))));
#else
  const float mixed0 = float(bfloat(m00 * r0 + m01 * r1));
  const float mixed1 = float(bfloat(m10 * r0 + m11 * r1));
#endif
#if TF_CTL == 2
  const float upd0 = yv;
  const float upd1 = yv;
#else
  const float upd0 = float(bfloat(post0 * yv));
  const float upd1 = float(bfloat(post1 * yv));
#endif
  return ushort2(sn_to_bits(mixed0 + upd0), sn_to_bits(mixed1 + upd1));
}

inline void sn_post_vec8(uint4 a0, uint4 a1, uint4 yv, float post0, float post1, float m00, float m01, float m10,
                         float m11, thread uint4& o0, thread uint4& o1) {
  for (uint w = 0; w < 4u; ++w) {
    const ushort2 p0 = sn_post_pair(sn_nv(sn_bf(ushort(a0[w] & 0xffffu))), sn_nv(sn_bf(ushort(a1[w] & 0xffffu))),
                                    sn_nv(sn_bf(ushort(yv[w] & 0xffffu))), post0, post1, m00, m01, m10, m11);
    const ushort2 p1 = sn_post_pair(sn_nv(sn_bf(ushort(a0[w] >> 16))), sn_nv(sn_bf(ushort(a1[w] >> 16))),
                                    sn_nv(sn_bf(ushort(yv[w] >> 16))), post0, post1, m00, m01, m10, m11);
    o0[w] = uint(p0.x) | (uint(p1.x) << 16);
    o1[w] = uint(p0.y) | (uint(p1.y) << 16);
  }
}

inline void sn_load_post_coeff(const device bfloat* COEFF_IN, uint r, thread float& post0, thread float& post1,
                                thread float& m00, thread float& m01, thread float& m10, thread float& m11) {
  const device ushort* co = (const device ushort*)(COEFF_IN + ulong(r) * 6u);
  post0 = sn_nv(sn_bf(co[0])); post1 = sn_nv(sn_bf(co[1])); m00 = sn_nv(sn_bf(co[2]));
#if TF_CTL == 5
  m01 = sn_nv(sn_bf(co[4])); m10 = sn_nv(sn_bf(co[3]));
#else
  m01 = sn_nv(sn_bf(co[3])); m10 = sn_nv(sn_bf(co[4]));
#endif
  m11 = sn_nv(sn_bf(co[5]));
}

inline void sn_pre_norm_from_tg(
    threadgroup ushort* tg_r,
    threadgroup float* tg_red,
    threadgroup float* tg_dot,
    threadgroup float* tg_pre,
    const device float* W,
    const device float* B,
    const device bfloat* SUB_W,
    device bfloat* COEFF_OUT,
    device bfloat* NORM_OUT,
    device float* XS_OUT,
    constant SnMhcPreNormArgs& a,
    uint r,
    uint t,
    uint lane,
    uint sg) {
  const uint H = a.hidden;
  const uint two_h = 2u * H;

  float ss = 0.0f;
  float dot[8];
  for (uint m = 0; m < 8; ++m) dot[m] = 0.0f;

#if TF_CTL != 5
  if ((two_h & 3u) == 0u && ((ulong(W) & 15u) == 0u)) {
    const threadgroup ushort4* tg4 = (const threadgroup ushort4*)tg_r;
    const device float4* W4 = (const device float4*)W;
    const uint n4 = two_h >> 2;
    for (uint j4 = t; j4 < n4; j4 += SN_TG) {
      const ushort4 rs = tg4[j4];
      const float r0 = sn_nv(sn_bf(rs.x)), r1 = sn_nv(sn_bf(rs.y));
      const float r2 = sn_nv(sn_bf(rs.z)), r3 = sn_nv(sn_bf(rs.w));
      ss = sn_acc(sn_acc(sn_acc(sn_acc(ss, r0 * r0), r1 * r1), r2 * r2), r3 * r3);
      for (uint m = 0; m < 8; ++m) {
        const float4 wv = W4[ulong(m) * n4 + j4];
        dot[m] = sn_acc(sn_acc(sn_acc(sn_acc(dot[m], r0 * sn_nv(wv.x)), r1 * sn_nv(wv.y)),
                               r2 * sn_nv(wv.z)), r3 * sn_nv(wv.w));
      }
    }
  } else
#endif
  {
    for (uint j = t; j < two_h; j += SN_TG) {
      const float rv = sn_nv(sn_bf(tg_r[j]));
      ss = sn_acc(ss, rv * rv);
      for (uint m = 0; m < 8; ++m) {
#if TF_CTL == 5
        dot[m] = sn_acc(dot[m], rv * sn_nv(W[ulong(j) * 8u + m]));
#else
        dot[m] = sn_acc(dot[m], rv * sn_nv(W[ulong(m) * two_h + j]));
#endif
      }
    }
  }

  const float ss_sg = sn_simd_sum(ss);
  float dot_sg[8];
  for (uint m = 0; m < 8; ++m) dot_sg[m] = sn_simd_sum(dot[m]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0u) {
    tg_red[sg] = ss_sg;
    for (uint m = 0; m < 8; ++m) tg_dot[sg * 8u + m] = dot_sg[m];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sg == 0u) {
    float z_lane = 0.0f;
    if (lane < 8u) {
      float tot_ss = tg_red[0];
      for (uint s = 1; s < SN_SG; ++s) tot_ss = sn_acc(tot_ss, tg_red[s]);
      const float inv_rms = metal::precise::rsqrt(tot_ss / float(two_h) + a.eps);
      float dm = tg_dot[lane];
      for (uint s = 1; s < SN_SG; ++s) dm = sn_acc(dm, tg_dot[s * 8u + lane]);
      z_lane = dm * inv_rms + sn_nv(B[lane]);
    }
    float sig = 0.0f;
    if (lane < 4u) {
      const float num = (lane < 2u) ? 1.0f : 2.0f;
      sig = metal::precise::divide(num, 1.0f + metal::precise::exp(-z_lane));
    }
    const float pre0 = simd_shuffle(sig, 0), pre1 = simd_shuffle(sig, 1);
    const float post0 = simd_shuffle(sig, 2), post1 = simd_shuffle(sig, 3);
    float a0 = simd_shuffle(z_lane, ushort(4u + (lane & 1u) * 2u));
    float a1 = simd_shuffle(z_lane, ushort(5u + (lane & 1u) * 2u));
#if TF_CTL == 2
    const uint iters = 0u;
#else
    const uint iters = a.iters;
#endif
    for (uint it = 0; it < iters; ++it) {
      const float rn = sn_lse2(a0, a1);
      a0 -= rn;
      a1 -= rn;
      const float p0 = simd_shuffle_xor(a0, 1), p1 = simd_shuffle_xor(a1, 1);
      const float cn = (lane == 0u) ? sn_lse2(a0, p0) : sn_lse2(p1, a1);
      const float c0 = simd_shuffle(cn, 0), c1 = simd_shuffle(cn, 1);
      a0 -= c0;
      a1 -= c1;
    }
    const float e0 = metal::precise::exp(a0), e1 = metal::precise::exp(a1);
    const float mix00 = simd_shuffle(e0, 0), mix01 = simd_shuffle(e1, 0);
    const float mix10 = simd_shuffle(e0, 1), mix11 = simd_shuffle(e1, 1);
    if (lane == 0u) {
      device bfloat* co = COEFF_OUT + ulong(r) * 6u;
      co[0] = bfloat(post0); co[1] = bfloat(post1);
      co[2] = bfloat(mix00); co[3] = bfloat(mix01); co[4] = bfloat(mix10); co[5] = bfloat(mix11);
      tg_pre[0] = float(bfloat(pre0)); tg_pre[1] = float(bfloat(pre1));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float p0 = tg_pre[0], p1 = tg_pre[1];
  float br_reg[16];
  float ss_b = 0.0f;
  const device ushort* sub_u = (const device ushort*)SUB_W;
  device ushort* norm_u = (device ushort*)(NORM_OUT + ulong(r) * H);
#if TF_CTL == 4
  const bool vec4_h = false;
#else
  const bool vec4_h = ((H & 3u) == 0u) && ((ulong(sub_u) & 7u) == 0u) && ((ulong(norm_u) & 7u) == 0u);
#endif
  uint k = 0;
  if (vec4_h) {
    const threadgroup ushort4 *t0_4 = (const threadgroup ushort4*)tg_r, *t1_4 = (const threadgroup ushort4*)(tg_r + H);
    const uint n4 = H >> 2;
    for (uint d4 = t; d4 < n4; d4 += SN_TG, k += 4u) {
      const ushort4 r0_s = t0_4[d4], r1_s = t1_4[d4];
      for (uint c = 0; c < 4u; ++c) {
        float br = float(bfloat(p0 * sn_nv(sn_bf(r0_s[c])) + p1 * sn_nv(sn_bf(r1_s[c]))));
        if (a.pre_scale != 1.0f) br = float(bfloat(a.pre_scale * br));
        br_reg[k + c] = br; ss_b = sn_acc(ss_b, br * br);
      }
    }
  } else {
    for (uint d = t; d < H; d += SN_TG, ++k) {
      float br = float(bfloat(p0 * sn_nv(sn_bf(tg_r[d])) + p1 * sn_nv(sn_bf(tg_r[H + d]))));
      if (a.pre_scale != 1.0f) br = float(bfloat(a.pre_scale * br));
      br_reg[k] = br; ss_b = sn_acc(ss_b, br * br);
    }
  }

  const float ss_b_sg = sn_simd_sum(ss_b);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0u) tg_red[sg] = ss_b_sg;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0u) {
    float tot_b = tg_red[0];
    for (uint s = 1; s < SN_SG; ++s) tot_b = sn_acc(tot_b, tg_red[s]);
    tg_pre[2] = metal::precise::rsqrt(tot_b / float(H) + a.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float inv_b = tg_pre[2];
  k = 0;
  if (vec4_h) {
    const device ushort4* sub4 = (const device ushort4*)sub_u;
    device ushort4* out4 = (device ushort4*)norm_u;
    threadgroup ushort4* tg4 = (threadgroup ushort4*)tg_r;
    const uint n4 = H >> 2;
    for (uint d4 = t; d4 < n4; d4 += SN_TG, k += 4u) {
      const ushort4 sws = sub4[d4];
      ushort4 ov;
      for (uint c = 0; c < 4u; ++c) {
        const float sw = sn_nv(sn_bf(sws[c]));
        ov[c] = sn_to_bits((br_reg[k + c] * inv_b) * ((a.shift_norm != 0u) ? (1.0f + sw) : sw));
      }
      out4[d4] = ov;
      if (a.xs_group != 0u) tg4[d4] = ov;
    }
  } else {
    for (uint d = t; d < H; d += SN_TG, ++k) {
#if TF_CTL == 4
      if (d + 1u == H) continue;
#endif
      const float sw = sn_nv(sn_bf(as_type<ushort>(SUB_W[d])));
      const bfloat out_b = bfloat((br_reg[k] * inv_b) * ((a.shift_norm != 0u) ? (1.0f + sw) : sw));
      NORM_OUT[ulong(r) * H + d] = out_b;
      if (a.xs_group != 0u) tg_r[d] = as_type<ushort>(out_b);
    }
  }

  if (a.xs_group != 0u) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((a.xs_group == 128u && (H & 127u) == 0u) || (a.xs_group == 64u && (H & 63u) == 0u)) {
      const uint num_groups = H / a.xs_group, sub_groups = SN_TG >> 3, u = t & 7u;
      const uint q_cnt = (a.xs_group == 128u) ? 2u : 1u, per_t = q_cnt * 8u;
      for (uint g0 = 0; g0 < num_groups; g0 += sub_groups) {
        const uint g = g0 + (t >> 3);
        float xv[16];
        if (g < num_groups) {
          const threadgroup uint4* p8 = (const threadgroup uint4*)(tg_r + g * a.xs_group + u * per_t);
          for (uint q = 0; q < q_cnt; ++q) {
            const uint4 w = p8[q];
            for (uint c = 0; c < 4u; ++c) {
              xv[q * 8u + c * 2u] = sn_bf(ushort(w[c] & 0xffffu));
              xv[q * 8u + c * 2u + 1u] = sn_bf(ushort(w[c] >> 16));
            }
          }
        }
        float gs = 0.0f;
        for (uint turn = 0; turn < 8u; ++turn) {
          if (g < num_groups && u == turn) for (uint i = 0; i < per_t; ++i) gs = sn_acc(gs, xv[i]);
          gs = simd_shuffle(gs, ushort((lane & ~7u) | turn));
        }
        if (g < num_groups && u == 0u) XS_OUT[ulong(g) * a.xs_stride + r] = gs;
      }
    } else {
      const uint num_groups = (H + a.xs_group - 1u) / a.xs_group;
      for (uint g = t; g < num_groups; g += SN_TG) {
        float gs = 0.0f;
        const uint base = g * a.xs_group, lim = min(base + a.xs_group, H);
        for (uint i = base; i < lim; ++i) gs = sn_acc(gs, sn_bf(tg_r[i]));
        XS_OUT[ulong(g) * a.xs_stride + r] = gs;
      }
    }
  }
}

[[kernel, max_total_threads_per_threadgroup(SN_TG)]] void sn_mhc_enter(
    const device bfloat* H_IN [[buffer(0)]],
    const device bfloat* PENDING [[buffer(1)]],
    device bfloat* R_OUT [[buffer(2)]],
    device bfloat* H_OUT [[buffer(3)]],
    constant SnMhcEnterArgs& a [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint t [[thread_index_in_threadgroup]]) {
  const uint r = tg.y;
  if (r >= a.rows) return;
  const uint H = a.hidden;
  const bool vec8 = ((H & 7u) == 0u);
  const uint idx = tg.x * SN_TG + t;
  if (vec8) {
    const uint chunks = H >> 3;
    if (idx >= chunks) return;
    const ulong off_h = (ulong(r) * H + (idx << 3)) >> 3;
    const uint4 hv = ((const device uint4*)H_IN)[off_h];
    uint4 out_v = hv;
    const bool has_p = (a.has_pending != 0u);
#if TF_CTL == 1
    const bool check_nv = true;
#else
    const bool check_nv = false;
#endif
    if (has_p || check_nv) {
      const uint4 pv = has_p ? ((const device uint4*)PENDING)[off_h] : uint4(0u);
      for (uint w = 0; w < 4u; ++w) {
        const uint h0 = hv[w] & 0xffffu;
        const uint h1 = hv[w] >> 16;
        const float f0 = sn_nv(sn_bf(ushort(h0)));
        const float f1 = sn_nv(sn_bf(ushort(h1)));
        const float v0 = has_p ? (f0 + sn_nv(sn_bf(ushort(pv[w] & 0xffffu)))) : f0;
        const float v1 = has_p ? (f1 + sn_nv(sn_bf(ushort(pv[w] >> 16)))) : f1;
        out_v[w] = uint(sn_to_bits(v0)) | (uint(sn_to_bits(v1)) << 16);
      }
    }
    const ulong r0_off = (ulong(r) * 2u * H + (idx << 3)) >> 3;
    const ulong r1_off = (ulong(r) * 2u * H + H + (idx << 3)) >> 3;
#if TF_CTL == 4
    if (idx + 1u == chunks) {
      device ushort* dst0 = (device ushort*)(R_OUT + ulong(r) * 2u * H + (idx << 3));
      device ushort* dst1 = (device ushort*)(R_OUT + ulong(r) * 2u * H + H + (idx << 3));
      for (uint q = 0; q < 7u; ++q) {
        const ushort val = ushort((out_v[q >> 1u] >> ((q & 1u) << 4u)) & 0xffffu);
        dst0[q] = val;
        dst1[q] = val;
        if (a.write_h != 0u) ((device ushort*)(H_OUT + ulong(r) * H + (idx << 3)))[q] = val;
      }
      return;
    }
#endif
    ((device uint4*)R_OUT)[r0_off] = out_v;
#if TF_CTL == 2
    ((device uint4*)R_OUT)[r1_off] = uint4(0u);
#else
    ((device uint4*)R_OUT)[r1_off] = out_v;
#endif
    if (a.write_h != 0u) ((device uint4*)H_OUT)[off_h] = out_v;
    return;
  }
  if (idx >= H) return;
#if TF_CTL == 4
  if (idx + 1u == H) return;
#endif
  const float fv = sn_nv(sn_bf(as_type<ushort>(H_IN[ulong(r) * H + idx])));
  const float v = (a.has_pending != 0u) ? (fv + sn_nv(sn_bf(as_type<ushort>(PENDING[ulong(r) * H + idx])))) : fv;
  const bfloat hb = bfloat(v);
  R_OUT[ulong(r) * 2u * H + idx] = hb;
#if TF_CTL == 2
  R_OUT[ulong(r) * 2u * H + H + idx] = bfloat(0.0f);
#else
  R_OUT[ulong(r) * 2u * H + H + idx] = hb;
#endif
  if (a.write_h != 0u) H_OUT[ulong(r) * H + idx] = hb;
}

[[kernel, max_total_threads_per_threadgroup(SN_TG)]] void sn_mhc_pre_norm(
    const device bfloat* R_IN [[buffer(0)]],
    const device float* W [[buffer(1)]],
    const device float* B [[buffer(2)]],
    const device bfloat* SUB_W [[buffer(3)]],
    device bfloat* COEFF_OUT [[buffer(4)]],
    device bfloat* NORM_OUT [[buffer(5)]],
    device float* XS_OUT [[buffer(6)]],
    constant SnMhcPreNormArgs& a [[buffer(7)]],
    uint r [[threadgroup_position_in_grid]],
    uint t [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  alignas(16) threadgroup ushort tg_r[8192];
  threadgroup float tg_red[SN_SG];
  threadgroup float tg_dot[SN_SG * 8u];
  threadgroup float tg_pre[3];
  if (r >= a.rows || a.hidden == 0u || a.hidden > 4096u) return;

  const uint two_h = 2u * a.hidden;
  const device ushort* src = (const device ushort*)(R_IN + ulong(r) * two_h);
  if ((a.hidden & 7u) == 0u && ((ulong(src) & 15u) == 0u)) {
    const device uint4* src4 = (const device uint4*)src;
    threadgroup uint4* dst4 = (threadgroup uint4*)tg_r;
    const uint n4 = two_h >> 3;
    for (uint j = t; j < n4; j += SN_TG) dst4[j] = src4[j];
  } else {
    for (uint j = t; j < two_h; j += SN_TG) tg_r[j] = src[j];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  sn_pre_norm_from_tg(tg_r, tg_red, tg_dot, tg_pre, W, B, SUB_W, COEFF_OUT, NORM_OUT, XS_OUT, a, r, t, lane, sg);
}

[[kernel, max_total_threads_per_threadgroup(SN_TG)]] void sn_mhc_post_merge(
    const device bfloat* R_IN [[buffer(0)]],
    const device bfloat* Y_IN [[buffer(1)]],
    const device bfloat* COEFF_IN [[buffer(2)]],
    device bfloat* OUT [[buffer(3)]],
    constant SnMhcPostMergeArgs& a [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint t [[thread_index_in_threadgroup]]) {
  const uint r = tg.y;
  if (r >= a.rows) return;
  const uint H = a.hidden;
  float post0, post1, m00, m01, m10, m11;
  sn_load_post_coeff(COEFF_IN, r, post0, post1, m00, m01, m10, m11);

  const uint idx = tg.x * SN_TG + t;
  if ((H & 7u) == 0u) {
    const uint chunks = H >> 3;
    if (idx >= chunks) return;
    const uint d = idx << 3;
    const uint4 v0 = ((const device uint4*)(R_IN + ulong(r) * 2u * H + d))[0];
    const uint4 v1 = ((const device uint4*)(R_IN + ulong(r) * 2u * H + H + d))[0];
    const uint4 vy = ((const device uint4*)(Y_IN + ulong(r) * H + d))[0];
    uint4 o0, o1, oe;
    sn_post_vec8(v0, v1, vy, post0, post1, m00, m01, m10, m11, o0, o1);
    if (a.exit != 0u) {
      for (uint w = 0; w < 4u; ++w) {
        const ushort e0 = sn_to_bits(0.5f * (sn_bf(ushort(o0[w] & 0xffffu)) + sn_bf(ushort(o1[w] & 0xffffu))));
        const ushort e1 = sn_to_bits(0.5f * (sn_bf(ushort(o0[w] >> 16)) + sn_bf(ushort(o1[w] >> 16))));
        oe[w] = uint(e0) | (uint(e1) << 16);
      }
    }
#if TF_CTL == 4
    if (idx + 1u == chunks) {
      if (a.exit != 0u) {
        device ushort* dst = (device ushort*)(OUT + ulong(r) * H + d);
        for (uint q = 0; q < 7u; ++q) dst[q] = ushort((oe[q >> 1u] >> ((q & 1u) << 4u)) & 0xffffu);
      } else {
        device ushort* d0 = (device ushort*)(OUT + ulong(r) * 2u * H + d);
        device ushort* d1 = (device ushort*)(OUT + ulong(r) * 2u * H + H + d);
        for (uint q = 0; q < 7u; ++q) {
          d0[q] = ushort((o0[q >> 1u] >> ((q & 1u) << 4u)) & 0xffffu);
          d1[q] = ushort((o1[q >> 1u] >> ((q & 1u) << 4u)) & 0xffffu);
        }
      }
      return;
    }
#endif
    if (a.exit != 0u) {
      ((device uint4*)(OUT + ulong(r) * H + d))[0] = oe;
    } else {
      ((device uint4*)(OUT + ulong(r) * 2u * H + d))[0] = o0;
      ((device uint4*)(OUT + ulong(r) * 2u * H + H + d))[0] = o1;
    }
    return;
  }

  if (idx >= H) return;
#if TF_CTL == 4
  if (idx + 1u == H) return;
#endif
  const float r0 = sn_nv(sn_bf(as_type<ushort>(R_IN[ulong(r) * 2u * H + idx])));
  const float r1 = sn_nv(sn_bf(as_type<ushort>(R_IN[ulong(r) * 2u * H + H + idx])));
  const float yv = sn_nv(sn_bf(as_type<ushort>(Y_IN[ulong(r) * H + idx])));
  const ushort2 rn = sn_post_pair(r0, r1, yv, post0, post1, m00, m01, m10, m11);
  if (a.exit != 0u) {
    OUT[ulong(r) * H + idx] = bfloat(0.5f * (sn_bf(rn.x) + sn_bf(rn.y)));
  } else {
    OUT[ulong(r) * 2u * H + idx] = as_type<bfloat>(rn.x);
    OUT[ulong(r) * 2u * H + H + idx] = as_type<bfloat>(rn.y);
  }
}

[[kernel, max_total_threads_per_threadgroup(SN_TG)]] void sn_mhc_post_and_pre_norm(
    const device bfloat* R_IN [[buffer(0)]],
    const device bfloat* Y_IN [[buffer(1)]],
    const device bfloat* COEFF_IN [[buffer(2)]],
    const device float* W_NEXT [[buffer(3)]],
    const device float* B_NEXT [[buffer(4)]],
    const device bfloat* SUB_W_NEXT [[buffer(5)]],
    device bfloat* R_OUT [[buffer(6)]],
    device bfloat* COEFF_OUT [[buffer(7)]],
    device bfloat* NORM_OUT [[buffer(8)]],
    device float* XS_OUT [[buffer(9)]],
    constant SnMhcPreNormArgs& a [[buffer(10)]],
    uint r [[threadgroup_position_in_grid]],
    uint t [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  alignas(16) threadgroup ushort tg_r[8192];
  threadgroup float tg_red[SN_SG];
  threadgroup float tg_dot[SN_SG * 8u];
  threadgroup float tg_pre[3];
  if (r >= a.rows || a.hidden == 0u || a.hidden > 4096u) return;

  const uint H = a.hidden;
  float post0, post1, m00, m01, m10, m11;
  sn_load_post_coeff(COEFF_IN, r, post0, post1, m00, m01, m10, m11);

  const device ushort* r0_ptr = (const device ushort*)(R_IN + ulong(r) * 2u * H);
  const device ushort* r1_ptr = r0_ptr + H;
  const device ushort* y_ptr = (const device ushort*)(Y_IN + ulong(r) * H);
  device ushort* ro0 = (device ushort*)(R_OUT + ulong(r) * 2u * H);
  device ushort* ro1 = ro0 + H;

  if ((H & 7u) == 0u && ((ulong(r0_ptr) & 15u) == 0u) && ((ulong(y_ptr) & 15u) == 0u) &&
      ((ulong(ro0) & 15u) == 0u)) {
    const device uint4 *r0_v = (const device uint4*)r0_ptr, *r1_v = (const device uint4*)r1_ptr;
    const device uint4* y_v = (const device uint4*)y_ptr;
    threadgroup uint4 *t0_v = (threadgroup uint4*)tg_r, *t1_v = (threadgroup uint4*)(tg_r + H);
    device uint4 *o0_v = (device uint4*)ro0, *o1_v = (device uint4*)ro1;
    const uint n4 = H >> 3;
    for (uint i = t; i < n4; i += SN_TG) {
      uint4 o0, o1;
      sn_post_vec8(r0_v[i], r1_v[i], y_v[i], post0, post1, m00, m01, m10, m11, o0, o1);
      t0_v[i] = o0; t1_v[i] = o1;
#if TF_CTL == 4
      if ((i << 3) + 7u == H - 1u) {
        const uint d = i << 3;
        for (uint q = 0; q < 7u; ++q) {
          ro0[d + q] = ushort((o0[q >> 1u] >> ((q & 1u) << 4u)) & 0xffffu);
          ro1[d + q] = ushort((o1[q >> 1u] >> ((q & 1u) << 4u)) & 0xffffu);
        }
        continue;
      }
#endif
      o0_v[i] = o0; o1_v[i] = o1;
    }
  } else {
    for (uint d = t; d < H; d += SN_TG) {
      const ushort2 rn = sn_post_pair(sn_nv(sn_bf(r0_ptr[d])), sn_nv(sn_bf(r1_ptr[d])),
                                      sn_nv(sn_bf(y_ptr[d])), post0, post1, m00, m01, m10, m11);
      tg_r[d] = rn.x;
      tg_r[H + d] = rn.y;
#if TF_CTL == 4
      if (d + 1u == H) continue;
#endif
      ro0[d] = rn.x;
      ro1[d] = rn.y;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  sn_pre_norm_from_tg(tg_r, tg_red, tg_dot, tg_pre, W_NEXT, B_NEXT, SUB_W_NEXT, COEFF_OUT, NORM_OUT, XS_OUT, a, r, t,
                      lane, sg);
}
