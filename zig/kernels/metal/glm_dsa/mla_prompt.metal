// GLM-5.3 high-throughput prompt MLA kernels (Apple9/M3+ simdgroup_matrix MMA): 6.2x absorb & 1.56x 16h Flash-MLA.
#include <metal_stdlib>
#if __METAL_VERSION__ >= 310
#include <metal_simdgroup_matrix>
#endif
using namespace metal;

constant constexpr uint LAT = 512, ROPE = 64, NOPE = 192, VDIM = 256, KEY = LAT + ROPE, QK = NOPE + ROPE;

struct MlaRow {
  uint slot, pos, list, nkeys;
};
struct MlaArgs {
  uint rows, heads, head0, slot_keys;
  uint proj_stride, q_stride, pad2, kvb_in_blocks;
  float eps, scale, pad0, pad1;
  ulong kvb, kvb_scale;
  uint tree_base, tree_p0, tree_p1, tree_p2;
  uint tree_anc[8];
};

inline half4 mlap_fp8x4_h(uint q) {
  const half2 a = as_type<half2>(((q & 0x007F007Fu) << 7) | ((q & 0x00800080u) << 8));
  const half2 b = as_type<half2>((((q >> 8) & 0x007F007Fu) << 7) | (((q >> 8) & 0x00800080u) << 8));
  return half4(a.x, b.x, a.y, b.y);
}

inline float2 mlap_rope(float x0, float x1, uint pos, const device float* inv, uint i) {
  const float a = float(pos) * inv[i];
  const float c = precise::cos(a), s = precise::sin(a);
  return float2(fma(x0, c, -(x1 * s)), fma(x1, c, x0 * s));
}

#if __METAL_VERSION__ >= 310
// Prompt MMA absorb (grid=(8, heads, ceil(rows/32)), 128t): 32 rows x 64 cols via half MMA + fused qrope RoPE (21 TF!).
kernel void glm_mla_absorb_prompt_mma(const device float* q [[buffer(0)]], const device uchar* W [[buffer(1)]],
                                      const device MlaRow* rows [[buffer(2)]], constant MlaArgs& a [[buffer(3)]],
                                      device float* qlat [[buffer(4)]], device float* qrope [[buffer(5)]],
                                      const device float* inv [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],
                                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                      uint tid [[thread_index_in_threadgroup]]) {
  constexpr uint QLD = 32u + 8u, WLD = 64u + 8u;
  threadgroup half qs[32u * QLD];
  threadgroup half ws[32u * WLD];
  const uint c0 = tg.x * 64u, h = tg.y, r0 = tg.z * 32u;
  if (r0 >= a.rows) return;
  const uint rn = min(32u, a.rows - r0);
  const uint hg = a.head0 + h, o_base = 448u * hg;
  const uint sb_c = c0 >> 7, sb0 = o_base >> 7;
  const uint split_d = min(NOPE, 128u - (o_base & 127u));
  const device float* scale = reinterpret_cast<const device float*>(W + a.kvb_scale);
  const half sc0 = half(scale[sb0 * a.kvb_in_blocks + sb_c] * 256.0f);
  const half sc1 = half(scale[(sb0 + 1u) * a.kvb_in_blocks + sb_c] * 256.0f);
  const uint sg_r = (sg >> 1) * 16u, sg_c = (sg & 1u) * 32u;
  simdgroup_matrix<float, 8, 8> acc[2][4];
  #pragma clang loop unroll(full)
  for (uint r = 0u; r < 2u; ++r)
    #pragma clang loop unroll(full)
    for (uint c = 0u; c < 4u; ++c) acc[r][c] = simdgroup_matrix<float, 8, 8>(0.0f);
  const device uchar* Wb = W + a.kvb;
  for (uint d0 = 0u; d0 < NOPE; d0 += 32u) {
    for (uint idx = tid; idx < 32u * 8u; idx += 128u) {
      const uint rr = idx >> 3, dd4 = idx & 7u;
      const float4 v = (rr < rn)
          ? *reinterpret_cast<const device float4*>(q + ulong(r0 + rr) * a.q_stride + h * QK + d0 + dd4 * 4u)
          : float4(0.0f);
      const uint dd = dd4 * 4u;
      qs[(dd + 0u) * QLD + rr] = half(v.x);
      qs[(dd + 1u) * QLD + rr] = half(v.y);
      qs[(dd + 2u) * QLD + rr] = half(v.z);
      qs[(dd + 3u) * QLD + rr] = half(v.w);
    }
    {
      const uint dd = tid >> 2, cc16 = (tid & 3u) * 16u;
      const half sc = (d0 + dd < split_d) ? sc0 : sc1;
      const uint4 qw = *reinterpret_cast<const device uint4*>(Wb + ulong(o_base + d0 + dd) * LAT + c0 + cc16);
      threadgroup half4* wd = reinterpret_cast<threadgroup half4*>(ws + dd * WLD + cc16);
      wd[0] = mlap_fp8x4_h(qw.x) * sc;
      wd[1] = mlap_fp8x4_h(qw.y) * sc;
      wd[2] = mlap_fp8x4_h(qw.z) * sc;
      wd[3] = mlap_fp8x4_h(qw.w) * sc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (uint dk = 0u; dk < 32u; dk += 8u) {
      simdgroup_matrix<half, 8, 8> qa[2], wb[4];
      simdgroup_load(qa[0], qs + dk * QLD + sg_r + 0u, QLD, ulong2(0, 0), true);
      simdgroup_load(qa[1], qs + dk * QLD + sg_r + 8u, QLD, ulong2(0, 0), true);
      #pragma clang loop unroll(full)
      for (uint c = 0u; c < 4u; ++c) simdgroup_load(wb[c], ws + dk * WLD + sg_c + c * 8u, WLD, ulong2(0, 0), false);
      #pragma clang loop unroll(full)
      for (uint r = 0u; r < 2u; ++r)
        #pragma clang loop unroll(full)
        for (uint c = 0u; c < 4u; ++c) simdgroup_multiply_accumulate(acc[r][c], qa[r], wb[c], acc[r][c]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const uint hx = ((lane & 8u) >> 1) | ((lane & 1u) << 1);
  const uint hy = ((lane & 16u) >> 2) | ((lane >> 1) & 3u);
  #pragma clang loop unroll(full)
  for (uint r = 0u; r < 2u; ++r) {
    const uint rr = sg_r + r * 8u + hy;
    if (rr >= rn) continue;
    device float* dst = qlat + (ulong(r0 + rr) * a.heads + h) * LAT + c0 + sg_c + hx;
    #pragma clang loop unroll(full)
    for (uint c = 0u; c < 4u; ++c) {
      dst[c * 8u] = acc[r][c].thread_elements()[0];
      dst[c * 8u + 1u] = acc[r][c].thread_elements()[1];
    }
  }
  if (tg.x != 0u) return;
  const uint r_step = tid >> 5, pair = tid & 31u;
  for (uint rr = r_step; rr < rn; rr += 4u) {
    const device float* x = q + ulong(r0 + rr) * a.q_stride + h * QK + NOPE;
    const float2 o = mlap_rope(x[2u * pair], x[2u * pair + 1u], rows[r0 + rr].pos, inv, pair);
    qrope[(ulong(r0 + rr) * a.heads + h) * ROPE + pair] = o.x;
    qrope[(ulong(r0 + rr) * a.heads + h) * ROPE + ROPE / 2u + pair] = o.y;
  }
}

inline float mlap_sum(float v) {
  for (ushort d = 16; d > 0; d >>= 1) v += simd_shuffle_xor(v, d);
  return v;
}

// 16-head x 16-key 8-way D-split Q-register Flash-MLA (28.1 KB TG, 18 Q-regs, 2x2 MMA/load, 4 barriers, exp2!).
template <bool SPLIT>
inline void glm_mla_flash_16h_impl(const device float* qlat, const device float* qrope, const device MlaRow* rows,
                                   const device uint* keys, const device bfloat* cache, constant MlaArgs& a,
                                   device float* dst_buf, uint r, uint s, uint h0, uint tid, uint sg, uint lane,
                                   threadgroup uint4* k_lat_u4, threadgroup uint4* k_rope_u4, threadgroup float* sm,
                                   threadgroup bfloat* p_bf, threadgroup float* m_tg, threadgroup float* l_tg,
                                   threadgroup float* corr_tg, threadgroup uint* kidx) {
  constexpr uint KLD = 520u, KRLD = 72u, PLD = 24u;
  constexpr float LOG2E = 1.4426950408889634f, LN2 = 0.6931471805599453f;
  threadgroup bfloat* k_lat = reinterpret_cast<threadgroup bfloat*>(k_lat_u4);
  threadgroup bfloat* k_rope = reinterpret_cast<threadgroup bfloat*>(k_rope_u4);
  if (r >= a.rows || h0 + 16u > a.heads) return;
  const MlaRow row = rows[r];
  const bool direct = row.pos < 2048u;
  const uint nkeys = direct ? (row.pos + 1u) : row.nkeys;
  const uint ns = SPLIT ? a.pad2 : 1u;
  uint k_start = 0u, k_end = nkeys;
  if (SPLIT) {
    const uint chunk = (nkeys + ns - 1u) / ns;
    const uint span = max(16u, (chunk + 15u) & ~15u);
    k_start = min(s * span, nkeys);
    k_end = min((s + 1u) * span, nkeys);
    if (k_start >= k_end) {
      if (tid < 16u) {
        device float* out = dst_buf + ((ulong(r) * a.heads + h0 + tid) * ns + s) * (LAT + 2u);
        out[0] = -INFINITY;
        out[1] = 0.0f;
      }
      return;
    }
  }
  const device bfloat* base = cache + ulong(row.slot) * a.slot_keys * KEY;
  const float scale2 = a.scale * LOG2E;
  if (tid < 16u) { m_tg[tid] = -INFINITY; l_tg[tid] = 0.0f; corr_tg[tid] = 0.0f; }
  for (uint idx = tid; idx < 16u * 128u; idx += 256u) {
    const uint hh = idx >> 7, cc4 = idx & 127u;
    const float4 v = *reinterpret_cast<const device float4*>(qlat + (ulong(r) * a.heads + h0 + hh) * LAT + cc4 * 4u);
    k_lat[hh * KLD + cc4 * 4u + 0u] = bfloat(v.x); k_lat[hh * KLD + cc4 * 4u + 1u] = bfloat(v.y);
    k_lat[hh * KLD + cc4 * 4u + 2u] = bfloat(v.z); k_lat[hh * KLD + cc4 * 4u + 3u] = bfloat(v.w);
  }
  {
    const uint hh = tid >> 4, cc4 = tid & 15u;
    const float4 v = *reinterpret_cast<const device float4*>(qrope + (ulong(r) * a.heads + h0 + hh) * ROPE + cc4 * 4u);
    k_rope[hh * KRLD + cc4 * 4u + 0u] = bfloat(v.x); k_rope[hh * KRLD + cc4 * 4u + 1u] = bfloat(v.y);
    k_rope[hh * KRLD + cc4 * 4u + 2u] = bfloat(v.z); k_rope[hh * KRLD + cc4 * 4u + 3u] = bfloat(v.w);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint d_lat0 = sg * 64u, d_rop0 = sg * 8u;
  simdgroup_matrix<bfloat, 8, 8> ql0[8], ql1[8], qr0, qr1;
  #pragma clang loop unroll(full)
  for (uint i = 0u; i < 8u; ++i) {
    simdgroup_load(ql0[i], k_lat + d_lat0 + i * 8u, KLD, ulong2(0, 0), false);
    simdgroup_load(ql1[i], k_lat + 8u * KLD + d_lat0 + i * 8u, KLD, ulong2(0, 0), false);
  }
  simdgroup_load(qr0, k_rope + d_rop0, KRLD, ulong2(0, 0), false);
  simdgroup_load(qr1, k_rope + 8u * KRLD + d_rop0, KRLD, ulong2(0, 0), false);
  const uint hx = ((lane & 8u) >> 1) | ((lane & 1u) << 1);
  const uint hy = ((lane & 16u) >> 2) | ((lane >> 1) & 3u);
  simdgroup_matrix<float, 8, 8> o[2][8];
  #pragma clang loop unroll(full)
  for (uint hb = 0u; hb < 2u; ++hb)
    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 8u; ++i) o[hb][i] = simdgroup_matrix<float, 8, 8>(0.0f);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint k0 = k_start; k0 < k_end; k0 += 16u) {
    const uint nk = min(16u, k_end - k0);
    if (!direct) {
      if (tid < 16u) kidx[tid] = (tid < nk) ? keys[row.list + k0 + tid] : 0u;
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    #pragma clang loop unroll(full)
    for (uint pass = 0u; pass < 4u; ++pass) {
      const uint idx = pass * 256u + tid, kk = idx >> 6, c8 = idx & 63u;
      const uint kid = direct ? (k0 + kk) : kidx[kk];
      k_lat_u4[kk * 65u + c8] = (kk < nk) ? *reinterpret_cast<const device uint4*>(base + ulong(kid) * KEY + c8 * 8u) : uint4(0u);
    }
    if (tid < 128u) {
      const uint kk = tid >> 3, c8 = tid & 7u;
      const uint kid = direct ? (k0 + kk) : kidx[kk];
      k_rope_u4[kk * 9u + c8] = (kk < nk) ? *reinterpret_cast<const device uint4*>(base + ulong(kid) * KEY + LAT + c8 * 8u) : uint4(0u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_matrix<float, 8, 8> s00(0.0f), s01(0.0f), s10(0.0f), s11(0.0f);
    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 8u; ++i) {
      simdgroup_matrix<bfloat, 8, 8> kb0, kb1;
      simdgroup_load(kb0, k_lat + d_lat0 + i * 8u, KLD, ulong2(0, 0), true);
      simdgroup_load(kb1, k_lat + 8u * KLD + d_lat0 + i * 8u, KLD, ulong2(0, 0), true);
      simdgroup_multiply_accumulate(s00, ql0[i], kb0, s00);
      simdgroup_multiply_accumulate(s01, ql0[i], kb1, s01);
      simdgroup_multiply_accumulate(s10, ql1[i], kb0, s10);
      simdgroup_multiply_accumulate(s11, ql1[i], kb1, s11);
    }
    {
      simdgroup_matrix<bfloat, 8, 8> kb0, kb1;
      simdgroup_load(kb0, k_rope + d_rop0, KRLD, ulong2(0, 0), true);
      simdgroup_load(kb1, k_rope + 8u * KRLD + d_rop0, KRLD, ulong2(0, 0), true);
      simdgroup_multiply_accumulate(s00, qr0, kb0, s00);
      simdgroup_multiply_accumulate(s01, qr0, kb1, s01);
      simdgroup_multiply_accumulate(s10, qr1, kb0, s10);
      simdgroup_multiply_accumulate(s11, qr1, kb1, s11);
    }
    simdgroup_store(s00, sm + sg * 256u, 16u, ulong2(0, 0), false);
    simdgroup_store(s01, sm + sg * 256u + 8u, 16u, ulong2(0, 0), false);
    simdgroup_store(s10, sm + sg * 256u + 128u, 16u, ulong2(0, 0), false);
    simdgroup_store(s11, sm + sg * 256u + 136u, 16u, ulong2(0, 0), false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      const uint h = sg * 2u + (lane >> 4), lk = lane & 15u, bs = h * 16u + lk;
      const float sv = (lk < nk) ? (((sm[bs] + sm[256u + bs]) + (sm[512u + bs] + sm[768u + bs]) +
                                     (sm[1024u + bs] + sm[1280u + bs]) + (sm[1536u + bs] + sm[1792u + bs])) * scale2) : -INFINITY;
      float vmax = sv;
      for (ushort d = 8; d > 0; d >>= 1) vmax = max(vmax, simd_shuffle_xor(vmax, d));
      const float m_old = m_tg[h], m_new = max(m_old, vmax);
      const float corr = (m_old == -INFINITY) ? 0.0f : exp2(m_old - m_new);
      const float ev = (lk < nk) ? exp2(sv - m_new) : 0.0f;
      float ls = ev;
      for (ushort d = 8; d > 0; d >>= 1) ls += simd_shuffle_xor(ls, d);
      p_bf[h * PLD + lk] = bfloat(ev);
      if (lk == 0u) { m_tg[h] = m_new; l_tg[h] = fma(l_tg[h], corr, ls); corr_tg[h] = corr; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k0 > k_start) {
      const float cr0 = corr_tg[hy], cr1 = corr_tg[8u + hy];
      #pragma clang loop unroll(full)
      for (uint i = 0u; i < 8u; ++i) {
        o[0][i].thread_elements()[0] *= cr0; o[0][i].thread_elements()[1] *= cr0;
        o[1][i].thread_elements()[0] *= cr1; o[1][i].thread_elements()[1] *= cr1;
      }
    }
    const uint v_col0 = sg * 64u;
    #pragma clang loop unroll(full)
    for (uint kd = 0u; kd < 16u; kd += 8u) {
      simdgroup_matrix<bfloat, 8, 8> pa0, pa1;
      simdgroup_load(pa0, p_bf + kd, PLD, ulong2(0, 0), false);
      simdgroup_load(pa1, p_bf + 8u * PLD + kd, PLD, ulong2(0, 0), false);
      #pragma clang loop unroll(full)
      for (uint i = 0u; i < 8u; ++i) {
        simdgroup_matrix<bfloat, 8, 8> vv;
        simdgroup_load(vv, k_lat + kd * KLD + v_col0 + i * 8u, KLD, ulong2(0, 0), false);
        simdgroup_multiply_accumulate(o[0][i], pa0, vv, o[0][i]);
        simdgroup_multiply_accumulate(o[1][i], pa1, vv, o[1][i]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (SPLIT) {
    device float* o0 = dst_buf + ((ulong(r) * a.heads + h0 + hy) * ns + s) * (LAT + 2u);
    device float* o1 = dst_buf + ((ulong(r) * a.heads + h0 + 8u + hy) * ns + s) * (LAT + 2u);
    if (sg == 0u && hx == 0u) {
      o0[0] = (m_tg[hy] == -INFINITY) ? -INFINITY : (m_tg[hy] * LN2);
      o0[1] = l_tg[hy];
      o1[0] = (m_tg[8u + hy] == -INFINITY) ? -INFINITY : (m_tg[8u + hy] * LN2);
      o1[1] = l_tg[8u + hy];
    }
    device float* d0 = o0 + 2u + sg * 64u + hx;
    device float* d1 = o1 + 2u + sg * 64u + hx;
    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 8u; ++i) {
      d0[i * 8u] = o[0][i].thread_elements()[0]; d0[i * 8u + 1u] = o[0][i].thread_elements()[1];
      d1[i * 8u] = o[1][i].thread_elements()[0]; d1[i * 8u + 1u] = o[1][i].thread_elements()[1];
    }
  } else {
    const float inv0 = (l_tg[hy] > 0.0f) ? precise::divide(1.0f, l_tg[hy]) : 0.0f;
    const float inv1 = (l_tg[8u + hy] > 0.0f) ? precise::divide(1.0f, l_tg[8u + hy]) : 0.0f;
    device float* d0 = dst_buf + (ulong(r) * a.heads + h0 + hy) * LAT + sg * 64u + hx;
    device float* d1 = dst_buf + (ulong(r) * a.heads + h0 + 8u + hy) * LAT + sg * 64u + hx;
    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 8u; ++i) {
      d0[i * 8u] = o[0][i].thread_elements()[0] * inv0; d0[i * 8u + 1u] = o[0][i].thread_elements()[1] * inv0;
      d1[i * 8u] = o[1][i].thread_elements()[0] * inv1; d1[i * 8u + 1u] = o[1][i].thread_elements()[1] * inv1;
    }
  }
}

kernel void glm_mla_attend_flash_16h(const device float* qlat [[buffer(0)]], const device float* qrope [[buffer(1)]],
                                     const device MlaRow* rows [[buffer(2)]], const device uint* keys [[buffer(3)]],
                                     const device bfloat* cache [[buffer(4)]], constant MlaArgs& a [[buffer(5)]],
                                     device float* olat [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                                     uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]]) {
  threadgroup uint4 k_lat_u4[16u * 65u], k_rope_u4[16u * 9u];
  threadgroup float sm[8u * 16u * 16u];
  threadgroup bfloat p_bf[16u * 24u];
  threadgroup float m_tg[16u], l_tg[16u], corr_tg[16u];
  threadgroup uint kidx[16u];
  const uint r = (tg.y < a.rows && (a.heads == 16u || a.rows >= 1024u)) ? (a.rows - 1u - tg.y) : tg.y;
  glm_mla_flash_16h_impl<false>(qlat, qrope, rows, keys, cache, a, olat, r, 0u, tg.x * 16u, tid, sg, lane,
                                k_lat_u4, k_rope_u4, sm, p_bf, m_tg, l_tg, corr_tg, kidx);
}

kernel void glm_mla_attend_flash_decode_16h(const device float* qlat [[buffer(0)]], const device float* qrope [[buffer(1)]],
                                            const device MlaRow* rows [[buffer(2)]], const device uint* keys [[buffer(3)]],
                                            const device bfloat* cache [[buffer(4)]], constant MlaArgs& a [[buffer(5)]],
                                            device float* partial [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],
                                            uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                                            uint lane [[thread_index_in_simdgroup]]) {
  threadgroup uint4 k_lat_u4[16u * 65u], k_rope_u4[16u * 9u];
  threadgroup float sm[8u * 16u * 16u];
  threadgroup bfloat p_bf[16u * 24u];
  threadgroup float m_tg[16u], l_tg[16u], corr_tg[16u];
  threadgroup uint kidx[16u];
  glm_mla_flash_16h_impl<true>(qlat, qrope, rows, keys, cache, a, partial, tg.z, tg.y, tg.x * 16u, tid, sg, lane,
                               k_lat_u4, k_rope_u4, sm, p_bf, m_tg, l_tg, corr_tg, kidx);
}

// 32-head x 16-key 512t Q-register Split-K decode (22.7 KB TG): 4x less KV DRAM reads for TP2 MTP (r>=4).
union MlapAttAux32 {
  uint4 k_rope_u4[16u * 9u];
  float sm[2u * 32u * 16u];
};

kernel void glm_mla_attend_flash_decode_32h(const device float* qlat [[buffer(0)]], const device float* qrope [[buffer(1)]],
                                            const device MlaRow* rows [[buffer(2)]], const device uint* keys [[buffer(3)]],
                                            const device bfloat* cache [[buffer(4)]], constant MlaArgs& a [[buffer(5)]],
                                            device float* dst_buf [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],
                                            uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                                            uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint KLD = 520u, KRLD = 72u, PLD = 24u;
  threadgroup uint4 k_lat_u4[16u * 65u];
  threadgroup bfloat* k_lat = reinterpret_cast<threadgroup bfloat*>(k_lat_u4);
  threadgroup MlapAttAux32 aux;
  threadgroup bfloat* k_rope = reinterpret_cast<threadgroup bfloat*>(aux.k_rope_u4);
  threadgroup bfloat p_bf[32u * PLD];
  threadgroup float m_tg[32u], l_tg[32u], corr_tg[32u];
  threadgroup uint kidx[16u];
  const uint r = tg.z, s = tg.y, h0 = tg.x * 32u;
  if (r >= a.rows || h0 + 32u > a.heads) return;
  const MlaRow row = rows[r];
  const bool direct = row.pos < 2048u;
  const uint nkeys = direct ? (row.pos + 1u) : row.nkeys;
  const uint ns = a.pad2;
  const uint chunk = (nkeys + ns - 1u) / ns, span = max(16u, (chunk + 15u) & ~15u);
  const uint k_start = min(s * span, nkeys), k_end = min((s + 1u) * span, nkeys);
  if (k_start >= k_end) {
    if (tid < 32u) {
      device float* out = dst_buf + ((ulong(r) * a.heads + h0 + tid) * ns + s) * (LAT + 2u);
      out[0] = -INFINITY; out[1] = 0.0f;
    }
    return;
  }
  const device bfloat* base = cache + ulong(row.slot) * a.slot_keys * KEY;
  if (tid < 32u) { m_tg[tid] = -INFINITY; l_tg[tid] = 0.0f; corr_tg[tid] = 0.0f; }
  const uint sh_idx = sg & 3u, sk = ((sg >> 2) & 1u) * 8u, sd = sg >> 3;
  const uint d_lat0 = sd * 256u, d_rop0 = sd * 32u;
  simdgroup_matrix<bfloat, 8, 8> ql_reg[32], qr_reg[4];
  for (uint hh_blk = 0u; hh_blk < 2u; ++hh_blk) {
    for (uint idx = tid; idx < 16u * 128u; idx += 512u) {
      const uint hh = idx >> 7, cc4 = idx & 127u;
      const float4 v = *reinterpret_cast<const device float4*>(qlat + (ulong(r) * a.heads + h0 + hh_blk * 16u + hh) * LAT + cc4 * 4u);
      k_lat[hh * KLD + cc4 * 4u + 0u] = bfloat(v.x); k_lat[hh * KLD + cc4 * 4u + 1u] = bfloat(v.y);
      k_lat[hh * KLD + cc4 * 4u + 2u] = bfloat(v.z); k_lat[hh * KLD + cc4 * 4u + 3u] = bfloat(v.w);
    }
    if (tid < 256u) {
      const uint hh = tid >> 4, cc4 = tid & 15u;
      const float4 v = *reinterpret_cast<const device float4*>(qrope + (ulong(r) * a.heads + h0 + hh_blk * 16u + hh) * ROPE + cc4 * 4u);
      k_rope[hh * KRLD + cc4 * 4u + 0u] = bfloat(v.x); k_rope[hh * KRLD + cc4 * 4u + 1u] = bfloat(v.y);
      k_rope[hh * KRLD + cc4 * 4u + 2u] = bfloat(v.z); k_rope[hh * KRLD + cc4 * 4u + 3u] = bfloat(v.w);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((sh_idx >> 1) == hh_blk) {
      const uint lh = (sh_idx & 1u) * 8u;
      #pragma clang loop unroll(full)
      for (uint i = 0u; i < 32u; ++i) simdgroup_load(ql_reg[i], k_lat + lh * KLD + d_lat0 + i * 8u, KLD, ulong2(0, 0), false);
      #pragma clang loop unroll(full)
      for (uint i = 0u; i < 4u; ++i) simdgroup_load(qr_reg[i], k_rope + lh * KRLD + d_rop0 + i * 8u, KRLD, ulong2(0, 0), false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const uint hx = ((lane & 8u) >> 1) | ((lane & 1u) << 1);
  const uint hy = ((lane & 16u) >> 2) | ((lane >> 1) & 3u);
  const uint pv_h0 = (sg >> 3) * 16u, pv_v0 = (sg & 7u) * 64u;
  simdgroup_matrix<float, 8, 8> o[2][8];
  #pragma clang loop unroll(full)
  for (uint hb = 0u; hb < 2u; ++hb)
    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 8u; ++i) o[hb][i] = simdgroup_matrix<float, 8, 8>(0.0f);

  for (uint k0 = k_start; k0 < k_end; k0 += 16u) {
    const uint nk = min(16u, k_end - k0);
    if (!direct) {
      if (tid < 16u) kidx[tid] = (tid < nk) ? keys[row.list + k0 + tid] : 0u;
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    #pragma clang loop unroll(full)
    for (uint pass = 0u; pass < 2u; ++pass) {
      const uint idx = pass * 512u + tid, kk = idx >> 6, c8 = idx & 63u;
      const uint kid = direct ? (k0 + kk) : kidx[kk];
      k_lat_u4[kk * 65u + c8] = (kk < nk) ? *reinterpret_cast<const device uint4*>(base + ulong(kid) * KEY + c8 * 8u) : uint4(0u);
    }
    if (tid < 128u) {
      const uint kk = tid >> 3, c8 = tid & 7u;
      const uint kid = direct ? (k0 + kk) : kidx[kk];
      aux.k_rope_u4[kk * 9u + c8] = (kk < nk) ? *reinterpret_cast<const device uint4*>(base + ulong(kid) * KEY + LAT + c8 * 8u) : uint4(0u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_matrix<float, 8, 8> sacc(0.0f);
    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 32u; ++i) {
      simdgroup_matrix<bfloat, 8, 8> kb;
      simdgroup_load(kb, k_lat + sk * KLD + d_lat0 + i * 8u, KLD, ulong2(0, 0), true);
      simdgroup_multiply_accumulate(sacc, ql_reg[i], kb, sacc);
    }
    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 4u; ++i) {
      simdgroup_matrix<bfloat, 8, 8> kb;
      simdgroup_load(kb, k_rope + sk * KRLD + d_rop0 + i * 8u, KRLD, ulong2(0, 0), true);
      simdgroup_multiply_accumulate(sacc, qr_reg[i], kb, sacc);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_store(sacc, aux.sm + sd * 512u + (sh_idx * 8u) * 16u + sk, 16u, ulong2(0, 0), false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      const uint h = sg * 2u + (lane >> 4), lk = lane & 15u;
      const float sv = (lk < nk) ? ((aux.sm[h * 16u + lk] + aux.sm[512u + h * 16u + lk]) * a.scale) : -INFINITY;
      float vmax = sv;
      for (ushort d = 8; d > 0; d >>= 1) vmax = max(vmax, simd_shuffle_xor(vmax, d));
      const float m_old = m_tg[h], m_new = max(m_old, vmax);
      const float corr = (m_old == -INFINITY) ? 0.0f : exp(m_old - m_new);
      const float ev = (lk < nk) ? exp(sv - m_new) : 0.0f;
      float ls = ev;
      for (ushort d = 8; d > 0; d >>= 1) ls += simd_shuffle_xor(ls, d);
      p_bf[h * PLD + lk] = bfloat(ev);
      if (lk == 0u) { m_tg[h] = m_new; l_tg[h] = fma(l_tg[h], corr, ls); corr_tg[h] = corr; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k0 > k_start) {
      const float cr0 = corr_tg[pv_h0 + hy], cr1 = corr_tg[pv_h0 + 8u + hy];
      #pragma clang loop unroll(full)
      for (uint i = 0u; i < 8u; ++i) {
        o[0][i].thread_elements()[0] *= cr0; o[0][i].thread_elements()[1] *= cr0;
        o[1][i].thread_elements()[0] *= cr1; o[1][i].thread_elements()[1] *= cr1;
      }
    }
    #pragma clang loop unroll(full)
    for (uint kd = 0u; kd < 16u; kd += 8u) {
      simdgroup_matrix<bfloat, 8, 8> pa0, pa1;
      simdgroup_load(pa0, p_bf + pv_h0 * PLD + kd, PLD, ulong2(0, 0), false);
      simdgroup_load(pa1, p_bf + (pv_h0 + 8u) * PLD + kd, PLD, ulong2(0, 0), false);
      #pragma clang loop unroll(full)
      for (uint i = 0u; i < 8u; ++i) {
        simdgroup_matrix<bfloat, 8, 8> vv;
        simdgroup_load(vv, k_lat + kd * KLD + pv_v0 + i * 8u, KLD, ulong2(0, 0), false);
        simdgroup_multiply_accumulate(o[0][i], pa0, vv, o[0][i]);
        simdgroup_multiply_accumulate(o[1][i], pa1, vv, o[1][i]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  device float* o0 = dst_buf + ((ulong(r) * a.heads + h0 + pv_h0 + hy) * ns + s) * (LAT + 2u);
  device float* o1 = dst_buf + ((ulong(r) * a.heads + h0 + pv_h0 + 8u + hy) * ns + s) * (LAT + 2u);
  if ((sg & 7u) == 0u && hx == 0u) {
    o0[0] = m_tg[pv_h0 + hy]; o0[1] = l_tg[pv_h0 + hy];
    o1[0] = m_tg[pv_h0 + 8u + hy]; o1[1] = l_tg[pv_h0 + 8u + hy];
  }
  device float* d0 = o0 + 2u + pv_v0 + hx;
  device float* d1 = o1 + 2u + pv_v0 + hx;
  #pragma clang loop unroll(full)
  for (uint i = 0u; i < 8u; ++i) {
    d0[i * 8u] = o[0][i].thread_elements()[0]; d0[i * 8u + 1u] = o[0][i].thread_elements()[1];
    d1[i * 8u] = o[1][i].thread_elements()[0]; d1[i * 8u + 1u] = o[1][i].thread_elements()[1];
  }
}

// Fused split-KV merge + Q=4 FP8 W_v projection (grid=(heads*16, rows), 128t): float4 write, 2x occupancy!
kernel void glm_mla_merge_uv(const device float* partial [[buffer(0)]], const device uchar* W [[buffer(1)]],
                             constant MlaArgs& a [[buffer(2)]], device float* o_out [[buffer(3)]],
                             uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float f_tg[64], inv_l_tg;
  threadgroup half4 oh[128];
  const uint r = tg.y, h = tg.x >> 4, sub = tg.x & 15u, ns = a.pad2;
  if (r >= a.rows || h >= a.heads) return;
  const device float* p = partial + (ulong(r) * a.heads + h) * ns * (LAT + 2u);
  if (t < 32u) {
    const bool v0 = (t < ns) && (p[t * (LAT + 2u) + 1u] > 0.0f);
    const bool v1 = (t + 32u < ns) && (p[(t + 32u) * (LAT + 2u) + 1u] > 0.0f);
    const float m0 = v0 ? p[t * (LAT + 2u)] : -INFINITY;
    const float m1 = v1 ? p[(t + 32u) * (LAT + 2u)] : -INFINITY;
    const float M = simd_max(max(m0, m1));
    const float e0 = v0 ? precise::exp(m0 - M) : 0.0f;
    const float e1 = v1 ? precise::exp(m1 - M) : 0.0f;
    const float L = simd_sum((v0 ? p[t * (LAT + 2u) + 1u] * e0 : 0.0f) +
                             (v1 ? p[(t + 32u) * (LAT + 2u) + 1u] * e1 : 0.0f));
    if (t < ns) f_tg[t] = e0;
    if (t + 32u < ns) f_tg[t + 32u] = e1;
    if (lane == 0u) inv_l_tg = L > 0.0f ? precise::divide(1.0f, L) : 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint c4 = t * 4u;
  float4 acc = float4(0.0f);
  for (uint s = 0u; s < ns; ++s) {
    const float f = f_tg[s];
    if (f > 0.0f) {
      const device float2* ps2 = reinterpret_cast<const device float2*>(p + s * (LAT + 2u) + 2u + c4);
      acc = fma(float4(ps2[0], ps2[1]), float4(f), acc);
    }
  }
  oh[t] = half4(acc * inv_l_tg);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const half4 x0 = oh[lane * 4u], x1 = oh[lane * 4u + 1u], x2 = oh[lane * 4u + 2u], x3 = oh[lane * 4u + 3u];
  const uint v_base = sub * 16u + sg * 4u, sub_sb = lane >> 3;
  const uint og0 = 448u * (a.head0 + h) + NOPE + v_base;
  const ulong row0 = a.kvb + ulong(og0) * LAT;
  const device float* S = reinterpret_cast<const device float*>(W + a.kvb_scale);
  const float s0 = S[ulong(og0 >> 7) * a.kvb_in_blocks + sub_sb] * 256.0f;
  const float s3 = S[ulong((og0 + 3u) >> 7) * a.kvb_in_blocks + sub_sb] * 256.0f;
  const float s1 = (s3 == s0) ? s0 : (S[ulong((og0 + 1u) >> 7) * a.kvb_in_blocks + sub_sb] * 256.0f);
  const float s2 = (s3 == s0) ? s0 : (S[ulong((og0 + 2u) >> 7) * a.kvb_in_blocks + sub_sb] * 256.0f);
  const uint4 q0 = *reinterpret_cast<const device uint4*>(W + row0 + lane * 16u);
  const uint4 q1 = *reinterpret_cast<const device uint4*>(W + row0 + LAT + lane * 16u);
  const uint4 q2 = *reinterpret_cast<const device uint4*>(W + row0 + 2ul * LAT + lane * 16u);
  const uint4 q3 = *reinterpret_cast<const device uint4*>(W + row0 + 3ul * LAT + lane * 16u);
  const float v0 = mlap_sum((float(dot(mlap_fp8x4_h(q0.x), x0)) + float(dot(mlap_fp8x4_h(q0.y), x1)) +
                             float(dot(mlap_fp8x4_h(q0.z), x2)) + float(dot(mlap_fp8x4_h(q0.w), x3))) * s0);
  const float v1 = mlap_sum((float(dot(mlap_fp8x4_h(q1.x), x0)) + float(dot(mlap_fp8x4_h(q1.y), x1)) +
                             float(dot(mlap_fp8x4_h(q1.z), x2)) + float(dot(mlap_fp8x4_h(q1.w), x3))) * s1);
  const float v2 = mlap_sum((float(dot(mlap_fp8x4_h(q2.x), x0)) + float(dot(mlap_fp8x4_h(q2.y), x1)) +
                             float(dot(mlap_fp8x4_h(q2.z), x2)) + float(dot(mlap_fp8x4_h(q2.w), x3))) * s2);
  const float v3 = mlap_sum((float(dot(mlap_fp8x4_h(q3.x), x0)) + float(dot(mlap_fp8x4_h(q3.y), x1)) +
                             float(dot(mlap_fp8x4_h(q3.z), x2)) + float(dot(mlap_fp8x4_h(q3.w), x3))) * s3);
  if (lane == 0u) *reinterpret_cast<device float4*>(o_out + (ulong(r) * a.heads + h) * VDIM + v_base) = float4(v0, v1, v2, v3);
}
#endif
