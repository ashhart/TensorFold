// TFQ3 FlashNext prompt prefill kernels using tfp::mma_16x32 (M5 NAX and pre-M5 simdgroup_matrix).
template <typename T> using pw_frag = tfp::frag<T>;

inline void pw_tile_4b(thread pw_frag<half>& f, thread uint& st, uint wc, constant FnPwV3Tables* T) {
  half4 q0, q1; pw_v3_q8_4b(st, wc, T, q0, q1);
  f[0]=q0.x; f[1]=q0.y; f[2]=q0.z; f[3]=q0.w; f[4]=q1.x; f[5]=q1.y; f[6]=q1.z; f[7]=q1.w;
}
inline void pw_tile_3b(thread pw_frag<half>& f, thread uint& st, uint m24, constant FnPwV3Tables* T) {
  half4 q0, q1; pw_v3_q8_3b(st, m24, T, q0, q1);
  f[0]=q0.x; f[1]=q0.y; f[2]=q0.z; f[3]=q0.w; f[4]=q1.x; f[5]=q1.y; f[6]=q1.z; f[7]=q1.w;
}
inline void pw_tile_5b(thread pw_frag<half>& f, thread uint& st, uint hi8, uint lo32, constant FnPwV3Tables* T) {
  half4 q0, q1; pw_v3_q8_5b(st, hi8, lo32, T, q0, q1);
  f[0]=q0.x; f[1]=q0.y; f[2]=q0.z; f[3]=q0.w; f[4]=q1.x; f[5]=q1.y; f[6]=q1.z; f[7]=q1.w;
}

[[kernel]] void fn_pw_prefill_fwht_x(
    const device bfloat* XS [[buffer(0)]],
    device half* X_rot      [[buffer(1)]],
    constant FnPwDims& d    [[buffer(2)]],
    uint2 gid               [[threadgroup_position_in_grid]],
    ushort lane             [[thread_index_in_simdgroup]]
) {
  uint pair = gid.y;
  if (pair >= d.slots) return;
  uint c0 = gid.x * 128u;
  const device bfloat* src = XS + ulong(pair) * 2560ul + c0;
  float4 v;
  v.x = float(src[lane +  0u]) * pw_sgn(0x12345678u, c0 + lane +  0u);
  v.y = float(src[lane + 32u]) * pw_sgn(0x12345678u, c0 + lane + 32u);
  v.z = float(src[lane + 64u]) * pw_sgn(0x12345678u, c0 + lane + 64u);
  v.w = float(src[lane + 96u]) * pw_sgn(0x12345678u, c0 + lane + 96u);
  pw_fwht128_simd4(v, lane);
  device half* dst = X_rot + ulong(pair) * 2560ul + c0;
  dst[lane +  0u] = half(v.x); dst[lane + 32u] = half(v.y);
  dst[lane + 64u] = half(v.z); dst[lane + 96u] = half(v.w);
}

[[kernel]] void fn_pw_tile_scan(
    const device int* offsets [[buffer(0)]],
    device uint* tile_meta    [[buffer(1)]],
    device uint* tile_cnt     [[buffer(2)]],
    constant FnPwDims& d      [[buffer(3)]],
    ushort tid                [[thread_index_in_threadgroup]],
    ushort sg                 [[simdgroup_index_in_threadgroup]],
    ushort lane               [[thread_index_in_simdgroup]]
) {
  threadgroup uint sg_tot[16];
  int lo = 0, hi = 0; uint nt = 0u;
  if (uint(tid) < d.experts) {
    lo = offsets[tid];
    hi = offsets[tid + 1u];
    nt = (hi > lo) ? uint((hi - lo + 31) >> 5) : 0u;
  }
  uint excl = simd_prefix_exclusive_sum(nt);
  if (lane == 31u) sg_tot[sg] = excl + nt;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint sg_base = 0u;
  for (ushort s = 0u; s < sg; ++s) sg_base += sg_tot[s];
  uint out_off = sg_base + excl;
  if (uint(tid) < d.experts) {
    for (uint k = 0u; k < nt; ++k) tile_meta[out_off + k] = (uint(tid) << 18) | uint(lo + int(k << 5));
  }
  if (tid == 511u) tile_cnt[0] = sg_base + excl + nt;
}

inline void pw_frag_get_tg(thread pw_frag<half>& f, const threadgroup half* p, int ld, int r, int c, short2 home) {
  tfp::frag_get(f, p, ld, r, c, home);
}

template <typename T>
inline void pw_frag_put_tg(thread const pw_frag<float>& f, threadgroup T* p, int ld, int r, int c, short2 home, int nr) {
  FN_PW_UNROLL for (short e = 0; e < 8; e++) {
    int rr = r + home.y + (e >> 2) * 8;
    if (rr < nr) p[rr * ld + (c + home.x + TF_COL(e))] = T(f[e]);
  }
}

#ifdef TF_SIMD_FRAGS
inline void pw_stage_b(threadgroup half* dst, thread const pw_frag<half>& b, short2 m5h) {
  threadgroup half* q = dst + m5h.y * 16 + m5h.x;
  FN_PW_UNROLL for (short e = 0; e < 8; e++) q[(e >> 2) * 128 + (e & 3)] = b[e];
}
#define FN_NAX_STEP32(K_IDX) { \
  pw_stage_b(tg_b[sg], b0, m5h); pw_stage_b(tg_b[sg] + 256, b1, m5h); \
  simdgroup_barrier(mem_flags::mem_threadgroup); \
  pw_frag<half> tb0, tb1, a0; \
  tfp::frag_get_t(tb0, tg_b[sg], 16, 0, 0, home); tfp::frag_get_t(tb1, tg_b[sg] + 256, 16, 0, 0, home); \
  pw_frag_get_tg(a0, tg_a, 128, 0, (K_IDX) * 16, home); tfp::mma_16x32<false, true>(bc0_lo, bc0_hi, a0, tb0, tb1); \
  if (nb > 16u) { pw_frag<half> a1; pw_frag_get_tg(a1, tg_a, 128, 16, (K_IDX) * 16, home); \
    tfp::mma_16x32<false, true>(bc1_lo, bc1_hi, a1, tb0, tb1); } }
#else
#define FN_NAX_STEP32(K_IDX) { \
  pw_frag<half> a0; pw_frag_get_tg(a0, tg_a, 128, 0, (K_IDX) * 16, home); \
  tfp::mma_16x32<false, true>(bc0_lo, bc0_hi, a0, b0, b1); \
  if (nb > 16u) { pw_frag<half> a1; pw_frag_get_tg(a1, tg_a, 128, 16, (K_IDX) * 16, home); \
    tfp::mma_16x32<false, true>(bc1_lo, bc1_hi, a1, b0, b1); } }
#endif

#define FN_NAX_SCALE_ACC(SV) \
  { float fsv = float(SV); \
    c0_lo += bc0_lo * fsv; c0_hi += bc0_hi * fsv; \
    if (nb > 16u) { c1_lo += bc1_lo * fsv; c1_hi += bc1_hi * fsv; } }

template <bool HAS_5B>
[[kernel]] void fn_pw_prefill_gu_t(
    const device uint* W           [[buffer(0)]],
    const device half* scales      [[buffer(1)]],
    const device FnPwExpertMeta* M [[buffer(2)]],
    constant FnPwV3Tables* T       [[buffer(3)]],
    const device half* X_rot       [[buffer(4)]],
    const device int* offsets      [[buffer(5)]],
    device half* ACT_rot           [[buffer(6)]],
    constant FnPwDims& d           [[buffer(7)]],
    const device uint* tile_meta   [[buffer(8)]],
    const device uint* tile_cnt    [[buffer(9)]],
    uint2 gid                      [[threadgroup_position_in_grid]],
    ushort tid                     [[thread_index_in_threadgroup]],
    ushort sg                      [[simdgroup_index_in_threadgroup]],
    ushort lane                    [[thread_index_in_simdgroup]]
) {
  uint tidx = gid.y;
  if (tidx >= tile_cnt[0]) return;
  uint meta = tile_meta[tidx];
  uint e = meta >> 18;
  int base = int(meta & 0x3FFFFu);
  int hi = offsets[e + 1u];
  uint nb = uint(metal::min(32, hi - base));
  uint rb = gid.x, r0 = rb * 128u;

  threadgroup half tg_a[32 * 128];
  threadgroup half tg_g[32 * 128];
#ifdef TF_SIMD_FRAGS
  threadgroup half tg_b[8][512];
#endif

  FnPwExpertMeta em = M[e];
  ushort is_up = sg >> 2;
  ushort sg_n  = sg & 3u;
  uint b_bits = em.b_bits;
  uint wpb = b_bits * 2u;
  ulong w_off = is_up ? em.up_word_off : em.gate_word_off;
  uint sc_off = is_up ? em.up_sc_off   : em.gate_sc_off;
  uint strip0 = rb * 8u + uint(sg_n) * 2u;
  const device uint* w_s0 = W + w_off + ulong(strip0 + 0u) * 20ul * wpb * 32ul;
  const device uint* w_s1 = W + w_off + ulong(strip0 + 1u) * 20ul * wpb * 32ul;
  const device half* sc0  = scales + ulong(sc_off) + ulong(rb) * 20ul;
  uint st0 = 0x13579BDFu ^ (uint(lane) * 0x9E3779B1u);
  uint st1 = st0;
  short2 home = tfp::frag_home(lane);
#ifdef TF_SIMD_FRAGS
  short2 m5h = pw_frag_home(lane);
#endif
  pw_frag<float> c0_lo = 0.0f, c0_hi = 0.0f, c1_lo = 0.0f, c1_hi = 0.0f;
  uint row_l = tid >> 3, col_l = (tid & 7u) * 16u;

  for (uint bc = 0u; bc < 20u; ++bc) {
    if (row_l < nb) {
      const device half4* src4 = (const device half4*)(X_rot + ulong(base + int(row_l)) * 2560ul + (bc << 7) + col_l);
      threadgroup half4* dst4 = (threadgroup half4*)(tg_a + row_l * 128u + col_l);
      dst4[0] = src4[0]; dst4[1] = src4[1]; dst4[2] = src4[2]; dst4[3] = src4[3];
    } else if (row_l < ((nb + 15u) & ~15u)) {
      threadgroup half4* dst4 = (threadgroup half4*)(tg_a + row_l * 128u + col_l);
      dst4[0] = half4(0.0h); dst4[1] = half4(0.0h); dst4[2] = half4(0.0h); dst4[3] = half4(0.0h);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    pw_frag<float> bc0_lo = 0.0f, bc0_hi = 0.0f, bc1_lo = 0.0f, bc1_hi = 0.0f;
    pw_frag<half> b0, b1;
    if (b_bits == 4u) {
      const device uint4* p0 = (const device uint4*)(w_s0 + ulong(bc) * 8u * 32u);
      const device uint4* p1 = (const device uint4*)(w_s1 + ulong(bc) * 8u * 32u);
      for (uint h = 0u; h < 2u; ++h) {
        uint4 v0 = p0[h * 32u + lane], v1 = p1[h * 32u + lane];
        int kb = int(h * 4u);
        pw_tile_4b(b0, st0, v0.x, T); pw_tile_4b(b1, st1, v1.x, T); FN_NAX_STEP32(kb + 0);
        pw_tile_4b(b0, st0, v0.y, T); pw_tile_4b(b1, st1, v1.y, T); FN_NAX_STEP32(kb + 1);
        pw_tile_4b(b0, st0, v0.z, T); pw_tile_4b(b1, st1, v1.z, T); FN_NAX_STEP32(kb + 2);
        pw_tile_4b(b0, st0, v0.w, T); pw_tile_4b(b1, st1, v1.w, T); FN_NAX_STEP32(kb + 3);
      }
    } else if (HAS_5B && b_bits == 5u) {
      for (uint h = 0u; h < 2u; ++h) {
        const device uint* p0 = w_s0 + (ulong(bc) * 10u + h * 5u) * 32u + lane;
        const device uint* p1 = w_s1 + (ulong(bc) * 10u + h * 5u) * 32u + lane;
        uint a0=p0[0],a1=p0[32],a2=p0[64],a3=p0[96],a4=p0[128];
        uint q0=p1[0],q1=p1[32],q2=p1[64],q3=p1[96],q4=p1[128];
        int kb = int(h * 4u);
        pw_tile_5b(b0,st0,a0>>24,(a0<<8)|(a1>>24),T); pw_tile_5b(b1,st1,q0>>24,(q0<<8)|(q1>>24),T); FN_NAX_STEP32(kb + 0);
        pw_tile_5b(b0,st0,(a1>>16)&255u,(a1<<16)|(a2>>16),T); pw_tile_5b(b1,st1,(q1>>16)&255u,(q1<<16)|(q2>>16),T); FN_NAX_STEP32(kb + 1);
        pw_tile_5b(b0,st0,(a2>>8)&255u,(a2<<24)|(a3>>8),T); pw_tile_5b(b1,st1,(q2>>8)&255u,(q2<<24)|(q3>>8),T); FN_NAX_STEP32(kb + 2);
        pw_tile_5b(b0,st0,a3&255u,a4,T); pw_tile_5b(b1,st1,q3&255u,q4,T); FN_NAX_STEP32(kb + 3);
      }
    } else {
      for (uint h = 0u; h < 2u; ++h) {
        const device uint* p0 = w_s0 + (ulong(bc) * 6u + h * 3u) * 32u + lane;
        const device uint* p1 = w_s1 + (ulong(bc) * 6u + h * 3u) * 32u + lane;
        uint a0 = p0[0], a1 = p0[32], a2 = p0[64];
        uint q0 = p1[0], q1 = p1[32], q2 = p1[64];
        int kb = int(h * 4u);
        pw_tile_3b(b0,st0,a0>>8,T); pw_tile_3b(b1,st1,q0>>8,T); FN_NAX_STEP32(kb + 0);
        pw_tile_3b(b0,st0,((a0<<16)|(a1>>16))&0xFFFFFFu,T); pw_tile_3b(b1,st1,((q0<<16)|(q1>>16))&0xFFFFFFu,T); FN_NAX_STEP32(kb + 1);
        pw_tile_3b(b0,st0,((a1<<8)|(a2>>24))&0xFFFFFFu,T); pw_tile_3b(b1,st1,((q1<<8)|(q2>>24))&0xFFFFFFu,T); FN_NAX_STEP32(kb + 2);
        pw_tile_3b(b0,st0,a2&0xFFFFFFu,T); pw_tile_3b(b1,st1,q2&0xFFFFFFu,T); FN_NAX_STEP32(kb + 3);
      }
    }
    FN_NAX_SCALE_ACC(sc0[bc]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  threadgroup half* dst_tg = is_up ? tg_a : tg_g;
  pw_frag_put_tg(c0_lo, dst_tg, 128,  0, int(sg_n * 32u),       home, int(nb));
  pw_frag_put_tg(c0_hi, dst_tg, 128,  0, int(sg_n * 32u + 16u), home, int(nb));
  if (nb > 16u) {
    pw_frag_put_tg(c1_lo, dst_tg, 128, 16, int(sg_n * 32u),       home, int(nb));
    pw_frag_put_tg(c1_hi, dst_tg, 128, 16, int(sg_n * 32u + 16u), home, int(nb));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint m = uint(sg); m < nb; m += 8u) {
    const threadgroup half* gs = tg_g + m * 128u;
    const threadgroup half* us = tg_a + m * 128u;
    float4 gv = float4(gs[lane], gs[lane + 32u], gs[lane + 64u], gs[lane + 96u]);
    float4 uv = float4(us[lane], us[lane + 32u], us[lane + 64u], us[lane + 96u]);
    pw_fwht128_simd4(gv, lane);
    pw_fwht128_simd4(uv, lane);
    float4 z;
    z.x = float(fn_pw_swiglu(gv.x * pw_sgn(0x9E3779B9u, r0 + lane +  0u), uv.x * pw_sgn(0x9E3779B9u, r0 + lane +  0u))) * pw_sgn(0x12345678u, r0 + lane +  0u);
    z.y = float(fn_pw_swiglu(gv.y * pw_sgn(0x9E3779B9u, r0 + lane + 32u), uv.y * pw_sgn(0x9E3779B9u, r0 + lane + 32u))) * pw_sgn(0x12345678u, r0 + lane + 32u);
    z.z = float(fn_pw_swiglu(gv.z * pw_sgn(0x9E3779B9u, r0 + lane + 64u), uv.z * pw_sgn(0x9E3779B9u, r0 + lane + 64u))) * pw_sgn(0x12345678u, r0 + lane + 64u);
    z.w = float(fn_pw_swiglu(gv.w * pw_sgn(0x9E3779B9u, r0 + lane + 96u), uv.w * pw_sgn(0x9E3779B9u, r0 + lane + 96u))) * pw_sgn(0x12345678u, r0 + lane + 96u);
    pw_fwht128_simd4(z, lane);
    device half* dst = ACT_rot + ulong(base + int(m)) * 640ul + r0;
    dst[lane +  0u] = half(z.x); dst[lane + 32u] = half(z.y);
    dst[lane + 64u] = half(z.z); dst[lane + 96u] = half(z.w);
  }
}

template <bool HAS_5B>
[[kernel]] void fn_pw_prefill_down_t(
    const device uint* W           [[buffer(0)]],
    const device half* scales      [[buffer(1)]],
    const device FnPwExpertMeta* M [[buffer(2)]],
    constant FnPwV3Tables* T       [[buffer(3)]],
    const device half* ACT_rot     [[buffer(4)]],
    const device int* offsets      [[buffer(5)]],
    device bfloat* DS_out          [[buffer(6)]],
    constant FnPwDims& d           [[buffer(7)]],
    const device uint* tile_meta   [[buffer(8)]],
    const device uint* tile_cnt    [[buffer(9)]],
    uint2 gid                      [[threadgroup_position_in_grid]],
    ushort tid                     [[thread_index_in_threadgroup]],
    ushort sg                      [[simdgroup_index_in_threadgroup]],
    ushort lane                    [[thread_index_in_simdgroup]]
) {
  uint tidx = gid.y;
  if (tidx >= tile_cnt[0]) return;
  uint meta = tile_meta[tidx];
  uint e = meta >> 18;
  int base = int(meta & 0x3FFFFu);
  int hi = offsets[e + 1u];
  uint nb = uint(metal::min(32, hi - base));
  uint rb = gid.x, r0 = rb * 128u;

  threadgroup float tg_f[32 * 128];
  threadgroup half* tg_a = (threadgroup half*)tg_f;
#ifdef TF_SIMD_FRAGS
  threadgroup half tg_b[4][512];
#endif

  FnPwExpertMeta em = M[e];
  uint b_bits = em.b_bits;
  uint wpb = b_bits * 2u;
  uint strip0 = rb * 8u + uint(sg) * 2u;
  const device uint* w_s0 = W + em.down_word_off + ulong(strip0 + 0u) * 5ul * wpb * 32ul;
  const device uint* w_s1 = W + em.down_word_off + ulong(strip0 + 1u) * 5ul * wpb * 32ul;
  const device half* sc0  = scales + ulong(em.down_sc_off) + ulong(rb) * 5ul;
  uint st0 = 0x13579BDFu ^ (uint(lane) * 0x9E3779B1u);
  uint st1 = st0;
  short2 home = tfp::frag_home(lane);
#ifdef TF_SIMD_FRAGS
  short2 m5h = pw_frag_home(lane);
#endif
  pw_frag<float> c0_lo = 0.0f, c0_hi = 0.0f, c1_lo = 0.0f, c1_hi = 0.0f;
  uint row_l = tid >> 2, col_l = (tid & 3u) * 32u;

  for (uint bc = 0u; bc < 5u; ++bc) {
    if (row_l < nb) {
      const device half4* src4 = (const device half4*)(ACT_rot + ulong(base + int(row_l)) * 640ul + (bc << 7) + col_l);
      threadgroup half4* dst4 = (threadgroup half4*)(tg_a + row_l * 128u + col_l);
      FN_PW_UNROLL for (uint j = 0u; j < 8u; ++j) dst4[j] = src4[j];
    } else if (row_l < ((nb + 15u) & ~15u)) {
      threadgroup half4* dst4 = (threadgroup half4*)(tg_a + row_l * 128u + col_l);
      FN_PW_UNROLL for (uint j = 0u; j < 8u; ++j) dst4[j] = half4(0.0h);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    pw_frag<float> bc0_lo = 0.0f, bc0_hi = 0.0f, bc1_lo = 0.0f, bc1_hi = 0.0f;
    pw_frag<half> b0, b1;
    if (b_bits == 4u) {
      const device uint4* p0 = (const device uint4*)(w_s0 + ulong(bc) * 8u * 32u);
      const device uint4* p1 = (const device uint4*)(w_s1 + ulong(bc) * 8u * 32u);
      for (uint h = 0u; h < 2u; ++h) {
        uint4 v0 = p0[h * 32u + lane], v1 = p1[h * 32u + lane];
        int kb = int(h * 4u);
        pw_tile_4b(b0, st0, v0.x, T); pw_tile_4b(b1, st1, v1.x, T); FN_NAX_STEP32(kb + 0);
        pw_tile_4b(b0, st0, v0.y, T); pw_tile_4b(b1, st1, v1.y, T); FN_NAX_STEP32(kb + 1);
        pw_tile_4b(b0, st0, v0.z, T); pw_tile_4b(b1, st1, v1.z, T); FN_NAX_STEP32(kb + 2);
        pw_tile_4b(b0, st0, v0.w, T); pw_tile_4b(b1, st1, v1.w, T); FN_NAX_STEP32(kb + 3);
      }
    } else if (HAS_5B && b_bits == 5u) {
      for (uint h = 0u; h < 2u; ++h) {
        const device uint* p0 = w_s0 + (ulong(bc) * 10u + h * 5u) * 32u + lane;
        const device uint* p1 = w_s1 + (ulong(bc) * 10u + h * 5u) * 32u + lane;
        uint a0=p0[0],a1=p0[32],a2=p0[64],a3=p0[96],a4=p0[128];
        uint q0=p1[0],q1=p1[32],q2=p1[64],q3=p1[96],q4=p1[128];
        int kb = int(h * 4u);
        pw_tile_5b(b0,st0,a0>>24,(a0<<8)|(a1>>24),T); pw_tile_5b(b1,st1,q0>>24,(q0<<8)|(q1>>24),T); FN_NAX_STEP32(kb + 0);
        pw_tile_5b(b0,st0,(a1>>16)&255u,(a1<<16)|(a2>>16),T); pw_tile_5b(b1,st1,(q1>>16)&255u,(q1<<16)|(q2>>16),T); FN_NAX_STEP32(kb + 1);
        pw_tile_5b(b0,st0,(a2>>8)&255u,(a2<<24)|(a3>>8),T); pw_tile_5b(b1,st1,(q2>>8)&255u,(q2<<24)|(q3>>8),T); FN_NAX_STEP32(kb + 2);
        pw_tile_5b(b0,st0,a3&255u,a4,T); pw_tile_5b(b1,st1,q3&255u,q4,T); FN_NAX_STEP32(kb + 3);
      }
    } else {
      for (uint h = 0u; h < 2u; ++h) {
        const device uint* p0 = w_s0 + (ulong(bc) * 6u + h * 3u) * 32u + lane;
        const device uint* p1 = w_s1 + (ulong(bc) * 6u + h * 3u) * 32u + lane;
        uint a0 = p0[0], a1 = p0[32], a2 = p0[64];
        uint q0 = p1[0], q1 = p1[32], q2 = p1[64];
        int kb = int(h * 4u);
        pw_tile_3b(b0,st0,a0>>8,T); pw_tile_3b(b1,st1,q0>>8,T); FN_NAX_STEP32(kb + 0);
        pw_tile_3b(b0,st0,((a0<<16)|(a1>>16))&0xFFFFFFu,T); pw_tile_3b(b1,st1,((q0<<16)|(q1>>16))&0xFFFFFFu,T); FN_NAX_STEP32(kb + 1);
        pw_tile_3b(b0,st0,((a1<<8)|(a2>>24))&0xFFFFFFu,T); pw_tile_3b(b1,st1,((q1<<8)|(q2>>24))&0xFFFFFFu,T); FN_NAX_STEP32(kb + 2);
        pw_tile_3b(b0,st0,a2&0xFFFFFFu,T); pw_tile_3b(b1,st1,q2&0xFFFFFFu,T); FN_NAX_STEP32(kb + 3);
      }
    }
    FN_NAX_SCALE_ACC(sc0[bc]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  pw_frag_put_tg(c0_lo, tg_f, 128,  0, int(sg * 32u),       home, int(nb));
  pw_frag_put_tg(c0_hi, tg_f, 128,  0, int(sg * 32u + 16u), home, int(nb));
  if (nb > 16u) {
    pw_frag_put_tg(c1_lo, tg_f, 128, 16, int(sg * 32u),       home, int(nb));
    pw_frag_put_tg(c1_hi, tg_f, 128, 16, int(sg * 32u + 16u), home, int(nb));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint m = uint(sg); m < nb; m += 4u) {
    const threadgroup float* ys = tg_f + m * 128u;
    float4 v = float4(ys[lane], ys[lane + 32u], ys[lane + 64u], ys[lane + 96u]);
    pw_fwht128_simd4(v, lane);
    device bfloat* dst = DS_out + ulong(base + int(m)) * 2560ul + r0;
    dst[lane +  0u] = bfloat(v.x * pw_sgn(0x9E3779B9u, r0 + lane +  0u));
    dst[lane + 32u] = bfloat(v.y * pw_sgn(0x9E3779B9u, r0 + lane + 32u));
    dst[lane + 64u] = bfloat(v.z * pw_sgn(0x9E3779B9u, r0 + lane + 64u));
    dst[lane + 96u] = bfloat(v.w * pw_sgn(0x9E3779B9u, r0 + lane + 96u));
  }
}

template [[host_name("fn_pw_prefill_gu_543")]]   [[kernel]] decltype(fn_pw_prefill_gu_t<true>)    fn_pw_prefill_gu_t<true>;
template [[host_name("fn_pw_prefill_down_543")]] [[kernel]] decltype(fn_pw_prefill_down_t<true>)  fn_pw_prefill_down_t<true>;
template [[host_name("fn_pw_prefill_gu_43")]]    [[kernel]] decltype(fn_pw_prefill_gu_t<false>)   fn_pw_prefill_gu_t<false>;
template [[host_name("fn_pw_prefill_down_43")]]  [[kernel]] decltype(fn_pw_prefill_down_t<false>) fn_pw_prefill_down_t<false>;
