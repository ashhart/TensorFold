// Qwen27 TFQ3 dense prefill kernels using tfp::mma_16x32 (M5 NAX and pre-M5 simdgroup_matrix).
template <typename T> using pw_frag = tfp::frag<T>;

inline void pw_tile_4b(thread pw_frag<half>& f, thread uint& st, uint wc, constant Q27PwV3Tables* T) {
  half4 q0, q1; pw_v3_q8_4b(st, wc, T, q0, q1);
  f[0]=q0.x; f[1]=q0.y; f[2]=q0.z; f[3]=q0.w; f[4]=q1.x; f[5]=q1.y; f[6]=q1.z; f[7]=q1.w;
}

inline void pw_tile_3b(thread pw_frag<half>& f, thread uint& st, uint m24, constant Q27PwV3Tables* T) {
  half4 q0, q1; pw_v3_q8_3b(st, m24, T, q0, q1);
  f[0]=q0.x; f[1]=q0.y; f[2]=q0.z; f[3]=q0.w; f[4]=q1.x; f[5]=q1.y; f[6]=q1.z; f[7]=q1.w;
}

inline void pw_tile_5b(thread pw_frag<half>& f, thread uint& st, uint hi8, uint lo32, constant Q27PwV3Tables* T) {
  half4 q0, q1; pw_v3_q8_5b(st, hi8, lo32, T, q0, q1);
  f[0]=q0.x; f[1]=q0.y; f[2]=q0.z; f[3]=q0.w; f[4]=q1.x; f[5]=q1.y; f[6]=q1.z; f[7]=q1.w;
}

inline void pw_frag_get_tg(thread pw_frag<half>& f, const threadgroup half* p, int ld, int r, int c, short2 home) {
  tfp::frag_get(f, p, ld, r, c, home);
}

template <typename T>
inline void pw_frag_put_tg(thread const pw_frag<float>& f, threadgroup T* p, int ld, int r, int c, short2 home, int nr) {
  Q27_PW_UNROLL for (short e = 0; e < 8; e++) {
    int rr = r + home.y + (e >> 2) * 8;
    if (rr < nr) p[rr * ld + (c + home.x + TF_COL(e))] = T(f[e]);
  }
}

#ifdef TF_SIMD_FRAGS
inline void pw_stage_b(threadgroup half* dst, thread const pw_frag<half>& b, short2 m5h) {
  threadgroup half* q = dst + m5h.y * 16 + m5h.x;
  Q27_PW_UNROLL for (short e = 0; e < 8; e++) q[(e >> 2) * 128 + (e & 3)] = b[e];
}
#define Q27_NAX_STEP32(K_IDX) { \
  pw_stage_b(tg_b[sg], b0, m5h); pw_stage_b(tg_b[sg] + 256, b1, m5h); \
  simdgroup_barrier(mem_flags::mem_threadgroup); \
  pw_frag<half> tb0, tb1, a0; \
  tfp::frag_get_t(tb0, tg_b[sg], 16, 0, 0, home); tfp::frag_get_t(tb1, tg_b[sg] + 256, 16, 0, 0, home); \
  pw_frag_get_tg(a0, tg_a, 128, 0, (K_IDX) * 16, home); tfp::mma_16x32<false, true>(bc0_lo, bc0_hi, a0, tb0, tb1); \
  if (nb > 16u) { pw_frag<half> a1; pw_frag_get_tg(a1, tg_a, 128, 16, (K_IDX) * 16, home); \
    tfp::mma_16x32<false, true>(bc1_lo, bc1_hi, a1, tb0, tb1); } }
#else
#define Q27_NAX_STEP32(K_IDX) { \
  pw_frag<half> a0; pw_frag_get_tg(a0, tg_a, 128, 0, (K_IDX) * 16, home); \
  tfp::mma_16x32<false, true>(bc0_lo, bc0_hi, a0, b0, b1); \
  if (nb > 16u) { pw_frag<half> a1; pw_frag_get_tg(a1, tg_a, 128, 16, (K_IDX) * 16, home); \
    tfp::mma_16x32<false, true>(bc1_lo, bc1_hi, a1, b0, b1); } }
#endif

#define Q27_NAX_SCALE_ACC(SV) \
  { float fsv = float(SV); \
    c0_lo += bc0_lo * fsv; c0_hi += bc0_hi * fsv; \
    if (nb > 16u) { c1_lo += bc1_lo * fsv; c1_hi += bc1_hi * fsv; } }

template <uint BITS>
[[kernel]] void q27_pw_dense_prefill_t(
    const device uint* W       [[buffer(0)]],
    const device half* scales  [[buffer(1)]],
    constant Q27PwV3Tables* T  [[buffer(2)]],
    const device half* X_rot   [[buffer(3)]],
    device bfloat* OUT         [[buffer(4)]],
    constant Q27PwDenseDims& d [[buffer(5)]],
    uint2 gid                  [[threadgroup_position_in_grid]],
    ushort tid                 [[thread_index_in_threadgroup]],
    ushort sg                  [[simdgroup_index_in_threadgroup]],
    ushort lane                [[thread_index_in_simdgroup]]
) {
  uint rb = gid.x, r0 = rb * 128u;
  uint m0 = gid.y * 32u;
  if (m0 >= d.rows) return;
  uint nb = metal::min(32u, d.rows - m0);
  uint n_bc = d.k >> 7;
  constexpr uint wpb = BITS * 2u;
  uint strip0 = rb * 8u + uint(sg) * 2u;
  const device uint* w_s0 = W + ulong(strip0 + 0u) * ulong(n_bc) * wpb * 32ul;
  const device uint* w_s1 = W + ulong(strip0 + 1u) * ulong(n_bc) * wpb * 32ul;
  const device half* sc0  = scales + ulong(rb) * ulong(n_bc);
  uint st0 = 0x13579BDFu ^ (uint(lane) * 0x9E3779B1u);
  uint st1 = st0;
  short2 home = tfp::frag_home(lane);
#ifdef TF_SIMD_FRAGS
  short2 m5h = pw_frag_home(lane);
#endif
  pw_frag<float> c0_lo = 0.0f, c0_hi = 0.0f, c1_lo = 0.0f, c1_hi = 0.0f;
  threadgroup float tg_f[32 * 128];
  threadgroup half* tg_a = (threadgroup half*)tg_f;
#ifdef TF_SIMD_FRAGS
  threadgroup half tg_b[4][512];
#endif
  uint row_l = tid >> 2, col_l = (tid & 3u) * 32u;

  for (uint bc = 0u; bc < n_bc; ++bc) {
    if (row_l < nb) {
      const device half4* src4 = (const device half4*)(X_rot + ulong(m0 + row_l) * ulong(d.k) + (bc << 7) + col_l);
      threadgroup half4* dst4 = (threadgroup half4*)(tg_a + row_l * 128u + col_l);
      Q27_PW_UNROLL for (uint j = 0u; j < 8u; ++j) dst4[j] = src4[j];
    } else if (row_l < ((nb + 15u) & ~15u)) {
      threadgroup half4* dst4 = (threadgroup half4*)(tg_a + row_l * 128u + col_l);
      Q27_PW_UNROLL for (uint j = 0u; j < 8u; ++j) dst4[j] = half4(0.0h);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    pw_frag<float> bc0_lo = 0.0f, bc0_hi = 0.0f, bc1_lo = 0.0f, bc1_hi = 0.0f;
    pw_frag<half> b0, b1;
    if constexpr (BITS == 4u) {
      const device uint4* p0 = (const device uint4*)(w_s0 + ulong(bc) * 8u * 32u);
      const device uint4* p1 = (const device uint4*)(w_s1 + ulong(bc) * 8u * 32u);
      for (uint h = 0u; h < 2u; ++h) {
        uint4 v0 = p0[h * 32u + lane], v1 = p1[h * 32u + lane];
        int kb = int(h * 4u);
        pw_tile_4b(b0, st0, v0.x, T); pw_tile_4b(b1, st1, v1.x, T); Q27_NAX_STEP32(kb + 0);
        pw_tile_4b(b0, st0, v0.y, T); pw_tile_4b(b1, st1, v1.y, T); Q27_NAX_STEP32(kb + 1);
        pw_tile_4b(b0, st0, v0.z, T); pw_tile_4b(b1, st1, v1.z, T); Q27_NAX_STEP32(kb + 2);
        pw_tile_4b(b0, st0, v0.w, T); pw_tile_4b(b1, st1, v1.w, T); Q27_NAX_STEP32(kb + 3);
      }
    } else if constexpr (BITS == 3u) {
      for (uint h = 0u; h < 2u; ++h) {
        const device uint* p0 = w_s0 + (ulong(bc) * 6u + h * 3u) * 32u + lane;
        const device uint* p1 = w_s1 + (ulong(bc) * 6u + h * 3u) * 32u + lane;
        uint a0 = p0[0], a1 = p0[32], a2 = p0[64];
        uint q0 = p1[0], q1 = p1[32], q2 = p1[64];
        int kb = int(h * 4u);
        pw_tile_3b(b0, st0, a0 >> 8, T);
        pw_tile_3b(b1, st1, q0 >> 8, T);
        Q27_NAX_STEP32(kb + 0);
        pw_tile_3b(b0, st0, ((a0 << 16) | (a1 >> 16)) & 0xFFFFFFu, T);
        pw_tile_3b(b1, st1, ((q0 << 16) | (q1 >> 16)) & 0xFFFFFFu, T);
        Q27_NAX_STEP32(kb + 1);
        pw_tile_3b(b0, st0, ((a1 << 8) | (a2 >> 24)) & 0xFFFFFFu, T);
        pw_tile_3b(b1, st1, ((q1 << 8) | (q2 >> 24)) & 0xFFFFFFu, T);
        Q27_NAX_STEP32(kb + 2);
        pw_tile_3b(b0, st0, a2 & 0xFFFFFFu, T);
        pw_tile_3b(b1, st1, q2 & 0xFFFFFFu, T);
        Q27_NAX_STEP32(kb + 3);
      }
    } else {
      for (uint h = 0u; h < 2u; ++h) {
        const device uint* p0 = w_s0 + (ulong(bc) * 10u + h * 5u) * 32u + lane;
        const device uint* p1 = w_s1 + (ulong(bc) * 10u + h * 5u) * 32u + lane;
        uint a0 = p0[0], a1 = p0[32], a2 = p0[64], a3 = p0[96], a4 = p0[128];
        uint q0 = p1[0], q1 = p1[32], q2 = p1[64], q3 = p1[96], q4 = p1[128];
        int kb = int(h * 4u);
        pw_tile_5b(b0, st0, a0 >> 24, (a0 << 8) | (a1 >> 24), T);
        pw_tile_5b(b1, st1, q0 >> 24, (q0 << 8) | (q1 >> 24), T);
        Q27_NAX_STEP32(kb + 0);
        pw_tile_5b(b0, st0, (a1 >> 16) & 255u, (a1 << 16) | (a2 >> 16), T);
        pw_tile_5b(b1, st1, (q1 >> 16) & 255u, (q1 << 16) | (q2 >> 16), T);
        Q27_NAX_STEP32(kb + 1);
        pw_tile_5b(b0, st0, (a2 >> 8) & 255u, (a2 << 24) | (a3 >> 8), T);
        pw_tile_5b(b1, st1, (q2 >> 8) & 255u, (q2 << 24) | (q3 >> 8), T);
        Q27_NAX_STEP32(kb + 2);
        pw_tile_5b(b0, st0, a3 & 255u, a4, T);
        pw_tile_5b(b1, st1, q3 & 255u, q4, T);
        Q27_NAX_STEP32(kb + 3);
      }
    }
    Q27_NAX_SCALE_ACC(sc0[bc]);
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
    uint sr0 = q27_pw_sr0(r0, d.sr_split);
    device bfloat* dst = OUT + ulong(m0 + m) * ulong(d.ldo) + r0;
    dst[lane +  0u] = bfloat(v.x * pw_sgn(0x9E3779B9u, sr0 + lane +  0u));
    dst[lane + 32u] = bfloat(v.y * pw_sgn(0x9E3779B9u, sr0 + lane + 32u));
    dst[lane + 64u] = bfloat(v.z * pw_sgn(0x9E3779B9u, sr0 + lane + 64u));
    dst[lane + 96u] = bfloat(v.w * pw_sgn(0x9E3779B9u, sr0 + lane + 96u));
  }
}

template [[host_name("q27_pw_dense_prefill_3b")]] [[kernel]] decltype(q27_pw_dense_prefill_t<3u>) q27_pw_dense_prefill_t<3u>;
template [[host_name("q27_pw_dense_prefill_4b")]] [[kernel]] decltype(q27_pw_dense_prefill_t<4u>) q27_pw_dense_prefill_t<4u>;
template [[host_name("q27_pw_dense_prefill_5b")]] [[kernel]] decltype(q27_pw_dense_prefill_t<5u>) q27_pw_dense_prefill_t<5u>;
