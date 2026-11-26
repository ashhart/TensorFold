// Qwen27 TFQ3 dense decode kernels (3/4/5-bit fused inv-FWHT and 2D split-K=8).

template <uint BITS>
inline uint q27_pw_init_st(const device uint* w_base, uint bc0, ushort lane) {
  if (bc0 == 0u) return 0x13579BDFu ^ (uint(lane) * 0x9E3779B1u);
  if constexpr (BITS == 4u) return ((const device uint4*)w_base)[(ulong(bc0) * 2ul - 1ul) * 32ul + lane].w;
  return w_base[(ulong(bc0) * (BITS * 2u) - 1ul) * 32ul + lane];
}

template <uint BITS>
inline void pw_dense_mv_range1(
    const device uint* w_base, const device half* sc_base, uint bc0, uint bc1,
    constant Q27PwV3Tables* T, const device half* X_r, ushort lane, short2 home,
    thread float& acc0, thread float& acc1
) {
  uint st = q27_pw_init_st<BITS>(w_base, bc0, lane);
  acc0 = 0.0f; acc1 = 0.0f;
  if constexpr (BITS == 4u) {
    const device uint4* w4 = (const device uint4*)w_base;
    for (uint bc = bc0; bc < bc1; ++bc) {
      half4 ba0 = half4(0.0h), ba1 = half4(0.0h);
      uint4 v = w4[(bc * 2u + 0u) * 32u + lane];
      uint cb = (bc << 7) + uint(home.x);
      half4 q0, q1;
      pw_v3_q8_4b(st, v.x, T, q0, q1); Q27_PW_DOT1(q0, q1, X_r, cb +   0u, ba0, ba1);
      pw_v3_q8_4b(st, v.y, T, q0, q1); Q27_PW_DOT1(q0, q1, X_r, cb +  16u, ba0, ba1);
      pw_v3_q8_4b(st, v.z, T, q0, q1); Q27_PW_DOT1(q0, q1, X_r, cb +  32u, ba0, ba1);
      pw_v3_q8_4b(st, v.w, T, q0, q1); Q27_PW_DOT1(q0, q1, X_r, cb +  48u, ba0, ba1);
      v = w4[(bc * 2u + 1u) * 32u + lane];
      pw_v3_q8_4b(st, v.x, T, q0, q1); Q27_PW_DOT1(q0, q1, X_r, cb +  64u, ba0, ba1);
      pw_v3_q8_4b(st, v.y, T, q0, q1); Q27_PW_DOT1(q0, q1, X_r, cb +  80u, ba0, ba1);
      pw_v3_q8_4b(st, v.z, T, q0, q1); Q27_PW_DOT1(q0, q1, X_r, cb +  96u, ba0, ba1);
      pw_v3_q8_4b(st, v.w, T, q0, q1); Q27_PW_DOT1(q0, q1, X_r, cb + 112u, ba0, ba1);
      float sc = float(sc_base[bc]);
      acc0 = fma(pw_fin(ba0), sc, acc0); acc1 = fma(pw_fin(ba1), sc, acc1);
    }
  } else if constexpr (BITS == 3u) {
    for (uint bc = bc0; bc < bc1; ++bc) {
      half4 ba0 = half4(0.0h), ba1 = half4(0.0h);
      for (uint h = 0u; h < 2u; ++h) {
        const device uint* blk = w_base + (ulong(bc) * 6u + h * 3u) * 32u + lane;
        uint a0 = blk[0], a1 = blk[32], a2 = blk[64];
        uint cb = (bc << 7) + (h << 6) + uint(home.x);
        half4 q0, q1;
        pw_v3_q8_3b(st, a0 >> 8,                               T, q0, q1); Q27_PW_DOT1(q0, q1, X_r, cb +  0u, ba0, ba1);
        pw_v3_q8_3b(st, ((a0 << 16) | (a1 >> 16)) & 0xFFFFFFu, T, q0, q1); Q27_PW_DOT1(q0, q1, X_r, cb + 16u, ba0, ba1);
        pw_v3_q8_3b(st, ((a1 <<  8) | (a2 >> 24)) & 0xFFFFFFu, T, q0, q1); Q27_PW_DOT1(q0, q1, X_r, cb + 32u, ba0, ba1);
        pw_v3_q8_3b(st, a2 & 0xFFFFFFu,                        T, q0, q1); Q27_PW_DOT1(q0, q1, X_r, cb + 48u, ba0, ba1);
      }
      float sc = float(sc_base[bc]);
      acc0 = fma(pw_fin(ba0), sc, acc0); acc1 = fma(pw_fin(ba1), sc, acc1);
    }
  } else {
    for (uint bc = bc0; bc < bc1; ++bc) {
      half4 ba0 = half4(0.0h), ba1 = half4(0.0h);
      for (uint h = 0u; h < 2u; ++h) {
        const device uint* blk = w_base + (ulong(bc) * 10u + h * 5u) * 32u + lane;
        uint w0 = blk[0], w1 = blk[32], w2 = blk[64], w3 = blk[96], w4 = blk[128];
        uint cb = (bc << 7) + (h << 6) + uint(home.x);
        half4 q0, q1;
        pw_v3_q8_5b(st, w0 >> 24,          (w0 <<  8) | (w1 >> 24), T, q0, q1); Q27_PW_DOT1(q0, q1, X_r, cb +  0u, ba0, ba1);
        pw_v3_q8_5b(st, (w1 >> 16) & 255u, (w1 << 16) | (w2 >> 16), T, q0, q1); Q27_PW_DOT1(q0, q1, X_r, cb + 16u, ba0, ba1);
        pw_v3_q8_5b(st, (w2 >>  8) & 255u, (w2 << 24) | (w3 >>  8), T, q0, q1); Q27_PW_DOT1(q0, q1, X_r, cb + 32u, ba0, ba1);
        pw_v3_q8_5b(st, w3 & 255u,          w4,                     T, q0, q1); Q27_PW_DOT1(q0, q1, X_r, cb + 48u, ba0, ba1);
      }
      float sc = float(sc_base[bc]);
      acc0 = fma(pw_fin(ba0), sc, acc0); acc1 = fma(pw_fin(ba1), sc, acc1);
    }
  }
  acc0 += simd_shuffle_xor(acc0, 1); acc0 += simd_shuffle_xor(acc0, 8);
  acc1 += simd_shuffle_xor(acc1, 1); acc1 += simd_shuffle_xor(acc1, 8);
}

template <uint BITS>
inline void pw_dense_mv_range2(
    const device uint* w_base, const device half* sc_base, uint bc0, uint bc1,
    constant Q27PwV3Tables* T, const device half* x0, const device half* x1,
    ushort lane, short2 home, thread float2& acc0, thread float2& acc1
) {
  uint st = q27_pw_init_st<BITS>(w_base, bc0, lane);
  acc0 = float2(0.0f); acc1 = float2(0.0f);
  if constexpr (BITS == 4u) {
    const device uint4* w4 = (const device uint4*)w_base;
    for (uint bc = bc0; bc < bc1; ++bc) {
      PwH2 ba0 = {half4(0.0h), half4(0.0h)}, ba1 = {half4(0.0h), half4(0.0h)};
      uint4 v = w4[(bc * 2u + 0u) * 32u + lane];
      uint cb = (bc << 7) + uint(home.x);
      half4 q0, q1;
      pw_v3_q8_4b(st, v.x, T, q0, q1); Q27_PW_DOT2(q0, q1, x0, x1, cb +   0u, ba0, ba1);
      pw_v3_q8_4b(st, v.y, T, q0, q1); Q27_PW_DOT2(q0, q1, x0, x1, cb +  16u, ba0, ba1);
      pw_v3_q8_4b(st, v.z, T, q0, q1); Q27_PW_DOT2(q0, q1, x0, x1, cb +  32u, ba0, ba1);
      pw_v3_q8_4b(st, v.w, T, q0, q1); Q27_PW_DOT2(q0, q1, x0, x1, cb +  48u, ba0, ba1);
      v = w4[(bc * 2u + 1u) * 32u + lane];
      pw_v3_q8_4b(st, v.x, T, q0, q1); Q27_PW_DOT2(q0, q1, x0, x1, cb +  64u, ba0, ba1);
      pw_v3_q8_4b(st, v.y, T, q0, q1); Q27_PW_DOT2(q0, q1, x0, x1, cb +  80u, ba0, ba1);
      pw_v3_q8_4b(st, v.z, T, q0, q1); Q27_PW_DOT2(q0, q1, x0, x1, cb +  96u, ba0, ba1);
      pw_v3_q8_4b(st, v.w, T, q0, q1); Q27_PW_DOT2(q0, q1, x0, x1, cb + 112u, ba0, ba1);
      float sc = float(sc_base[bc]);
      acc0 = fma(pw_fin(ba0), sc, acc0); acc1 = fma(pw_fin(ba1), sc, acc1);
    }
  } else if constexpr (BITS == 3u) {
    for (uint bc = bc0; bc < bc1; ++bc) {
      PwH2 ba0 = {half4(0.0h), half4(0.0h)}, ba1 = {half4(0.0h), half4(0.0h)};
      for (uint h = 0u; h < 2u; ++h) {
        const device uint* blk = w_base + (ulong(bc) * 6u + h * 3u) * 32u + lane;
        uint a0 = blk[0], a1 = blk[32], a2 = blk[64];
        uint cb = (bc << 7) + (h << 6) + uint(home.x);
        half4 q0, q1;
        pw_v3_q8_3b(st, a0 >> 8,                               T, q0, q1); Q27_PW_DOT2(q0, q1, x0, x1, cb +  0u, ba0, ba1);
        pw_v3_q8_3b(st, ((a0 << 16) | (a1 >> 16)) & 0xFFFFFFu, T, q0, q1); Q27_PW_DOT2(q0, q1, x0, x1, cb + 16u, ba0, ba1);
        pw_v3_q8_3b(st, ((a1 <<  8) | (a2 >> 24)) & 0xFFFFFFu, T, q0, q1); Q27_PW_DOT2(q0, q1, x0, x1, cb + 32u, ba0, ba1);
        pw_v3_q8_3b(st, a2 & 0xFFFFFFu,                        T, q0, q1); Q27_PW_DOT2(q0, q1, x0, x1, cb + 48u, ba0, ba1);
      }
      float sc = float(sc_base[bc]);
      acc0 = fma(pw_fin(ba0), sc, acc0); acc1 = fma(pw_fin(ba1), sc, acc1);
    }
  } else {
    for (uint bc = bc0; bc < bc1; ++bc) {
      PwH2 ba0 = {half4(0.0h), half4(0.0h)}, ba1 = {half4(0.0h), half4(0.0h)};
      for (uint h = 0u; h < 2u; ++h) {
        const device uint* blk = w_base + (ulong(bc) * 10u + h * 5u) * 32u + lane;
        uint w0 = blk[0], w1 = blk[32], w2 = blk[64], w3 = blk[96], w4 = blk[128];
        uint cb = (bc << 7) + (h << 6) + uint(home.x);
        half4 q0, q1;
        pw_v3_q8_5b(st, w0 >> 24,          (w0 <<  8) | (w1 >> 24), T, q0, q1); Q27_PW_DOT2(q0, q1, x0, x1, cb +  0u, ba0, ba1);
        pw_v3_q8_5b(st, (w1 >> 16) & 255u, (w1 << 16) | (w2 >> 16), T, q0, q1); Q27_PW_DOT2(q0, q1, x0, x1, cb + 16u, ba0, ba1);
        pw_v3_q8_5b(st, (w2 >>  8) & 255u, (w2 << 24) | (w3 >>  8), T, q0, q1); Q27_PW_DOT2(q0, q1, x0, x1, cb + 32u, ba0, ba1);
        pw_v3_q8_5b(st, w3 & 255u,          w4,                     T, q0, q1); Q27_PW_DOT2(q0, q1, x0, x1, cb + 48u, ba0, ba1);
      }
      float sc = float(sc_base[bc]);
      acc0 = fma(pw_fin(ba0), sc, acc0); acc1 = fma(pw_fin(ba1), sc, acc1);
    }
  }
  acc0 += simd_shuffle_xor(acc0, 1); acc0 += simd_shuffle_xor(acc0, 8);
  acc1 += simd_shuffle_xor(acc1, 1); acc1 += simd_shuffle_xor(acc1, 8);
}

template <uint BITS>
inline void pw_dense_mv_range4(
    const device uint* w_base, const device half* sc_base, uint bc0, uint bc1,
    constant Q27PwV3Tables* T,
    const device half* x0, const device half* x1, const device half* x2, const device half* x3,
    ushort lane, short2 home, thread float4& acc0, thread float4& acc1
) {
  uint st = q27_pw_init_st<BITS>(w_base, bc0, lane);
  acc0 = float4(0.0f); acc1 = float4(0.0f);
  if constexpr (BITS == 4u) {
    const device uint4* w4 = (const device uint4*)w_base;
    for (uint bc = bc0; bc < bc1; ++bc) {
      PwH4 ba0 = {half4(0.0h), half4(0.0h), half4(0.0h), half4(0.0h)};
      PwH4 ba1 = {half4(0.0h), half4(0.0h), half4(0.0h), half4(0.0h)};
      uint4 v = w4[(bc * 2u + 0u) * 32u + lane];
      uint cb = (bc << 7) + uint(home.x);
      half4 q0, q1;
      pw_v3_q8_4b(st, v.x, T, q0, q1); Q27_PW_DOT4(q0, q1, x0, x1, x2, x3, cb +   0u, ba0, ba1);
      pw_v3_q8_4b(st, v.y, T, q0, q1); Q27_PW_DOT4(q0, q1, x0, x1, x2, x3, cb +  16u, ba0, ba1);
      pw_v3_q8_4b(st, v.z, T, q0, q1); Q27_PW_DOT4(q0, q1, x0, x1, x2, x3, cb +  32u, ba0, ba1);
      pw_v3_q8_4b(st, v.w, T, q0, q1); Q27_PW_DOT4(q0, q1, x0, x1, x2, x3, cb +  48u, ba0, ba1);
      v = w4[(bc * 2u + 1u) * 32u + lane];
      pw_v3_q8_4b(st, v.x, T, q0, q1); Q27_PW_DOT4(q0, q1, x0, x1, x2, x3, cb +  64u, ba0, ba1);
      pw_v3_q8_4b(st, v.y, T, q0, q1); Q27_PW_DOT4(q0, q1, x0, x1, x2, x3, cb +  80u, ba0, ba1);
      pw_v3_q8_4b(st, v.z, T, q0, q1); Q27_PW_DOT4(q0, q1, x0, x1, x2, x3, cb +  96u, ba0, ba1);
      pw_v3_q8_4b(st, v.w, T, q0, q1); Q27_PW_DOT4(q0, q1, x0, x1, x2, x3, cb + 112u, ba0, ba1);
      float sc = float(sc_base[bc]);
      acc0 = fma(pw_fin(ba0), sc, acc0); acc1 = fma(pw_fin(ba1), sc, acc1);
    }
  } else if constexpr (BITS == 3u) {
    for (uint bc = bc0; bc < bc1; ++bc) {
      PwH4 ba0 = {half4(0.0h), half4(0.0h), half4(0.0h), half4(0.0h)};
      PwH4 ba1 = {half4(0.0h), half4(0.0h), half4(0.0h), half4(0.0h)};
      for (uint h = 0u; h < 2u; ++h) {
        const device uint* blk = w_base + (ulong(bc) * 6u + h * 3u) * 32u + lane;
        uint a0 = blk[0], a1 = blk[32], a2 = blk[64];
        uint cb = (bc << 7) + (h << 6) + uint(home.x);
        half4 q0, q1;
        pw_v3_q8_3b(st, a0 >> 8,                               T, q0, q1); Q27_PW_DOT4(q0, q1, x0, x1, x2, x3, cb +  0u, ba0, ba1);
        pw_v3_q8_3b(st, ((a0 << 16) | (a1 >> 16)) & 0xFFFFFFu, T, q0, q1); Q27_PW_DOT4(q0, q1, x0, x1, x2, x3, cb + 16u, ba0, ba1);
        pw_v3_q8_3b(st, ((a1 <<  8) | (a2 >> 24)) & 0xFFFFFFu, T, q0, q1); Q27_PW_DOT4(q0, q1, x0, x1, x2, x3, cb + 32u, ba0, ba1);
        pw_v3_q8_3b(st, a2 & 0xFFFFFFu,                        T, q0, q1); Q27_PW_DOT4(q0, q1, x0, x1, x2, x3, cb + 48u, ba0, ba1);
      }
      float sc = float(sc_base[bc]);
      acc0 = fma(pw_fin(ba0), sc, acc0); acc1 = fma(pw_fin(ba1), sc, acc1);
    }
  } else {
    for (uint bc = bc0; bc < bc1; ++bc) {
      PwH4 ba0 = {half4(0.0h), half4(0.0h), half4(0.0h), half4(0.0h)};
      PwH4 ba1 = {half4(0.0h), half4(0.0h), half4(0.0h), half4(0.0h)};
      for (uint h = 0u; h < 2u; ++h) {
        const device uint* blk = w_base + (ulong(bc) * 10u + h * 5u) * 32u + lane;
        uint w0 = blk[0], w1 = blk[32], w2 = blk[64], w3 = blk[96], w4 = blk[128];
        uint cb = (bc << 7) + (h << 6) + uint(home.x);
        half4 q0, q1;
        pw_v3_q8_5b(st, w0 >> 24,          (w0 <<  8) | (w1 >> 24), T, q0, q1); Q27_PW_DOT4(q0, q1, x0, x1, x2, x3, cb +  0u, ba0, ba1);
        pw_v3_q8_5b(st, (w1 >> 16) & 255u, (w1 << 16) | (w2 >> 16), T, q0, q1); Q27_PW_DOT4(q0, q1, x0, x1, x2, x3, cb + 16u, ba0, ba1);
        pw_v3_q8_5b(st, (w2 >>  8) & 255u, (w2 << 24) | (w3 >>  8), T, q0, q1); Q27_PW_DOT4(q0, q1, x0, x1, x2, x3, cb + 32u, ba0, ba1);
        pw_v3_q8_5b(st, w3 & 255u,          w4,                     T, q0, q1); Q27_PW_DOT4(q0, q1, x0, x1, x2, x3, cb + 48u, ba0, ba1);
      }
      float sc = float(sc_base[bc]);
      acc0 = fma(pw_fin(ba0), sc, acc0); acc1 = fma(pw_fin(ba1), sc, acc1);
    }
  }
  acc0 += simd_shuffle_xor(acc0, 1); acc0 += simd_shuffle_xor(acc0, 8);
  acc1 += simd_shuffle_xor(acc1, 1); acc1 += simd_shuffle_xor(acc1, 8);
}

// Fused 128-row inv-FWHT decode (TG=256: 8 strips/TG, writes directly to OUT).
template <uint BITS>
[[kernel]] void q27_pw_dec_fused1_t(
    const device uint* W       [[buffer(0)]],
    const device half* scales  [[buffer(1)]],
    constant Q27PwV3Tables* T  [[buffer(2)]],
    const device half* X_rot   [[buffer(3)]],
    device bfloat* OUT         [[buffer(4)]],
    constant Q27PwDenseDims& d [[buffer(5)]],
    uint gid                   [[threadgroup_position_in_grid]],
    ushort sg                  [[simdgroup_index_in_threadgroup]],
    ushort lane                [[thread_index_in_simdgroup]]
) {
  uint br = gid, r0 = br * 128u, strip = br * 8u + uint(sg), n_bc = d.k >> 7;
  constexpr uint wpb = BITS * 2u;
  const device uint* w_base = W + ulong(strip) * ulong(n_bc) * wpb * 32ul;
  const device half* s_base = scales + ulong(br) * ulong(n_bc);
  short2 home = pw_frag_home(lane);
  threadgroup float tg_y[128];
  float a0, a1;
  pw_dense_mv_range1<BITS>(w_base, s_base, 0u, n_bc, T, X_rot, lane, home, a0, a1);
  if ((lane & 9u) == 0u) {
    tg_y[uint(sg) * 16u + uint(home.y)]      = a0;
    tg_y[uint(sg) * 16u + uint(home.y) + 8u] = a1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u) {
    float4 v = float4(tg_y[lane], tg_y[lane + 32u], tg_y[lane + 64u], tg_y[lane + 96u]);
    pw_fwht128_simd4(v, lane);
    uint sr0 = q27_pw_sr0(r0, d.sr_split);
    device bfloat* dst = OUT + r0;
    dst[lane +  0u] = bfloat(v.x * pw_sgn(0x9E3779B9u, sr0 + lane +  0u));
    dst[lane + 32u] = bfloat(v.y * pw_sgn(0x9E3779B9u, sr0 + lane + 32u));
    dst[lane + 64u] = bfloat(v.z * pw_sgn(0x9E3779B9u, sr0 + lane + 64u));
    dst[lane + 96u] = bfloat(v.w * pw_sgn(0x9E3779B9u, sr0 + lane + 96u));
  }
}

template <uint BITS>
[[kernel]] void q27_pw_dec_fused2_t(
    const device uint* W       [[buffer(0)]],
    const device half* scales  [[buffer(1)]],
    constant Q27PwV3Tables* T  [[buffer(2)]],
    const device half* X_rot   [[buffer(3)]],
    device bfloat* OUT         [[buffer(4)]],
    constant Q27PwDenseDims& d [[buffer(5)]],
    uint gid                   [[threadgroup_position_in_grid]],
    ushort sg                  [[simdgroup_index_in_threadgroup]],
    ushort lane                [[thread_index_in_simdgroup]]
) {
  uint br = gid, r0 = br * 128u, strip = br * 8u + uint(sg), n_bc = d.k >> 7;
  constexpr uint wpb = BITS * 2u;
  const device uint* w_base = W + ulong(strip) * ulong(n_bc) * wpb * 32ul;
  const device half* s_base = scales + ulong(br) * ulong(n_bc);
  short2 home = pw_frag_home(lane);
  threadgroup float tg_y[2][128];
  float2 a0, a1;
  pw_dense_mv_range2<BITS>(w_base, s_base, 0u, n_bc, T, X_rot, X_rot + ulong(d.k), lane, home, a0, a1);
  if ((lane & 9u) == 0u) {
    uint idx = uint(sg) * 16u + uint(home.y);
    tg_y[0][idx] = a0.x; tg_y[0][idx + 8u] = a1.x;
    tg_y[1][idx] = a0.y; tg_y[1][idx + 8u] = a1.y;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg < 2u) {
    float4 v = float4(tg_y[sg][lane], tg_y[sg][lane + 32u], tg_y[sg][lane + 64u], tg_y[sg][lane + 96u]);
    pw_fwht128_simd4(v, lane);
    uint sr0 = q27_pw_sr0(r0, d.sr_split);
    device bfloat* dst = OUT + ulong(sg) * ulong(d.ldo) + r0;
    dst[lane +  0u] = bfloat(v.x * pw_sgn(0x9E3779B9u, sr0 + lane +  0u));
    dst[lane + 32u] = bfloat(v.y * pw_sgn(0x9E3779B9u, sr0 + lane + 32u));
    dst[lane + 64u] = bfloat(v.z * pw_sgn(0x9E3779B9u, sr0 + lane + 64u));
    dst[lane + 96u] = bfloat(v.w * pw_sgn(0x9E3779B9u, sr0 + lane + 96u));
  }
}

template <uint BITS>
[[kernel]] void q27_pw_dec_fused4_t(
    const device uint* W       [[buffer(0)]],
    const device half* scales  [[buffer(1)]],
    constant Q27PwV3Tables* T  [[buffer(2)]],
    const device half* X_rot   [[buffer(3)]],
    device bfloat* OUT         [[buffer(4)]],
    constant Q27PwDenseDims& d [[buffer(5)]],
    uint gid                   [[threadgroup_position_in_grid]],
    ushort sg                  [[simdgroup_index_in_threadgroup]],
    ushort lane                [[thread_index_in_simdgroup]]
) {
  uint br = gid, r0 = br * 128u, strip = br * 8u + uint(sg), n_bc = d.k >> 7;
  constexpr uint wpb = BITS * 2u;
  const device uint* w_base = W + ulong(strip) * ulong(n_bc) * wpb * 32ul;
  const device half* s_base = scales + ulong(br) * ulong(n_bc);
  short2 home = pw_frag_home(lane);
  threadgroup float tg_y[4][128];
  for (uint rb = 0u; rb < d.rows; rb += 4u) {
    uint nc = metal::min(4u, d.rows - rb);
    const device half* x0 = X_rot + ulong(rb + 0u) * ulong(d.k);
    const device half* x1 = X_rot + ulong(rb + ((nc > 1u) ? 1u : 0u)) * ulong(d.k);
    const device half* x2 = X_rot + ulong(rb + ((nc > 2u) ? 2u : 0u)) * ulong(d.k);
    const device half* x3 = X_rot + ulong(rb + ((nc > 3u) ? 3u : 0u)) * ulong(d.k);
    float4 a0, a1;
    pw_dense_mv_range4<BITS>(w_base, s_base, 0u, n_bc, T, x0, x1, x2, x3, lane, home, a0, a1);
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
      pw_fwht128_simd4(v, lane);
      uint sr0 = q27_pw_sr0(r0, d.sr_split);
      device bfloat* dst = OUT + ulong(rb + uint(sg)) * ulong(d.ldo) + r0;
      dst[lane +  0u] = bfloat(v.x * pw_sgn(0x9E3779B9u, sr0 + lane +  0u));
      dst[lane + 32u] = bfloat(v.y * pw_sgn(0x9E3779B9u, sr0 + lane + 32u));
      dst[lane + 64u] = bfloat(v.z * pw_sgn(0x9E3779B9u, sr0 + lane + 64u));
      dst[lane + 96u] = bfloat(v.w * pw_sgn(0x9E3779B9u, sr0 + lane + 96u));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// Zero-SRAM 2D split-K=8 dense decode (grid=(n/32, 8) of TG=64, reduced in q27_pw_fwht_y).
template <uint BITS>
[[kernel]] void q27_pw_dec_2dsk8_1_t(
    const device uint* W       [[buffer(0)]],
    const device half* scales  [[buffer(1)]],
    constant Q27PwV3Tables* T  [[buffer(2)]],
    const device half* X_rot   [[buffer(3)]],
    device float* YR_out       [[buffer(4)]],
    constant Q27PwDenseDims& d [[buffer(5)]],
    uint2 gid                  [[threadgroup_position_in_grid]],
    ushort sg                  [[simdgroup_index_in_threadgroup]],
    ushort lane                [[thread_index_in_simdgroup]]
) {
  uint strip = gid.x * 2u + uint(sg), kp = gid.y, n_bc = d.k >> 7;
  uint chunk = n_bc >> 3;
  uint bc0 = kp * chunk, bc1 = bc0 + chunk;
  constexpr uint wpb = BITS * 2u;
  const device uint* w_base = W + ulong(strip) * ulong(n_bc) * wpb * 32ul;
  const device half* s_base = scales + ulong(strip >> 3) * ulong(n_bc);
  short2 home = pw_frag_home(lane);
  float a0, a1;
  pw_dense_mv_range1<BITS>(w_base, s_base, bc0, bc1, T, X_rot, lane, home, a0, a1);
  if ((lane & 9u) == 0u) {
    ulong pbase = ulong(kp) * ulong(d.n);
    YR_out[pbase + strip * 16u + uint(home.y)]      = a0;
    YR_out[pbase + strip * 16u + uint(home.y) + 8u] = a1;
  }
}

template <uint BITS>
[[kernel]] void q27_pw_dec_2dsk8_2_t(
    const device uint* W       [[buffer(0)]],
    const device half* scales  [[buffer(1)]],
    constant Q27PwV3Tables* T  [[buffer(2)]],
    const device half* X_rot   [[buffer(3)]],
    device float* YR_out       [[buffer(4)]],
    constant Q27PwDenseDims& d [[buffer(5)]],
    uint2 gid                  [[threadgroup_position_in_grid]],
    ushort sg                  [[simdgroup_index_in_threadgroup]],
    ushort lane                [[thread_index_in_simdgroup]]
) {
  uint strip = gid.x * 2u + uint(sg), kp = gid.y, n_bc = d.k >> 7;
  uint chunk = n_bc >> 3;
  uint bc0 = kp * chunk, bc1 = bc0 + chunk;
  constexpr uint wpb = BITS * 2u;
  const device uint* w_base = W + ulong(strip) * ulong(n_bc) * wpb * 32ul;
  const device half* s_base = scales + ulong(strip >> 3) * ulong(n_bc);
  short2 home = pw_frag_home(lane);
  float2 a0, a1;
  pw_dense_mv_range2<BITS>(w_base, s_base, bc0, bc1, T, X_rot, X_rot + ulong(d.k), lane, home, a0, a1);
  if ((lane & 9u) == 0u) {
    uint r = strip * 16u + uint(home.y);
    ulong pbase = ulong(kp) * 2ul * ulong(d.n);
    YR_out[pbase + r] = a0.x; YR_out[pbase + r + 8u] = a1.x;
    YR_out[pbase + ulong(d.n) + r] = a0.y; YR_out[pbase + ulong(d.n) + r + 8u] = a1.y;
  }
}

template <uint BITS>
[[kernel]] void q27_pw_dec_2dsk8_4_t(
    const device uint* W       [[buffer(0)]],
    const device half* scales  [[buffer(1)]],
    constant Q27PwV3Tables* T  [[buffer(2)]],
    const device half* X_rot   [[buffer(3)]],
    device float* YR_out       [[buffer(4)]],
    constant Q27PwDenseDims& d [[buffer(5)]],
    uint2 gid                  [[threadgroup_position_in_grid]],
    ushort sg                  [[simdgroup_index_in_threadgroup]],
    ushort lane                [[thread_index_in_simdgroup]]
) {
  uint strip = gid.x * 2u + uint(sg), kp = gid.y, n_bc = d.k >> 7;
  uint chunk = n_bc >> 3;
  uint bc0 = kp * chunk, bc1 = bc0 + chunk;
  constexpr uint wpb = BITS * 2u;
  const device uint* w_base = W + ulong(strip) * ulong(n_bc) * wpb * 32ul;
  const device half* s_base = scales + ulong(strip >> 3) * ulong(n_bc);
  short2 home = pw_frag_home(lane);
  ulong pbase = ulong(kp) * ulong(d.rows) * ulong(d.n);
  for (uint rb = 0u; rb < d.rows; rb += 4u) {
    uint nc = metal::min(4u, d.rows - rb);
    const device half* x0 = X_rot + ulong(rb + 0u) * ulong(d.k);
    const device half* x1 = X_rot + ulong(rb + ((nc > 1u) ? 1u : 0u)) * ulong(d.k);
    const device half* x2 = X_rot + ulong(rb + ((nc > 2u) ? 2u : 0u)) * ulong(d.k);
    const device half* x3 = X_rot + ulong(rb + ((nc > 3u) ? 3u : 0u)) * ulong(d.k);
    float4 a0, a1;
    pw_dense_mv_range4<BITS>(w_base, s_base, bc0, bc1, T, x0, x1, x2, x3, lane, home, a0, a1);
    if ((lane & 9u) == 0u) {
      uint r = strip * 16u + uint(home.y);
      YR_out[pbase + ulong(rb + 0u) * ulong(d.n) + r] = a0.x; YR_out[pbase + ulong(rb + 0u) * ulong(d.n) + r + 8u] = a1.x;
      if (nc > 1u) { YR_out[pbase + ulong(rb + 1u) * ulong(d.n) + r] = a0.y; YR_out[pbase + ulong(rb + 1u) * ulong(d.n) + r + 8u] = a1.y; }
      if (nc > 2u) { YR_out[pbase + ulong(rb + 2u) * ulong(d.n) + r] = a0.z; YR_out[pbase + ulong(rb + 2u) * ulong(d.n) + r + 8u] = a1.z; }
      if (nc > 3u) { YR_out[pbase + ulong(rb + 3u) * ulong(d.n) + r] = a0.w; YR_out[pbase + ulong(rb + 3u) * ulong(d.n) + r + 8u] = a1.w; }
    }
  }
}

template [[host_name("q27_pw_dec_fused1_3b")]]  [[kernel]] decltype(q27_pw_dec_fused1_t<3u>)  q27_pw_dec_fused1_t<3u>;
template [[host_name("q27_pw_dec_fused1_4b")]]  [[kernel]] decltype(q27_pw_dec_fused1_t<4u>)  q27_pw_dec_fused1_t<4u>;
template [[host_name("q27_pw_dec_fused1_5b")]]  [[kernel]] decltype(q27_pw_dec_fused1_t<5u>)  q27_pw_dec_fused1_t<5u>;
template [[host_name("q27_pw_dec_fused2_3b")]]  [[kernel]] decltype(q27_pw_dec_fused2_t<3u>)  q27_pw_dec_fused2_t<3u>;
template [[host_name("q27_pw_dec_fused2_4b")]]  [[kernel]] decltype(q27_pw_dec_fused2_t<4u>)  q27_pw_dec_fused2_t<4u>;
template [[host_name("q27_pw_dec_fused2_5b")]]  [[kernel]] decltype(q27_pw_dec_fused2_t<5u>)  q27_pw_dec_fused2_t<5u>;
template [[host_name("q27_pw_dec_fused4_3b")]]  [[kernel]] decltype(q27_pw_dec_fused4_t<3u>)  q27_pw_dec_fused4_t<3u>;
template [[host_name("q27_pw_dec_fused4_4b")]]  [[kernel]] decltype(q27_pw_dec_fused4_t<4u>)  q27_pw_dec_fused4_t<4u>;
template [[host_name("q27_pw_dec_fused4_5b")]]  [[kernel]] decltype(q27_pw_dec_fused4_t<5u>)  q27_pw_dec_fused4_t<5u>;
template [[host_name("q27_pw_dec_2dsk8_1_3b")]] [[kernel]] decltype(q27_pw_dec_2dsk8_1_t<3u>) q27_pw_dec_2dsk8_1_t<3u>;
template [[host_name("q27_pw_dec_2dsk8_1_4b")]] [[kernel]] decltype(q27_pw_dec_2dsk8_1_t<4u>) q27_pw_dec_2dsk8_1_t<4u>;
template [[host_name("q27_pw_dec_2dsk8_1_5b")]] [[kernel]] decltype(q27_pw_dec_2dsk8_1_t<5u>) q27_pw_dec_2dsk8_1_t<5u>;
template [[host_name("q27_pw_dec_2dsk8_2_3b")]] [[kernel]] decltype(q27_pw_dec_2dsk8_2_t<3u>) q27_pw_dec_2dsk8_2_t<3u>;
template [[host_name("q27_pw_dec_2dsk8_2_4b")]] [[kernel]] decltype(q27_pw_dec_2dsk8_2_t<4u>) q27_pw_dec_2dsk8_2_t<4u>;
template [[host_name("q27_pw_dec_2dsk8_2_5b")]] [[kernel]] decltype(q27_pw_dec_2dsk8_2_t<5u>) q27_pw_dec_2dsk8_2_t<5u>;
template [[host_name("q27_pw_dec_2dsk8_4_3b")]] [[kernel]] decltype(q27_pw_dec_2dsk8_4_t<3u>) q27_pw_dec_2dsk8_4_t<3u>;
template [[host_name("q27_pw_dec_2dsk8_4_4b")]] [[kernel]] decltype(q27_pw_dec_2dsk8_4_t<4u>) q27_pw_dec_2dsk8_4_t<4u>;
template [[host_name("q27_pw_dec_2dsk8_4_5b")]] [[kernel]] decltype(q27_pw_dec_2dsk8_4_t<5u>) q27_pw_dec_2dsk8_4_t<5u>;
