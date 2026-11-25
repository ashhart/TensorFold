// TFQ3 FlashNext MoE decode matvec helpers (3b/4b/5b routed experts).

template <uint N_BC>
inline void pw_mv_nosplit1_5b(
    const device uint* w_base, const device half* sc_base, constant FnPwV3Tables* T,
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
    acc0 = fma(pw_fin(ba0), sc, acc0); acc1 = fma(pw_fin(ba1), sc, acc1);
  }
  acc0 += simd_shuffle_xor(acc0, 1); acc0 += simd_shuffle_xor(acc0, 8);
  acc1 += simd_shuffle_xor(acc1, 1); acc1 += simd_shuffle_xor(acc1, 8);
}

template <uint N_BC>
inline void pw_mv_nosplit4_5b(
    const device uint* w_base, const device half* sc_base, constant FnPwV3Tables* T,
    const device half* x0, const device half* x1, const device half* x2, const device half* x3,
    ushort lane, short2 home, thread float4& acc0, thread float4& acc1
) {
  uint st = 0x13579BDFu ^ (uint(lane) * 0x9E3779B1u);
  acc0 = float4(0.0f); acc1 = float4(0.0f);
  for (uint bc = 0u; bc < N_BC; ++bc) {
    float4 ba0 = float4(0.0f), ba1 = float4(0.0f);
    for (uint h = 0u; h < 2u; ++h) {
      const device uint* blk = w_base + (ulong(bc) * 10u + h * 5u) * 32u + lane;
      uint w0 = blk[0], w1 = blk[32], w2 = blk[64], w3 = blk[96], w4 = blk[128];
      uint cb = (bc << 7) + (h << 6) + uint(home.x);
      half4 q0, q1;
      pw_v3_q8_5b(st, w0 >> 24,          (w0 <<  8) | (w1 >> 24), T, q0, q1); PW_MDOT4(q0, q1, x0, x1, x2, x3, cb +  0u, ba0, ba1);
      pw_v3_q8_5b(st, (w1 >> 16) & 255u, (w1 << 16) | (w2 >> 16), T, q0, q1); PW_MDOT4(q0, q1, x0, x1, x2, x3, cb + 16u, ba0, ba1);
      pw_v3_q8_5b(st, (w2 >>  8) & 255u, (w2 << 24) | (w3 >>  8), T, q0, q1); PW_MDOT4(q0, q1, x0, x1, x2, x3, cb + 32u, ba0, ba1);
      pw_v3_q8_5b(st, w3 & 255u,          w4,                     T, q0, q1); PW_MDOT4(q0, q1, x0, x1, x2, x3, cb + 48u, ba0, ba1);
    }
    float sc = float(sc_base[bc]);
    acc0 = fma(pw_fin(ba0), sc, acc0); acc1 = fma(pw_fin(ba1), sc, acc1);
  }
  acc0 += simd_shuffle_xor(acc0, 1); acc0 += simd_shuffle_xor(acc0, 8);
  acc1 += simd_shuffle_xor(acc1, 1); acc1 += simd_shuffle_xor(acc1, 8);
}

template <uint N_BC, bool HAS_5B>
inline void pw_mv_nosplit1(
    const device uint* w_base, const device half* sc_base, uint b_bits, constant FnPwV3Tables* T,
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
      acc0 = fma(pw_fin(ba0), sc, acc0); acc1 = fma(pw_fin(ba1), sc, acc1);
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
      acc0 = fma(pw_fin(ba0), sc, acc0); acc1 = fma(pw_fin(ba1), sc, acc1);
    }
  }
  acc0 += simd_shuffle_xor(acc0, 1); acc0 += simd_shuffle_xor(acc0, 8);
  acc1 += simd_shuffle_xor(acc1, 1); acc1 += simd_shuffle_xor(acc1, 8);
}

template <uint N_BC, bool HAS_5B>
inline void pw_mv_nosplit2(
    const device uint* w_base, const device half* sc_base, uint b_bits, constant FnPwV3Tables* T,
    const device half* x0, const device half* x1, ushort lane, short2 home, thread float2& acc0, thread float2& acc1
) {
  uint st = 0x13579BDFu ^ (uint(lane) * 0x9E3779B1u);
  acc0 = float2(0.0f); acc1 = float2(0.0f);
  if (b_bits == 4u) {
    const device uint4* w4 = (const device uint4*)w_base;
    for (uint bc = 0u; bc < N_BC; ++bc) {
      float2 ba0 = float2(0.0f), ba1 = float2(0.0f);
      uint4 v = w4[(bc * 2u + 0u) * 32u + lane];
      uint cb = (bc << 7) + uint(home.x);
      half4 q0, q1;
      pw_v3_q8_4b(st, v.x, T, q0, q1); PW_MDOT2(q0, q1, x0, x1, cb +   0u, ba0, ba1);
      pw_v3_q8_4b(st, v.y, T, q0, q1); PW_MDOT2(q0, q1, x0, x1, cb +  16u, ba0, ba1);
      pw_v3_q8_4b(st, v.z, T, q0, q1); PW_MDOT2(q0, q1, x0, x1, cb +  32u, ba0, ba1);
      pw_v3_q8_4b(st, v.w, T, q0, q1); PW_MDOT2(q0, q1, x0, x1, cb +  48u, ba0, ba1);
      v = w4[(bc * 2u + 1u) * 32u + lane];
      pw_v3_q8_4b(st, v.x, T, q0, q1); PW_MDOT2(q0, q1, x0, x1, cb +  64u, ba0, ba1);
      pw_v3_q8_4b(st, v.y, T, q0, q1); PW_MDOT2(q0, q1, x0, x1, cb +  80u, ba0, ba1);
      pw_v3_q8_4b(st, v.z, T, q0, q1); PW_MDOT2(q0, q1, x0, x1, cb +  96u, ba0, ba1);
      pw_v3_q8_4b(st, v.w, T, q0, q1); PW_MDOT2(q0, q1, x0, x1, cb + 112u, ba0, ba1);
      float sc = float(sc_base[bc]);
      acc0 = fma(pw_fin(ba0), sc, acc0); acc1 = fma(pw_fin(ba1), sc, acc1);
    }
  } else if (HAS_5B && b_bits == 5u) {
    for (uint bc = 0u; bc < N_BC; ++bc) {
      float2 ba0 = float2(0.0f), ba1 = float2(0.0f);
      for (uint h = 0u; h < 2u; ++h) {
        const device uint* blk = w_base + (ulong(bc) * 10u + h * 5u) * 32u + lane;
        uint w0 = blk[0], w1 = blk[32], w2 = blk[64], w3 = blk[96], w4 = blk[128];
        uint cb = (bc << 7) + (h << 6) + uint(home.x);
        half4 q0, q1;
        pw_v3_q8_5b(st, w0 >> 24,          (w0 <<  8) | (w1 >> 24), T, q0, q1); PW_MDOT2(q0, q1, x0, x1, cb +  0u, ba0, ba1);
        pw_v3_q8_5b(st, (w1 >> 16) & 255u, (w1 << 16) | (w2 >> 16), T, q0, q1); PW_MDOT2(q0, q1, x0, x1, cb + 16u, ba0, ba1);
        pw_v3_q8_5b(st, (w2 >>  8) & 255u, (w2 << 24) | (w3 >>  8), T, q0, q1); PW_MDOT2(q0, q1, x0, x1, cb + 32u, ba0, ba1);
        pw_v3_q8_5b(st, w3 & 255u,          w4,                     T, q0, q1); PW_MDOT2(q0, q1, x0, x1, cb + 48u, ba0, ba1);
      }
      float sc = float(sc_base[bc]);
      acc0 = fma(pw_fin(ba0), sc, acc0); acc1 = fma(pw_fin(ba1), sc, acc1);
    }
  } else {
    for (uint bc = 0u; bc < N_BC; ++bc) {
      float2 ba0 = float2(0.0f), ba1 = float2(0.0f);
      for (uint h = 0u; h < 2u; ++h) {
        const device uint* blk = w_base + (ulong(bc) * 6u + h * 3u) * 32u + lane;
        uint w0 = blk[0], w1 = blk[32], w2 = blk[64];
        uint cb = (bc << 7) + (h << 6) + uint(home.x);
        half4 q0, q1;
        pw_v3_q8_3b(st, w0 >> 8,                               T, q0, q1); PW_MDOT2(q0, q1, x0, x1, cb +  0u, ba0, ba1);
        pw_v3_q8_3b(st, ((w0 << 16) | (w1 >> 16)) & 0xFFFFFFu, T, q0, q1); PW_MDOT2(q0, q1, x0, x1, cb + 16u, ba0, ba1);
        pw_v3_q8_3b(st, ((w1 <<  8) | (w2 >> 24)) & 0xFFFFFFu, T, q0, q1); PW_MDOT2(q0, q1, x0, x1, cb + 32u, ba0, ba1);
        pw_v3_q8_3b(st, w2 & 0xFFFFFFu,                        T, q0, q1); PW_MDOT2(q0, q1, x0, x1, cb + 48u, ba0, ba1);
      }
      float sc = float(sc_base[bc]);
      acc0 = fma(pw_fin(ba0), sc, acc0); acc1 = fma(pw_fin(ba1), sc, acc1);
    }
  }
  acc0 += simd_shuffle_xor(acc0, 1); acc0 += simd_shuffle_xor(acc0, 8);
  acc1 += simd_shuffle_xor(acc1, 1); acc1 += simd_shuffle_xor(acc1, 8);
}

template <uint N_BC, bool HAS_5B>
inline void pw_mv_nosplit4(
    const device uint* w_base, const device half* sc_base, uint b_bits, constant FnPwV3Tables* T,
    const device half* x0, const device half* x1, const device half* x2, const device half* x3,
    ushort lane, short2 home, thread float4& acc0, thread float4& acc1
) {
  uint st = 0x13579BDFu ^ (uint(lane) * 0x9E3779B1u);
  acc0 = float4(0.0f); acc1 = float4(0.0f);
  if (b_bits == 4u) {
    const device uint4* w4 = (const device uint4*)w_base;
    for (uint bc = 0u; bc < N_BC; ++bc) {
      float4 ba0 = float4(0.0f), ba1 = float4(0.0f);
      uint4 v = w4[(bc * 2u + 0u) * 32u + lane];
      uint cb = (bc << 7) + uint(home.x);
      half4 q0, q1;
      pw_v3_q8_4b(st, v.x, T, q0, q1); PW_MDOT4(q0, q1, x0, x1, x2, x3, cb +   0u, ba0, ba1);
      pw_v3_q8_4b(st, v.y, T, q0, q1); PW_MDOT4(q0, q1, x0, x1, x2, x3, cb +  16u, ba0, ba1);
      pw_v3_q8_4b(st, v.z, T, q0, q1); PW_MDOT4(q0, q1, x0, x1, x2, x3, cb +  32u, ba0, ba1);
      pw_v3_q8_4b(st, v.w, T, q0, q1); PW_MDOT4(q0, q1, x0, x1, x2, x3, cb +  48u, ba0, ba1);
      v = w4[(bc * 2u + 1u) * 32u + lane];
      pw_v3_q8_4b(st, v.x, T, q0, q1); PW_MDOT4(q0, q1, x0, x1, x2, x3, cb +  64u, ba0, ba1);
      pw_v3_q8_4b(st, v.y, T, q0, q1); PW_MDOT4(q0, q1, x0, x1, x2, x3, cb +  80u, ba0, ba1);
      pw_v3_q8_4b(st, v.z, T, q0, q1); PW_MDOT4(q0, q1, x0, x1, x2, x3, cb +  96u, ba0, ba1);
      pw_v3_q8_4b(st, v.w, T, q0, q1); PW_MDOT4(q0, q1, x0, x1, x2, x3, cb + 112u, ba0, ba1);
      float sc = float(sc_base[bc]);
      acc0 = fma(pw_fin(ba0), sc, acc0); acc1 = fma(pw_fin(ba1), sc, acc1);
    }
  } else if (HAS_5B && b_bits == 5u) {
    pw_mv_nosplit4_5b<N_BC>(w_base, sc_base, T, x0, x1, x2, x3, lane, home, acc0, acc1);
    return;
  } else {
    for (uint bc = 0u; bc < N_BC; ++bc) {
      float4 ba0 = float4(0.0f), ba1 = float4(0.0f);
      for (uint h = 0u; h < 2u; ++h) {
        const device uint* blk = w_base + (ulong(bc) * 6u + h * 3u) * 32u + lane;
        uint w0 = blk[0], w1 = blk[32], w2 = blk[64];
        uint cb = (bc << 7) + (h << 6) + uint(home.x);
        half4 q0, q1;
        pw_v3_q8_3b(st, w0 >> 8,                               T, q0, q1); PW_MDOT4(q0, q1, x0, x1, x2, x3, cb +  0u, ba0, ba1);
        pw_v3_q8_3b(st, ((w0 << 16) | (w1 >> 16)) & 0xFFFFFFu, T, q0, q1); PW_MDOT4(q0, q1, x0, x1, x2, x3, cb + 16u, ba0, ba1);
        pw_v3_q8_3b(st, ((w1 <<  8) | (w2 >> 24)) & 0xFFFFFFu, T, q0, q1); PW_MDOT4(q0, q1, x0, x1, x2, x3, cb + 32u, ba0, ba1);
        pw_v3_q8_3b(st, w2 & 0xFFFFFFu,                        T, q0, q1); PW_MDOT4(q0, q1, x0, x1, x2, x3, cb + 48u, ba0, ba1);
      }
      float sc = float(sc_base[bc]);
      acc0 = fma(pw_fin(ba0), sc, acc0); acc1 = fma(pw_fin(ba1), sc, acc1);
    }
  }
  acc0 += simd_shuffle_xor(acc0, 1); acc0 += simd_shuffle_xor(acc0, 8);
  acc1 += simd_shuffle_xor(acc1, 1); acc1 += simd_shuffle_xor(acc1, 8);
}
