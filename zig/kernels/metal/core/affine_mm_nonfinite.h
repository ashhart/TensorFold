// Nonfinite group detection and the shared two-step g128 accumulator for affine_mm.metal.
#ifdef TF_AM_EXACT_NONFINITE
// Row sums identify candidate groups; scanning candidates keeps finite overflow on the original path.
template <int TM, int TL>
inline void am_nonfinite(thread const AmRows<TM>& rows, int k, thread bool (&raw)[TL][2]) {
  TF_UNROLL
  for (short m = 0; m < TL; m++) {
    const int g = k / TF_GROUP;
    TF_UNROLL
    for (short h = 0; h < 2; h++) {
      raw[m][h] = !isfinite(rows.s[m][h][g]);
      if (raw[m][h]) {
        raw[m][h] = false;
        for (short j = 0; j < TF_GROUP; j++) raw[m][h] |= !isfinite(float(rows.x[m][h][k + j]));
      }
    }
  }
}
#endif

// The fragment of x at column c from a lane's two row starts (8-byte loads on M5).
inline frag<bfloat> am_frag(const device bfloat* r0, const device bfloat* r1, int c, short2 home) {
#ifndef TF_SIMD_FRAGS
  return frag<bfloat>(*(const device bfloat4*)(r0 + c + home.x), *(const device bfloat4*)(r1 + c + home.x));
#else
  frag<bfloat> f;
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    f[e] = (e < 4 ? r0 : r1)[c + home.x + TF_COL(e)];
  }
  return f;
#endif
}

// A g128 group's raw and offset sums span two steps; their epilogue follows the second.
template <int TM, int TL>
inline void am_step128(thread frag<float> (&acc)[TM][2], thread frag<float> (&part)[TM][2], thread const AmRows<TM>& rows,
                       int k, const threadgroup bfloat* tile, const threadgroup float* sb,
#ifdef TF_AM_EXACT_NONFINITE
                       thread frag<float> (&exact)[TM][2], thread bool (&raw)[TM][2],
#endif
                       int tn, short2 home) {
  const bool opens = k % 128 == 0;
  TF_UNROLL
  for (short q = 0; q < 4; q++) {
    frag<bfloat> b0, b1, a[TL];
#if defined(TF_GLM_M3_FAST) && defined(TF_SIMD_FRAGS)
    frag_get(b0, tile, PAD, 16 * q, tn, home);
    frag_get(b1, tile, PAD, 16 * q, tn + 16, home);
#else
    frag_get_t(b0, tile, PAD, tn, 16 * q, home);
    frag_get_t(b1, tile, PAD, tn + 16, 16 * q, home);
#endif
    TF_UNROLL
    for (short m = 0; m < TL; m++) {
      a[m] = am_frag(rows.x[m][0], rows.x[m][1], k + 16 * q, home);
    }
#ifdef TF_AM_EXACT_NONFINITE
    if (opens && q == 0) am_nonfinite<TM, TM>(rows, k, raw);
#endif
    TF_UNROLL
    for (short m = 0; m < TL; m++) {
      if (opens && q == 0) {
        am_first(part[m][0], part[m][1], a[m], b0, b1);
      } else {
        mma_16x32<false, true>(part[m][0], part[m][1], a[m], b0, b1);
      }
#ifdef TF_AM_EXACT_NONFINITE
      if (simd_any(raw[m][0] || raw[m][1])) {
        frag<bfloat> r0, r1;
        TF_UNROLL
        for (short e = 0; e < 8; e++) {
          r0[e] = bfloat(b0[e] - OFF);
          r1[e] = bfloat(b1[e] - OFF);
        }
        if (opens && q == 0) {
          am_first(exact[m][0], exact[m][1], a[m], r0, r1);
        } else {
          mma_16x32<false, true>(exact[m][0], exact[m][1], a[m], r0, r1);
        }
      }
#endif
    }
  }
  if (opens) return;
  TF_UNROLL
  for (short m = 0; m < TL; m++) {
    const float xr0 = rows.s[m][0][k / 128], xr1 = rows.s[m][1][k / 128];
    TF_UNROLL
    for (short j = 0; j < 2; j++) {
      TF_UNROLL
      for (short c = 0; c < 4; c++) {
        const int col = tn + 16 * j + home.x + TF_COL(c);
        const float s = sb[col];
#ifdef TF_AM_EXACT_NONFINITE
        float b0 = sb[64 + col], b1 = b0;
        if (raw[m][0]) b0 += OFF * s;
        if (raw[m][1]) b1 += OFF * s;
        const float p0 = raw[m][0] ? exact[m][j][c] : part[m][j][c];
        const float p1 = raw[m][1] ? exact[m][j][4 + c] : part[m][j][4 + c];
        acc[m][j][c] += s * p0 + b0 * xr0;
        acc[m][j][4 + c] += s * p1 + b1 * xr1;
#else
        const float b = sb[64 + col];
        acc[m][j][c] += s * part[m][j][c] + b * xr0;
        acc[m][j][4 + c] += s * part[m][j][4 + c] + b * xr1;
#endif
      }
    }
  }
}
