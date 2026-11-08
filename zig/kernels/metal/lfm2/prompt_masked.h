// Exceptional P x V only: preserve the tensor product of finite values and restore visible IEEE terms per row.
namespace tfp {
inline float lfm2_fragment_at(thread const frag<float>& f, ushort row, ushort col) {
#ifndef TF_SIMD_FRAGS
  const ushort owner = ((row & 4) << 2) | ((row & 3) << 1) | ((col & 4) >> 2) | (col & 8);
  const ushort element = (row >> 3) * 4 + (col & 3);
#else
  const ushort owner = ((row & 4) << 2) | ((row & 3) << 1) | ((col & 2) >> 1) | ((col & 4) << 1);
  const ushort element = (row >> 3) * 4 + (col & 1) + ((col & 8) >> 2);
#endif
  return simd_shuffle(f[element], owner);
}
inline void lfm2_masked_pv(thread frag<float>& lo, thread frag<float>& hi, thread const frag<bfloat>& prob,
    thread const frag<bfloat>& v0, thread const frag<bfloat>& v1, short2 home,
    uint key0, uint count0, uint count1) {
  frag<float> a, b, p;
  for (ushort e = 0; e < 8; e++) p[e] = float(prob[e]);
  frag<bfloat> clean0, clean1;
  bool exceptional = false;
  TF_UNROLL
  for (ushort e = 0; e < 8; e++) {
    a[e] = float(v0[e]); b[e] = float(v1[e]);
    exceptional = exceptional || !isfinite(a[e]) || !isfinite(b[e]);
    clean0[e] = isfinite(a[e]) ? v0[e] : bfloat(copysign(0.0f, a[e]));
    clean1[e] = isfinite(b[e]) ? v1[e] : bfloat(copysign(0.0f, b[e]));
  }
  if (!simd_any(exceptional)) {
    mma_16x32<false, false>(lo, hi, prob, v0, v1);
    return;
  }
  mma_16x32<false, false>(lo, hi, prob, clean0, clean1);
  TF_UNROLL
  for (ushort key = 0; key < 16; key++) {
    const float pa = lfm2_fragment_at(p, home.y, key), pb = lfm2_fragment_at(p, home.y + 8, key);
    const bool va = (key0 + key < count0);
    const bool vb = (key0 + key < count1);
    TF_UNROLL
    for (ushort col = 0; col < 4; col++) {
      const float x = lfm2_fragment_at(a, key, home.x + TF_COL(col));
      const float y = lfm2_fragment_at(b, key, home.x + TF_COL(col));
      if (!isfinite(x)) { if (va) lo[col] += pa * x; if (vb) lo[4 + col] += pb * x; }
      if (!isfinite(y)) { if (va) hi[col] += pa * y; if (vb) hi[4 + col] += pb * y; }
    }
  }
}
} // namespace tfp
