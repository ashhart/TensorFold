// Exceptional prompt tiles: finite entries keep the original product; only visible exceptional terms are restored.
namespace tfp {

// A logical fragment element, in the tensor-unit layout or the pre-M5 SIMD layout.
inline float nemo_fragment_at(thread const frag<float>& f, ushort row, ushort col) {
#ifndef TF_SIMD_FRAGS
  const ushort owner = ((row & 4) << 2) | ((row & 3) << 1) | ((col & 4) >> 2) | (col & 8);
  const ushort element = (row >> 3) * 4 + (col & 3);
#else
  const ushort owner = ((row & 4) << 2) | ((row & 3) << 1) | ((col & 2) >> 1) | ((col & 4) << 1);
  const ushort element = (row >> 3) * 4 + (col & 1) + ((col & 8) >> 2);
#endif
  return simd_shuffle(f[element], owner);
}

template <typename T>
inline void nemo_masked_pv(thread frag<float>& lo, thread frag<float>& hi, thread const frag<float>& p,
    thread const frag<T>& v0, thread const frag<T>& v1, ushort lane, short2 home,
    int key0, int key_end, int query0) {
  frag<float> a, b;
  frag<T> clean0, clean1;
  bool exceptional = false;
  TF_UNROLL
  for (ushort e = 0; e < 8; e++) {
    a[e] = float(v0[e]); b[e] = float(v1[e]);
    exceptional = exceptional || !isfinite(a[e]) || !isfinite(b[e]);
    clean0[e] = isfinite(a[e]) ? v0[e] : T(copysign(0.0f, a[e]));
    clean1[e] = isfinite(b[e]) ? v1[e] : T(copysign(0.0f, b[e]));
  }
  if (!simd_any(exceptional)) {
    attention_pv(lo, hi, p, v0, v1, lane);
    return;
  }
  attention_pv(lo, hi, p, clean0, clean1, lane);
  TF_UNROLL
  for (ushort key = 0; key < 16; key++) {
    const float pa = nemo_fragment_at(p, home.y, key), pb = nemo_fragment_at(p, home.y + 8, key);
    TF_UNROLL
    for (ushort col = 0; col < 4; col++) {
      const float va = nemo_fragment_at(a, key, home.x + TF_COL(col));
      const float vb = nemo_fragment_at(b, key, home.x + TF_COL(col));
      const int position = key0 + key;
      if (position < key_end) {
        if (!isfinite(va)) {
          if (position <= query0) lo[col] += pa * va;
          if (position <= query0 + 8) lo[4 + col] += pb * va;
        }
        if (!isfinite(vb)) {
          if (position <= query0) hi[col] += pa * vb;
          if (position <= query0 + 8) hi[4 + col] += pb * vb;
        }
      }
    }
  }
}
}  // namespace tfp
