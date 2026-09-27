
#define PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
// the bf16 at index e (0..7) of 8 packed bf16 as fp32
inline float bf8(uint4 v, int e) {
  const uint w = v[e / 2];
  return as_type<float>((e % 2) ? (w & 0xFFFF0000u) : (w << 16));
}
// a row's 8 inputs summed left to right
inline float sum8(uint4 v, float one) {
  float t = bf8(v, 0);
  for (int e = 1; e < 8; e++) t = fma(bf8(v, e), one, t);
  return t;
}
// 2^-4s
inline float pre(int s) { return as_type<float>(uint(127 - 4 * s) << 23); }

inline uint4 scale8(const device bfloat* X, const device bfloat* E, int r, int j, int K) {
  uint4 out;
  for (int h = 0; h < 4; h++) {
    const int k = 8 * j + 2 * h;
    const bfloat a = bfloat(float(X[size_t(r) * K + k]) * float(E[k]));
    const bfloat b = bfloat(float(X[size_t(r) * K + k + 1]) * float(E[k + 1]));
    out[h] = uint(as_type<ushort>(a)) | (uint(as_type<ushort>(b)) << 16);
  }
  return out;
}
