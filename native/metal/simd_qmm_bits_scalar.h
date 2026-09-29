
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

// code j of B-bit codes packed from bit 0 of v (j a compile-time constant after unrolling)
template <int B>
inline uint code_at(const thread uint* v, const int j) {
  const int bit = j * B, word = bit >> 5, shift = bit & 31;
  uint c = v[word] >> shift;
  if (shift + B > 32) c |= v[word + 1] << (32 - shift);
  return c & ((1u << B) - 1u);
}
// float(c) for c < 2^23 without a convert: the same value
inline float cf(uint c) { return as_type<float>(0x4B000000u | c) - 8388608.0f; }
