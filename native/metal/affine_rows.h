
#define PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
// code i of a 32-code block held in BITS words (i is a compile-time constant after unrolling)
template <int BITS>
inline uint code_at(const thread uint* w, const int i) {
  const int bit = i * BITS, word = bit >> 5, shift = bit & 31;
  uint v = w[word] >> shift;
  if (shift + BITS > 32) v |= w[word + 1] << (32 - shift);
  return v & ((1u << BITS) - 1u);
}
// the bf16 in the low (h = 0) or high (h = 1) half of v, as fp32
inline float bf_half(uint v, int h) { return as_type<float>(h ? (v & 0xFFFF0000u) : (v << 16)); }
