// DeepSeek-V4.1's keyed draws on the GPU: the host sampler's keys and Gumbel noise in float (prepended to kernels).
#include <metal_stdlib>
using namespace metal;

// splitmix64's finalizer, as the host's keyed sampler mixes its keys.
inline ulong tf_key_mix(ulong v) {
  v ^= v >> 30;
  v *= 0xBF58476D1CE4E5B9ul;
  v ^= v >> 27;
  v *= 0x94D049BB133111EBul;
  return v ^ (v >> 31);
}

inline bool tf_key_finite(float v) {
  return (as_type<uint>(v) & 0x7f800000u) != 0x7f800000u;
}

// Gumbel noise -log(-log(u)) of the host's uniform (bits >> 11) / 2^53 + 2^-54 in float: -log(u) by series near u = 1.
inline float tf_key_gumbel(ulong bits) {
  const ulong m = bits >> 11;
  const float w = (float((1ul << 53) - m) - 0.5f) * 0x1p-53f;  // 1 - u
  float l;
  if (w < 0.0625f) {
    float term = w, sum = w;
    for (uint k = 2; k <= 8u; ++k) {
      term *= w;
      sum += term / float(k);
    }
    l = sum;
  } else {
    l = -precise::log(float(m) * 0x1p-53f + 0x1p-54f);
  }
  return -precise::log(l);
}

// The better of two draws: the larger score, then the earlier of (value desc, id asc), as the host breaks a tie.
inline void tf_key_beats(thread float& s, thread float& v, thread uint& i, float os, float ov, uint oi) {
  if (os > s || (os == s && (ov > v || (ov == v && oi < i)))) {
    s = os;
    v = ov;
    i = oi;
  }
}

// The host's key before the id of a draw at `position` (sampling.zig keyBase).
inline ulong tf_key_base(ulong seed, ulong position) {
  return tf_key_mix(tf_key_mix(seed + 0x9E3779B97F4A7C15ul) ^ (position * 0xD1B54A32D192ED03ul));
}
