
constant float FP4_VALUES[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                 -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
inline float e8m0(uint8_t s) {
  return as_type<float>(s == 0 ? 0x400000u : (uint(s) << 23));
}
inline float fp4dot16(const device uint8_t* w, const thread float* xt, float scale) {
  const device uint16_t* ws = (const device uint16_t*)w;
  float accum = 0.0f;
  for (int i = 0; i < 4; i++)
    accum += (xt[4 * i] * FP4_VALUES[ws[i] & 15] + xt[4 * i + 1] * FP4_VALUES[(ws[i] >> 4) & 15] +
              xt[4 * i + 2] * FP4_VALUES[(ws[i] >> 8) & 15] + xt[4 * i + 3] * FP4_VALUES[(ws[i] >> 12) & 15]);
  return scale * accum;
}

inline float log1p_accurate(float y) {
  const float u = 1.0f + y;
  return u == 1.0f ? y : metal::precise::log(u) * y / (u - 1.0f);
}
inline float sqrtsoftplus(float l) {
  return metal::precise::sqrt(metal::max(l, 0.0f) + log1p_accurate(metal::precise::exp(-metal::abs(l))));
}
