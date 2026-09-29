
template <typename T>
inline float load16f(const device T* x, thread float* xt) {
  float sum = 0.0f;
  for (int i = 0; i < 16; i += 4) {
    const float a = float(x[i]), b = float(x[i + 1]), c = float(x[i + 2]), d = float(x[i + 3]);
    sum += a + b + c + d;
    xt[i] = a; xt[i + 1] = b / 16.0f; xt[i + 2] = c / 256.0f; xt[i + 3] = d / 4096.0f;
  }
  return sum;
}
inline float qdot16(const device uint8_t* w, const thread float* xt, float scale, float bias, float sum) {
  const device uint16_t* ws = (const device uint16_t*)w;
  float accum = 0.0f;
  for (int i = 0; i < 4; i++)
    accum += xt[4 * i] * float(ws[i] & 0x000f) + xt[4 * i + 1] * float(ws[i] & 0x00f0) +
             xt[4 * i + 2] * float(ws[i] & 0x0f00) + xt[4 * i + 3] * float(ws[i] & 0xf000);
  return scale * accum + sum * bias;
}
