
inline float bfsum4(const device bfloat* x) {
  return float(bfloat(float(bfloat(float(bfloat(float(x[0]) + float(x[1]))) + float(x[2]))) + float(x[3])));
}
inline float bfsum8(const device bfloat* x) {
  bfloat s = bfloat(float(x[0]) + float(x[1]));
  s = bfloat(float(s) + float(x[2])); s = bfloat(float(s) + float(x[3])); s = bfloat(float(s) + float(x[4]));
  s = bfloat(float(s) + float(x[5])); s = bfloat(float(s) + float(x[6])); s = bfloat(float(s) + float(x[7]));
  return float(s);
}
template <int B, int V> inline float loadv(const device bfloat* x, thread float* xt);
template <> inline float loadv<4, 16>(const device bfloat* x, thread float* xt) {
  float sum = 0.0f;
  for (int i = 0; i < 16; i += 4) {
    const bfloat a = x[i], b = x[i + 1], c = x[i + 2], d = x[i + 3];
    sum += float(bfloat(float(bfloat(float(bfloat(float(a) + float(b))) + float(c))) + float(d)));
    xt[i] = float(a); xt[i + 1] = float(b) / 16.0f; xt[i + 2] = float(c) / 256.0f; xt[i + 3] = float(d) / 4096.0f;
  }
  return sum;
}
template <> inline float loadv<8, 8>(const device bfloat* x, thread float* xt) {
  float sum = 0.0f;
  for (int i = 0; i < 8; i++) { sum += float(x[i]); xt[i] = float(x[i]); }
  return sum;
}
template <> inline float loadv<8, 16>(const device bfloat* x, thread float* xt) {
  float sum = 0.0f;
  for (int i = 0; i < 16; i++) { sum += float(x[i]); xt[i] = float(x[i]); }
  return sum;
}
template <> inline float loadv<8, 32>(const device bfloat* x, thread float* xt) {
  float sum = 0.0f;
  for (int i = 0; i < 32; i++) { sum += float(x[i]); xt[i] = float(x[i]); }
  return sum;
}
template <> inline float loadv<6, 8>(const device bfloat* x, thread float* xt) {
  float sum = 0.0f;
  for (int i = 0; i < 8; i += 4) {
    sum += bfsum4(x + i);
    xt[i] = float(x[i]); xt[i + 1] = float(x[i + 1]) / 64.0f; xt[i + 2] = float(x[i + 2]) / 16.0f;
    xt[i + 3] = float(x[i + 3]) / 4.0f;
  }
  return sum;
}
template <> inline float loadv<5, 16>(const device bfloat* x, thread float* xt) {
  float sum = 0.0f;
  for (int i = 0; i < 16; i += 8) {
    sum += bfsum8(x + i);
    xt[i] = float(x[i]); xt[i + 1] = float(x[i + 1]) / 32.0f; xt[i + 2] = float(x[i + 2]) / 4.0f;
    xt[i + 3] = float(x[i + 3]) / 128.0f; xt[i + 4] = float(x[i + 4]) / 16.0f; xt[i + 5] = float(x[i + 5]) / 2.0f;
    xt[i + 6] = float(x[i + 6]) / 64.0f; xt[i + 7] = float(x[i + 7]) / 8.0f;
  }
  return sum;
}
template <int B, int V> inline float qdotv(const device uint8_t* w, const thread float* xt, float scale, float bias,
                                           float sum);
template <> inline float qdotv<4, 16>(const device uint8_t* w, const thread float* xt, float scale, float bias,
                                      float sum) {
  const device uint16_t* ws = (const device uint16_t*)w;
  float accum = 0.0f;
  for (int i = 0; i < 4; i++)
    accum += xt[4 * i] * float(ws[i] & 0x000f) + xt[4 * i + 1] * float(ws[i] & 0x00f0) +
             xt[4 * i + 2] * float(ws[i] & 0x0f00) + xt[4 * i + 3] * float(ws[i] & 0xf000);
  return scale * accum + sum * bias;
}
template <> inline float qdotv<8, 8>(const device uint8_t* w, const thread float* xt, float scale, float bias,
                                     float sum) {
  float accum = 0.0f;
  for (int i = 0; i < 8; i++) accum += xt[i] * w[i];
  return scale * accum + sum * bias;
}
template <> inline float qdotv<8, 16>(const device uint8_t* w, const thread float* xt, float scale, float bias,
                                      float sum) {
  float accum = 0.0f;
  for (int i = 0; i < 16; i++) accum += xt[i] * w[i];
  return scale * accum + sum * bias;
}
template <> inline float qdotv<8, 32>(const device uint8_t* w, const thread float* xt, float scale, float bias,
                                      float sum) {
  float accum = 0.0f;
  for (int i = 0; i < 32; i++) accum += xt[i] * w[i];
  return scale * accum + sum * bias;
}
template <> inline float qdotv<6, 8>(const device uint8_t* w, const thread float* xt, float scale, float bias,
                                     float sum) {
  float accum = 0.0f;
  for (int i = 0; i < 2; i++) {
    xt += 4 * i;
    w += 3 * i;
    accum += (w[0] & 0x3f) * xt[0];
    accum += (w[0] & 0xc0) * xt[1];
    accum += (w[1] & 0x0f) * (xt[1] * 256.0f);
    accum += (w[1] & 0xf0) * xt[2];
    accum += (w[2] & 0x03) * (xt[2] * 256.0f);
    accum += (w[2] & 0xfc) * xt[3];
  }
  return scale * accum + sum * bias;
}
template <> inline float qdotv<5, 16>(const device uint8_t* w, const thread float* xt, float scale, float bias,
                                      float sum) {
  float accum = 0.0f;
  for (int i = 0; i < 2; i++) {
    xt += 8 * i;
    w += 5 * i;
    accum += (w[0] & 0x1f) * xt[0];
    accum += (w[0] & 0xe0) * xt[1];
    accum += (w[1] & 0x3) * (xt[1] * 256.0f);
    accum += (w[1] & 0x7c) * xt[2];
    accum += (w[1] & 0x80) * xt[3];
    accum += (w[2] & 0xf) * (xt[3] * 256.0f);
    accum += (w[2] & 0xf0) * xt[4];
    accum += (w[3] & 0x1) * (xt[4] * 256.0f);
    accum += (w[3] & 0x3e) * xt[5];
    accum += (w[3] & 0xc0) * xt[6];
    accum += (w[4] & 0x7) * (xt[6] * 256.0f);
    accum += (w[4] & 0xf8) * xt[7];
  }
  return scale * accum + sum * bias;
}
