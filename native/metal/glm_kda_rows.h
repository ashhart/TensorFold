
// f_b / g_b for one output: MLX 0.32.2's one-row qmv_quad at 4 or 8 bits (the caller does the quad_sum)
template <int BITS, int PER>
inline float quad_dot(device const bfloat* x, device const uint8_t* wb, float s, float bb);
template <>
inline float quad_dot<4, 32>(device const bfloat* x, device const uint8_t* wb, float s, float bb) {
  constexpr int PER = 32;
  float xt[PER];
  float sum = 0.0f;
  for (int i = 0; i < PER; i += 4) {
    const bfloat a = x[i], b = x[i + 1], c = x[i + 2], e = x[i + 3];
    sum += float(bfloat(float(bfloat(float(bfloat(float(a) + float(b))) + float(c))) + float(e)));
    xt[i] = float(a); xt[i + 1] = float(b) / 16.0f; xt[i + 2] = float(c) / 256.0f; xt[i + 3] = float(e) / 4096.0f;
  }
  device const uint16_t* ws = (device const uint16_t*)wb;
  float accum = 0.0f;
  for (int i = 0; i < PER / 4; i++)
    accum += xt[4 * i] * float(ws[i] & 0x000f) + xt[4 * i + 1] * float(ws[i] & 0x00f0) +
             xt[4 * i + 2] * float(ws[i] & 0x0f00) + xt[4 * i + 3] * float(ws[i] & 0xf000);
  float result = 0.0f;
  result += s * accum + sum * bb;
  return result;
}
template <>
inline float quad_dot<4, 16>(device const bfloat* x, device const uint8_t* wb, float s, float bb) {
  constexpr int PER = 16;
  float xt[PER];
  float sum = 0.0f;
  for (int i = 0; i < PER; i += 4) {
    const bfloat a = x[i], b = x[i + 1], c = x[i + 2], e = x[i + 3];
    sum += float(bfloat(float(bfloat(float(bfloat(float(a) + float(b))) + float(c))) + float(e)));
    xt[i] = float(a); xt[i + 1] = float(b) / 16.0f; xt[i + 2] = float(c) / 256.0f; xt[i + 3] = float(e) / 4096.0f;
  }
  device const uint16_t* ws = (device const uint16_t*)wb;
  float accum = 0.0f;
  for (int i = 0; i < PER / 4; i++)
    accum += xt[4 * i] * float(ws[i] & 0x000f) + xt[4 * i + 1] * float(ws[i] & 0x00f0) +
             xt[4 * i + 2] * float(ws[i] & 0x0f00) + xt[4 * i + 3] * float(ws[i] & 0xf000);
  float result = 0.0f;
  result += s * accum + sum * bb;
  return result;
}
template <>
inline float quad_dot<8, 32>(device const bfloat* x, device const uint8_t* wb, float s, float bb) {
  constexpr int PER = 32;
  float xt[PER];
  float sum = 0.0f;
  for (int i = 0; i < PER; i++) { sum += float(x[i]); xt[i] = float(x[i]); }
  float accum = 0.0f;
  for (int i = 0; i < PER; i++) accum += xt[i] * wb[i];
  float result = 0.0f;
  result += s * accum + sum * bb;
  return result;
}
template <>
inline float quad_dot<8, 16>(device const bfloat* x, device const uint8_t* wb, float s, float bb) {
  constexpr int PER = 16;
  float xt[PER];
  float sum = 0.0f;
  for (int i = 0; i < PER; i++) { sum += float(x[i]); xt[i] = float(x[i]); }
  float accum = 0.0f;
  for (int i = 0; i < PER; i++) accum += xt[i] * wb[i];
  float result = 0.0f;
  result += s * accum + sum * bb;
  return result;
}
template <typename U>
inline U mlx_sigmoid_precise(U x) {
  U e = static_cast<U>(metal::precise::exp(metal::abs(x)));
  U y = static_cast<U>(1) / (static_cast<U>(1) + e);
  return (x < 0) ? y : (static_cast<U>(1) - y);
}
// `(x * x).sum(-1)` rounds the square before the add (no fma), as in #2105.
#pragma clang fp contract(off)
inline float sq_acc(float acc, float v) {
  return v * v + acc;
}
#pragma clang fp contract(on)
