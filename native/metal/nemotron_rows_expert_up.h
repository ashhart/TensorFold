
// MLX's 4-bit qmv_fast inner loop (quantized.h: load_vector, qdot). A lane's 16 inputs, pre-divided by 1, 16, 256,
// 4096 so the masked nibbles need no shift, and their sum with each run of 4 summed in bf16 first (MLX's
// x[i] + x[i + 1] + x[i + 2] + x[i + 3] on bfloat16_t).
inline float tf_load16(const device bfloat* x, thread float* xt) {
  float sum = 0.0f;
  for (int i = 0; i < 16; i += 4) {
    const bfloat a = x[i], b = x[i + 1], c = x[i + 2], d = x[i + 3];
    sum += float(bfloat(float(bfloat(float(bfloat(float(a) + float(b))) + float(c))) + float(d)));
    xt[i] = float(a); xt[i + 1] = float(b) / 16.0f; xt[i + 2] = float(c) / 256.0f; xt[i + 3] = float(d) / 4096.0f;
  }
  return sum;
}
// the lane's 16 inputs times its 8 bytes of one weight row: scale * sum(x q) + bias * sum(x)
inline float tf_qdot16(const device uint8_t* w, const thread float* xt, float scale, float bias, float sum) {
  const device uint16_t* ws = (const device uint16_t*)w;
  float accum = 0.0f;
  for (int i = 0; i < 4; i++)
    accum += (xt[4 * i] * (ws[i] & 0x000f) + xt[4 * i + 1] * (ws[i] & 0x00f0) +
              xt[4 * i + 2] * (ws[i] & 0x0f00) + xt[4 * i + 3] * (ws[i] & 0xf000));
  return scale * accum + sum * bias;
}
// One input row x [K] times RPS weight rows (w: the first row's words, rows K / 2 bytes apart; sc, bi: its group
// scales and biases, rows K / GS apart), fp32 over the simdgroup. Lane l takes inputs 16 l .. 16 l + 15 of each
// 512-input step, then lanes below (K % 512) / 16 one more 16-input chunk; the lane sums add up in simd_sum.
// Reads nothing but its own row of x.
template <int K, int GS, int RPS>
inline void tf_rowdot(const device uint8_t* w, const device bfloat* sc, const device bfloat* bi,
                      const device bfloat* x, uint lane, thread float* acc) {
  constexpr int KB = K / 2;
  constexpr int KG = K / GS;
  constexpr int FULL = K / 512 * 512;
  w += lane * 8;
  sc += lane / (GS / 16);
  bi += lane / (GS / 16);
  x += lane * 16;
  for (int j = 0; j < RPS; j++) acc[j] = 0.0f;
  for (int k0 = 0; k0 < FULL; k0 += 512) {
    float xt[16];
    const float sum = tf_load16(x, xt);
    for (int j = 0; j < RPS; j++) acc[j] += tf_qdot16(w + j * KB, xt, float(sc[j * KG]), float(bi[j * KG]), sum);
    w += 256; sc += 512 / GS; bi += 512 / GS; x += 512;
  }
  if (FULL < K && int(lane) < (K - FULL) / 16) {
    float xt[16];
    const float sum = tf_load16(x, xt);
    for (int j = 0; j < RPS; j++) acc[j] += tf_qdot16(w + j * KB, xt, float(sc[j * KG]), float(bi[j * KG]), sum);
  }
  for (int j = 0; j < RPS; j++) acc[j] = simd_sum(acc[j]);
}
