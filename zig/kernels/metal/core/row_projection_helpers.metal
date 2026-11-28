// 2^-e, exact: a code read in place at bit e times its input scaled by 2^-e is the same product.
inline float tf_rp_p2(int e) { return as_type<float>(uint(127 - e) << 23); }

// An input in its code's place, exact: at 4 bits divided by 16^(i % 4), at 2 bits by 4^i.
template <int BITS>
inline float tf_rp_x(float v, int i) {
  if (BITS == 4) return i % 4 == 0 ? v : i % 4 == 1 ? v / 16.0f : i % 4 == 2 ? v / 256.0f : v / 4096.0f;
  if (BITS == 2) return v * tf_rp_p2(2 * i);
  return v;
}

// Four more inputs into a group sum: fp32 in order, or rounded to bf16 at each add (BF16SUM, Nemotron's experts).
template <bool BF16SUM>
inline float tf_rp_sum4(float sum, bfloat a, bfloat b, bfloat c, bfloat d) {
  if (BF16SUM) return sum + float(bfloat(float(bfloat(float(bfloat(float(a) + float(b))) + float(c))) + float(d)));
  sum += float(a); sum += float(b); sum += float(c); sum += float(d);
  return sum;
}

// 16 bf16 inputs and their group sum: as floats in their codes' places, or as bf16 (tf_rp_in places them where used).
template <int BITS, bool BF16SUM>
inline float tf_rp_load16(const device bfloat* x, thread float* xt) {
  float sum = 0.0f;
  for (int i = 0; i < 16; i += 4) {
    const bfloat a = x[i], b = x[i + 1], c = x[i + 2], d = x[i + 3];
    sum = tf_rp_sum4<BF16SUM>(sum, a, b, c, d);
    xt[i] = tf_rp_x<BITS>(float(a), i);
    xt[i + 1] = tf_rp_x<BITS>(float(b), i + 1);
    xt[i + 2] = tf_rp_x<BITS>(float(c), i + 2);
    xt[i + 3] = tf_rp_x<BITS>(float(d), i + 3);
  }
  return sum;
}

template <int BITS, bool BF16SUM>
inline float tf_rp_load16(const device bfloat* x, thread bfloat* xt) {
  float sum = 0.0f;
  for (int i = 0; i < 16; i += 4) {
    const bfloat a = x[i], b = x[i + 1], c = x[i + 2], d = x[i + 3];
    sum = tf_rp_sum4<BF16SUM>(sum, a, b, c, d);
    xt[i] = a; xt[i + 1] = b; xt[i + 2] = c; xt[i + 3] = d;
  }
  return sum;
}

// A loaded input as a float in its code's place.
template <int BITS>
inline float tf_rp_in(float v, int) { return v; }
template <int BITS>
inline float tf_rp_in(bfloat v, int i) { return tf_rp_x<BITS>(float(v), i); }

// MB rows' dots with 16 codes (3, 5, 6 bits: 8, 8, 4 codes in 3, 5, 3 bytes), unpacked once; sums in one order.
template <int MB, int BITS, typename X>
inline void tf_rp_qdot16(const device uint8_t* w, const thread X (*xt)[16], float scale, float bias,
                         const thread float* sum, thread float* out) {
  float q[16];
  if (BITS == 2) {
    const uint v = *(const device uint*)w;
    for (int i = 0; i < 16; i++) q[i] = float(v & (3u << (2 * i)));
  } else if (BITS == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < 4; i++) {
      const int v = ws[i];
      q[4 * i] = float(v & 0x000f); q[4 * i + 1] = float(v & 0x00f0); q[4 * i + 2] = float(v & 0x0f00); q[4 * i + 3] = float(v & 0xf000);
    }
  } else if (BITS == 6) {
    for (int i = 0; i < 4; i++) {
      const device uint8_t* p = w + 3 * i;
      q[4 * i] = float(p[0] & 0x3f);
      q[4 * i + 1] = float(((p[0] >> 6) & 0x03) | ((p[1] & 0x0f) << 2));
      q[4 * i + 2] = float(((p[1] >> 4) & 0x0f) | ((p[2] & 0x03) << 4));
      q[4 * i + 3] = float((p[2] >> 2) & 0x3f);
    }
  } else if (BITS == 3) {
    for (int i = 0; i < 2; i++) {
      const device uint8_t* p = w + 3 * i;
      const uint v = uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16);
      for (int j = 0; j < 8; j++) q[8 * i + j] = float((v >> (3 * j)) & 7u);
    }
  } else if (BITS == 5) {
    for (int i = 0; i < 2; i++) {
      const device uint8_t* p = w + 5 * i;
      const uint lo = uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16) | (uint(p[3]) << 24), hi = p[4];
      for (int j = 0; j < 6; j++) q[8 * i + j] = float((lo >> (5 * j)) & 31u);
      q[8 * i + 6] = float(((lo >> 30) | (hi << 2)) & 31u);
      q[8 * i + 7] = float(hi >> 3);
    }
  } else {
    for (int i = 0; i < 16; i++) q[i] = float(w[i]);
  }
  for (int b = 0; b < MB; b++) {
    float accum = 0.0f;
    for (int i = 0; i < 4; i++)
      accum += (tf_rp_in<BITS>(xt[b][4 * i], 4 * i) * q[4 * i] + tf_rp_in<BITS>(xt[b][4 * i + 1], 4 * i + 1) * q[4 * i + 1] +
                tf_rp_in<BITS>(xt[b][4 * i + 2], 4 * i + 2) * q[4 * i + 2] +
                tf_rp_in<BITS>(xt[b][4 * i + 3], 4 * i + 3) * q[4 * i + 3]);
    out[b] += scale * accum + sum[b] * bias;
  }
}
