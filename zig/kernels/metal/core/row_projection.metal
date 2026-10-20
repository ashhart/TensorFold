// Affine 2- to 8-bit row projections without tensor units: RPS columns a simdgroup, MB rows a pass over the weights.
#include <metal_stdlib>
using namespace metal;

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

// RPS columns of MB rows at x.x[b], lanes splitting K by 16 (512 a pass); 3 hold bf16, 4+ go in pairs; same sums.
template <int MB> struct TfRpRows { const device bfloat* x[MB]; };

template <int RPS, int MB, int BITS, bool BF16SUM>
inline void tf_rp_rows_at(const device uint8_t* w, const device bfloat* sc, const device bfloat* bi, thread const TfRpRows<MB>& x,
                          int K, int GS, uint lane, thread float (*acc)[RPS]) {
  typedef typename metal::conditional<(MB >= 3), bfloat, float>::type X;
  static_assert(MB <= 3 || MB % 2 == 0, "4+ rows run in pairs");
  const int KB = K * BITS / 8;
  const int KG = K / GS;
  const int FULL = K / 512 * 512;
  const int per_lane = GS / 16;
  w += lane * (16 * BITS / 8);
  sc += lane / per_lane;
  bi += lane / per_lane;
  const int xl = int(lane) * 16;
  float col[MB];
  for (int j = 0; j < RPS; j++)
    for (int b = 0; b < MB; b++) acc[b][j] = 0.0f;
  for (int k0 = 0; k0 <= FULL; k0 += 512) {
    if (k0 == FULL && !(FULL < K && int(lane) < (K - FULL) / 16)) break;
    if (MB > 3) {
      for (int p = 0; p < MB; p += 2) {
        float xt[2][16];
        float sum[2];
        for (int b = 0; b < 2; b++) sum[b] = tf_rp_load16<BITS, BF16SUM>(x.x[p + b] + xl + k0, xt[b]);
        for (int j = 0; j < RPS; j++) {
          float two[2] = {acc[p][j], acc[p + 1][j]};
          tf_rp_qdot16<2, BITS>(w + j * KB, xt, float(sc[j * KG]), float(bi[j * KG]), sum, two);
          acc[p][j] = two[0];
          acc[p + 1][j] = two[1];
        }
      }
    } else {
      X xt[MB][16];
      float sum[MB];
      for (int b = 0; b < MB; b++) sum[b] = tf_rp_load16<BITS, BF16SUM>(x.x[b] + xl + k0, xt[b]);
      for (int j = 0; j < RPS; j++) {
        for (int b = 0; b < MB; b++) col[b] = acc[b][j];
        tf_rp_qdot16<MB, BITS>(w + j * KB, xt, float(sc[j * KG]), float(bi[j * KG]), sum, col);
        for (int b = 0; b < MB; b++) acc[b][j] = col[b];
      }
    }
    w += 512 * BITS / 8; sc += 512 / GS; bi += 512 / GS;
  }
  for (int j = 0; j < RPS; j++)
    for (int b = 0; b < MB; b++) acc[b][j] = simd_sum(acc[b][j]);
}

// Contiguous callers use the same reduction with their original row starts.
template <int RPS, int MB, int BITS, bool BF16SUM>
inline void tf_rp_rows(const device uint8_t* w, const device bfloat* sc, const device bfloat* bi, const device bfloat* x0,
                       int K, int GS, uint lane, thread float (*acc)[RPS]) {
  TfRpRows<MB> x;
  for (int b = 0; b < MB; b++) x.x[b] = x0 + b * K;
  tf_rp_rows_at<RPS, MB, BITS, BF16SUM>(w, sc, bi, x, K, GS, lane, acc);
}

// A value out: bf16, relu squared with RELU2.
template <bool RELU2>
inline bfloat tf_rp_out(float v) {
  if (RELU2) {
    const float h = metal::max(float(bfloat(v)), 0.0f);
    return bfloat(h * h);
  }
  return bfloat(v);
}

// dims: K, N, GS, rows, OUT's row stride when STRIDED (else N). X [rows, K]; W [N, K*BITS/32] words; S, B [N, K/GS].
template <int RPS, int SG, int MB, bool RELU2, int BITS, bool BF16SUM, bool STRIDED = false>
[[kernel]] void tf_row_projection(const device bfloat* X [[buffer(0)]],
                                  const device uint32_t* W [[buffer(1)]],
                                  const device bfloat* S [[buffer(2)]],
                                  const device bfloat* B [[buffer(3)]],
                                  constant int* dims [[buffer(4)]],
                                  device bfloat* OUT [[buffer(5)]],
                                  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
                                  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
                                  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  const int K = dims[0], N = dims[1], GS = dims[2], rows = dims[3], LD = STRIDED ? dims[4] : N;
  const uint lane = thread_index_in_simdgroup;
  const int row0 = (int(threadgroup_position_in_grid.x) * SG + int(simdgroup_index_in_threadgroup)) * RPS;
  if (row0 >= N) return;
  const size_t at = size_t(row0);
  const device uint8_t* wr = (const device uint8_t*)W + at * (K * BITS / 8);
  const device bfloat* sr = S + at * (K / GS);
  const device bfloat* br = B + at * (K / GS);
  int m = 0;
  #pragma clang loop unroll(disable)
  for (; m + MB <= rows; m += MB) {
    float acc[MB][RPS];
    tf_rp_rows<RPS, MB, BITS, BF16SUM>(wr, sr, br, X + size_t(m) * K, K, GS, lane, acc);
    if (lane == 0)
      for (int b = 0; b < MB; b++)
        for (int j = 0; j < RPS; j++) OUT[size_t(m + b) * LD + row0 + j] = tf_rp_out<RELU2>(acc[b][j]);
  }
  #pragma clang loop unroll(disable)
  for (; m < rows; m++) {
    float acc[1][RPS];
    tf_rp_rows<RPS, 1, BITS, BF16SUM>(wr, sr, br, X + size_t(m) * K, K, GS, lane, acc);
    if (lane == 0)
      for (int j = 0; j < RPS; j++) OUT[size_t(m) * LD + row0 + j] = tf_rp_out<RELU2>(acc[0][j]);
  }
}

// dims: K, N, GS, slots, experts, repeat; W, S, B hold the experts back to back; slot y takes expert ids[y] on x row y / repeat (an id past the count: a zero row).
template <int RPS, int SG, bool RELU2, int BITS, bool BF16SUM>
[[kernel]] void tf_row_projection_indexed(const device bfloat* X [[buffer(0)]],
                                          const device uint32_t* W [[buffer(1)]],
                                          const device bfloat* S [[buffer(2)]],
                                          const device bfloat* B [[buffer(3)]],
                                          constant int* dims [[buffer(4)]],
                                          device bfloat* OUT [[buffer(5)]],
                                          const device uint* ids [[buffer(6)]],
                                          uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
                                          uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
                                          uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  const int K = dims[0], N = dims[1], GS = dims[2], experts = dims[4], repeat = dims[5];
  const uint lane = thread_index_in_simdgroup;
  const int row0 = (int(threadgroup_position_in_grid.x) * SG + int(simdgroup_index_in_threadgroup)) * RPS;
  if (row0 >= N) return;
  const uint slot = threadgroup_position_in_grid.y;
  const uint expert = ids[slot];
  device bfloat* out = OUT + size_t(slot) * N + row0;
  if (expert >= uint(experts)) {
    if (lane < uint(RPS)) out[lane] = bfloat(0.0f);
    return;
  }
  const size_t at = size_t(expert) * N + size_t(row0);
  const device uint8_t* wr = (const device uint8_t*)W + at * (K * BITS / 8);
  const device bfloat* sr = S + at * (K / GS);
  const device bfloat* br = B + at * (K / GS);
  float acc[1][RPS];
  tf_rp_rows<RPS, 1, BITS, BF16SUM>(wr, sr, br, X + size_t(slot / repeat) * K, K, GS, lane, acc);
  if (lane == 0)
    for (int j = 0; j < RPS; j++) out[j] = tf_rp_out<RELU2>(acc[0][j]);
}

template [[host_name("tf_row_projection_q2_f32")]] [[kernel]] decltype(tf_row_projection<4, 2, 2, false, 2, false>) tf_row_projection<4, 2, 2, false, 2, false>;
template [[host_name("tf_row_projection_indexed_q2_f32")]] [[kernel]] decltype(tf_row_projection_indexed<4, 2, false, 2, false>) tf_row_projection_indexed<4, 2, false, 2, false>;
template [[host_name("tf_row_projection_relu2_q2_f32")]] [[kernel]] decltype(tf_row_projection<4, 2, 2, true, 2, false>) tf_row_projection<4, 2, 2, true, 2, false>;
template [[host_name("tf_row_projection_indexed_relu2_q2_f32")]] [[kernel]] decltype(tf_row_projection_indexed<4, 2, true, 2, false>) tf_row_projection_indexed<4, 2, true, 2, false>;
template [[host_name("tf_row_projection_q2_bf16")]] [[kernel]] decltype(tf_row_projection<4, 2, 2, false, 2, true>) tf_row_projection<4, 2, 2, false, 2, true>;
template [[host_name("tf_row_projection_indexed_q2_bf16")]] [[kernel]] decltype(tf_row_projection_indexed<4, 2, false, 2, true>) tf_row_projection_indexed<4, 2, false, 2, true>;
template [[host_name("tf_row_projection_relu2_q2_bf16")]] [[kernel]] decltype(tf_row_projection<4, 2, 2, true, 2, true>) tf_row_projection<4, 2, 2, true, 2, true>;
template [[host_name("tf_row_projection_indexed_relu2_q2_bf16")]] [[kernel]] decltype(tf_row_projection_indexed<4, 2, true, 2, true>) tf_row_projection_indexed<4, 2, true, 2, true>;
template [[host_name("tf_row_projection_q4_f32")]] [[kernel]] decltype(tf_row_projection<4, 2, 2, false, 4, false>) tf_row_projection<4, 2, 2, false, 4, false>;
template [[host_name("tf_row_projection_indexed_q4_f32")]] [[kernel]] decltype(tf_row_projection_indexed<4, 2, false, 4, false>) tf_row_projection_indexed<4, 2, false, 4, false>;
template [[host_name("tf_row_projection_relu2_q4_f32")]] [[kernel]] decltype(tf_row_projection<4, 2, 2, true, 4, false>) tf_row_projection<4, 2, 2, true, 4, false>;
template [[host_name("tf_row_projection_indexed_relu2_q4_f32")]] [[kernel]] decltype(tf_row_projection_indexed<4, 2, true, 4, false>) tf_row_projection_indexed<4, 2, true, 4, false>;
template [[host_name("tf_row_projection_q4_bf16")]] [[kernel]] decltype(tf_row_projection<4, 2, 2, false, 4, true>) tf_row_projection<4, 2, 2, false, 4, true>;
template [[host_name("tf_row_projection_indexed_q4_bf16")]] [[kernel]] decltype(tf_row_projection_indexed<4, 2, false, 4, true>) tf_row_projection_indexed<4, 2, false, 4, true>;
template [[host_name("tf_row_projection_relu2_q4_bf16")]] [[kernel]] decltype(tf_row_projection<4, 2, 2, true, 4, true>) tf_row_projection<4, 2, 2, true, 4, true>;
template [[host_name("tf_row_projection_indexed_relu2_q4_bf16")]] [[kernel]] decltype(tf_row_projection_indexed<4, 2, true, 4, true>) tf_row_projection_indexed<4, 2, true, 4, true>;
template [[host_name("tf_row_projection_q6_f32")]] [[kernel]] decltype(tf_row_projection<4, 2, 2, false, 6, false>) tf_row_projection<4, 2, 2, false, 6, false>;
template [[host_name("tf_row_projection_indexed_q6_f32")]] [[kernel]] decltype(tf_row_projection_indexed<4, 2, false, 6, false>) tf_row_projection_indexed<4, 2, false, 6, false>;
template [[host_name("tf_row_projection_relu2_q6_f32")]] [[kernel]] decltype(tf_row_projection<4, 2, 2, true, 6, false>) tf_row_projection<4, 2, 2, true, 6, false>;
template [[host_name("tf_row_projection_indexed_relu2_q6_f32")]] [[kernel]] decltype(tf_row_projection_indexed<4, 2, true, 6, false>) tf_row_projection_indexed<4, 2, true, 6, false>;
template [[host_name("tf_row_projection_q6_bf16")]] [[kernel]] decltype(tf_row_projection<4, 2, 2, false, 6, true>) tf_row_projection<4, 2, 2, false, 6, true>;
template [[host_name("tf_row_projection_indexed_q6_bf16")]] [[kernel]] decltype(tf_row_projection_indexed<4, 2, false, 6, true>) tf_row_projection_indexed<4, 2, false, 6, true>;
template [[host_name("tf_row_projection_relu2_q6_bf16")]] [[kernel]] decltype(tf_row_projection<4, 2, 2, true, 6, true>) tf_row_projection<4, 2, 2, true, 6, true>;
template [[host_name("tf_row_projection_indexed_relu2_q6_bf16")]] [[kernel]] decltype(tf_row_projection_indexed<4, 2, true, 6, true>) tf_row_projection_indexed<4, 2, true, 6, true>;
template [[host_name("tf_row_projection_q8_f32")]] [[kernel]] decltype(tf_row_projection<4, 2, 2, false, 8, false>) tf_row_projection<4, 2, 2, false, 8, false>;
template [[host_name("tf_row_projection_indexed_q8_f32")]] [[kernel]] decltype(tf_row_projection_indexed<4, 2, false, 8, false>) tf_row_projection_indexed<4, 2, false, 8, false>;
template [[host_name("tf_row_projection_relu2_q8_f32")]] [[kernel]] decltype(tf_row_projection<4, 2, 2, true, 8, false>) tf_row_projection<4, 2, 2, true, 8, false>;
template [[host_name("tf_row_projection_indexed_relu2_q8_f32")]] [[kernel]] decltype(tf_row_projection_indexed<4, 2, true, 8, false>) tf_row_projection_indexed<4, 2, true, 8, false>;
template [[host_name("tf_row_projection_q8_bf16")]] [[kernel]] decltype(tf_row_projection<4, 2, 2, false, 8, true>) tf_row_projection<4, 2, 2, false, 8, true>;
template [[host_name("tf_row_projection_indexed_q8_bf16")]] [[kernel]] decltype(tf_row_projection_indexed<4, 2, false, 8, true>) tf_row_projection_indexed<4, 2, false, 8, true>;
template [[host_name("tf_row_projection_relu2_q8_bf16")]] [[kernel]] decltype(tf_row_projection<4, 2, 2, true, 8, true>) tf_row_projection<4, 2, 2, true, 8, true>;
template [[host_name("tf_row_projection_indexed_relu2_q8_bf16")]] [[kernel]] decltype(tf_row_projection_indexed<4, 2, true, 8, true>) tf_row_projection_indexed<4, 2, true, 8, true>;
template [[host_name("tf_row_projection_q3_f32")]] [[kernel]] decltype(tf_row_projection<4, 2, 2, false, 3, false>) tf_row_projection<4, 2, 2, false, 3, false>;
template [[host_name("tf_row_projection_indexed_q3_f32")]] [[kernel]] decltype(tf_row_projection_indexed<4, 2, false, 3, false>) tf_row_projection_indexed<4, 2, false, 3, false>;
template [[host_name("tf_row_projection_relu2_q3_f32")]] [[kernel]] decltype(tf_row_projection<4, 2, 2, true, 3, false>) tf_row_projection<4, 2, 2, true, 3, false>;
template [[host_name("tf_row_projection_indexed_relu2_q3_f32")]] [[kernel]] decltype(tf_row_projection_indexed<4, 2, true, 3, false>) tf_row_projection_indexed<4, 2, true, 3, false>;
template [[host_name("tf_row_projection_q3_bf16")]] [[kernel]] decltype(tf_row_projection<4, 2, 2, false, 3, true>) tf_row_projection<4, 2, 2, false, 3, true>;
template [[host_name("tf_row_projection_indexed_q3_bf16")]] [[kernel]] decltype(tf_row_projection_indexed<4, 2, false, 3, true>) tf_row_projection_indexed<4, 2, false, 3, true>;
template [[host_name("tf_row_projection_relu2_q3_bf16")]] [[kernel]] decltype(tf_row_projection<4, 2, 2, true, 3, true>) tf_row_projection<4, 2, 2, true, 3, true>;
template [[host_name("tf_row_projection_indexed_relu2_q3_bf16")]] [[kernel]] decltype(tf_row_projection_indexed<4, 2, true, 3, true>) tf_row_projection_indexed<4, 2, true, 3, true>;
template [[host_name("tf_row_projection_q5_f32")]] [[kernel]] decltype(tf_row_projection<4, 2, 2, false, 5, false>) tf_row_projection<4, 2, 2, false, 5, false>;
template [[host_name("tf_row_projection_indexed_q5_f32")]] [[kernel]] decltype(tf_row_projection_indexed<4, 2, false, 5, false>) tf_row_projection_indexed<4, 2, false, 5, false>;
template [[host_name("tf_row_projection_relu2_q5_f32")]] [[kernel]] decltype(tf_row_projection<4, 2, 2, true, 5, false>) tf_row_projection<4, 2, 2, true, 5, false>;
template [[host_name("tf_row_projection_indexed_relu2_q5_f32")]] [[kernel]] decltype(tf_row_projection_indexed<4, 2, true, 5, false>) tf_row_projection_indexed<4, 2, true, 5, false>;
template [[host_name("tf_row_projection_q5_bf16")]] [[kernel]] decltype(tf_row_projection<4, 2, 2, false, 5, true>) tf_row_projection<4, 2, 2, false, 5, true>;
template [[host_name("tf_row_projection_indexed_q5_bf16")]] [[kernel]] decltype(tf_row_projection_indexed<4, 2, false, 5, true>) tf_row_projection_indexed<4, 2, false, 5, true>;
template [[host_name("tf_row_projection_relu2_q5_bf16")]] [[kernel]] decltype(tf_row_projection<4, 2, 2, true, 5, true>) tf_row_projection<4, 2, 2, true, 5, true>;
template [[host_name("tf_row_projection_indexed_relu2_q5_bf16")]] [[kernel]] decltype(tf_row_projection_indexed<4, 2, true, 5, true>) tf_row_projection_indexed<4, 2, true, 5, true>;
template [[host_name("tf_row_projection_strided_q2_f32")]] [[kernel]] decltype(tf_row_projection<4, 2, 2, false, 2, false, true>) tf_row_projection<4, 2, 2, false, 2, false, true>;
template [[host_name("tf_row_projection_strided_q2_bf16")]] [[kernel]] decltype(tf_row_projection<4, 2, 2, false, 2, true, true>) tf_row_projection<4, 2, 2, false, 2, true, true>;
template [[host_name("tf_row_projection_strided_q3_f32")]] [[kernel]] decltype(tf_row_projection<4, 2, 2, false, 3, false, true>) tf_row_projection<4, 2, 2, false, 3, false, true>;
template [[host_name("tf_row_projection_strided_q3_bf16")]] [[kernel]] decltype(tf_row_projection<4, 2, 2, false, 3, true, true>) tf_row_projection<4, 2, 2, false, 3, true, true>;
template [[host_name("tf_row_projection_strided_q4_f32")]] [[kernel]] decltype(tf_row_projection<4, 2, 2, false, 4, false, true>) tf_row_projection<4, 2, 2, false, 4, false, true>;
template [[host_name("tf_row_projection_strided_q4_bf16")]] [[kernel]] decltype(tf_row_projection<4, 2, 2, false, 4, true, true>) tf_row_projection<4, 2, 2, false, 4, true, true>;
template [[host_name("tf_row_projection_strided_q5_f32")]] [[kernel]] decltype(tf_row_projection<4, 2, 2, false, 5, false, true>) tf_row_projection<4, 2, 2, false, 5, false, true>;
template [[host_name("tf_row_projection_strided_q5_bf16")]] [[kernel]] decltype(tf_row_projection<4, 2, 2, false, 5, true, true>) tf_row_projection<4, 2, 2, false, 5, true, true>;
template [[host_name("tf_row_projection_strided_q6_f32")]] [[kernel]] decltype(tf_row_projection<4, 2, 2, false, 6, false, true>) tf_row_projection<4, 2, 2, false, 6, false, true>;
template [[host_name("tf_row_projection_strided_q6_bf16")]] [[kernel]] decltype(tf_row_projection<4, 2, 2, false, 6, true, true>) tf_row_projection<4, 2, 2, false, 6, true, true>;
template [[host_name("tf_row_projection_strided_q8_f32")]] [[kernel]] decltype(tf_row_projection<4, 2, 2, false, 8, false, true>) tf_row_projection<4, 2, 2, false, 8, false, true>;
template [[host_name("tf_row_projection_strided_q8_bf16")]] [[kernel]] decltype(tf_row_projection<4, 2, 2, false, 8, true, true>) tf_row_projection<4, 2, 2, false, 8, true, true>;
