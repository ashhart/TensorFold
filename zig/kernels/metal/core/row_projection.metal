// Affine 4-bit row projections for chips without tensor units: RPS columns a simdgroup, MB rows a weight read, each row its own fp32 sums.
#include <metal_stdlib>
using namespace metal;

// 16 bf16 inputs as scaled floats (nibble positions) and their sum, rounded to bf16 at each add as the expert kernels do.
inline float tf_rp_load16(const device bfloat* x, thread float* xt) {
  float sum = 0.0f;
  for (int i = 0; i < 16; i += 4) {
    const bfloat a = x[i], b = x[i + 1], c = x[i + 2], d = x[i + 3];
    sum += float(bfloat(float(bfloat(float(bfloat(float(a) + float(b))) + float(c))) + float(d)));
    xt[i] = float(a); xt[i + 1] = float(b) / 16.0f; xt[i + 2] = float(c) / 256.0f; xt[i + 3] = float(d) / 4096.0f;
  }
  return sum;
}

// MB rows' dots with one weight word group (16 nibbles), the words unpacked to floats once; each row's sums in one order.
template <int MB>
inline void tf_rp_qdot16(const device uint8_t* w, const thread float (*xt)[16], float scale, float bias,
                         const thread float* sum, thread float* out) {
  const device uint16_t* ws = (const device uint16_t*)w;
  float q[16];
  for (int i = 0; i < 4; i++) {
    const int v = ws[i];
    q[4 * i] = float(v & 0x000f); q[4 * i + 1] = float(v & 0x00f0); q[4 * i + 2] = float(v & 0x0f00); q[4 * i + 3] = float(v & 0xf000);
  }
  for (int b = 0; b < MB; b++) {
    float accum = 0.0f;
    for (int i = 0; i < 4; i++)
      accum += (xt[b][4 * i] * q[4 * i] + xt[b][4 * i + 1] * q[4 * i + 1] +
                xt[b][4 * i + 2] * q[4 * i + 2] + xt[b][4 * i + 3] * q[4 * i + 3]);
    out[b] += scale * accum + sum[b] * bias;
  }
}

// RPS output columns of MB rows: lanes split K in 16-input pieces (512 a pass, then a tail), one simd_sum a (row, column).
template <int RPS, int MB>
inline void tf_rp_rows(const device uint8_t* w, const device bfloat* sc, const device bfloat* bi, const device bfloat* x0,
                       int K, int GS, uint lane, thread float (*acc)[RPS]) {
  const int KB = K / 2;
  const int KG = K / GS;
  const int FULL = K / 512 * 512;
  const int per_lane = GS / 16;
  w += lane * 8;
  sc += lane / per_lane;
  bi += lane / per_lane;
  const int xl = int(lane) * 16;
  float col[MB];
  for (int j = 0; j < RPS; j++)
    for (int b = 0; b < MB; b++) acc[b][j] = 0.0f;
  for (int k0 = 0; k0 <= FULL; k0 += 512) {
    if (k0 == FULL && !(FULL < K && int(lane) < (K - FULL) / 16)) break;
    float xt[MB][16];
    float sum[MB];
    for (int b = 0; b < MB; b++) sum[b] = tf_rp_load16(x0 + b * K + xl + k0, xt[b]);
    for (int j = 0; j < RPS; j++) {
      for (int b = 0; b < MB; b++) col[b] = acc[b][j];
      tf_rp_qdot16<MB>(w + j * KB, xt, float(sc[j * KG]), float(bi[j * KG]), sum, col);
      for (int b = 0; b < MB; b++) acc[b][j] = col[b];
    }
    w += 256; sc += 512 / GS; bi += 512 / GS;
  }
  for (int j = 0; j < RPS; j++)
    for (int b = 0; b < MB; b++) acc[b][j] = simd_sum(acc[b][j]);
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

// dims: K, N, GS, rows. X [rows, K] bf16; W [N, K/8] words; S, B [N, K/GS] bf16; OUT [rows, N] bf16.
template <int RPS, int SG, int MB, bool RELU2>
[[kernel]] void tf_row_projection(const device bfloat* X [[buffer(0)]],
                                  const device uint32_t* W [[buffer(1)]],
                                  const device bfloat* S [[buffer(2)]],
                                  const device bfloat* B [[buffer(3)]],
                                  constant int* dims [[buffer(4)]],
                                  device bfloat* OUT [[buffer(5)]],
                                  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
                                  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
                                  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  const int K = dims[0], N = dims[1], GS = dims[2], rows = dims[3];
  const uint lane = thread_index_in_simdgroup;
  const int row0 = (int(threadgroup_position_in_grid.x) * SG + int(simdgroup_index_in_threadgroup)) * RPS;
  if (row0 >= N) return;
  const size_t at = size_t(row0);
  const device uint8_t* wr = (const device uint8_t*)W + at * (K / 2);
  const device bfloat* sr = S + at * (K / GS);
  const device bfloat* br = B + at * (K / GS);
  int m = 0;
  #pragma clang loop unroll(disable)
  for (; m + MB <= rows; m += MB) {
    float acc[MB][RPS];
    tf_rp_rows<RPS, MB>(wr, sr, br, X + size_t(m) * K, K, GS, lane, acc);
    if (lane == 0)
      for (int b = 0; b < MB; b++)
        for (int j = 0; j < RPS; j++) OUT[size_t(m + b) * N + row0 + j] = tf_rp_out<RELU2>(acc[b][j]);
  }
  #pragma clang loop unroll(disable)
  for (; m < rows; m++) {
    float acc[1][RPS];
    tf_rp_rows<RPS, 1>(wr, sr, br, X + size_t(m) * K, K, GS, lane, acc);
    if (lane == 0)
      for (int j = 0; j < RPS; j++) OUT[size_t(m) * N + row0 + j] = tf_rp_out<RELU2>(acc[0][j]);
  }
}

template [[host_name("tf_row_projection")]] [[kernel]] decltype(tf_row_projection<4, 2, 2, false>) tf_row_projection<4, 2, 2, false>;
template [[host_name("tf_row_projection_relu2")]] [[kernel]] decltype(tf_row_projection<4, 2, 2, true>) tf_row_projection<4, 2, 2, true>;
