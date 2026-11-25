// Affine 2- to 8-bit row projections without tensor units: RPS columns a simdgroup, MB rows a pass over the weights.
#include <metal_stdlib>
using namespace metal;

// The kernel contract's planted control (TF_RP_BAD_ROWS 1) gives every SHARED row row 0's inputs.
#ifndef TF_RP_BAD_ROWS
#define TF_RP_BAD_ROWS 0
#endif

#include "row_projection_helpers.metal"

// RPS columns of MB rows at x.x[b], K split by 16 (512 a pass); 3 or SHARED hold bf16, 4+ else in pairs; same sums.
template <int MB> struct TfRpRows { const device bfloat* x[MB]; };

template <int RPS, int MB, int BITS, bool BF16SUM, bool SHARED = false>
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
    if (MB > 3 && !SHARED) {
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
template <int RPS, int MB, int BITS, bool BF16SUM, bool SHARED = false>
inline void tf_rp_rows(const device uint8_t* w, const device bfloat* sc, const device bfloat* bi, const device bfloat* x0,
                       int K, int GS, uint lane, thread float (*acc)[RPS]) {
  TfRpRows<MB> x;
  for (int b = 0; b < MB; b++) x.x[b] = x0 + (SHARED && TF_RP_BAD_ROWS ? 0 : b) * K;
  tf_rp_rows_at<RPS, MB, BITS, BF16SUM, SHARED>(w, sc, bi, x, K, GS, lane, acc);
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
template <int RPS, int SG, int MB, bool RELU2, int BITS, bool BF16SUM, bool STRIDED, bool WIDE>
inline void tf_row_projection_body(const device bfloat* X, const device uint32_t* W,
                                    const device bfloat* S, const device bfloat* B, constant int* dims,
                                    device bfloat* OUT, uint simdgroup_index_in_threadgroup,
                                    uint thread_index_in_simdgroup, uint3 threadgroup_position_in_grid) {
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
    tf_rp_rows<RPS, MB, BITS, BF16SUM, WIDE>(wr, sr, br, X + size_t(m) * K, K, GS, lane, acc);
    if (lane == 0)
      for (int b = 0; b < MB; b++)
        for (int j = 0; j < RPS; j++) OUT[size_t(m + b) * LD + row0 + j] = tf_rp_out<RELU2>(acc[b][j]);
  }
  if (WIDE && m + 4 <= rows) {
    float acc[4][RPS];
    tf_rp_rows<RPS, 4, BITS, BF16SUM, true>(wr, sr, br, X + size_t(m) * K, K, GS, lane, acc);
    if (lane == 0)
      for (int b = 0; b < 4; b++)
        for (int j = 0; j < RPS; j++) OUT[size_t(m + b) * LD + row0 + j] = tf_rp_out<RELU2>(acc[b][j]);
    m += 4;
  }
  if (WIDE && m + 2 <= rows) {
    float acc[2][RPS];
    tf_rp_rows<RPS, 2, BITS, BF16SUM>(wr, sr, br, X + size_t(m) * K, K, GS, lane, acc);
    if (lane == 0)
      for (int b = 0; b < 2; b++)
        for (int j = 0; j < RPS; j++) OUT[size_t(m + b) * LD + row0 + j] = tf_rp_out<RELU2>(acc[b][j]);
    m += 2;
  }
  #pragma clang loop unroll(disable)
  for (; m < rows; m++) {
    float acc[1][RPS];
    tf_rp_rows<RPS, 1, BITS, BF16SUM>(wr, sr, br, X + size_t(m) * K, K, GS, lane, acc);
    if (lane == 0)
      for (int j = 0; j < RPS; j++) OUT[size_t(m) * LD + row0 + j] = tf_rp_out<RELU2>(acc[0][j]);
  }
}

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
  tf_row_projection_body<RPS, SG, MB, RELU2, BITS, BF16SUM, STRIDED, false>(X, W, S, B, dims, OUT,
    simdgroup_index_in_threadgroup, thread_index_in_simdgroup, threadgroup_position_in_grid);
}

// Windows of 4+ rows (core/row_projection.zig picks it): 8s, then a 4, a 2 and a last row; 64 threads reserved.
template <int RPS, int SG, bool RELU2, int BITS, bool BF16SUM, bool STRIDED = false>
[[max_total_threads_per_threadgroup(64)]]
[[kernel]] void tf_row_projection_wide(const device bfloat* X [[buffer(0)]],
                                  const device uint32_t* W [[buffer(1)]],
                                  const device bfloat* S [[buffer(2)]],
                                  const device bfloat* B [[buffer(3)]],
                                  constant int* dims [[buffer(4)]],
                                  device bfloat* OUT [[buffer(5)]],
                                  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
                                  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
                                  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  tf_row_projection_body<RPS, SG, 8, RELU2, BITS, BF16SUM, STRIDED, true>(X, W, S, B, dims, OUT,
    simdgroup_index_in_threadgroup, thread_index_in_simdgroup, threadgroup_position_in_grid);
}

#include "row_projection_tail.metal"
template [[host_name("tf_row_projection_wide_q2_f32")]] [[kernel]] decltype(tf_row_projection_wide<4, 2, false, 2, false, false>) tf_row_projection_wide<4, 2, false, 2, false, false>;
template [[host_name("tf_row_projection_wide_relu2_q2_f32")]] [[kernel]] decltype(tf_row_projection_wide<4, 2, true, 2, false, false>) tf_row_projection_wide<4, 2, true, 2, false, false>;
template [[host_name("tf_row_projection_wide_strided_q2_f32")]] [[kernel]] decltype(tf_row_projection_wide<4, 2, false, 2, false, true>) tf_row_projection_wide<4, 2, false, 2, false, true>;
template [[host_name("tf_row_projection_wide_q2_bf16")]] [[kernel]] decltype(tf_row_projection_wide<4, 2, false, 2, true, false>) tf_row_projection_wide<4, 2, false, 2, true, false>;
template [[host_name("tf_row_projection_wide_relu2_q2_bf16")]] [[kernel]] decltype(tf_row_projection_wide<4, 2, true, 2, true, false>) tf_row_projection_wide<4, 2, true, 2, true, false>;
template [[host_name("tf_row_projection_wide_strided_q2_bf16")]] [[kernel]] decltype(tf_row_projection_wide<4, 2, false, 2, true, true>) tf_row_projection_wide<4, 2, false, 2, true, true>;
template [[host_name("tf_row_projection_wide_q3_f32")]] [[kernel]] decltype(tf_row_projection_wide<4, 2, false, 3, false, false>) tf_row_projection_wide<4, 2, false, 3, false, false>;
template [[host_name("tf_row_projection_wide_relu2_q3_f32")]] [[kernel]] decltype(tf_row_projection_wide<4, 2, true, 3, false, false>) tf_row_projection_wide<4, 2, true, 3, false, false>;
template [[host_name("tf_row_projection_wide_strided_q3_f32")]] [[kernel]] decltype(tf_row_projection_wide<4, 2, false, 3, false, true>) tf_row_projection_wide<4, 2, false, 3, false, true>;
template [[host_name("tf_row_projection_wide_q3_bf16")]] [[kernel]] decltype(tf_row_projection_wide<4, 2, false, 3, true, false>) tf_row_projection_wide<4, 2, false, 3, true, false>;
template [[host_name("tf_row_projection_wide_relu2_q3_bf16")]] [[kernel]] decltype(tf_row_projection_wide<4, 2, true, 3, true, false>) tf_row_projection_wide<4, 2, true, 3, true, false>;
template [[host_name("tf_row_projection_wide_strided_q3_bf16")]] [[kernel]] decltype(tf_row_projection_wide<4, 2, false, 3, true, true>) tf_row_projection_wide<4, 2, false, 3, true, true>;
template [[host_name("tf_row_projection_wide_q4_f32")]] [[kernel]] decltype(tf_row_projection_wide<4, 2, false, 4, false, false>) tf_row_projection_wide<4, 2, false, 4, false, false>;
template [[host_name("tf_row_projection_wide_relu2_q4_f32")]] [[kernel]] decltype(tf_row_projection_wide<4, 2, true, 4, false, false>) tf_row_projection_wide<4, 2, true, 4, false, false>;
template [[host_name("tf_row_projection_wide_strided_q4_f32")]] [[kernel]] decltype(tf_row_projection_wide<4, 2, false, 4, false, true>) tf_row_projection_wide<4, 2, false, 4, false, true>;
template [[host_name("tf_row_projection_wide_q4_bf16")]] [[kernel]] decltype(tf_row_projection_wide<4, 2, false, 4, true, false>) tf_row_projection_wide<4, 2, false, 4, true, false>;
template [[host_name("tf_row_projection_wide_relu2_q4_bf16")]] [[kernel]] decltype(tf_row_projection_wide<4, 2, true, 4, true, false>) tf_row_projection_wide<4, 2, true, 4, true, false>;
template [[host_name("tf_row_projection_wide_strided_q4_bf16")]] [[kernel]] decltype(tf_row_projection_wide<4, 2, false, 4, true, true>) tf_row_projection_wide<4, 2, false, 4, true, true>;
template [[host_name("tf_row_projection_wide_q5_f32")]] [[kernel]] decltype(tf_row_projection_wide<4, 2, false, 5, false, false>) tf_row_projection_wide<4, 2, false, 5, false, false>;
template [[host_name("tf_row_projection_wide_relu2_q5_f32")]] [[kernel]] decltype(tf_row_projection_wide<4, 2, true, 5, false, false>) tf_row_projection_wide<4, 2, true, 5, false, false>;
template [[host_name("tf_row_projection_wide_strided_q5_f32")]] [[kernel]] decltype(tf_row_projection_wide<4, 2, false, 5, false, true>) tf_row_projection_wide<4, 2, false, 5, false, true>;
template [[host_name("tf_row_projection_wide_q5_bf16")]] [[kernel]] decltype(tf_row_projection_wide<4, 2, false, 5, true, false>) tf_row_projection_wide<4, 2, false, 5, true, false>;
template [[host_name("tf_row_projection_wide_relu2_q5_bf16")]] [[kernel]] decltype(tf_row_projection_wide<4, 2, true, 5, true, false>) tf_row_projection_wide<4, 2, true, 5, true, false>;
template [[host_name("tf_row_projection_wide_strided_q5_bf16")]] [[kernel]] decltype(tf_row_projection_wide<4, 2, false, 5, true, true>) tf_row_projection_wide<4, 2, false, 5, true, true>;
template [[host_name("tf_row_projection_wide_q6_f32")]] [[kernel]] decltype(tf_row_projection_wide<4, 2, false, 6, false, false>) tf_row_projection_wide<4, 2, false, 6, false, false>;
template [[host_name("tf_row_projection_wide_relu2_q6_f32")]] [[kernel]] decltype(tf_row_projection_wide<4, 2, true, 6, false, false>) tf_row_projection_wide<4, 2, true, 6, false, false>;
template [[host_name("tf_row_projection_wide_strided_q6_f32")]] [[kernel]] decltype(tf_row_projection_wide<4, 2, false, 6, false, true>) tf_row_projection_wide<4, 2, false, 6, false, true>;
template [[host_name("tf_row_projection_wide_q6_bf16")]] [[kernel]] decltype(tf_row_projection_wide<4, 2, false, 6, true, false>) tf_row_projection_wide<4, 2, false, 6, true, false>;
template [[host_name("tf_row_projection_wide_relu2_q6_bf16")]] [[kernel]] decltype(tf_row_projection_wide<4, 2, true, 6, true, false>) tf_row_projection_wide<4, 2, true, 6, true, false>;
template [[host_name("tf_row_projection_wide_strided_q6_bf16")]] [[kernel]] decltype(tf_row_projection_wide<4, 2, false, 6, true, true>) tf_row_projection_wide<4, 2, false, 6, true, true>;
template [[host_name("tf_row_projection_wide_q8_f32")]] [[kernel]] decltype(tf_row_projection_wide<4, 2, false, 8, false, false>) tf_row_projection_wide<4, 2, false, 8, false, false>;
template [[host_name("tf_row_projection_wide_relu2_q8_f32")]] [[kernel]] decltype(tf_row_projection_wide<4, 2, true, 8, false, false>) tf_row_projection_wide<4, 2, true, 8, false, false>;
template [[host_name("tf_row_projection_wide_strided_q8_f32")]] [[kernel]] decltype(tf_row_projection_wide<4, 2, false, 8, false, true>) tf_row_projection_wide<4, 2, false, 8, false, true>;
template [[host_name("tf_row_projection_wide_q8_bf16")]] [[kernel]] decltype(tf_row_projection_wide<4, 2, false, 8, true, false>) tf_row_projection_wide<4, 2, false, 8, true, false>;
template [[host_name("tf_row_projection_wide_relu2_q8_bf16")]] [[kernel]] decltype(tf_row_projection_wide<4, 2, true, 8, true, false>) tf_row_projection_wide<4, 2, true, 8, true, false>;
template [[host_name("tf_row_projection_wide_strided_q8_bf16")]] [[kernel]] decltype(tf_row_projection_wide<4, 2, false, 8, true, true>) tf_row_projection_wide<4, 2, false, 8, true, true>;
