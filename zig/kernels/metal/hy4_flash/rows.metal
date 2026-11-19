
// Hy4-flash lane windows (compiled after the chip's core row projection source, whose helpers it calls).
// Each output column's per-row arithmetic is tf_rp_rows's, as in the plain and wide kernels, so rows give the same
// bits however many share a pass. Here every weight is read once for up to 8 rows, with RPS columns a simdgroup and
// SG simdgroups a threadgroup, and no 64-thread cap, so a 2- to 8-row window costs about what one row does.
template <int RPS, int MB, int BITS, bool SH>
inline void tf_hy4_rows_block(const device uint8_t* wr, const device bfloat* sr, const device bfloat* br,
                              const device bfloat* X, int K, int N, int GS, int m, int row0, uint lane,
                              device bfloat* OUT) {
  float acc[MB][RPS];
  tf_rp_rows<RPS, MB, BITS, false, SH && (MB >= 3)>(wr, sr, br, X + size_t(m) * K, K, GS, lane, acc);
  if (lane == 0)
    for (int b = 0; b < MB; b++)
      for (int j = 0; j < RPS; j++) OUT[size_t(m + b) * N + row0 + j] = tf_rp_out<false>(acc[b][j]);
}

// dims: K, N, GS, rows (as tf_row_projection). X [rows, K]; W [N, K*BITS/32] words; S, B [N, K/GS].
// SH: 3+ row blocks unpack each code once for all their rows (the core's shared branch) instead of once a pair.
template <int RPS, int SG, int BITS, bool SH = false>
[[kernel]] void tf_hy4_rows(const device bfloat* X [[buffer(0)]], const device uint32_t* W [[buffer(1)]],
                            const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]],
                            constant int* dims [[buffer(4)]], device bfloat* OUT [[buffer(5)]],
                            uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                            uint3 tg [[threadgroup_position_in_grid]]) {
  const int K = dims[0], N = dims[1], GS = dims[2], rows = dims[3];
  const int row0 = (int(tg.x) * SG + int(sg)) * RPS;
  if (row0 >= N) return;
  const size_t at = size_t(row0);
  const device uint8_t* wr = (const device uint8_t*)W + at * (K * BITS / 8);
  const device bfloat* sr = S + at * (K / GS);
  const device bfloat* br = B + at * (K / GS);
  int m = 0;
  #pragma clang loop unroll(disable)
  for (; m + 8 <= rows; m += 8) tf_hy4_rows_block<RPS, 8, BITS, SH>(wr, sr, br, X, K, N, GS, m, row0, lane, OUT);
  if (m + 6 <= rows) {
    tf_hy4_rows_block<RPS, 6, BITS, SH>(wr, sr, br, X, K, N, GS, m, row0, lane, OUT);
    m += 6;
  } else if (m + 4 <= rows) {
    tf_hy4_rows_block<RPS, 4, BITS, SH>(wr, sr, br, X, K, N, GS, m, row0, lane, OUT);
    m += 4;
  }
  if (m + 2 <= rows) {
    tf_hy4_rows_block<RPS, 2, BITS, SH>(wr, sr, br, X, K, N, GS, m, row0, lane, OUT);
    m += 2;
  }
  if (m < rows) tf_hy4_rows_block<RPS, 1, BITS, SH>(wr, sr, br, X, K, N, GS, m, row0, lane, OUT);
}

template [[host_name("tf_hy4_rows_r2_q4")]] [[kernel]] decltype(tf_hy4_rows<2, 4, 4>) tf_hy4_rows<2, 4, 4>;
template [[host_name("tf_hy4_rows_r2_q8")]] [[kernel]] decltype(tf_hy4_rows<2, 4, 8>) tf_hy4_rows<2, 4, 8>;
template [[host_name("tf_hy4_rows_r4_q4")]] [[kernel]] decltype(tf_hy4_rows<4, 4, 4>) tf_hy4_rows<4, 4, 4>;
template [[host_name("tf_hy4_rows_r4_q8")]] [[kernel]] decltype(tf_hy4_rows<4, 4, 8>) tf_hy4_rows<4, 4, 8>;
template [[host_name("tf_hy4_rows_r2s_q4")]] [[kernel]] decltype(tf_hy4_rows<2, 4, 4, true>) tf_hy4_rows<2, 4, 4, true>;
template [[host_name("tf_hy4_rows_r2s_q8")]] [[kernel]] decltype(tf_hy4_rows<2, 4, 8, true>) tf_hy4_rows<2, 4, 8, true>;
template [[host_name("tf_hy4_rows_r1s_q4")]] [[kernel]] decltype(tf_hy4_rows<1, 8, 4, true>) tf_hy4_rows<1, 8, 4, true>;
template [[host_name("tf_hy4_rows_r1s_q8")]] [[kernel]] decltype(tf_hy4_rows<1, 8, 8, true>) tf_hy4_rows<1, 8, 8, true>;
template [[host_name("tf_hy4_rows_r1_q4")]] [[kernel]] decltype(tf_hy4_rows<1, 8, 4>) tf_hy4_rows<1, 8, 4>;
template [[host_name("tf_hy4_rows_r1_q8")]] [[kernel]] decltype(tf_hy4_rows<1, 8, 8>) tf_hy4_rows<1, 8, 8>;

// Greedy pick on the GPU: the first index of a row's largest BF16 logit (NaN never wins), as the host's scan.
// a.x: vocabulary, a.y: where row 0's token goes in OUT.
[[kernel]] void tf_hy4_argmax(const device ushort* logits [[buffer(0)]], device uint* out [[buffer(1)]],
                              constant uint2& a [[buffer(2)]], uint row [[threadgroup_position_in_grid]],
                              uint t [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]]) {
  const device ushort* x = logits + size_t(row) * a.x;
  float best = -INFINITY;
  uint id = 0xffffffffu;
  for (uint i = t; i < a.x; i += 1024) {
    const float v = as_type<float>(uint(x[i]) << 16);
    if (v > best) { best = v; id = i; }
  }
  for (ushort o = 16; o > 0; o >>= 1) {
    const float ov = simd_shuffle_xor(best, o);
    const uint oi = simd_shuffle_xor(id, o);
    if (ov > best || (ov == best && oi < id)) { best = ov; id = oi; }
  }
  threadgroup float bv[32];
  threadgroup uint bi[32];
  if (lane == 0) { bv[sg] = best; bi[sg] = id; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    best = bv[lane]; id = bi[lane];
    for (ushort o = 16; o > 0; o >>= 1) {
      const float ov = simd_shuffle_xor(best, o);
      const uint oi = simd_shuffle_xor(id, o);
      if (ov > best || (ov == best && oi < id)) { best = ov; id = oi; }
    }
    if (lane == 0) out[a.y + row] = id;
  }
}

// tf_add_bf16 (A + B in bfloat) feeding tf_draft_rms in one launch: the routed and shared expert branches summed as
// each norm thread reads them, with the norm's thread split, sums and roundings, so the same bits and no add pass.
struct Hy4Norm { uint rows, heads, dim, x_stride, y_stride, round_normalized; float eps; };
[[kernel]] void tf_hy4_add_rms(device const bfloat* A [[buffer(0)]], device const bfloat* B [[buffer(1)]],
                               device const bfloat* W [[buffer(2)]], constant Hy4Norm& a [[buffer(3)]],
                               device bfloat* Y [[buffer(4)]], uint2 block [[threadgroup_position_in_grid]],
                               uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]], uint2 threads [[threads_per_threadgroup]]) {
  threadgroup float sums[32], inverse[1];
  float value = 0.0f;
  const size_t base = size_t(block.y) * a.x_stride + block.x * a.dim;
  for (uint first = 4 * t; first < a.dim; first += 4 * threads.x)
    for (uint j = 0; j < 4 && first + j < a.dim; j++) {
      const bfloat s = A[base + first + j] + B[base + first + j];
      float x = float(s);
      value += x * x;
    }
  value = simd_sum(value);
  if (sg == 0) sums[lane] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) sums[sg] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float total = simd_sum(sums[lane]);
    if (lane == 0) inverse[0] = precise::rsqrt(total / float(a.dim) + a.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint first = 4 * t; first < a.dim; first += 4 * threads.x)
    for (uint j = 0; j < 4 && first + j < a.dim; j++) {
      const uint col = first + j;
      const bfloat s = A[base + col] + B[base + col];
      float normalized = float(s) * inverse[0];
      normalized = float(bfloat(normalized));
      Y[size_t(block.y) * a.y_stride + block.x * a.dim + col] = bfloat(normalized * float(W[col]));
    }
}

// Decode-sized passes (1-8 rows) of an MLX-affine 4- or 8-bit [n, k] matrix (groups of 64): fp32 sums, each weight
// read once with 16-byte loads. A simdgroup owns QR output columns; lane l takes 32 consecutive inputs of every
// 1024 (half a group): it holds those columns' weight words, scales and biases, then streams each row's 32 inputs
// through them. y[r][n] = sum over groups of scale * (x . q) + bias * sum(x). dims: K, N, GS (64), rows.
template <int BITS, int QR>
[[kernel]] void tf_hy4_qmv(const device bfloat* X [[buffer(0)]], const device uint32_t* W [[buffer(1)]],
                           const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]],
                           constant int* dims [[buffer(4)]], device bfloat* OUT [[buffer(5)]],
                           uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                           uint3 tg [[threadgroup_position_in_grid]]) {
  const int K = dims[0], N = dims[1], rows = min(dims[3], 8);
  constexpr int SG = 4, WPL = 32 * BITS / 32; // weight words a lane takes a step (4 at 4 bits, 8 at 8)
  const int col0 = (int(tg.x) * SG + int(sg)) * QR;
  if (col0 >= N) return;
  const int words_row = K * BITS / 32, groups_row = K / 64;
  float acc[8][QR];
  for (int r = 0; r < 8; ++r)
    for (int j = 0; j < QR; ++j) acc[r][j] = 0.0f;
  for (int k0 = int(lane) * 32; k0 < K; k0 += 1024) {
    uint w[QR][WPL];
    float scale[QR], bias[QR];
    const int g = k0 / 64;
    for (int j = 0; j < QR; ++j) {
      const int n = col0 + j;
      const device uint4* wp = (const device uint4*)(W + size_t(n) * words_row + k0 * BITS / 32);
      for (int v = 0; v < WPL / 4; ++v) {
        const uint4 w4 = wp[v];
        w[j][4 * v] = w4.x; w[j][4 * v + 1] = w4.y; w[j][4 * v + 2] = w4.z; w[j][4 * v + 3] = w4.w;
      }
      scale[j] = float(S[size_t(n) * groups_row + g]);
      bias[j] = float(B[size_t(n) * groups_row + g]);
    }
    for (int r = 0; r < rows; ++r) {
      const device uint4* xp = (const device uint4*)(X + size_t(r) * K + k0);
      float dot[QR];
      for (int j = 0; j < QR; ++j) dot[j] = 0.0f;
      float xsum = 0.0f;
      for (int v = 0; v < 4; ++v) {
        const uint4 x4 = xp[v];
        const uint xw[4] = {x4.x, x4.y, x4.z, x4.w};
        for (int u = 0; u < 4; ++u) {
          const int i = v * 8 + u * 2; // input index within the lane's 32
          const float x0 = as_type<float>(xw[u] << 16), x1 = as_type<float>(xw[u] & 0xffff0000u);
          xsum += x0 + x1;
          for (int j = 0; j < QR; ++j) {
            float q0, q1;
            if (BITS == 4) {
              const uint word = w[j][i / 8];
              q0 = float((word >> (4 * (i % 8))) & 0xfu);
              q1 = float((word >> (4 * (i % 8 + 1))) & 0xfu);
            } else {
              const uint word = w[j][i / 4];
              q0 = float((word >> (8 * (i % 4))) & 0xffu);
              q1 = float((word >> (8 * (i % 4 + 1))) & 0xffu);
            }
            dot[j] = fma(x1, q1, fma(x0, q0, dot[j]));
          }
        }
      }
      for (int j = 0; j < QR; ++j) acc[r][j] = fma(scale[j], dot[j], fma(bias[j], xsum, acc[r][j]));
    }
  }
  for (int r = 0; r < rows; ++r) {
    for (int j = 0; j < QR; ++j) {
      const float v = simd_sum(acc[r][j]);
      if (lane == 0) OUT[size_t(r) * N + col0 + j] = bfloat(v);
    }
  }
}

template [[host_name("tf_hy4_qmv_q4")]] [[kernel]] decltype(tf_hy4_qmv<4, 4>) tf_hy4_qmv<4, 4>;
template [[host_name("tf_hy4_qmv_q8")]] [[kernel]] decltype(tf_hy4_qmv<8, 4>) tf_hy4_qmv<8, 4>;

// tf_hy4_qmv with the weights unpacked once a step for every row: two output columns a simdgroup, their 64 codes
// as floats in registers, each row's 32 inputs streamed past them (multi-row passes pay one unpack, not one a row).
template <int BITS>
[[kernel]] void tf_hy4_qmv2(const device bfloat* X [[buffer(0)]], const device uint32_t* W [[buffer(1)]],
                            const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]],
                            constant int* dims [[buffer(4)]], device bfloat* OUT [[buffer(5)]],
                            uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                            uint3 tg [[threadgroup_position_in_grid]]) {
  const int K = dims[0], N = dims[1], rows = min(dims[3], 8);
  constexpr int SG = 4, QR = 2, WPL = BITS;
  const int col0 = (int(tg.x) * SG + int(sg)) * QR;
  if (col0 >= N) return;
  const int words_row = K * BITS / 32, groups_row = K / 64;
  float acc[8][QR];
  for (int r = 0; r < 8; ++r)
    for (int j = 0; j < QR; ++j) acc[r][j] = 0.0f;
  for (int k0 = int(lane) * 32; k0 < K; k0 += 1024) {
    float q[QR][32];
    float scale[QR], bias[QR];
    const int g = k0 / 64;
    for (int j = 0; j < QR; ++j) {
      const int n = col0 + j;
      const device uint4* wp = (const device uint4*)(W + size_t(n) * words_row + k0 * BITS / 32);
      for (int v = 0; v < WPL / 4; ++v) {
        const uint4 w4 = wp[v];
        const uint words[4] = {w4.x, w4.y, w4.z, w4.w};
        for (int u = 0; u < 4; ++u) {
          if (BITS == 4) {
            for (int b = 0; b < 8; ++b) q[j][(v * 4 + u) * 8 + b] = float((words[u] >> (4 * b)) & 0xfu);
          } else {
            for (int b = 0; b < 4; ++b) q[j][(v * 4 + u) * 4 + b] = float((words[u] >> (8 * b)) & 0xffu);
          }
        }
      }
      scale[j] = float(S[size_t(n) * groups_row + g]);
      bias[j] = float(B[size_t(n) * groups_row + g]);
    }
    for (int r = 0; r < rows; ++r) {
      const device uint4* xp = (const device uint4*)(X + size_t(r) * K + k0);
      float dot[QR] = {0.0f, 0.0f};
      float xsum = 0.0f;
      for (int v = 0; v < 4; ++v) {
        const uint4 x4 = xp[v];
        const uint xw[4] = {x4.x, x4.y, x4.z, x4.w};
        for (int u = 0; u < 4; ++u) {
          const int i = v * 8 + u * 2;
          const float x0 = as_type<float>(xw[u] << 16), x1 = as_type<float>(xw[u] & 0xffff0000u);
          xsum += x0 + x1;
          for (int j = 0; j < QR; ++j) dot[j] = fma(x1, q[j][i + 1], fma(x0, q[j][i], dot[j]));
        }
      }
      for (int j = 0; j < QR; ++j) acc[r][j] = fma(scale[j], dot[j], fma(bias[j], xsum, acc[r][j]));
    }
  }
  for (int r = 0; r < rows; ++r)
    for (int j = 0; j < QR; ++j) {
      const float v = simd_sum(acc[r][j]);
      if (lane == 0) OUT[size_t(r) * N + col0 + j] = bfloat(v);
    }
}

template [[host_name("tf_hy4_qmv2_q4")]] [[kernel]] decltype(tf_hy4_qmv2<4>) tf_hy4_qmv2<4>;
template [[host_name("tf_hy4_qmv2_q8")]] [[kernel]] decltype(tf_hy4_qmv2<8>) tf_hy4_qmv2<8>;

// tf_hy4_qmv2 for decode passes with the row count fixed at compile time (R = 1 to 4) and QR output columns a
// simdgroup: each lane's 32 inputs are read once for QR columns, and the QR weight loads of a step issue together.
// The same sums per (row, column) as tf_hy4_qmv2 (fma over the 32 codes in order, then scale and bias, then the
// simdgroup sum), so the same bits.
template <int BITS, int R, int QR>
[[kernel]] void tf_hy4_qmv3(const device bfloat* X [[buffer(0)]], const device uint32_t* W [[buffer(1)]],
                            const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]],
                            constant int* dims [[buffer(4)]], device bfloat* OUT [[buffer(5)]],
                            uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                            uint3 tg [[threadgroup_position_in_grid]]) {
  const int K = dims[0], N = dims[1];
  constexpr int SG = 4;
  const int col0 = (int(tg.x) * SG + int(sg)) * QR;
  if (col0 >= N) return;
  const int words_row = K * BITS / 32, groups_row = K / 64;
  float acc[R][QR];
  for (int r = 0; r < R; ++r)
    for (int j = 0; j < QR; ++j) acc[r][j] = 0.0f;
  for (int k0 = int(lane) * 32; k0 < K; k0 += 1024) {
    const int g = k0 / 64;
    uint4 wv[QR][BITS / 4];
    float scale[QR], bias[QR];
    for (int j = 0; j < QR; ++j) {
      const int n = col0 + j;
      const device uint4* wp = (const device uint4*)(W + size_t(n) * words_row + k0 * BITS / 32);
      for (int v = 0; v < BITS / 4; ++v) wv[j][v] = wp[v];
      scale[j] = float(S[size_t(n) * groups_row + g]);
      bias[j] = float(B[size_t(n) * groups_row + g]);
    }
    for (int r = 0; r < R; ++r) {
      const device uint4* xp = (const device uint4*)(X + size_t(r) * K + k0);
      float xs[32];
      float xsum = 0.0f;
      for (int v = 0; v < 4; ++v) {
        const uint4 x4 = xp[v];
        const uint xw[4] = {x4.x, x4.y, x4.z, x4.w};
        for (int u = 0; u < 4; ++u) {
          xs[v * 8 + u * 2] = as_type<float>(xw[u] << 16);
          xs[v * 8 + u * 2 + 1] = as_type<float>(xw[u] & 0xffff0000u);
          xsum += xs[v * 8 + u * 2] + xs[v * 8 + u * 2 + 1];
        }
      }
      for (int j = 0; j < QR; ++j) {
        float dot = 0.0f;
        for (int v = 0; v < BITS / 4; ++v) {
          const uint words[4] = {wv[j][v].x, wv[j][v].y, wv[j][v].z, wv[j][v].w};
          for (int u = 0; u < 4; ++u) {
            if (BITS == 4) {
              for (int b = 0; b < 8; ++b) dot = fma(xs[(v * 4 + u) * 8 + b], float((words[u] >> (4 * b)) & 0xfu), dot);
            } else {
              for (int b = 0; b < 4; ++b) dot = fma(xs[(v * 4 + u) * 4 + b], float((words[u] >> (8 * b)) & 0xffu), dot);
            }
          }
        }
        acc[r][j] = fma(scale[j], dot, fma(bias[j], xsum, acc[r][j]));
      }
    }
  }
  for (int r = 0; r < R; ++r)
    for (int j = 0; j < QR; ++j) {
      const float v = simd_sum(acc[r][j]);
      if (lane == 0) OUT[size_t(r) * N + col0 + j] = bfloat(v);
    }
}

template [[host_name("tf_hy4_qmv3_r1_q4")]] [[kernel]] decltype(tf_hy4_qmv3<4, 1, 4>) tf_hy4_qmv3<4, 1, 4>;
template [[host_name("tf_hy4_qmv3_r2_q4")]] [[kernel]] decltype(tf_hy4_qmv3<4, 2, 4>) tf_hy4_qmv3<4, 2, 4>;
template [[host_name("tf_hy4_qmv3_r1_q8")]] [[kernel]] decltype(tf_hy4_qmv3<8, 1, 4>) tf_hy4_qmv3<8, 1, 4>;
template [[host_name("tf_hy4_qmv3_r2_q8")]] [[kernel]] decltype(tf_hy4_qmv3<8, 2, 4>) tf_hy4_qmv3<8, 2, 4>;
template [[host_name("tf_hy4_qmv3_r3_q4")]] [[kernel]] decltype(tf_hy4_qmv3<4, 3, 4>) tf_hy4_qmv3<4, 3, 4>;
template [[host_name("tf_hy4_qmv3_r4_q4")]] [[kernel]] decltype(tf_hy4_qmv3<4, 4, 4>) tf_hy4_qmv3<4, 4, 4>;
template [[host_name("tf_hy4_qmv3_r3_q8")]] [[kernel]] decltype(tf_hy4_qmv3<8, 3, 4>) tf_hy4_qmv3<8, 3, 4>;
template [[host_name("tf_hy4_qmv3_r4_q8")]] [[kernel]] decltype(tf_hy4_qmv3<8, 4, 4>) tf_hy4_qmv3<8, 4, 4>;
template [[host_name("tf_hy4_qmv3x8_r1_q4")]] [[kernel]] decltype(tf_hy4_qmv3<4, 1, 8>) tf_hy4_qmv3<4, 1, 8>;
template [[host_name("tf_hy4_qmv3x8_r2_q4")]] [[kernel]] decltype(tf_hy4_qmv3<4, 2, 8>) tf_hy4_qmv3<4, 2, 8>;
template [[host_name("tf_hy4_qmv3x8_r1_q8")]] [[kernel]] decltype(tf_hy4_qmv3<8, 1, 8>) tf_hy4_qmv3<8, 1, 8>;
template [[host_name("tf_hy4_qmv3x8_r2_q8")]] [[kernel]] decltype(tf_hy4_qmv3<8, 2, 8>) tf_hy4_qmv3<8, 2, 8>;
template [[host_name("tf_hy4_qmv3x8_r3_q4")]] [[kernel]] decltype(tf_hy4_qmv3<4, 3, 8>) tf_hy4_qmv3<4, 3, 8>;
template [[host_name("tf_hy4_qmv3x8_r4_q4")]] [[kernel]] decltype(tf_hy4_qmv3<4, 4, 8>) tf_hy4_qmv3<4, 4, 8>;
template [[host_name("tf_hy4_qmv3x8_r3_q8")]] [[kernel]] decltype(tf_hy4_qmv3<8, 3, 8>) tf_hy4_qmv3<8, 3, 8>;
template [[host_name("tf_hy4_qmv3x8_r4_q8")]] [[kernel]] decltype(tf_hy4_qmv3<8, 4, 8>) tf_hy4_qmv3<8, 4, 8>;

// ---- Routed experts for 1-2 row passes: each (row, pick) pair's gate/up and down as GEMVs over that expert's
// MLX-layout matrices (fp32 sums, each weight read once), the clipped SwiGLU rounded as the tensor gather's epilogue.
struct Hy4ExpertShape { uint pairs, picks, k, n; float limit; };

// QR = 2 output columns a simdgroup: their codes unpacked once a step, the row's 32 inputs a lane streamed past.
template <int BITS>
inline void hy4_qmv_cols2(const device bfloat* x, const device uint32_t* W, const device bfloat* S,
                          const device bfloat* B, int K, int col0, uint lane, thread float (&acc)[2]) {
  const int words_row = K * BITS / 32, groups_row = K / 64;
  acc[0] = 0.0f; acc[1] = 0.0f;
  for (int k0 = int(lane) * 32; k0 < K; k0 += 1024) {
    const int g = k0 / 64;
    const device uint4* xp = (const device uint4*)(x + k0);
    float xs[32];
    float xsum = 0.0f;
    for (int v = 0; v < 4; ++v) {
      const uint4 x4 = xp[v];
      const uint xw[4] = {x4.x, x4.y, x4.z, x4.w};
      for (int u = 0; u < 4; ++u) {
        xs[v * 8 + u * 2] = as_type<float>(xw[u] << 16);
        xs[v * 8 + u * 2 + 1] = as_type<float>(xw[u] & 0xffff0000u);
        xsum += xs[v * 8 + u * 2] + xs[v * 8 + u * 2 + 1];
      }
    }
    for (int j = 0; j < 2; ++j) {
      const int n = col0 + j;
      const device uint4* wp = (const device uint4*)(W + size_t(n) * words_row + k0 * BITS / 32);
      float dot = 0.0f;
      for (int v = 0; v < BITS / 4; ++v) {
        const uint4 w4 = wp[v];
        const uint words[4] = {w4.x, w4.y, w4.z, w4.w};
        for (int u = 0; u < 4; ++u) {
          if (BITS == 4) {
            for (int b = 0; b < 8; ++b) dot = fma(xs[(v * 4 + u) * 8 + b], float((words[u] >> (4 * b)) & 0xfu), dot);
          } else {
            for (int b = 0; b < 4; ++b) dot = fma(xs[(v * 4 + u) * 4 + b], float((words[u] >> (8 * b)) & 0xffu), dot);
          }
        }
      }
      acc[j] = fma(float(S[size_t(n) * groups_row + g]), dot, fma(float(B[size_t(n) * groups_row + g]), xsum, acc[j]));
    }
  }
  acc[0] = simd_sum(acc[0]);
  acc[1] = simd_sum(acc[1]);
}

inline bfloat hy4_expert_swiglu(float g, float u, float limit) {
  const float lim = float(bfloat(limit));
  const bfloat gt = bfloat(metal::min(float(bfloat(g)), lim));
  const bfloat up = bfloat(metal::min(metal::max(float(bfloat(u)), -lim), lim));
  const bfloat sl = bfloat(precise::divide(float(gt), 1.0f + precise::exp(-float(gt))));
  return sl * up;
}

// act[pair][n] = swiglu(gate_e x_row, up_e x_row); x = latent [rows][k]; grid (n / 8, pairs), 4 simdgroups.
template <int BITS>
[[kernel]] void tf_hy4_expert_glu(const device bfloat* latent [[buffer(0)]], const device int* pick [[buffer(1)]],
                                  const device uint32_t* GW [[buffer(2)]], const device bfloat* GS [[buffer(3)]],
                                  const device bfloat* GB [[buffer(4)]], const device uint32_t* UW [[buffer(5)]],
                                  const device bfloat* US [[buffer(6)]], const device bfloat* UB [[buffer(7)]],
                                  device bfloat* act [[buffer(8)]], constant Hy4ExpertShape& a [[buffer(9)]],
                                  uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                  uint3 tg [[threadgroup_position_in_grid]]) {
  const uint p = tg.y;
  if (p >= a.pairs) return;
  const int col0 = (int(tg.x) * 4 + int(sg)) * 2;
  if (col0 >= int(a.n)) return;
  const int K = int(a.k), N = int(a.n);
  const ulong e = ulong(pick[p]);
  const ulong wstride = ulong(N) * (K * BITS / 32), sstride = ulong(N) * (K / 64);
  const device bfloat* x = latent + ulong(p / a.picks) * K;
  float g[2], u[2];
  hy4_qmv_cols2<BITS>(x, GW + e * wstride, GS + e * sstride, GB + e * sstride, K, col0, lane, g);
  hy4_qmv_cols2<BITS>(x, UW + e * wstride, US + e * sstride, UB + e * sstride, K, col0, lane, u);
  if (lane == 0)
    for (int j = 0; j < 2; ++j) act[ulong(p) * N + col0 + j] = hy4_expert_swiglu(g[j], u[j], a.limit);
}

// out[pair][n] (fp32) = down_e act[pair]; grid (n / 8, pairs).
template <int BITS>
[[kernel]] void tf_hy4_expert_down(const device bfloat* act [[buffer(0)]], const device int* pick [[buffer(1)]],
                                   const device uint32_t* DW [[buffer(2)]], const device bfloat* DS [[buffer(3)]],
                                   const device bfloat* DB [[buffer(4)]], device float* out [[buffer(5)]],
                                   constant Hy4ExpertShape& a [[buffer(6)]],
                                   uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                   uint3 tg [[threadgroup_position_in_grid]]) {
  const uint p = tg.y;
  if (p >= a.pairs) return;
  const int col0 = (int(tg.x) * 4 + int(sg)) * 2;
  if (col0 >= int(a.n)) return;
  const int K = int(a.k), N = int(a.n);
  const ulong e = ulong(pick[p]);
  const ulong wstride = ulong(N) * (K * BITS / 32), sstride = ulong(N) * (K / 64);
  float d[2];
  hy4_qmv_cols2<BITS>(act + ulong(p) * K, DW + e * wstride, DS + e * sstride, DB + e * sstride, K, col0, lane, d);
  if (lane == 0)
    for (int j = 0; j < 2; ++j) out[ulong(p) * N + col0 + j] = d[j];
}

// tf_hy4_expert_down with tf_hy4_routed_sum folded in: threadgroup (column block, row) takes the row's picks in
// order and adds weight * down(act) as the routed sum does (fma in slot order), writing the bf16 weighted row.
// The same bits as the two launches, without the fp32 pair outputs.
template <int BITS>
[[kernel]] void tf_hy4_expert_down_sum(const device bfloat* act [[buffer(0)]], const device int* pick [[buffer(1)]],
                                       const device uint32_t* DW [[buffer(2)]], const device bfloat* DS [[buffer(3)]],
                                       const device bfloat* DB [[buffer(4)]], device bfloat* weighted [[buffer(5)]],
                                       constant Hy4ExpertShape& a [[buffer(6)]], const device float* wts [[buffer(7)]],
                                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                       uint3 tg [[threadgroup_position_in_grid]]) {
  const uint r = tg.y;
  if (r * a.picks >= a.pairs) return;
  const int col0 = (int(tg.x) * 4 + int(sg)) * 2;
  if (col0 >= int(a.n)) return;
  const int K = int(a.k), N = int(a.n);
  const ulong wstride = ulong(N) * (K * BITS / 32), sstride = ulong(N) * (K / 64);
  float sum[2] = {0.0f, 0.0f};
  for (uint slot = 0; slot < a.picks; ++slot) {
    const uint p = r * a.picks + slot;
    const ulong e = ulong(pick[p]);
    float d[2];
    hy4_qmv_cols2<BITS>(act + ulong(p) * K, DW + e * wstride, DS + e * sstride, DB + e * sstride, K, col0, lane, d);
    const float w = wts[p];
    sum[0] = fma(w, d[0], sum[0]);
    sum[1] = fma(w, d[1], sum[1]);
  }
  if (lane == 0)
    for (int j = 0; j < 2; ++j) weighted[ulong(r) * N + col0 + j] = bfloat(sum[j]);
}

template [[host_name("tf_hy4_expert_down_sum_q4")]] [[kernel]] decltype(tf_hy4_expert_down_sum<4>) tf_hy4_expert_down_sum<4>;
template [[host_name("tf_hy4_expert_down_sum_q8")]] [[kernel]] decltype(tf_hy4_expert_down_sum<8>) tf_hy4_expert_down_sum<8>;

template [[host_name("tf_hy4_expert_glu_q4")]] [[kernel]] decltype(tf_hy4_expert_glu<4>) tf_hy4_expert_glu<4>;
template [[host_name("tf_hy4_expert_glu_q8")]] [[kernel]] decltype(tf_hy4_expert_glu<8>) tf_hy4_expert_glu<8>;
template [[host_name("tf_hy4_expert_down_q4")]] [[kernel]] decltype(tf_hy4_expert_down<4>) tf_hy4_expert_down<4>;
template [[host_name("tf_hy4_expert_down_q8")]] [[kernel]] decltype(tf_hy4_expert_down<8>) tf_hy4_expert_down<8>;

// ---- Routed experts for 3-16 row passes, grouped by expert: the router window's jobs (ids[j], member pairs at
// members[j * 16 ..], -1 padded, count[0] jobs), each active expert's codes unpacked once for all its member rows.
// A member row's sums run in hy4_qmv_cols2's order, so every pair's result equals the 1-2 row path's bits.
template <int BITS, int M>
inline void hy4_qmv_cols2_group(const device bfloat* x0, thread const int (&xrow)[M], int count, int ldx,
                                const device uint32_t* W, const device bfloat* S, const device bfloat* B, int K,
                                int col0, uint lane, thread float (&acc)[M][2]) {
  const int words_row = K * BITS / 32, groups_row = K / 64;
  for (int m = 0; m < M; ++m) { acc[m][0] = 0.0f; acc[m][1] = 0.0f; }
  for (int k0 = int(lane) * 32; k0 < K; k0 += 1024) {
    const int g = k0 / 64;
    float q[2][32];
    float sc[2], bi[2];
    for (int j = 0; j < 2; ++j) {
      const int n = col0 + j;
      const device uint4* wp = (const device uint4*)(W + size_t(n) * words_row + k0 * BITS / 32);
      for (int v = 0; v < BITS / 4; ++v) {
        const uint4 w4 = wp[v];
        const uint words[4] = {w4.x, w4.y, w4.z, w4.w};
        for (int u = 0; u < 4; ++u) {
          if (BITS == 4) {
            for (int b = 0; b < 8; ++b) q[j][(v * 4 + u) * 8 + b] = float((words[u] >> (4 * b)) & 0xfu);
          } else {
            for (int b = 0; b < 4; ++b) q[j][(v * 4 + u) * 4 + b] = float((words[u] >> (8 * b)) & 0xffu);
          }
        }
      }
      sc[j] = float(S[size_t(n) * groups_row + g]);
      bi[j] = float(B[size_t(n) * groups_row + g]);
    }
    for (int m = 0; m < M; ++m) {
      if (m >= count) break;
      const device uint4* xp = (const device uint4*)(x0 + size_t(xrow[m]) * ldx + k0);
      float xs[32];
      float xsum = 0.0f;
      for (int v = 0; v < 4; ++v) {
        const uint4 x4 = xp[v];
        const uint xw[4] = {x4.x, x4.y, x4.z, x4.w};
        for (int u = 0; u < 4; ++u) {
          xs[v * 8 + u * 2] = as_type<float>(xw[u] << 16);
          xs[v * 8 + u * 2 + 1] = as_type<float>(xw[u] & 0xffff0000u);
          xsum += xs[v * 8 + u * 2] + xs[v * 8 + u * 2 + 1];
        }
      }
      for (int j = 0; j < 2; ++j) {
        float dot = 0.0f;
        for (int i = 0; i < 32; ++i) dot = fma(xs[i], q[j][i], dot);
        acc[m][j] = fma(sc[j], dot, fma(bi[j], xsum, acc[m][j]));
      }
    }
  }
  for (int m = 0; m < M; ++m) {
    if (m >= count) break;
    acc[m][0] = simd_sum(acc[m][0]);
    acc[m][1] = simd_sum(acc[m][1]);
  }
}

// act[pair][n] for every member pair of job tg.y; grid (n / 8, job bound), 4 simdgroups.
template <int BITS, int M>
[[kernel]] void tf_hy4_expert_glu_groups(const device bfloat* latent [[buffer(0)]], const device int* ids [[buffer(1)]],
                                         const device int* members [[buffer(2)]], const device int* count [[buffer(3)]],
                                         const device uint32_t* GW [[buffer(4)]], const device bfloat* GS [[buffer(5)]],
                                         const device bfloat* GB [[buffer(6)]], const device uint32_t* UW [[buffer(7)]],
                                         const device bfloat* US [[buffer(8)]], const device bfloat* UB [[buffer(9)]],
                                         device bfloat* act [[buffer(10)]], constant Hy4ExpertShape& a [[buffer(11)]],
                                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                         uint3 tg [[threadgroup_position_in_grid]]) {
  const int job = int(tg.y);
  if (job >= count[0]) return;
  const int col0 = (int(tg.x) * 4 + int(sg)) * 2;
  if (col0 >= int(a.n)) return;
  const int K = int(a.k), N = int(a.n);
  const ulong e = ulong(ids[job]);
  const ulong wstride = ulong(N) * (K * BITS / 32), sstride = ulong(N) * (K / 64);
  const device int* mem = members + job * 16;
  for (int c0 = 0; c0 < 16; c0 += M) {
    int pair[M], xrow[M];
    int n = 0;
    for (int m = 0; m < M; ++m) {
      const int p = mem[c0 + m];
      pair[m] = p;
      xrow[m] = p >= 0 ? p / int(a.picks) : 0;
      n += p >= 0 ? 1 : 0;
    }
    if (n == 0) break;
    float g[M][2], u[M][2];
    hy4_qmv_cols2_group<BITS, M>(latent, xrow, n, K, GW + e * wstride, GS + e * sstride, GB + e * sstride, K, col0, lane, g);
    hy4_qmv_cols2_group<BITS, M>(latent, xrow, n, K, UW + e * wstride, US + e * sstride, UB + e * sstride, K, col0, lane, u);
    if (lane == 0)
      for (int m = 0; m < M; ++m) {
        if (m >= n) break;
        for (int j = 0; j < 2; ++j) act[ulong(pair[m]) * N + col0 + j] = hy4_expert_swiglu(g[m][j], u[m][j], a.limit);
      }
    if (n < M) break;
  }
}

// out[pair][n] (fp32) = down_e act[pair] for every member pair of job tg.y; grid (n / 8, job bound).
template <int BITS, int M>
[[kernel]] void tf_hy4_expert_down_groups(const device bfloat* act [[buffer(0)]], const device int* ids [[buffer(1)]],
                                          const device int* members [[buffer(2)]], const device int* count [[buffer(3)]],
                                          const device uint32_t* DW [[buffer(4)]], const device bfloat* DS [[buffer(5)]],
                                          const device bfloat* DB [[buffer(6)]], device float* out [[buffer(7)]],
                                          constant Hy4ExpertShape& a [[buffer(8)]],
                                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                          uint3 tg [[threadgroup_position_in_grid]]) {
  const int job = int(tg.y);
  if (job >= count[0]) return;
  const int col0 = (int(tg.x) * 4 + int(sg)) * 2;
  if (col0 >= int(a.n)) return;
  const int K = int(a.k), N = int(a.n);
  const ulong e = ulong(ids[job]);
  const ulong wstride = ulong(N) * (K * BITS / 32), sstride = ulong(N) * (K / 64);
  const device int* mem = members + job * 16;
  for (int c0 = 0; c0 < 16; c0 += M) {
    int pair[M];
    int n = 0;
    for (int m = 0; m < M; ++m) {
      const int p = mem[c0 + m];
      pair[m] = p >= 0 ? p : 0;
      n += p >= 0 ? 1 : 0;
    }
    if (n == 0) break;
    float d[M][2];
    hy4_qmv_cols2_group<BITS, M>(act, pair, n, K, DW + e * wstride, DS + e * sstride, DB + e * sstride, K, col0, lane, d);
    if (lane == 0)
      for (int m = 0; m < M; ++m) {
        if (m >= n) break;
        for (int j = 0; j < 2; ++j) out[ulong(pair[m]) * N + col0 + j] = d[m][j];
      }
    if (n < M) break;
  }
}

template [[host_name("tf_hy4_expert_glu_groups_q4")]] [[kernel]] decltype(tf_hy4_expert_glu_groups<4, 4>) tf_hy4_expert_glu_groups<4, 4>;
template [[host_name("tf_hy4_expert_glu_groups_q8")]] [[kernel]] decltype(tf_hy4_expert_glu_groups<8, 4>) tf_hy4_expert_glu_groups<8, 4>;
template [[host_name("tf_hy4_expert_down_groups_q4")]] [[kernel]] decltype(tf_hy4_expert_down_groups<4, 4>) tf_hy4_expert_down_groups<4, 4>;
template [[host_name("tf_hy4_expert_down_groups_q8")]] [[kernel]] decltype(tf_hy4_expert_down_groups<8, 4>) tf_hy4_expert_down_groups<8, 4>;

// ---- MLA absorb and expand-and-gate for 1-4 row passes over kv_b's affine codes: per head 448 rows (192 key,
// then 256 value), each 512 latent values in groups of 64; w = fma(scale, code, bias) in fp32 as read. ----
struct Hy4MlaQArgs { uint rows, heads; };

// absorbed[row][head][c] (c < 512, stride 576) = sum_d query[row][head][d] * key[head][d][c], d < 192.
// Grid (2 halves of 256 columns, heads), 8 simdgroups splitting d (rows d = sg + 8 i, four loads in flight a step);
// a lane holds 8 adjacent columns; the 8 partials meet in two threadgroup steps in a fixed order.
template <int BITS>
[[kernel]] void tf_hy4_absorb_qmv(const device bfloat* query [[buffer(0)]], const device uint32_t* W [[buffer(1)]],
                                  const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]],
                                  device bfloat* absorbed [[buffer(4)]], constant Hy4MlaQArgs& a [[buffer(5)]],
                                  uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                  uint t [[thread_index_in_threadgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr int R = 4, SG = 8, D = 192, U = 4, WPR = 512 * BITS / 32;
  const int h = int(tg.y), heads = int(a.heads), rows = min(int(a.rows), R);
  const int c0 = int(tg.x) * 256 + int(lane) * 8, g = c0 / 64;
  threadgroup float qs[R][D];
  threadgroup float part[SG / 2][R][256];
  for (int i = int(t); i < rows * D; i += SG * 32) {
    const int r = i / D, d = i % D;
    qs[r][d] = float(query[(ulong(r) * heads + h) * 256 + d]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float acc[R][8];
  for (int r = 0; r < R; ++r)
    for (int k = 0; k < 8; ++k) acc[r][k] = 0.0f;
  const ulong base = ulong(h) * 448;
  for (int d0 = int(sg); d0 < D; d0 += SG * U) {
    uint2 words[U];
    float s[U], b[U];
    for (int u = 0; u < U; ++u) {
      const ulong row = base + d0 + SG * u;
      if (BITS == 4) words[u] = uint2(W[row * WPR + c0 / 8], 0u);
      else words[u] = *(const device uint2*)(W + row * WPR + c0 / 4);
      s[u] = float(S[row * 8 + g]);
      b[u] = float(B[row * 8 + g]);
    }
    for (int u = 0; u < U; ++u) {
      float w[8];
      if (BITS == 4) {
        for (int k = 0; k < 8; ++k) w[k] = fma(s[u], float((words[u].x >> (4 * k)) & 0xfu), b[u]);
      } else {
        for (int k = 0; k < 4; ++k) {
          w[k] = fma(s[u], float((words[u].x >> (8 * k)) & 0xffu), b[u]);
          w[4 + k] = fma(s[u], float((words[u].y >> (8 * k)) & 0xffu), b[u]);
        }
      }
      const int d = d0 + SG * u;
      for (int r = 0; r < R; ++r) {
        if (r >= rows) break;
        const float q = qs[r][d];
        for (int k = 0; k < 8; ++k) acc[r][k] = fma(q, w[k], acc[r][k]);
      }
    }
  }
  // simdgroups 4-7 hand theirs to 0-3, which then hold the pair sums
  if (sg >= SG / 2)
    for (int r = 0; r < R; ++r) {
      if (r >= rows) break;
      for (int k = 0; k < 8; ++k) part[sg - SG / 2][r][lane * 8 + k] = acc[r][k];
    }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg < SG / 2)
    for (int r = 0; r < R; ++r) {
      if (r >= rows) break;
      for (int k = 0; k < 8; ++k) acc[r][k] += part[sg][r][lane * 8 + k];
    }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg < SG / 2)
    for (int r = 0; r < R; ++r) {
      if (r >= rows) break;
      for (int k = 0; k < 8; ++k) part[sg][r][lane * 8 + k] = acc[r][k];
    }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = int(t); i < rows * 256; i += SG * 32) {
    const int r = i / 256, c = i % 256;
    const float v = ((part[0][r][c] + part[1][r][c]) + part[2][r][c]) + part[3][r][c];
    absorbed[(ulong(r) * heads + h) * 576 + tg.x * 256 + c] = bfloat(v);
  }
}

// values[row][head][e] = bf16(bf16(sum_c latent[row][head][c] * value[head][e][c]) * bf16(sigmoid(logit))), e < 256.
// Grid (256 / 8, heads), 4 simdgroups of two outputs; a lane holds 16 latent columns (one scale group's quarter).
template <int BITS>
[[kernel]] void tf_hy4_expand_gated_qmv(const device bfloat* latent [[buffer(0)]], const device uint32_t* W [[buffer(1)]],
                                        const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]],
                                        device bfloat* values [[buffer(4)]], constant Hy4MlaQArgs& a [[buffer(5)]],
                                        const device bfloat* logits [[buffer(6)]],
                                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                        uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr int R = 4, WPR = 512 * BITS / 32;
  const int h = int(tg.y), heads = int(a.heads), rows = min(int(a.rows), R);
  const int e0 = (int(tg.x) * 4 + int(sg)) * 2;
  const int c0 = int(lane) * 16, g = c0 / 64;
  float x[R][16], xs[R];
  for (int r = 0; r < R; ++r) {
    xs[r] = 0.0f;
    if (r >= rows) continue;
    const device uint4* xp = (const device uint4*)(latent + (ulong(r) * heads + h) * 512 + c0);
    for (int v = 0; v < 2; ++v) {
      const uint4 x4 = xp[v];
      const uint xw[4] = {x4.x, x4.y, x4.z, x4.w};
      for (int u = 0; u < 4; ++u) {
        x[r][v * 8 + u * 2] = as_type<float>(xw[u] << 16);
        x[r][v * 8 + u * 2 + 1] = as_type<float>(xw[u] & 0xffff0000u);
        xs[r] += x[r][v * 8 + u * 2] + x[r][v * 8 + u * 2 + 1];
      }
    }
  }
  for (int j = 0; j < 2; ++j) {
    const int e = e0 + j;
    const ulong row = ulong(h) * 448 + 192 + e;
    const float s = float(S[row * 8 + g]), b = float(B[row * 8 + g]);
    float q[16];
    if (BITS == 4) {
      const uint2 ww = *(const device uint2*)(W + row * WPR + c0 / 8);
      for (int k = 0; k < 8; ++k) {
        q[k] = float((ww.x >> (4 * k)) & 0xfu);
        q[8 + k] = float((ww.y >> (4 * k)) & 0xfu);
      }
    } else {
      const uint4 ww = *(const device uint4*)(W + row * WPR + c0 / 4);
      const uint words[4] = {ww.x, ww.y, ww.z, ww.w};
      for (int u = 0; u < 4; ++u)
        for (int k = 0; k < 4; ++k) q[u * 4 + k] = float((words[u] >> (8 * k)) & 0xffu);
    }
    for (int r = 0; r < R; ++r) {
      if (r >= rows) break;
      float dot = 0.0f;
      for (int i = 0; i < 16; ++i) dot = fma(x[r][i], q[i], dot);
      const float v = simd_sum(fma(s, dot, b * xs[r]));
      if (lane == 0) {
        const ulong at = (ulong(r) * heads + h) * 256 + e;
        const bfloat value = bfloat(v);
        const bfloat gate = bfloat(1.0f / (1.0f + precise::exp(-float(logits[at]))));
        values[at] = bfloat(float(value) * float(gate));
      }
    }
  }
}

template [[host_name("tf_hy4_absorb_qmv_q4")]] [[kernel]] decltype(tf_hy4_absorb_qmv<4>) tf_hy4_absorb_qmv<4>;
template [[host_name("tf_hy4_absorb_qmv_q8")]] [[kernel]] decltype(tf_hy4_absorb_qmv<8>) tf_hy4_absorb_qmv<8>;
template [[host_name("tf_hy4_expand_gated_qmv_q4")]] [[kernel]] decltype(tf_hy4_expand_gated_qmv<4>) tf_hy4_expand_gated_qmv<4>;
template [[host_name("tf_hy4_expand_gated_qmv_q8")]] [[kernel]] decltype(tf_hy4_expand_gated_qmv<8>) tf_hy4_expand_gated_qmv<8>;
