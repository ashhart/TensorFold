
// Hy4-flash lane windows (compiled after the chip's core row projection source, whose helpers it calls).
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
