// Independent row pairs reuse the shared row projection's arithmetic, with row pairs first in the grid.
template <int BITS>
[[kernel, max_total_threads_per_threadgroup(64)]] void lfm3_row_pairs(
    const device bfloat* X [[buffer(0)]], const device uint32_t* W [[buffer(1)]],
    const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]],
    constant int* dims [[buffer(4)]], device bfloat* OUT [[buffer(5)]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint2 tg [[threadgroup_position_in_grid]]) {
  const int K = dims[0], N = dims[1], GS = dims[2], rows = dims[3];
  const int first = int(tg.x) * 2, column = (int(tg.y) * 2 + int(sg)) * 4;
  if (first >= rows || column >= N) return;
#ifdef L3_PROJECTION_BAD_TAIL
  if (first + 1 >= rows) return;
#endif
#ifdef L3_PROJECTION_BAD_ROWS
  const int input = 0;
#else
  const int input = first;
#endif
  const device uint8_t* wr = (const device uint8_t*)W + size_t(column) * (K * BITS / 8);
  const device bfloat* sr = S + size_t(column) * (K / GS);
  const device bfloat* br = B + size_t(column) * (K / GS);
  if (first + 1 < rows) {
    float acc[2][4];
    tf_rp_rows<4, 2, BITS, false>(wr, sr, br, X + size_t(input) * K, K, GS, lane, acc);
    if (lane == 0) for (int row = 0; row < 2; ++row) for (int j = 0; j < 4; ++j)
      OUT[size_t(first + row) * N + column + j] = tf_rp_out<false>(acc[row][j]);
  } else {
    float acc[1][4];
    tf_rp_rows<4, 1, BITS, false>(wr, sr, br, X + size_t(input) * K, K, GS, lane, acc);
    if (lane == 0) for (int j = 0; j < 4; ++j) OUT[size_t(first) * N + column + j] = tf_rp_out<false>(acc[0][j]);
  }
}
template [[host_name("lfm3_row_pairs_q4")]] [[kernel]] decltype(lfm3_row_pairs<4>) lfm3_row_pairs<4>;
template [[host_name("lfm3_row_pairs_q6")]] [[kernel]] decltype(lfm3_row_pairs<6>) lfm3_row_pairs<6>;
template [[host_name("lfm3_row_pairs_q8")]] [[kernel]] decltype(lfm3_row_pairs<8>) lfm3_row_pairs<8>;
