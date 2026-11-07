// Two independent projections share a dispatch, retaining each shared row helper's arithmetic and BF16 stores.
template <int BITS>
[[kernel, max_total_threads_per_threadgroup(64)]] void lfm3_projection_dual(
    const device bfloat* X [[buffer(0)]], const device uint32_t* W0 [[buffer(1)]],
    const device bfloat* S0 [[buffer(2)]], const device bfloat* B0 [[buffer(3)]],
    constant int* dims [[buffer(4)]], device bfloat* OUT0 [[buffer(5)]],
    const device uint32_t* W1 [[buffer(6)]], const device bfloat* S1 [[buffer(7)]],
    const device bfloat* B1 [[buffer(8)]], device bfloat* OUT1 [[buffer(9)]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const int K = dims[0], N = dims[1], GS = dims[2], rows = dims[3];
  const int first = int(tg.x) * 2, column = (int(tg.y) * 2 + int(sg)) * 4;
  if (first >= rows || column >= N) return;
#ifdef L3_DUAL_BAD_TAIL
  if (tg.z == 1 && first + 1 >= rows) return;
#endif
#ifdef L3_DUAL_BAD_SECOND
  const bool second = false;
#else
  const bool second = tg.z != 0;
#endif
  const device uint32_t* W = second ? W1 : W0;
  const device bfloat* S = second ? S1 : S0;
  const device bfloat* B = second ? B1 : B0;
  device bfloat* OUT = tg.z != 0 ? OUT1 : OUT0;
  const device uint8_t* wr = (const device uint8_t*)W + size_t(column) * (K * BITS / 8);
  const device bfloat* sr = S + size_t(column) * (K / GS);
  const device bfloat* br = B + size_t(column) * (K / GS);
  if (first + 1 < rows) {
    float acc[2][4];
    tf_rp_rows<4, 2, BITS, false>(wr, sr, br, X + size_t(first) * K, K, GS, lane, acc);
    if (lane == 0) for (int row = 0; row < 2; ++row) for (int j = 0; j < 4; ++j)
      OUT[size_t(first + row) * N + column + j] = tf_rp_out<false>(acc[row][j]);
  } else {
    float acc[1][4];
    tf_rp_rows<4, 1, BITS, false>(wr, sr, br, X + size_t(first) * K, K, GS, lane, acc);
    if (lane == 0) for (int j = 0; j < 4; ++j) OUT[size_t(first) * N + column + j] = tf_rp_out<false>(acc[0][j]);
  }
}
template [[host_name("lfm3_projection_dual_q4")]] [[kernel]] decltype(lfm3_projection_dual<4>) lfm3_projection_dual<4>;
template [[host_name("lfm3_projection_dual_q6")]] [[kernel]] decltype(lfm3_projection_dual<6>) lfm3_projection_dual<6>;
template [[host_name("lfm3_projection_dual_q8")]] [[kernel]] decltype(lfm3_projection_dual<8>) lfm3_projection_dual<8>;
