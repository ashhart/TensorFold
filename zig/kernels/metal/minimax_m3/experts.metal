// A decode window's routed experts, expert by expert: weights read once a window, tf_row_projection_indexed's bits.

// One threadgroup an (expert, RPS * SG columns): its members 4, 2 then 1 at a time through tf_rp_rows_at.
template <int RPS, int SG, int BITS>
[[max_total_threads_per_threadgroup(64)]]
[[kernel]] void mm_experts_grouped(const device bfloat* X [[buffer(0)]], const device uint32_t* W [[buffer(1)]],
    const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]], constant int* dims [[buffer(4)]],
    device bfloat* OUT [[buffer(5)]], const device uint* G [[buffer(6)]], const device uint* M [[buffer(7)]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const int K = dims[0], N = dims[1], GS = dims[2], slots = dims[3], repeat = dims[5];
  const int row0 = (int(tg.x) * SG + int(sg)) * RPS;
  const uint g = tg.y, groups = G[0];
  if (row0 >= N || g >= groups) return;
  const uint expert = G[1 + 2 * g], first = G[2 + 2 * g], end = g + 1 < groups ? G[2 + 2 * (g + 1)] : uint(slots);
  const size_t at = size_t(expert) * N + size_t(row0);
  const device uint8_t* wr = (const device uint8_t*)W + at * (K * BITS / 8);
  const device bfloat* sr = S + at * (K / GS);
  const device bfloat* br = B + at * (K / GS);
  uint m = first;
  #pragma clang loop unroll(disable)
  for (; m + 4 <= end; m += 4) {
    TfRpRows<4> x;
    for (int b = 0; b < 4; b++) x.x[b] = X + size_t(M[m + b] / uint(repeat)) * K;
    float acc[4][RPS];
    tf_rp_rows_at<RPS, 4, BITS, false>(wr, sr, br, x, K, GS, lane, acc);
    if (lane == 0)
      for (int b = 0; b < 4; b++)
        for (int j = 0; j < RPS; j++) OUT[size_t(M[m + b]) * N + row0 + j] = bfloat(acc[b][j]);
  }
  #pragma clang loop unroll(disable)
  for (; m + 2 <= end; m += 2) {
    TfRpRows<2> x;
    for (int b = 0; b < 2; b++) x.x[b] = X + size_t(M[m + b] / uint(repeat)) * K;
    float acc[2][RPS];
    tf_rp_rows_at<RPS, 2, BITS, false>(wr, sr, br, x, K, GS, lane, acc);
    if (lane == 0)
      for (int b = 0; b < 2; b++)
        for (int j = 0; j < RPS; j++) OUT[size_t(M[m + b]) * N + row0 + j] = bfloat(acc[b][j]);
  }
  for (; m < end; m++) {
    TfRpRows<1> x;
    x.x[0] = X + size_t(M[m] / uint(repeat)) * K;
    float acc[1][RPS];
    tf_rp_rows_at<RPS, 1, BITS, false>(wr, sr, br, x, K, GS, lane, acc);
    if (lane == 0)
      for (int j = 0; j < RPS; j++) OUT[size_t(M[m]) * N + row0 + j] = bfloat(acc[0][j]);
  }
}

template [[host_name("mm_experts_grouped_q3")]] [[kernel]] decltype(mm_experts_grouped<4, 2, 3>) mm_experts_grouped<4, 2, 3>;
template [[host_name("mm_experts_grouped_q4")]] [[kernel]] decltype(mm_experts_grouped<4, 2, 4>) mm_experts_grouped<4, 2, 4>;
