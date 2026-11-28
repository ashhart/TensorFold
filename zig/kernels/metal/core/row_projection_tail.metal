// K, N, GS, slots, experts, repeat; W, S, B pack experts; slot y: expert ids[y], x row y/repeat (past the count: zero).
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
