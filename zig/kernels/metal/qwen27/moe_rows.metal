// Experts' decode rows on core/row_projection.metal's loop, the shared expert one more matrix; 64 threads, reserved.

// dims: K, N, GS, slots, experts, repeat; slot y takes matrix ids[y] on x row y / repeat, zeros for ids past `experts`.
template <int RPS, int SG, int BITS>
[[kernel, max_total_threads_per_threadgroup(64)]] void q27_rows_extra(const device bfloat* X [[buffer(0)]], const device uint32_t* W [[buffer(1)]],
                               const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]],
                               constant int* dims [[buffer(4)]], device bfloat* OUT [[buffer(5)]],
                               const device uint* ids [[buffer(6)]], const device uint32_t* W2 [[buffer(7)]],
                               const device bfloat* S2 [[buffer(8)]], const device bfloat* B2 [[buffer(9)]],
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
  if (expert > uint(experts)) {
    if (lane < uint(RPS)) out[lane] = bfloat(0.0f);
    return;
  }
  const bool extra = expert == uint(experts);
  const size_t at = (extra ? 0 : size_t(expert) * N) + size_t(row0);
  const device uint8_t* wr = (const device uint8_t*)(extra ? W2 : W) + at * (K * BITS / 8);
  const device bfloat* sr = (extra ? S2 : S) + at * (K / GS);
  const device bfloat* br = (extra ? B2 : B) + at * (K / GS);
  float acc[1][RPS];
  tf_rp_rows<RPS, 1, BITS, false>(wr, sr, br, X + size_t(slot / repeat) * K, K, GS, lane, acc);
  if (lane == 0)
    for (int j = 0; j < RPS; j++) out[j] = tf_rp_out<false>(acc[0][j]);
}

#define Q27_ROWS_EXTRA(B) template [[host_name("q27_rows_extra_q" #B)]] [[kernel]] decltype(q27_rows_extra<4, 2, B>) q27_rows_extra<4, 2, B>;
// the routed widths (widths.zig routed_bits), the ones the kernel contract checks
Q27_ROWS_EXTRA(4)
Q27_ROWS_EXTRA(8)
