// H3's products at 64 rows x 128 columns a threadgroup, each warp 32 x 64: every loaded fragment feeds two MMAs
// instead of one. Each output's K chain is MLX's, 32-deep steps in order.
namespace tfp {

template <bool BOUND>
inline void h3_step(thread frag<float> (&acc)[2][4], const device bfloat* A, const device bfloat* B, int lda, int ldb,
                    int kk, int kn, int rows, int cols, short2 home) {
  frag<bfloat> a[2][2], b[2][4];
  TF_UNROLL
  for (short m = 0; m < 2; m++) {
    TF_UNROLL
    for (short k = 0; k < 2; k++) {
      if (BOUND) frag_get_in(a[m][k], A, lda, 16 * m, kk + 16 * k, home, rows, kk + kn);
      else frag_get(a[m][k], A, lda, 16 * m, kk + 16 * k, home);
    }
  }
  TF_UNROLL
  for (short k = 0; k < 2; k++) {
    TF_UNROLL
    for (short n = 0; n < 4; n++) {
      if (BOUND) frag_get_in(b[k][n], B, ldb, 16 * n, kk + 16 * k, home, cols, kk + kn);
      else frag_get(b[k][n], B, ldb, 16 * n, kk + 16 * k, home);
    }
  }
  TF_UNROLL
  for (short m = 0; m < 2; m++) {
    TF_UNROLL
    for (short k = 0; k < 2; k++) {
      mma_16x32<false, true>(acc[m][0], acc[m][1], a[m][k], b[k][0], b[k][1]);
      mma_16x32<false, true>(acc[m][2], acc[m][3], a[m][k], b[k][2], b[k][3]);
    }
  }
}

// D [P[0], P[1]] = A [P[0], P[2]] * B^T [P[1], P[2]], N a multiple of 128; AM false bounds the rows.
template <bool AM>
inline void h3_gemm(const device bfloat* A, const device bfloat* B, device bfloat* D, const device int* P, uint3 tg,
                    uint sg, uint lane) {
  const int swz = P[8];
  const int ty = (int(tg.y) << swz) + (int(tg.x) & ((1 << swz) - 1)), tx = int(tg.x) >> swz;
  if (tx >= P[6] || ty >= P[7]) return;
  const int lda = P[3], ldb = P[4], ldd = P[5];
  const int row = ty * 64 + 32 * int(sg / 2), col = tx * 128 + 64 * int(sg % 2);
  A += long(P[13]) + long(P[10]) * tg.z + long(row) * lda;
  B += long(P[14]) + long(P[11]) * tg.z + long(col) * ldb;
  D += long(P[12]) * tg.z + long(row) * ldd + col;
  const int rows = AM ? 32 : min(32, P[0] - row), cols = 64;
  const bool full = AM || rows == 32;
  const bool live = rows > 0;
  const short2 home = frag_home(ushort(lane));
  frag<float> acc[2][4];
  TF_UNROLL
  for (short m = 0; m < 2; m++) TF_UNROLL for (short n = 0; n < 4; n++) acc[m][n] = frag<float>(0);
  const int kblocks = P[9];
#pragma clang loop unroll(disable)
  for (int kb = 0; kb < kblocks; kb++) {
    threadgroup_barrier(mem_flags::mem_none);
    if (full) {
#pragma clang loop unroll(disable)
      for (int kk = 0; kk < 256; kk += 32) h3_step<false>(acc, A, B, lda, ldb, kk, 32, rows, cols, home);
    } else if (live) {
#pragma clang loop unroll(disable)
      for (int kk = 0; kk < 256; kk += 32) h3_step<true>(acc, A, B, lda, ldb, kk, 32, rows, cols, home);
    }
    A += 256;
    B += 256;
  }
  const int rest = P[2] - kblocks * 256;
  if (rest > 0) {
    simdgroup_barrier(mem_flags::mem_none);
    if (live) {
#pragma clang loop unroll(disable)
      for (int kk = 0; kk < rest; kk += 32) h3_step<true>(acc, A, B, lda, ldb, kk, max(0, rest - kk), rows, cols, home);
    }
  }
  if (!live) return;
  TF_UNROLL
  for (short m = 0; m < 2; m++) {
    TF_UNROLL
    for (short n = 0; n < 4; n++) {
      if (full) frag_put(acc[m][n], D, ldd, 16 * m, 16 * n, home);
      else frag_put_in(acc[m][n], D, ldd, 16 * m, 16 * n, home, rows, cols);
    }
  }
}

}  // namespace tfp

[[max_total_threads_per_threadgroup(128)]]
[[kernel]] void h3_gemm_32x64(
  const device bfloat16_t* A [[buffer(0)]],
  const device bfloat16_t* B [[buffer(1)]],
  const device int32_t* P [[buffer(2)]],
  device bfloat16_t* D [[buffer(3)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  tfp::h3_gemm<true>(A, B, D, P, threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
}

[[max_total_threads_per_threadgroup(128)]]
[[kernel]] void h3_gemm_32x64_ragged(
  const device bfloat16_t* A [[buffer(0)]],
  const device bfloat16_t* B [[buffer(1)]],
  const device int32_t* P [[buffer(2)]],
  device bfloat16_t* D [[buffer(3)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  tfp::h3_gemm<false>(A, B, D, P, threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
}
