// H3's products, 64 x 128 a threadgroup and 32 x 64 a warp; each output's K chain is MLX's, 32-deep steps in order.
// The int8 form (fc1) takes A quantized a row and B a row per 256 values of K, both by h3_quantize_rows: each
// 256-deep block's int32 sums fold into fp32 at B's scales for that block, A's row scale last.
namespace tfp {

template <bool BOUND, typename T, typename C>
inline void h3_step(thread frag<C> (&acc)[2][4], const device T* A, const device T* B, int lda, int ldb, int kk, int kn,
                    int rows, int cols, short2 home) {
  frag<T> a[2][2], b[2][4];
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

// D [P[0], P[1]] = A [P[0], P[2]] * B^T [P[1], P[2]], N a multiple of 128; AM false bounds the rows. Int8 operands
// come with SA [P[0]], A's row scales, and SB [P[1], P[9]], B's per 256-deep block, K a multiple of 256.
template <bool AM, typename T, typename C>
inline void h3_gemm(const device T* A, const device T* B, device bfloat* D, const device int* P, const device float* SA,
                    const device float* SB, uint3 tg, uint sg, uint lane) {
  constexpr bool I8 = is_same_v<T, int8_t>;
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
  frag<C> acc[2][4];
  frag<float> total[2][4];
  TF_UNROLL
  for (short m = 0; m < 2; m++) TF_UNROLL for (short n = 0; n < 4; n++) acc[m][n] = frag<C>(0), total[m][n] = frag<float>(0);
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
    if (I8 && live) {
      // lane columns col + 16 n + home.x + (e & 3), each with its scale for this block
      TF_UNROLL
      for (short n = 0; n < 4; n++) {
        const device float* s = SB + long(col + 16 * n + home.x) * kblocks + kb;
        TF_UNROLL
        for (short m = 0; m < 2; m++) {
          TF_UNROLL
          for (short e = 0; e < 8; e++) total[m][n][e] += float(acc[m][n][e]) * s[(e & 3) * kblocks];
          acc[m][n] = frag<C>(0);
        }
      }
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
    // the lane's rows row + 16 m + home.y and 8 below; a ragged tile's dead rows read the last row's scale
    const int r = row + 16 * m + home.y, last = P[0] - 1;
    const float r0 = I8 ? SA[min(r, last)] : 1, r1 = I8 ? SA[min(r + 8, last)] : 1;
    TF_UNROLL
    for (short n = 0; n < 4; n++) {
      frag<float> f = frag<float>(acc[m][n]);
      if (I8) TF_UNROLL for (short e = 0; e < 8; e++) f[e] = total[m][n][e] * (e < 4 ? r0 : r1);
      if (full) frag_put(f, D, ldd, 16 * m, 16 * n, home);
      else frag_put_in(f, D, ldd, 16 * m, 16 * n, home, rows, cols);
    }
  }
}

}  // namespace tfp

// Each of P[0] rows of X (P[1] values) to int8 in groups of P[2], each at its absolute max over 127: Q = rint(X / S),
// S [row, group] its scale. One simdgroup a row, its groups in turn.
[[kernel]] void h3_quantize_rows(
  const device bfloat16_t* X [[buffer(0)]],
  const device int32_t* P [[buffer(1)]],
  device int8_t* Q [[buffer(2)]],
  device float* S [[buffer(3)]],
  uint at [[threadgroup_position_in_grid]],
  uint lane [[thread_index_in_simdgroup]]) {
  const int k = P[1], g = P[2];
  X += long(at) * k;
  Q += long(at) * k;
  S += long(at) * (k / g);
  for (int from = 0; from < k; from += g) {
    float most = 0;
    for (int i = from + int(lane); i < from + g; i += 32) most = max(most, abs(float(X[i])));
    most = simd_max(most);
    const float scale = most > 0 ? most / 127 : 1;
    for (int i = from + int(lane); i < from + g; i += 32) Q[i] = int8_t(rint(float(X[i]) / scale));
    if (lane == 0) S[from / g] = scale;
  }
}

[[max_total_threads_per_threadgroup(128)]]
[[kernel]] void h3_gemm_32x64(
  const device bfloat16_t* A [[buffer(0)]],
  const device bfloat16_t* B [[buffer(1)]],
  const device int32_t* P [[buffer(2)]],
  device bfloat16_t* D [[buffer(3)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  tfp::h3_gemm<true, bfloat, float>(A, B, D, P, nullptr, nullptr, threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
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
  tfp::h3_gemm<false, bfloat, float>(A, B, D, P, nullptr, nullptr, threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
}

[[max_total_threads_per_threadgroup(128)]]
[[kernel]] void h3_gemm_32x64_i8(
  const device int8_t* A [[buffer(0)]],
  const device int8_t* B [[buffer(1)]],
  const device int32_t* P [[buffer(2)]],
  device bfloat16_t* D [[buffer(3)]],
  const device float* SA [[buffer(4)]],
  const device float* SB [[buffer(5)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  tfp::h3_gemm<false, int8_t, int>(A, B, D, P, SA, SB, threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
}
