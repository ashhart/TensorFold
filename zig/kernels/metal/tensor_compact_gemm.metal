namespace tfp {
template <bool FULL>
inline void tensor_gemm_step16(thread frag<float>& lo, thread frag<float>& hi, const device float* A,
    const device float* B, int lda, int ldb, int kk, int kn, int rows, int cols, short2 home) {
  TF_UNROLL
  for (short k = 0; k < 2; k++) {
    frag<float> a, b0, b1;
    if (FULL) {
      frag_get(a, A, lda, 0, kk + 16 * k, home);
      frag_get(b0, B, ldb, 0, kk + 16 * k, home);
      frag_get(b1, B, ldb, 16, kk + 16 * k, home);
    } else {
      frag_get_in(a, A, lda, 0, kk + 16 * k, home, rows, kk + kn);
      frag_get_in(b0, B, ldb, 0, kk + 16 * k, home, cols, kk + kn);
      frag_get_in(b1, B, ldb, 16, kk + 16 * k, home, cols, kk + kn);
    }
#ifdef TF_NAX_EXACT_B
    mma_16x32<false, true, float, float, float, true>(lo, hi, a, b0, b1);
#else
    mma_16x32<false, true>(lo, hi, a, b0, b1);
#endif
  }
}

template <bool AM, bool AN>
inline void tensor_gemm16(const device float* A, const device float* B, device float* D,
    constant int* P, uint3 tg, uint sg, uint lane) {
  const int ty = (int(tg.y) << P[8]) + (int(tg.x) & ((1 << P[8]) - 1));
  const int tx = int(tg.x) >> P[8];
  if (tx >= P[6] || ty >= P[7]) return;
  const int row = ty * 32 + 16 * int(sg / 4), col = tx * 128 + 32 * int(sg % 4);
  const int rows = min(16, P[0] - row), cols = min(32, P[1] - col), depth = P[2];
  A += long(P[13]) + long(P[10]) * tg.z + long(row) * P[3];
  B += long(P[14]) + long(P[11]) * tg.z + long(col) * P[4];
  D += long(P[12]) * tg.z + long(row) * P[5] + col;
  const short2 home = frag_home(ushort(lane));
  frag<float> lo(0), hi(0);
  const bool full = (AM || rows == 16) && (AN || cols == 32);
#pragma clang loop unroll(disable)
  for (int kb = 0; kb < P[9]; kb++) {
    threadgroup_barrier(mem_flags::mem_none);
    if (rows <= 0 || cols <= 0) continue;
#pragma clang loop unroll(disable)
    for (int kk = 0; kk < 256; kk += 32) {
      if (full) tensor_gemm_step16<true>(lo, hi, A, B, P[3], P[4], kk, 32, rows, cols, home);
      else tensor_gemm_step16<false>(lo, hi, A, B, P[3], P[4], kk, 32, rows, cols, home);
    }
    A += 256;
    B += 256;
  }
  const int rest = depth - P[9] * 256;
  if (rest > 0 && rows > 0 && cols > 0) {
    simdgroup_barrier(mem_flags::mem_none);
#pragma clang loop unroll(disable)
    for (int kk = 0; kk < rest; kk += 32)
      tensor_gemm_step16<false>(lo, hi, A, B, P[3], P[4], kk, max(0, rest - kk), rows, cols, home);
  }
  if (rows > 0 && cols > 0) {
    frag_put_in(lo, D, P[5], 0, 0, home, rows, cols);
    frag_put_in(hi, D, P[5], 0, 16, home, rows, cols);
  }
}
}

#define TF_COMPACT_GEMM(NAME, AM, AN) \
  [[max_total_threads_per_threadgroup(256)]] kernel void NAME( \
      const device float* A, const device float* B, constant int* P, device float* D, \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], \
      uint3 tg [[threadgroup_position_in_grid]]) { \
    tfp::tensor_gemm16<AM, AN>(A, B, D, P, tg, sg, lane); \
  }
#ifdef TF_NAX_EXACT_B
TF_COMPACT_GEMM(qwv_gemm_f32_exact_compact_nt_00, false, false)
TF_COMPACT_GEMM(qwv_gemm_f32_exact_compact_nt_01, false, true)
TF_COMPACT_GEMM(qwv_gemm_f32_exact_compact_nt_10, true, false)
TF_COMPACT_GEMM(qwv_gemm_f32_exact_compact_nt_11, true, true)
#else
TF_COMPACT_GEMM(qwv_gemm_f32_compact_nt_00, false, false)
TF_COMPACT_GEMM(qwv_gemm_f32_compact_nt_01, false, true)
TF_COMPACT_GEMM(qwv_gemm_f32_compact_nt_10, true, false)
TF_COMPACT_GEMM(qwv_gemm_f32_compact_nt_11, true, true)
#endif
#undef TF_COMPACT_GEMM
