// NVFP4 prompt projections after gemm_nax.metal: x by the exact dequantized W, fp32 sums times the global, then bf16.

namespace tfp {
template <bool AM, bool AN>
inline void rc_fp4_gemm_t(const device bfloat16_t* A, const device bfloat16_t* B, device bfloat16_t* D,
                          const device int* P, float global, uint3 tg, uint sg, uint lane) {
  constexpr int BM = 64, BN = 128, BK = 256, WN = 4;
  const int swz = P[8];
  const int ty = (int(tg.y) << swz) + (int(tg.x) & ((1 << swz) - 1)), tx = int(tg.x) >> swz;
  if (tx >= P[6] || ty >= P[7]) {
    return;
  }
  const int lda = P[3], ldb = P[4], ldd = P[5];
  const int row = ty * BM + 32 * int(sg / WN), col = tx * BN + 32 * int(sg % WN);
  A += long(P[13]) + long(row) * lda;
  B += long(P[14]) + long(col) * ldb;
  D += long(row) * ldd + col;
  const int rows = AM ? 32 : min(32, P[0] - row), cols = AN ? 32 : min(32, P[1] - col);
  const bool full = (AM || rows == 32) && (AN || cols == 32);
  const short2 home = frag_home(ushort(lane));
  frag<float> acc[2][2];
  if (full) {
    gemm_block<bfloat16_t, false, true, true, BK>(acc, A, B, lda, ldb, P[2], P[9], rows, cols, home);
  } else {
    gemm_block<bfloat16_t, false, true, false, BK>(acc, A, B, lda, ldb, P[2], P[9], rows, cols, home);
  }
  TF_UNROLL
  for (short i = 0; i < 2; i++) {
    TF_UNROLL
    for (short j = 0; j < 2; j++) {
      acc[i][j] *= global;
    }
  }
  if (full) {
    gemm_store<bfloat16_t, true>(acc, D, ldd, rows, cols, home);
  } else if (rows > 0 && cols > 0) {
    gemm_store<bfloat16_t, false>(acc, D, ldd, rows, cols, home);
  }
}
}  // namespace tfp

#define RC_FP4_GEMM(NAME, AM, AN)                                                                                   \
  [[max_total_threads_per_threadgroup(256)]] [[kernel]] void NAME(                                                  \
      const device bfloat16_t* A [[buffer(0)]], const device bfloat16_t* B [[buffer(1)]],                          \
      const device int32_t* P [[buffer(2)]], device bfloat16_t* D [[buffer(3)]], constant float& g [[buffer(4)]],  \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],                        \
      uint3 tg [[threadgroup_position_in_grid]]) {                                                                 \
    tfp::rc_fp4_gemm_t<AM, AN>(A, B, D, P, g, tg, sg, lane);                                                       \
  }
RC_FP4_GEMM(rc_fp4_gemm_00, false, false)
RC_FP4_GEMM(rc_fp4_gemm_01, false, true)
RC_FP4_GEMM(rc_fp4_gemm_10, true, false)
RC_FP4_GEMM(rc_fp4_gemm_11, true, true)
