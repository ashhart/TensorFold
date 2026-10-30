// LFM3-Omni's bf16 products of many rows on the tensor unit, on the prefill GEMM's blocks; fp32 sums, one rounding.
#include "../prefill/gemm_nax.metal"

struct NaxArgs {
  int M, N, K, lda, ldb, ldd, kblocks, pad;
  long sa, sb, sd;
};

template <typename O, bool TA, bool TB>
inline void om_nax(const device bfloat* A, const device bfloat* B, device O* D, constant NaxArgs& a, uint3 tg, uint sg,
                   uint lane) {
  constexpr int BM = 64, BN = 128, BK = 256, WN = 4;
  const int row = int(tg.y) * BM + 32 * int(sg / WN), col = int(tg.x) * BN + 32 * int(sg % WN);
  A += a.sa * tg.z + (TA ? long(row) : long(row) * a.lda);
  B += a.sb * tg.z + (TB ? long(col) * a.ldb : long(col));
  D += a.sd * tg.z + long(row) * a.ldd + col;
  const int rows = min(32, a.M - row), cols = min(32, a.N - col);
  const short2 home = tfp::frag_home(ushort(lane));
  tfp::frag<float> acc[2][2];
  if (rows == 32 && cols == 32) {
    tfp::gemm_block<bfloat, TA, TB, true, BK>(acc, A, B, a.lda, a.ldb, a.K, a.kblocks, rows, cols, home);
    tfp::gemm_store<O, true>(acc, D, a.ldd, rows, cols, home);
  } else {
    tfp::gemm_block<bfloat, TA, TB, false, BK>(acc, A, B, a.lda, a.ldb, a.K, a.kblocks, rows, cols, home);
    if (rows > 0 && cols > 0) tfp::gemm_store<O, false>(acc, D, a.ldd, rows, cols, home);
  }
}

#define OM_NAX(NAME, O, TA, TB)                                                                                          \
  [[max_total_threads_per_threadgroup(256)]] kernel void NAME(                                                            \
      const device bfloat* A [[buffer(0)]], const device bfloat* B [[buffer(1)]], device O* D [[buffer(2)]],              \
      constant NaxArgs& a [[buffer(3)]], uint3 tg [[threadgroup_position_in_grid]],                                       \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {                              \
    om_nax<O, TA, TB>(A, B, D, a, tg, sg, lane);                                                                         \
  }
OM_NAX(om_nax_bb_nt, bfloat, false, true)
OM_NAX(om_nax_bf_nt, float, false, true)
OM_NAX(om_nax_bf_nn, float, false, false)
OM_NAX(om_nax_bf_tt, float, true, true)
