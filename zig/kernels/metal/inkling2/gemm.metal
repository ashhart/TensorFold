// Inkling 2 tower and draft GEMMs after gemm_nax.metal: y = x w^T, fp32 sums; P is gemm_nax's parameter array.

namespace tfp {

// gemm_nax with output type O apart from operand type T: an fp32 block stays fp32 for an epilogue's one rounding.
template <typename T, typename O, bool TA, bool TB, bool AM, bool AN>
inline void gemm_out(const device T* A, const device T* B, device O* D, const device int* P, uint3 tg, uint sg,
                     uint lane) {
  const int swz = P[8];
  const int ty = (int(tg.y) << swz) + (int(tg.x) & ((1 << swz) - 1)), tx = int(tg.x) >> swz;
  if (tx >= P[6] || ty >= P[7]) {
    return;
  }
  const int lda = P[3], ldb = P[4], ldd = P[5];
  const int row = ty * 64 + 32 * int(sg / 4), col = tx * 128 + 32 * int(sg % 4);
  A += long(P[13]) + long(P[10]) * tg.z + (TA ? long(row) : long(row) * lda);
  B += long(P[14]) + long(P[11]) * tg.z + (TB ? long(col) * ldb : long(col));
  D += long(P[12]) * tg.z + long(row) * ldd + col;
  const int rows = AM ? 32 : min(32, P[0] - row), cols = AN ? 32 : min(32, P[1] - col);
  const bool full = (AM || rows == 32) && (AN || cols == 32);
  const short2 home = frag_home(ushort(lane));
  frag<float> acc[2][2];
  if (full) {
    gemm_block<T, TA, TB, true, 256>(acc, A, B, lda, ldb, P[2], P[9], rows, cols, home);
    gemm_store<O, true>(acc, D, ldd, rows, cols, home);
  } else {
    gemm_block<T, TA, TB, false, 256>(acc, A, B, lda, ldb, P[2], P[9], rows, cols, home);
    if (rows > 0 && cols > 0) {
      gemm_store<O, false>(acc, D, ldd, rows, cols, home);
    }
  }
}

}  // namespace tfp

#define IK2_GEMM_OUT(NAME, T, O, TB, AM, AN)                                                                       \
  [[max_total_threads_per_threadgroup(256)]] [[kernel]] void NAME(                                                \
      const device T* A [[buffer(0)]], const device T* B [[buffer(1)]], const device int32_t* P [[buffer(2)]],     \
      device O* D [[buffer(3)]], uint sg [[simdgroup_index_in_threadgroup]],                                      \
      uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {                        \
    tfp::gemm_out<T, O, false, TB, AM, AN>(A, B, D, P, tg, sg, lane);                                              \
  }

// bf16 x w^T, bf16 out: the projections without a bias (one rounding of the fp32 sums)
TF_GEMM_NAX_KERNEL(ik2_gemm_bf16_nt_00, bfloat16_t, false, true, false, false)
TF_GEMM_NAX_KERNEL(ik2_gemm_bf16_nt_01, bfloat16_t, false, true, false, true)
TF_GEMM_NAX_KERNEL(ik2_gemm_bf16_nt_10, bfloat16_t, false, true, true, false)
TF_GEMM_NAX_KERNEL(ik2_gemm_bf16_nt_11, bfloat16_t, false, true, true, true)
// bf16 x w, bf16 out: the draft's unembedding folded with its Markov output projection
TF_GEMM_NAX_KERNEL(ik2_gemm_bf16_nn_00, bfloat16_t, false, false, false, false)
TF_GEMM_NAX_KERNEL(ik2_gemm_bf16_nn_01, bfloat16_t, false, false, false, true)
TF_GEMM_NAX_KERNEL(ik2_gemm_bf16_nn_10, bfloat16_t, false, false, true, false)
TF_GEMM_NAX_KERNEL(ik2_gemm_bf16_nn_11, bfloat16_t, false, false, true, true)
// bf16 x w^T, fp32 out: the tower's projections with a bias, which an epilogue adds before its one rounding
IK2_GEMM_OUT(ik2_gemm_bf16f_nt_00, bfloat16_t, float, true, false, false)
IK2_GEMM_OUT(ik2_gemm_bf16f_nt_01, bfloat16_t, float, true, false, true)
IK2_GEMM_OUT(ik2_gemm_bf16f_nt_10, bfloat16_t, float, true, true, false)
IK2_GEMM_OUT(ik2_gemm_bf16f_nt_11, bfloat16_t, float, true, true, true)
// q k^T per head in fp32
TF_GEMM_NAX_KERNEL(ik2_gemm_f32_nt_00, float, false, true, false, false)
TF_GEMM_NAX_KERNEL(ik2_gemm_f32_nt_01, float, false, true, false, true)
TF_GEMM_NAX_KERNEL(ik2_gemm_f32_nt_10, float, false, true, true, false)
TF_GEMM_NAX_KERNEL(ik2_gemm_f32_nt_11, float, false, true, true, true)
// p v per head in fp32
TF_GEMM_NAX_KERNEL(ik2_gemm_f32_nn_00, float, false, false, false, false)
TF_GEMM_NAX_KERNEL(ik2_gemm_f32_nn_01, float, false, false, false, true)
TF_GEMM_NAX_KERNEL(ik2_gemm_f32_nn_10, float, false, false, true, false)
TF_GEMM_NAX_KERNEL(ik2_gemm_f32_nn_11, float, false, false, true, true)
