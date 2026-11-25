// Inkling 2 tower GEMMs after gemm_nax.metal: y = x w^T, fp32 sums; P is gemm_nax's parameter array.

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

// Score products stay a full fp32 dot so an M5 does not cut an fp32 operand to 10 mantissa bits.
template <bool NT>
inline void ik2_f32_dot(const device float* A, const device float* B, const device int* P, device float* D, uint3 tid) {
  const int col = int(tid.x), row = int(tid.y), b = int(tid.z);
  const int m = P[0], n = P[1], K = P[2];
  if (col >= n || row >= m) return;
  const int lda = P[3], ldb = P[4], ldd = P[5], sa = P[10], sb = P[11], sd = P[12];
  const device float* a = A + long(b) * sa + long(row) * lda;
  const device float* brow = B + long(b) * sb;
  float s = 0;
  for (int k = 0; k < K; ++k) {
    s = fma(a[k], NT ? brow[long(col) * ldb + k] : brow[long(k) * ldb + col], s);
  }
  D[long(b) * sd + long(row) * ldd + col] = s;
}

#define IK2_F32_DOT(NAME, NT)                                                                                      \
  [[max_total_threads_per_threadgroup(256)]] [[kernel]] void NAME(                                               \
      const device float* A [[buffer(0)]], const device float* B [[buffer(1)]],                                  \
      const device int* P [[buffer(2)]], device float* D [[buffer(3)]], uint3 tid [[thread_position_in_grid]]) { \
    ik2_f32_dot<NT>(A, B, P, D, tid);                                                                              \
  }


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
// bf16 x w^T, fp32 out: the tower's projections with a bias, which an epilogue adds before its one rounding
IK2_GEMM_OUT(ik2_gemm_bf16f_nt_00, bfloat16_t, float, true, false, false)
IK2_GEMM_OUT(ik2_gemm_bf16f_nt_01, bfloat16_t, float, true, false, true)
IK2_GEMM_OUT(ik2_gemm_bf16f_nt_10, bfloat16_t, float, true, true, false)
IK2_GEMM_OUT(ik2_gemm_bf16f_nt_11, bfloat16_t, float, true, true, true)
// q k^T and p v, full fp32. The four edge names stay so the existing pipeline table still resolves.
IK2_F32_DOT(ik2_gemm_f32_nt_00, true)
IK2_F32_DOT(ik2_gemm_f32_nt_01, true)
IK2_F32_DOT(ik2_gemm_f32_nt_10, true)
IK2_F32_DOT(ik2_gemm_f32_nt_11, true)
IK2_F32_DOT(ik2_gemm_f32_nn_00, false)
IK2_F32_DOT(ik2_gemm_f32_nn_01, false)
IK2_F32_DOT(ik2_gemm_f32_nn_10, false)
IK2_F32_DOT(ik2_gemm_f32_nn_11, false)

#ifdef TF_SIMD_FRAGS
// y = x w^T for at most 8 rows without tensor units: one 8x8x8 MMA per K octet in order, the tile GEMM's bits.
[[max_total_threads_per_threadgroup(32)]] [[kernel]] void ik2_gemm_bf16_nt_rows(
    const device bfloat* A [[buffer(0)]], const device bfloat* B [[buffer(1)]], const device int32_t* P [[buffer(2)]],
    device bfloat* D [[buffer(3)]], uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  const int m = P[0], K = P[2], lda = P[3], ldb = P[4], ldd = P[5];
  const int col = int(tg.x) * 16;
  const uint qid = lane >> 2u;
  const uint fm = (qid & 4u) + ((lane >> 1u) & 3u);
  const uint fn = (qid & 2u) * 2u + (lane & 1u) * 2u;
  const bool live = int(fm) < m;
  const device bfloat* a = A + long(fm) * lda + fn;
  const device bfloat* b0 = B + long(col) * ldb;
  const device bfloat* b1 = b0 + 8 * long(ldb);
  simdgroup_matrix<float, 8, 8> acc0(0.0f), acc1(0.0f);
#pragma clang loop unroll_count(8)
  for (int k = 0; k < K; k += 8) {
    simdgroup_matrix<bfloat, 8, 8> fa, fb0, fb1;
    fa.thread_elements()[0] = live ? a[k] : bfloat(0.0f);
    fa.thread_elements()[1] = live ? a[k + 1] : bfloat(0.0f);
    simdgroup_load(fb0, b0 + k, ldb, ulong2(0, 0), true);
    simdgroup_load(fb1, b1 + k, ldb, ulong2(0, 0), true);
    simdgroup_multiply_accumulate(acc0, fa, fb0, acc0);
    simdgroup_multiply_accumulate(acc1, fa, fb1, acc1);
  }
  if (!live) return;
  device bfloat* d = D + long(fm) * ldd + col + fn;
  d[0] = bfloat(acc0.thread_elements()[0]);
  d[1] = bfloat(acc0.thread_elements()[1]);
  d[8] = bfloat(acc1.thread_elements()[0]);
  d[9] = bfloat(acc1.thread_elements()[1]);
}
#endif
