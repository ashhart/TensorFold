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
constant constexpr int IK2_DOT_KC = 16;  // k steps a staged chunk holds
constant constexpr int IK2_DOT_LD = 68;  // a staged row's pitch: 64 floats and 4 to stagger the banks

// This thread's 4 elements of each operand chunk (k in [k0, k0 + kc), `rows` and `cols` live); zero outside.
template <typename TA, bool NT>
inline void ik2_dot_fetch(const device TA* a, const device float* b, int k0, int kc, int rows, int cols, int lda,
                          int ldb, uint t, thread float* pa, thread float* pb) {
  for (uint i = 0; i < 4; ++i) {
    const uint e = t + i * 256;
    const uint ka = e % IK2_DOT_KC, ra = e / IK2_DOT_KC;
    pa[i] = (int(ka) < kc && int(ra) < rows) ? float(a[long(ra) * lda + k0 + ka]) : 0.0f;
    const uint kb = NT ? ka : e / 64, rb = NT ? ra : e % 64;
    pb[i] = (int(kb) < kc && int(rb) < cols) ? (NT ? b[long(rb) * ldb + k0 + kb] : b[long(k0 + kb) * ldb + rb]) : 0.0f;
  }
}

// A 64 x 64 tile a group, 4 x 4 outputs a thread, each one fma chain over k; SCORE stores bf16(bf16(sum) * scale).
template <bool NT, bool SCORE, typename TA, typename TD>
inline void ik2_f32_dot(const device TA* A, const device float* B, const device int* P, device TD* D,
                        threadgroup float* as, threadgroup float* bs, uint3 tg, uint t) {
  const int m = P[0], n = P[1], K = P[2], lda = P[3], ldb = P[4], ldd = P[5];
  const int row0 = int(tg.y) * 64, col0 = int(tg.x) * 64;
  const int rows = min(64, m - row0), cols = min(64, n - col0);
  const device TA* a = A + long(tg.z) * P[10] + long(row0) * lda;
  const device float* b = B + long(tg.z) * P[11] + (NT ? long(col0) * ldb : long(col0));
  const uint ty = t / 16, tx = t % 16;
  float acc[4][4] = {};
  float pa[4], pb[4];
  int kc = min(IK2_DOT_KC, K);
  ik2_dot_fetch<TA, NT>(a, b, 0, kc, rows, cols, lda, ldb, t, pa, pb);
  for (int k0 = 0; k0 < K; k0 += IK2_DOT_KC) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 4; ++i) {
      const uint e = t + i * 256;
      as[(e % IK2_DOT_KC) * IK2_DOT_LD + e / IK2_DOT_KC] = pa[i];
      bs[NT ? (e % IK2_DOT_KC) * IK2_DOT_LD + e / IK2_DOT_KC : (e / 64) * IK2_DOT_LD + e % 64] = pb[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int here = kc, next = k0 + IK2_DOT_KC;
    kc = min(IK2_DOT_KC, K - next);
    if (next < K) ik2_dot_fetch<TA, NT>(a, b, next, kc, rows, cols, lda, ldb, t, pa, pb);  // in flight during the fmas
    for (int kk = 0; kk < here; ++kk) {
      const float4 av = *reinterpret_cast<threadgroup const float4*>(as + kk * IK2_DOT_LD + ty * 4);
      const float4 bv = *reinterpret_cast<threadgroup const float4*>(bs + kk * IK2_DOT_LD + tx * 4);
      for (uint i = 0; i < 4; ++i) {
        for (uint j = 0; j < 4; ++j) acc[i][j] = fma(av[i], bv[j], acc[i][j]);
      }
    }
  }
  device TD* d = D + long(tg.z) * P[12] + long(row0) * ldd + col0;
  const float scale = as_type<float>(P[13]);
  for (uint i = 0; i < 4; ++i) {
    if (int(ty * 4 + i) >= rows) break;
    for (uint j = 0; j < 4; ++j) {
      if (int(tx * 4 + j) < cols) {
        d[long(ty * 4 + i) * ldd + tx * 4 + j] = TD(SCORE ? tf_vis_bf(tf_vis_bf(acc[i][j]) * scale) : acc[i][j]);
      }
    }
  }
}

#define IK2_F32_DOT(NAME, NT, SCORE, TA, TD)                                                                       \
  [[max_total_threads_per_threadgroup(256)]] [[kernel]] void NAME(                                               \
      const device TA* A [[buffer(0)]], const device float* B [[buffer(1)]],                                     \
      const device int* P [[buffer(2)]], device TD* D [[buffer(3)]],                                             \
      uint3 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {                       \
    threadgroup float as[IK2_DOT_KC * IK2_DOT_LD], bs[IK2_DOT_KC * IK2_DOT_LD];                                  \
    ik2_f32_dot<NT, SCORE, TA, TD>(A, B, P, D, as, bs, tg, t);                                                   \
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
// q k^T to bf16 scores (the scale and both roundings in the epilogue); the four edge names keep the pipeline table.
IK2_F32_DOT(ik2_gemm_score_nt_00, true, true, float, bfloat)
IK2_F32_DOT(ik2_gemm_score_nt_01, true, true, float, bfloat)
IK2_F32_DOT(ik2_gemm_score_nt_10, true, true, float, bfloat)
IK2_F32_DOT(ik2_gemm_score_nt_11, true, true, float, bfloat)
// p v: bf16 probabilities (exact in fp32) against fp32 values, fp32 sums
IK2_F32_DOT(ik2_gemm_prob_nn_00, false, false, bfloat, float)
IK2_F32_DOT(ik2_gemm_prob_nn_01, false, false, bfloat, float)
IK2_F32_DOT(ik2_gemm_prob_nn_10, false, false, bfloat, float)
IK2_F32_DOT(ik2_gemm_prob_nn_11, false, false, bfloat, float)

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
