// Prefill GEMMs on the M5 tensor unit, bit-identical to MLX 0.32.3's steel_gemm_fused_nax and steel_gemm_splitk_nax.
#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;
typedef half float16_t;
#include "../nax.h"

namespace tfp {

// One 32-deep K step of a simdgroup's 32x32 block; kk is the step's K offset from A and B, kn the K values left there.
template <typename T, bool TA, bool TB, bool BOUND>
inline void gemm_step(thread frag<float> (&acc)[2][2], const device T* A, const device T* B, int lda, int ldb, int kk,
                      int kn, int rows, int cols, short2 home) {
  frag<T> a[2][2], b[2][2];   // a[m][k], b[k][n]: 16x16 blocks of A and B in their stored orientation
  TF_UNROLL
  for (short i = 0; i < 2; i++) {
    TF_UNROLL
    for (short j = 0; j < 2; j++) {
      const int ar = TA ? kk + 16 * j : 16 * i, ac = TA ? 16 * i : kk + 16 * j;
      const int br = TB ? 16 * j : kk + 16 * i, bc = TB ? kk + 16 * i : 16 * j;
#ifdef TF_SIMD_FRAGS
      // simdgroup matrices: a transposed operand loads with its stored rows on the column pattern
      if (BOUND) {
        if (TA) frag_get_t_in(a[i][j], A, lda, ar, ac, home, kk + kn, rows);
        else frag_get_in(a[i][j], A, lda, ar, ac, home, rows, kk + kn);
        if (TB) frag_get_t_in(b[i][j], B, ldb, br, bc, home, cols, kk + kn);
        else frag_get_in(b[i][j], B, ldb, br, bc, home, kk + kn, cols);
      } else {
        if (TA) frag_get_t(a[i][j], A, lda, ar, ac, home);
        else frag_get(a[i][j], A, lda, ar, ac, home);
        if (TB) frag_get_t(b[i][j], B, ldb, br, bc, home);
        else frag_get(b[i][j], B, ldb, br, bc, home);
      }
#else
      if (BOUND) {
        frag_get_in(a[i][j], A, lda, ar, ac, home, TA ? kk + kn : rows, TA ? rows : kk + kn);
        frag_get_in(b[i][j], B, ldb, br, bc, home, TB ? cols : kk + kn, TB ? kk + kn : cols);
      } else {
        frag_get(a[i][j], A, lda, ar, ac, home);
        frag_get(b[i][j], B, ldb, br, bc, home);
      }
#endif
    }
  }
  TF_UNROLL
  for (short m = 0; m < 2; m++) {
    TF_UNROLL
    for (short k = 0; k < 2; k++) {
      mma_16x32<TA, TB>(acc[m][0], acc[m][1], a[m][k], b[k][0], b[k][1]);
    }
  }
}

// A simdgroup's 32x32 block over K: whole BK blocks first (barrier each), then 32-deep steps zero-padded past K.
template <typename T, bool TA, bool TB, bool FULL, int BK>
inline void gemm_block(thread frag<float> (&acc)[2][2], const device T* A, const device T* B, int lda, int ldb,
                       int K, int kblocks, int rows, int cols, short2 home) {
  TF_UNROLL
  for (short i = 0; i < 2; i++) {
    TF_UNROLL
    for (short j = 0; j < 2; j++) {
      acc[i][j] = frag<float>(0);
    }
  }
  const bool live = rows > 0 && cols > 0;
#pragma clang loop unroll(disable)
  for (int kb = 0; kb < kblocks; kb++) {
    threadgroup_barrier(mem_flags::mem_none);
    if (!FULL && !live) {
      continue;
    }
#pragma clang loop unroll(disable)
    for (int kk = 0; kk < BK; kk += 32) {
      gemm_step<T, TA, TB, !FULL>(acc, A, B, lda, ldb, kk, 32, rows, cols, home);
    }
    A += TA ? BK * lda : BK;
    B += TB ? BK : BK * ldb;
  }
  const int rest = K - kblocks * BK;
  if (rest > 0) {
    simdgroup_barrier(mem_flags::mem_none);
    if (!FULL && !live) {
      return;
    }
#pragma clang loop unroll(disable)
    for (int kk = 0; kk < rest; kk += 32) {
      gemm_step<T, TA, TB, true>(acc, A, B, lda, ldb, kk, max(0, rest - kk), rows, cols, home);
    }
  }
}

template <typename O, bool FULL>
inline void gemm_store(thread const frag<float> (&acc)[2][2], device O* D, int ldd, int rows, int cols, short2 home) {
  TF_UNROLL
  for (short i = 0; i < 2; i++) {
    TF_UNROLL
    for (short j = 0; j < 2; j++) {
      if (FULL) {
        frag_put(acc[i][j], D, ldd, 16 * i, 16 * j, home);
      } else {
        frag_put_in(acc[i][j], D, ldd, 16 * i, 16 * j, home, rows, cols);
      }
    }
  }
}

// D = A * B per batch, steel_gemm_fused_nax without epilogue; P holds the GEMM_PARAMS fields of check_prefill_ops.
template <typename T, bool TA, bool TB, bool AM, bool AN, bool AK, int BM, int BN, int BK, int WM, int WN>
inline void gemm_nax(const device T* A, const device T* B, device T* D, const device int* P, uint3 tg, uint sg,
                     uint lane) {
  static_assert(BM / WM == 32 && BN / WN == 32, "a simdgroup owns a 32x32 block");
  const int swz = P[8];
  const int ty = (int(tg.y) << swz) + (int(tg.x) & ((1 << swz) - 1)), tx = int(tg.x) >> swz;
  if (tx >= P[6] || ty >= P[7]) {
    return;
  }
  const int lda = P[3], ldb = P[4], ldd = P[5];
  const int row = ty * BM + 32 * int(sg / WN), col = tx * BN + 32 * int(sg % WN);
  A += long(P[13]) + long(P[10]) * tg.z + (TA ? long(row) : long(row) * lda);
  B += long(P[14]) + long(P[11]) * tg.z + (TB ? long(col) * ldb : long(col));
  D += long(P[12]) * tg.z + long(row) * ldd + col;
  const int rows = AM ? 32 : min(32, P[0] - row), cols = AN ? 32 : min(32, P[1] - col);
  const bool full = (AM || rows == 32) && (AN || cols == 32);
  const short2 home = frag_home(ushort(lane));
  frag<float> acc[2][2];
  if (full) {
    gemm_block<T, TA, TB, true, BK>(acc, A, B, lda, ldb, P[2], P[9], rows, cols, home);
    gemm_store<T, true>(acc, D, ldd, rows, cols, home);
  } else {
    gemm_block<T, TA, TB, false, BK>(acc, A, B, lda, ldb, P[2], P[9], rows, cols, home);
    if (rows > 0 && cols > 0) {
      gemm_store<T, false>(acc, D, ldd, rows, cols, home);
    }
  }
}

// One K partition's fp32 block, steel_gemm_splitk_nax; P holds the SPLITK_PARAMS fields of check_prefill_ops.
template <typename T, bool TA, bool TB, bool AM, bool AN, int BM, int BN, int BK, int WM, int WN>
inline void gemm_splitk_nax(const device T* A, const device T* B, device float* C, const device int* P, uint3 tg,
                            uint sg, uint lane) {
  static_assert(BM / WM == 32 && BN / WN == 32, "a simdgroup owns a 32x32 block");
  const int swz = P[11];
  const int span_n = P[6] << swz, span_m = (P[7] + (1 << swz) - 1) >> swz;
  const int part = int(tg.x) / (span_n * span_m), at = int(tg.x) % (span_n * span_m);
  const int gx = at % span_n, gy = at / span_n;
  const int ty = (gy << swz) + (gx & ((1 << swz) - 1)), tx = gx >> swz;
  if (tx >= P[6] || ty >= P[7]) {
    return;
  }
  const int lda = P[3], ldb = P[4], ldc = P[5];
  const int k0 = P[10] * part, kn = min(k0 + P[10], P[2]) - k0;
  const int row = ty * BM + 32 * int(sg / WN), col = tx * BN + 32 * int(sg % WN);
  A += long(P[12]) + (TA ? long(row) + long(k0) * lda : long(k0) + long(row) * lda);
  B += long(P[13]) + (TB ? long(k0) + long(col) * ldb : long(col) + long(k0) * ldb);
  C += long(P[9]) * part + long(row) * ldc + col;
  const int rows = AM ? 32 : min(32, P[0] - row), cols = AN ? 32 : min(32, P[1] - col);
  const bool full = (AM || rows == 32) && (AN || cols == 32);
  const short2 home = frag_home(ushort(lane));
  frag<float> acc[2][2];
  if (full) {
    gemm_block<T, TA, TB, true, BK>(acc, A, B, lda, ldb, kn, kn / BK, rows, cols, home);
    gemm_store<float, true>(acc, C, ldc, rows, cols, home);
  } else {
    gemm_block<T, TA, TB, false, BK>(acc, A, B, lda, ldb, kn, kn / BK, rows, cols, home);
    gemm_store<float, false>(acc, C, ldc, rows, cols, home);
  }
}

// D = the partitions' fp32 blocks summed in partition order from 0, steel_gemm_splitk_accum. P: parts stride ldd.
template <typename O>
inline void gemm_splitk_sum(const device float* C, device O* D, const device int* P, uint3 at) {
  const long i = long(at.x) + long(at.y) * P[2];
  float s = 0;
  for (int p = 0; p < P[0]; p++) {
    s += C[i + long(p) * P[1]];
  }
  D[i] = O(s);
}

}  // namespace tfp
// ---- entry points: generated by tools/zig/check_prefill_ops.py --write ----
[[max_total_threads_per_threadgroup(256)]]
[[kernel]] void custom_kernel_tf_gemm_nax_bf16_n_t_n_n_n_64_128_256_2_4_bfloat16_t_bfloat16_t_int32_t_bfloat16_t(
  const device bfloat16_t* A [[buffer(0)]],
  const device bfloat16_t* B [[buffer(1)]],
  const device int32_t* P [[buffer(2)]],
  device bfloat16_t* D [[buffer(3)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  tfp::gemm_nax<bfloat16_t, false, true, false, false, false, 64, 128, 256, 2, 4>(A, B, D, P, threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);

}
[[max_total_threads_per_threadgroup(256)]]
[[kernel]] void custom_kernel_tf_gemm_nax_bf16_n_t_n_t_n_64_128_256_2_4_bfloat16_t_bfloat16_t_int32_t_bfloat16_t(
  const device bfloat16_t* A [[buffer(0)]],
  const device bfloat16_t* B [[buffer(1)]],
  const device int32_t* P [[buffer(2)]],
  device bfloat16_t* D [[buffer(3)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  tfp::gemm_nax<bfloat16_t, false, true, false, true, false, 64, 128, 256, 2, 4>(A, B, D, P, threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);

}
[[max_total_threads_per_threadgroup(256)]]
[[kernel]] void custom_kernel_tf_gemm_nax_bf16_n_t_t_n_n_64_128_256_2_4_bfloat16_t_bfloat16_t_int32_t_bfloat16_t(
  const device bfloat16_t* A [[buffer(0)]],
  const device bfloat16_t* B [[buffer(1)]],
  const device int32_t* P [[buffer(2)]],
  device bfloat16_t* D [[buffer(3)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  tfp::gemm_nax<bfloat16_t, false, true, true, false, false, 64, 128, 256, 2, 4>(A, B, D, P, threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);

}
[[max_total_threads_per_threadgroup(256)]]
[[kernel]] void custom_kernel_tf_gemm_nax_bf16_n_t_t_t_n_64_128_256_2_4_bfloat16_t_bfloat16_t_int32_t_bfloat16_t(
  const device bfloat16_t* A [[buffer(0)]],
  const device bfloat16_t* B [[buffer(1)]],
  const device int32_t* P [[buffer(2)]],
  device bfloat16_t* D [[buffer(3)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  tfp::gemm_nax<bfloat16_t, false, true, true, true, false, 64, 128, 256, 2, 4>(A, B, D, P, threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);

}
[[max_total_threads_per_threadgroup(256)]]
[[kernel]] void custom_kernel_tf_gemm_nax_f32_n_n_n_n_n_64_128_256_2_4_float_float_int32_t_float(
  const device float* A [[buffer(0)]],
  const device float* B [[buffer(1)]],
  const device int32_t* P [[buffer(2)]],
  device float* D [[buffer(3)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  tfp::gemm_nax<float, false, false, false, false, false, 64, 128, 256, 2, 4>(A, B, D, P, threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);

}
[[max_total_threads_per_threadgroup(256)]]
[[kernel]] void custom_kernel_tf_gemm_nax_f32_n_n_t_n_n_64_128_256_2_4_float_float_int32_t_float(
  const device float* A [[buffer(0)]],
  const device float* B [[buffer(1)]],
  const device int32_t* P [[buffer(2)]],
  device float* D [[buffer(3)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  tfp::gemm_nax<float, false, false, true, false, false, 64, 128, 256, 2, 4>(A, B, D, P, threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);

}
[[max_total_threads_per_threadgroup(256)]]
[[kernel]] void custom_kernel_tf_gemm_nax_f32_n_n_t_n_t_64_128_256_2_4_float_float_int32_t_float(
  const device float* A [[buffer(0)]],
  const device float* B [[buffer(1)]],
  const device int32_t* P [[buffer(2)]],
  device float* D [[buffer(3)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  tfp::gemm_nax<float, false, false, true, false, true, 64, 128, 256, 2, 4>(A, B, D, P, threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);

}
[[max_total_threads_per_threadgroup(256)]]
[[kernel]] void custom_kernel_tf_gemm_nax_f32_t_t_t_t_n_64_128_256_2_4_float_float_int32_t_float(
  const device float* A [[buffer(0)]],
  const device float* B [[buffer(1)]],
  const device int32_t* P [[buffer(2)]],
  device float* D [[buffer(3)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  tfp::gemm_nax<float, true, true, true, true, false, 64, 128, 256, 2, 4>(A, B, D, P, threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);

}
[[max_total_threads_per_threadgroup(256)]]
[[kernel]] void custom_kernel_tf_gemm_nax_f32_t_t_t_t_t_64_128_256_2_4_float_float_int32_t_float(
  const device float* A [[buffer(0)]],
  const device float* B [[buffer(1)]],
  const device int32_t* P [[buffer(2)]],
  device float* D [[buffer(3)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  tfp::gemm_nax<float, true, true, true, true, true, 64, 128, 256, 2, 4>(A, B, D, P, threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);

}
[[max_total_threads_per_threadgroup(128)]]
[[kernel]] void custom_kernel_tf_gemm_splitk_nax_bf16_n_t_n_t_64_64_256_2_2_bfloat16_t_bfloat16_t_int32_t_float(
  const device bfloat16_t* A [[buffer(0)]],
  const device bfloat16_t* B [[buffer(1)]],
  const device int32_t* P [[buffer(2)]],
  device float* C [[buffer(3)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  tfp::gemm_splitk_nax<bfloat16_t, false, true, false, true, 64, 64, 256, 2, 2>(A, B, C, P, threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);

}
[[max_total_threads_per_threadgroup(128)]]
[[kernel]] void custom_kernel_tf_gemm_splitk_nax_bf16_n_t_t_t_64_64_256_2_2_bfloat16_t_bfloat16_t_int32_t_float(
  const device bfloat16_t* A [[buffer(0)]],
  const device bfloat16_t* B [[buffer(1)]],
  const device int32_t* P [[buffer(2)]],
  device float* C [[buffer(3)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  tfp::gemm_splitk_nax<bfloat16_t, false, true, true, true, 64, 64, 256, 2, 2>(A, B, C, P, threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);

}
[[kernel]] void custom_kernel_tf_gemm_splitk_sum_bf16_float_int32_t_bfloat16_t(
  const device float* C [[buffer(0)]],
  const device int32_t* P [[buffer(1)]],
  device bfloat16_t* D [[buffer(2)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
  tfp::gemm_splitk_sum<bfloat16_t>(C, D, P, thread_position_in_grid);

}
