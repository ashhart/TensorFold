// BF16 linear kernels for M5 and M3: row-invariant decode/verify (M=1..32) and NAX/SIMD prefill GEMM (M>32).
#include <metal_stdlib>
using namespace metal;
#include "../nax.h"

#ifndef TF_CTL
#define TF_CTL 0
#endif

#if TF_CTL == 4
typedef half sn_mv_acc_t;
#else
typedef float sn_mv_acc_t;
#endif

struct SnBf16MvArgs {
  uint rows, n, k, ldy;
  ulong y1_addr, y2_addr;
  uint split1, ldy1, split2, ldy2;
};

struct SnBf16GemmArgs {
  int32_t p[16];
  ulong y1_addr, y2_addr;
  uint split1, ldy1, split2, ldy2;
};

struct SnBf16SumArgs {
  int32_t p[6];
  ulong y1_addr, y2_addr;
  uint split1, ldy1, split2, ldy2;
};

inline device bfloat* sn_bf16_split_dest(
    device bfloat* Y0,
    uint ldy0,
    ulong y1_addr,
    uint split1,
    uint ldy1,
    ulong y2_addr,
    uint split2,
    uint ldy2,
    uint col,
    thread uint& col_out,
    thread uint& ldy_out) {
  if (y2_addr != 0ul && col >= split2) {
    col_out = col - split2;
    ldy_out = ldy2;
    return (device bfloat*)y2_addr;
  } else if (y1_addr != 0ul && col >= split1) {
    col_out = col - split1;
    ldy_out = ldy1;
    return (device bfloat*)y1_addr;
  }
  col_out = col;
  ldy_out = ldy0;
  return Y0;
}

#define SN_BF16_COLS(R) ((R) == 1 ? 1u : ((R) <= 4 ? 2u : 1u))

// Y[r, j] = bf16(sum_i X[r, i] * W[j, i]) in FP32; SK slices per output row reduce in fixed slice order.
template <uint ROWS, uint SK>
inline void sn_bf16_mv_impl(
    const device bfloat* X,
    const device bfloat* W,
    constant SnBf16MvArgs& a,
    device bfloat* Y,
    uint3 tg,
    uint sg,
    uint lane,
    threadgroup float* red) {
#pragma clang fp contract(off)
  constexpr uint COLS = SN_BF16_COLS(ROWS);
  constexpr uint SG = 8u;
  constexpr uint GROUPS = SG / SK;
  constexpr uint NO = GROUPS * COLS;
  const uint g_idx = sg / SK;
  const uint s = sg % SK;
  const uint j0 = tg.x * NO + g_idx * COLS;
  const uint vec_k = a.k >> 3;
  const uint k_end = (TF_CTL == 3 && vec_k > 0) ? (vec_k - 1u) : vec_k;
  const device uint4* X4 = (const device uint4*)X;

  sn_mv_acc_t acc[COLS][ROWS];
  _Pragma("clang loop unroll(full)")
  for (uint col = 0; col < COLS; ++col) {
    _Pragma("clang loop unroll(full)")
    for (uint r = 0; r < ROWS; ++r) acc[col][r] = sn_mv_acc_t(0);
  }

#if TF_CTL == 2
  if (tg.x == 0 && sg == 0 && lane == 0 && a.n > 0) {
    ((device ushort*)X)[0] ^= 1u;
  }
#endif

  const uint step = SK * 32u;
  if (j0 + COLS <= a.n) {
    const device uint4* W_ptr[COLS];
    _Pragma("clang loop unroll(full)")
    for (uint col = 0; col < COLS; ++col) {
      W_ptr[col] = (const device uint4*)(W + (size_t)(j0 + col) * a.k);
    }
    for (uint c = s * 32u + lane; c < k_end; c += step) {
      vec<sn_mv_acc_t, 4> fx0[ROWS], fx1[ROWS];
      _Pragma("clang loop unroll(full)")
      for (uint r = 0; r < ROWS; ++r) {
        const uint rr = (r < a.rows) ? r : 0u;
        const uint4 xv = X4[(size_t)rr * vec_k + c];
        fx0[r] = vec<sn_mv_acc_t, 4>(as_type<bfloat4>(xv.xy));
        fx1[r] = vec<sn_mv_acc_t, 4>(as_type<bfloat4>(xv.zw));
      }
      uint4 wv[COLS];
      _Pragma("clang loop unroll(full)")
      for (uint col = 0; col < COLS; ++col) wv[col] = W_ptr[col][c];
      _Pragma("clang loop unroll(full)")
      for (uint col = 0; col < COLS; ++col) {
        const vec<sn_mv_acc_t, 4> fw0 = vec<sn_mv_acc_t, 4>(as_type<bfloat4>(wv[col].xy));
        const vec<sn_mv_acc_t, 4> fw1 = vec<sn_mv_acc_t, 4>(as_type<bfloat4>(wv[col].zw));
        _Pragma("clang loop unroll(full)")
        for (uint r = 0; r < ROWS; ++r) {
          acc[col][r] += dot(fw0, fx0[r]) + dot(fw1, fx1[r]);
        }
      }
    }
  } else if (j0 < a.n) {
    for (uint c = s * 32u + lane; c < k_end; c += step) {
      vec<sn_mv_acc_t, 4> fx0[ROWS], fx1[ROWS];
      _Pragma("clang loop unroll(full)")
      for (uint r = 0; r < ROWS; ++r) {
        const uint rr = (r < a.rows) ? r : 0u;
        const uint4 xv = X4[(size_t)rr * vec_k + c];
        fx0[r] = vec<sn_mv_acc_t, 4>(as_type<bfloat4>(xv.xy));
        fx1[r] = vec<sn_mv_acc_t, 4>(as_type<bfloat4>(xv.zw));
      }
      _Pragma("clang loop unroll(full)")
      for (uint col = 0; col < COLS; ++col) {
        if (j0 + col < a.n) {
          const uint4 wv = ((const device uint4*)(W + (size_t)(j0 + col) * a.k))[c];
          const vec<sn_mv_acc_t, 4> fw0 = vec<sn_mv_acc_t, 4>(as_type<bfloat4>(wv.xy));
          const vec<sn_mv_acc_t, 4> fw1 = vec<sn_mv_acc_t, 4>(as_type<bfloat4>(wv.zw));
          _Pragma("clang loop unroll(full)")
          for (uint r = 0; r < ROWS; ++r) {
            acc[col][r] += dot(fw0, fx0[r]) + dot(fw1, fx1[r]);
          }
        }
      }
    }
  }

  if constexpr (SK == 1u) {
    _Pragma("clang loop unroll(full)")
    for (uint col = 0; col < COLS; ++col) {
      if (j0 + col < a.n) {
        uint j_loc, ldy_eff;
        device bfloat* Y_dst = sn_bf16_split_dest(
            Y, a.ldy, a.y1_addr, a.split1, a.ldy1, a.y2_addr, a.split2, a.ldy2, j0 + col, j_loc, ldy_eff);
        _Pragma("clang loop unroll(full)")
        for (uint r = 0; r < ROWS; ++r) {
          float v = float(simd_sum(acc[col][r]));
          if (lane == 0 && r < a.rows) {
#if TF_CTL == 1
            if (isnan(v) || isinf(v)) v = 0.0f;
#endif
            Y_dst[(size_t)r * ldy_eff + j_loc] = bfloat(v);
          }
        }
      }
    }
    return;
  }

  _Pragma("clang loop unroll(full)")
  for (uint col = 0; col < COLS; ++col) {
    _Pragma("clang loop unroll(full)")
    for (uint r = 0; r < ROWS; ++r) {
      const float v = float(simd_sum(acc[col][r]));
      if (j0 + col < a.n && lane == 0 && r < a.rows) {
        red[((g_idx * COLS + col) * SK + s) * ROWS + r] = v;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (s == 0) {
    _Pragma("clang loop unroll(full)")
    for (uint col = 0; col < COLS; ++col) {
      if (j0 + col < a.n) {
        uint j_loc, ldy_eff;
        device bfloat* Y_dst = sn_bf16_split_dest(
            Y, a.ldy, a.y1_addr, a.split1, a.ldy1, a.y2_addr, a.split2, a.ldy2, j0 + col, j_loc, ldy_eff);
        for (uint r = lane; r < a.rows; r += 32u) {
          sn_mv_acc_t total = sn_mv_acc_t(0);
          _Pragma("clang loop unroll(full)")
          for (uint k = 0; k < SK; ++k) {
            total += sn_mv_acc_t(red[((g_idx * COLS + col) * SK + k) * ROWS + r]);
          }
          float out_v = float(total);
#if TF_CTL == 1
          if (isnan(out_v) || isinf(out_v)) out_v = 0.0f;
#endif
          Y_dst[(size_t)r * ldy_eff + j_loc] = bfloat(out_v);
        }
      }
    }
  }
}

#define SN_BF16_MV(ROWS, SK)                                                                                     \
  [[max_total_threads_per_threadgroup(256)]]                                                                     \
  [[kernel]] void sn_bf16_mv_r##ROWS##_sk##SK(                                                                   \
      const device bfloat* X [[buffer(0)]],                                                                      \
      const device bfloat* W [[buffer(1)]],                                                                      \
      constant SnBf16MvArgs& a [[buffer(2)]],                                                                    \
      device bfloat* Y [[buffer(3)]],                                                                            \
      uint3 tg [[threadgroup_position_in_grid]],                                                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                                                                \
      uint lane [[thread_index_in_simdgroup]]) {                                                                 \
    threadgroup float red[SK > 1 ? 8 * SN_BF16_COLS(ROWS) * ROWS : 1];                                           \
    sn_bf16_mv_impl<ROWS, SK>(X, W, a, Y, tg, sg, lane, red);                                                    \
  }

#define SN_BF16_MV_ROW(ROWS) \
  SN_BF16_MV(ROWS, 1)        \
  SN_BF16_MV(ROWS, 2)        \
  SN_BF16_MV(ROWS, 4)        \
  SN_BF16_MV(ROWS, 8)

SN_BF16_MV_ROW(1)
SN_BF16_MV_ROW(2)
SN_BF16_MV_ROW(4)
SN_BF16_MV_ROW(8)
SN_BF16_MV_ROW(16)
SN_BF16_MV_ROW(32)

namespace sn_prefill {

template <bool TB, bool BOUND>
inline void gemm_step(
    thread tfp::frag<float> (&acc)[2][2],
    const device bfloat* A,
    const device bfloat* B,
    int lda,
    int ldb,
    int kk,
    int kn,
    int rows,
    int cols,
    short2 home) {
  tfp::frag<bfloat> a[2][2], b[2][2];
  TF_UNROLL
  for (short i = 0; i < 2; i++) {
    TF_UNROLL
    for (short j = 0; j < 2; j++) {
      const int ar = 16 * i, ac = kk + 16 * j;
      const int br = TB ? 16 * j : kk + 16 * i;
      const int bc = TB ? kk + 16 * i : 16 * j;
#ifdef TF_SIMD_FRAGS
      if (BOUND) {
        tfp::frag_get_in(a[i][j], A, lda, ar, ac, home, rows, kk + kn);
        if (TB) tfp::frag_get_t_in(b[i][j], B, ldb, br, bc, home, cols, kk + kn);
        else tfp::frag_get_in(b[i][j], B, ldb, br, bc, home, kk + kn, cols);
      } else {
        tfp::frag_get(a[i][j], A, lda, ar, ac, home);
        if (TB) tfp::frag_get_t(b[i][j], B, ldb, br, bc, home);
        else tfp::frag_get(b[i][j], B, ldb, br, bc, home);
      }
#else
      if (BOUND) {
        tfp::frag_get_in(a[i][j], A, lda, ar, ac, home, rows, kk + kn);
        tfp::frag_get_in(b[i][j], B, ldb, br, bc, home, TB ? cols : kk + kn, TB ? kk + kn : cols);
      } else {
        tfp::frag_get(a[i][j], A, lda, ar, ac, home);
        tfp::frag_get(b[i][j], B, ldb, br, bc, home);
      }
#endif
    }
  }
  TF_UNROLL
  for (short m = 0; m < 2; m++) {
    TF_UNROLL
    for (short k = 0; k < 2; k++) {
      tfp::mma_16x32<false, TB>(acc[m][0], acc[m][1], a[m][k], b[k][0], b[k][1]);
    }
  }
#if TF_CTL == 4
  TF_UNROLL
  for (short m = 0; m < 2; m++) {
    TF_UNROLL
    for (short n = 0; n < 2; n++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) acc[m][n][e] = float(half(acc[m][n][e]));
    }
  }
#endif
}

template <bool TB, bool FULL, int BK>
inline void gemm_block(
    thread tfp::frag<float> (&acc)[2][2],
    const device bfloat* A,
    const device bfloat* B,
    int lda,
    int ldb,
    int K,
    int kblocks,
    int rows,
    int cols,
    short2 home) {
  TF_UNROLL
  for (short i = 0; i < 2; i++) {
    TF_UNROLL
    for (short j = 0; j < 2; j++) acc[i][j] = tfp::frag<float>(0);
  }
  const bool live = rows > 0 && cols > 0;
#pragma clang loop unroll(disable)
  for (int kb = 0; kb < kblocks; kb++) {
    threadgroup_barrier(mem_flags::mem_none);
    if (!FULL && !live) continue;
#pragma clang loop unroll(disable)
    for (int kk = 0; kk < BK; kk += 32) {
      gemm_step<TB, !FULL>(acc, A, B, lda, ldb, kk, 32, rows, cols, home);
    }
    A += BK;
    B += TB ? BK : BK * ldb;
  }
  const int rest = K - kblocks * BK;
  if (rest > 0) {
    simdgroup_barrier(mem_flags::mem_none);
    if (!FULL && !live) return;
#pragma clang loop unroll(disable)
    for (int kk = 0; kk < rest; kk += 32) {
      gemm_step<TB, true>(acc, A, B, lda, ldb, kk, max(0, rest - kk), rows, cols, home);
    }
  }
}

template <typename O, bool FULL>
inline void gemm_store(
    thread tfp::frag<float> (&acc)[2][2],
    device O* D,
    int ldd,
    int rows,
    int cols,
    short2 home) {
#if TF_CTL == 1
  TF_UNROLL
  for (short i = 0; i < 2; i++) {
    TF_UNROLL
    for (short j = 0; j < 2; j++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        if (isnan(acc[i][j][e]) || isinf(acc[i][j][e])) acc[i][j][e] = 0.0f;
      }
    }
  }
#endif
  TF_UNROLL
  for (short i = 0; i < 2; i++) {
    TF_UNROLL
    for (short j = 0; j < 2; j++) {
      if (FULL) tfp::frag_put(acc[i][j], D, ldd, 16 * i, 16 * j, home);
      else tfp::frag_put_in(acc[i][j], D, ldd, 16 * i, 16 * j, home, rows, cols);
    }
  }
}

template <bool AM, bool AN, int BM, int BN, int BK, int WM, int WN>
inline void gemm_nax(
    const device bfloat* A,
    const device bfloat* B,
    device bfloat* D,
    constant SnBf16GemmArgs& args,
    uint3 tg,
    uint sg,
    uint lane) {
  static_assert(BM / WM == 32 && BN / WN == 32, "a simdgroup owns a 32x32 block");
  constant int32_t* P = args.p;
  const int swz = P[8];
  const int ty = (int(tg.y) << swz) + (int(tg.x) & ((1 << swz) - 1));
  const int tx = int(tg.x) >> swz;
  if (tx >= P[6] || ty >= P[7]) return;
#if TF_CTL == 2
  if (tx == 0 && ty == 0 && sg == 0 && lane == 0) ((device ushort*)A)[0] ^= 1u;
#endif
  constexpr bool TB = (TF_CTL == 3) ? false : true;
  const int lda = P[3], ldb = P[4], ldd = P[5];
  const int row = ty * BM + 32 * int(sg / WN);
  const int col = tx * BN + 32 * int(sg % WN);
  A += long(P[13]) + long(P[10]) * tg.z + long(row) * lda;
  B += long(P[14]) + long(P[11]) * tg.z + (TB ? long(col) * ldb : long(col));
  const int rows = AM ? 32 : min(32, P[0] - row);
  const int cols = AN ? 32 : min(32, P[1] - col);
  const bool full = (AM || rows == 32) && (AN || cols == 32);
  const short2 home = tfp::frag_home(ushort(lane));
  tfp::frag<float> acc[2][2];
  if (full) {
    gemm_block<TB, true, BK>(acc, A, B, lda, ldb, P[2], P[9], rows, cols, home);
  } else {
    gemm_block<TB, false, BK>(acc, A, B, lda, ldb, P[2], P[9], rows, cols, home);
  }
  if (full || (rows > 0 && cols > 0)) {
    uint col_loc, ldd_eff;
    device bfloat* D_dst = sn_bf16_split_dest(
        D, uint(ldd), args.y1_addr, args.split1, args.ldy1,
        args.y2_addr, args.split2, args.ldy2, uint(col), col_loc, ldd_eff);
    D_dst += long(P[12]) * tg.z + long(row) * int(ldd_eff) + int(col_loc);
    if (full) gemm_store<bfloat, true>(acc, D_dst, int(ldd_eff), rows, cols, home);
    else gemm_store<bfloat, false>(acc, D_dst, int(ldd_eff), rows, cols, home);
  }
}

template <bool AM, bool AN, int BM, int BN, int BK, int WM, int WN>
inline void gemm_splitk_nax(
    const device bfloat* A,
    const device bfloat* B,
    device float* C,
    const device int* P,
    uint3 tg,
    uint sg,
    uint lane) {
  static_assert(BM / WM == 32 && BN / WN == 32, "a simdgroup owns a 32x32 block");
  const int swz = P[11];
  const int span_n = P[6] << swz;
  const int span_m = (P[7] + (1 << swz) - 1) >> swz;
  const int part = int(tg.x) / (span_n * span_m);
  const int at = int(tg.x) % (span_n * span_m);
  const int gx = at % span_n, gy = at / span_n;
  const int ty = (gy << swz) + (gx & ((1 << swz) - 1)), tx = gx >> swz;
  if (tx >= P[6] || ty >= P[7]) return;
  const int lda = P[3], ldb = P[4], ldc = P[5];
  const int k0 = P[10] * part, kn = min(k0 + P[10], P[2]) - k0;
  const int row = ty * BM + 32 * int(sg / WN), col = tx * BN + 32 * int(sg % WN);
  A += long(P[12]) + long(k0) + long(row) * lda;
  B += long(P[13]) + long(k0) + long(col) * ldb;
  C += long(P[9]) * part + long(row) * ldc + col;
  const int rows = AM ? 32 : min(32, P[0] - row);
  const int cols = AN ? 32 : min(32, P[1] - col);
  const bool full = (AM || rows == 32) && (AN || cols == 32);
  const short2 home = tfp::frag_home(ushort(lane));
  tfp::frag<float> acc[2][2];
  if (full) {
    gemm_block<true, true, BK>(acc, A, B, lda, ldb, kn, kn / BK, rows, cols, home);
    gemm_store<float, true>(acc, C, ldc, rows, cols, home);
  } else {
    gemm_block<true, false, BK>(acc, A, B, lda, ldb, kn, kn / BK, rows, cols, home);
    if (rows > 0 && cols > 0) gemm_store<float, false>(acc, C, ldc, rows, cols, home);
  }
}

}  // namespace sn_prefill

#define SN_BF16_GEMM(NAME, AM, AN)                                                                               \
  [[max_total_threads_per_threadgroup(256)]]                                                                     \
  [[kernel]] void NAME(                                                                                          \
      const device bfloat* A [[buffer(0)]],                                                                      \
      const device bfloat* B [[buffer(1)]],                                                                      \
      constant SnBf16GemmArgs& P [[buffer(2)]],                                                                  \
      device bfloat* D [[buffer(3)]],                                                                            \
      uint sg [[simdgroup_index_in_threadgroup]],                                                                \
      uint lane [[thread_index_in_simdgroup]],                                                                   \
      uint3 tg [[threadgroup_position_in_grid]]) {                                                               \
    sn_prefill::gemm_nax<AM, AN, 64, 128, 256, 2, 4>(A, B, D, P, tg, sg, lane);                                  \
  }

SN_BF16_GEMM(sn_bf16_gemm_00, false, false)
SN_BF16_GEMM(sn_bf16_gemm_01, false, true)
SN_BF16_GEMM(sn_bf16_gemm_10, true, false)
SN_BF16_GEMM(sn_bf16_gemm_11, true, true)

#define SN_BF16_SPLITK(NAME, AM, AN)                                                                             \
  [[max_total_threads_per_threadgroup(128)]]                                                                     \
  [[kernel]] void NAME(                                                                                          \
      const device bfloat* A [[buffer(0)]],                                                                      \
      const device bfloat* B [[buffer(1)]],                                                                      \
      const device int32_t* P [[buffer(2)]],                                                                     \
      device float* C [[buffer(3)]],                                                                             \
      uint sg [[simdgroup_index_in_threadgroup]],                                                                \
      uint lane [[thread_index_in_simdgroup]],                                                                   \
      uint3 tg [[threadgroup_position_in_grid]]) {                                                               \
    sn_prefill::gemm_splitk_nax<AM, AN, 64, 64, 256, 2, 2>(A, B, C, P, tg, sg, lane);                            \
  }

SN_BF16_SPLITK(sn_bf16_gemm_splitk_00, false, false)
SN_BF16_SPLITK(sn_bf16_gemm_splitk_01, false, true)
SN_BF16_SPLITK(sn_bf16_gemm_splitk_10, true, false)
SN_BF16_SPLITK(sn_bf16_gemm_splitk_11, true, true)

[[max_total_threads_per_threadgroup(256)]]
[[kernel]] void sn_bf16_gemm_splitk_sum(
    const device float* C [[buffer(0)]],
    constant SnBf16SumArgs& args [[buffer(1)]],
    device bfloat* D [[buffer(2)]],
    uint2 at [[thread_position_in_grid]]) {
  if (int(at.x) >= args.p[3] || int(at.y) >= args.p[4]) return;
  const long ci = long(at.x) + long(at.y) * args.p[2];
  float s = 0.0f;
  for (int p = 0; p < args.p[0]; p++) s += C[ci + long(p) * args.p[1]];
  uint col_loc, ldy_eff;
  device bfloat* D_dst = sn_bf16_split_dest(
      D, uint(args.p[5]), args.y1_addr, args.split1, args.ldy1,
      args.y2_addr, args.split2, args.ldy2, at.x, col_loc, ldy_eff);
  D_dst[long(col_loc) + long(at.y) * int(ldy_eff)] = bfloat(s);
}
