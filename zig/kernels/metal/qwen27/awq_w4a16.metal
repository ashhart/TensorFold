// Zero-lossless AWQ W4A16 (g128, FP32 scale+bias) decode and prefill kernels for M5 and M3.
#include <metal_stdlib>
using namespace metal;
#include "../nax.h"

#ifndef TF_CTL
#define TF_CTL 0
#endif

#if TF_CTL == 4
typedef half sn_awq_acc_t;
#else
typedef float sn_awq_acc_t;
#endif

struct SnAwqArgs {
  int M, N, K, MP, LDY;
  uint split1;
  ulong y1_addr, y2_addr;
  uint ldy1, split2, ldy2, _pad;
};

inline device bfloat* sn_awq_split_dest(
    device bfloat* Y0,
    constant SnAwqArgs& a,
    uint col,
    thread uint& col_out,
    thread uint& ldy_out) {
  if (a.y2_addr != 0ul && col >= a.split2) {
    col_out = col - a.split2;
    ldy_out = a.ldy2;
    return (device bfloat*)a.y2_addr;
  } else if (a.y1_addr != 0ul && col >= a.split1) {
    col_out = col - a.split1;
    ldy_out = a.ldy1;
    return (device bfloat*)a.y1_addr;
  }
  col_out = col;
  ldy_out = uint(a.LDY);
  return Y0;
}

inline float sn_awq_xsum_inline(const device bfloat* X, int m, int g, int M, int K) {
#pragma clang fp contract(off)
  float acc = 0.0f;
  if (m < M) {
    const device bfloat* xp = X + (size_t)m * K + g * 128;
    for (int i = 0; i < 128; i++) acc += float(xp[i]);
  }
  return acc;
}

#ifndef TF_SIMD_FRAGS
using namespace mpp::tensor_ops;

inline void sn_awq_sb_fetch(
    const device float2* SB, int g, int g_end, int N, int n0, int lane, short fn,
    thread float2 (&sb)[2][4]) {
  const bool ok = g < g_end;
  const float2 my_sb = (ok && n0 + lane < N) ? SB[(size_t)g * N + n0 + lane] : float2(0.0f);
  for (int f = 0; f < 2; f++) {
    for (int j = 0; j < 4; j++) {
      sb[f][j] = simd_shuffle(my_sb, ushort(f * 16 + fn + j));
    }
  }
}

template <int SK, bool PRECOMPUTE_SUMS, bool ROW_BLOCKS>
inline void sn_awq_lane_impl(
    const device bfloat* X,
    const device uint* W,
    const device float2* SB,
    constant SnAwqArgs& a,
    device bfloat* Y,
    const device float* XS,
    uint sgi,
    uint lanei,
    uint2 tg,
    threadgroup float* part) {
#pragma clang fp contract(off)
  constexpr int NT = 32, GS = 128, NF = 2;
  const int row0 = ROW_BLOCKS ? int(tg.y) * 16 : 0;
  int M_local = a.M;
  if constexpr (ROW_BLOCKS) {
    const int rem = a.M - row0;
    if (rem <= 0) return;
    X += (size_t)row0 * a.K;
    if constexpr (PRECOMPUTE_SUMS) XS += row0;
    M_local = min(rem, 16);
  }
  const int tile = int(tg.x);
  const int lane = int(lanei), sg = int(sgi);
  const int KG = a.K / GS, n0 = tile * NT;
  if (n0 >= a.N) return;
  const short qid = short(lane) >> 2;
  const short fm = (qid & 4) | ((short(lane) >> 1) & 3);
  const short fn = ((qid & 2) | (short(lane) & 1)) * 4;
  const int g_begin = (sg * KG) / SK;
  int g_end = ((sg + 1) * KG) / SK;
#if TF_CTL == 3
  if (g_end == KG && g_end > g_begin) g_end -= 1;
#endif
#if TF_CTL == 2
  if (tile == 0 && (!ROW_BLOCKS || tg.y == 0) && sg == 0 && lane == 0) {
    ((device ushort*)X)[0] ^= 1u;
  }
#endif

  constexpr auto desc = matmul2d_descriptor(16, NT, GS, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tA((device bfloat*)X, dextents<int32_t, 2>(a.K, M_local));

  float C[16];
  for (int i = 0; i < 16; i++) C[i] = 0.0f;

  float2 sb_cur[2][4];
  sn_awq_sb_fetch(SB, g_begin, g_end, a.N, n0, lane, fn, sb_cur);
  const ushort x_leader = ushort(lane & 22);

  for (int g = g_begin; g < g_end; g++) {
    float2 sb_next[2][4];
    sn_awq_sb_fetch(SB, g + 1, g_end, a.N, n0, lane, fn, sb_next);
    tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> b(
        (device uchar*)W + ((size_t)tile * KG + g) * (NT * GS / 2), dextents<int32_t, 2>(GS, NT));
    auto a_slice = tA.slice(g * GS, 0);
    auto P = op.template get_destination_cooperative_tensor<decltype(a_slice), decltype(b), float>();
    op.run(a_slice, b, P);

    float xs0, xs1;
    if constexpr (PRECOMPUTE_SUMS) {
      xs0 = (fm < M_local) ? XS[(size_t)g * a.MP + fm] : 0.0f;
      xs1 = (fm + 8 < M_local) ? XS[(size_t)g * a.MP + fm + 8] : 0.0f;
    } else {
      float lx0 = 0.0f, lx1 = 0.0f;
      if ((lane & 9) == 0) {
        lx0 = sn_awq_xsum_inline(X, fm, g, M_local, a.K);
        lx1 = sn_awq_xsum_inline(X, fm + 8, g, M_local, a.K);
      }
      xs0 = simd_shuffle(lx0, x_leader);
      xs1 = simd_shuffle(lx1, x_leader);
    }

    for (int f = 0; f < NF; f++) {
      for (int r = 0; r < 2; r++) {
        const float xs = r ? xs1 : xs0;
        for (int j = 0; j < 4; j++) {
          const int i = f * 8 + r * 4 + j;
#if TF_CTL == 4
          C[i] = float(half(fma(sb_cur[f][j].x, P[i], fma(sb_cur[f][j].y, xs, C[i]))));
#else
          C[i] = fma(sb_cur[f][j].x, P[i], fma(sb_cur[f][j].y, xs, C[i]));
#endif
        }
      }
    }
    for (int f = 0; f < NF; f++)
      for (int j = 0; j < 4; j++) sb_cur[f][j] = sb_next[f][j];
  }

  if constexpr (SK > 1) {
    if (sg > 0) {
      for (int i = 0; i < 16; i++) part[((sg - 1) * 16 + i) * 32 + lane] = C[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      for (int s2 = 1; s2 < SK; s2++) {
        for (int i = 0; i < 16; i++) {
#if TF_CTL == 4
          C[i] = float(half(C[i]) + half(part[((s2 - 1) * 16 + i) * 32 + lane]));
#else
          C[i] += part[((s2 - 1) * 16 + i) * 32 + lane];
#endif
        }
      }
    }
  }

  if (sg == 0) {
    uint n0_loc, ldy_eff;
    device bfloat* Y_dst = sn_awq_split_dest(Y, a, uint(n0), n0_loc, ldy_eff);
    if constexpr (ROW_BLOCKS) Y_dst += (size_t)row0 * ldy_eff;
    for (int f = 0; f < NF; f++) {
      for (int r = 0; r < 2; r++) {
        const int m = fm + 8 * r;
        const int col0_glob = n0 + f * 16 + fn;
        const int col0_loc = int(n0_loc) + f * 16 + fn;
        if (m < M_local) {
          for (int j = 0; j < 4; j++) {
            if (col0_glob + j < a.N) {
              float out_v = C[f * 8 + r * 4 + j];
#if TF_CTL == 1
              if (isnan(out_v) || isinf(out_v)) out_v = 0.0f;
#endif
              Y_dst[(size_t)m * ldy_eff + col0_loc + j] = bfloat(out_v);
            }
          }
        }
      }
    }
  }
}

#define SN_AWQ_LANE(SK, PS, RB)                                                                                  \
  [[max_total_threads_per_threadgroup(SK * 32)]]                                                                 \
  [[kernel]] void sn_awq_lane_sk##SK##_p##PS##_r##RB(                                                            \
      const device bfloat* X [[buffer(0)]],                                                                      \
      const device uint* W [[buffer(1)]],                                                                        \
      const device float2* SB [[buffer(2)]],                                                                     \
      constant SnAwqArgs& a [[buffer(3)]],                                                                       \
      device bfloat* Y [[buffer(4)]],                                                                            \
      const device float* XS [[buffer(5)]],                                                                      \
      uint sg [[simdgroup_index_in_threadgroup]],                                                                \
      uint lane [[thread_index_in_simdgroup]],                                                                   \
      uint2 tg [[threadgroup_position_in_grid]]) {                                                               \
    threadgroup float part[(SK > 1 ? SK - 1 : 1) * 16 * 32];                                                     \
    sn_awq_lane_impl<SK, PS != 0, RB != 0>(X, W, SB, a, Y, XS, sg, lane, tg, part);                              \
  }

#define SN_AWQ_LANE_SK(SK) \
  SN_AWQ_LANE(SK, 0, 0)    \
  SN_AWQ_LANE(SK, 0, 1)    \
  SN_AWQ_LANE(SK, 1, 0)    \
  SN_AWQ_LANE(SK, 1, 1)

SN_AWQ_LANE_SK(1)
SN_AWQ_LANE_SK(2)
SN_AWQ_LANE_SK(4)
SN_AWQ_LANE_SK(8)
#endif

// M3 (and universal fallback) coalesced AWQ W4A16 decode/verify kernel: [N][K/8] u32, [N][KG] float2.
#define SN_AWQ_COLS(R) (((R) >= 2u && (R) <= 8u) ? 2u : 1u)

template <uint ROWS, uint SK>
inline void sn_awq_row_mv_impl(
    const device bfloat* X,
    const device uint* W,
    const device float2* SB,
    constant SnAwqArgs& a,
    device bfloat* Y,
    uint3 tg,
    uint sg,
    uint lane,
    threadgroup float* red) {
#pragma clang fp contract(off)
  constexpr uint COLS = SN_AWQ_COLS(ROWS);
  constexpr uint SG = 8u;
  constexpr uint GROUPS = SG / SK;
  constexpr uint NO = GROUPS * COLS;
  const uint g_idx = sg / SK;
  const uint s = sg % SK;
  const uint j0 = tg.x * NO + g_idx * COLS;
  const uint KG = uint(a.K) >> 7;
  const uint num_u4 = uint(a.K) >> 5;
  const uint end_u4 = (TF_CTL == 3 && num_u4 > 0u) ? (num_u4 - 1u) : num_u4;
  const device uint4* X4 = (const device uint4*)X;

  sn_awq_acc_t acc[COLS][ROWS];
  _Pragma("clang loop unroll(full)")
  for (uint col = 0; col < COLS; ++col) {
    _Pragma("clang loop unroll(full)")
    for (uint r = 0; r < ROWS; ++r) acc[col][r] = sn_awq_acc_t(0);
  }

  if (int(j0) < a.N) {
#if TF_CTL == 2
    if (tg.x == 0 && sg == 0 && lane == 0) {
      ((device ushort*)X)[0] ^= 1u;
    }
#endif
    const device uint4* W_ptr[COLS];
    const device float2* SB_ptr[COLS];
    _Pragma("clang loop unroll(full)")
    for (uint col = 0; col < COLS; ++col) {
      const uint jj = (int(j0 + col) < a.N) ? (j0 + col) : 0u;
      W_ptr[col] = (const device uint4*)(W + (size_t)jj * (uint(a.K) >> 3));
      SB_ptr[col] = SB + (size_t)jj * KG;
    }
    for (uint idx = s * 32u + lane; idx < end_u4; idx += SK * 32u) {
      uint4 wq[COLS];
      float2 sb[COLS];
      _Pragma("clang loop unroll(full)")
      for (uint col = 0; col < COLS; ++col) {
        wq[col] = W_ptr[col][idx];
        sb[col] = SB_ptr[col][idx >> 2];
      }
      _Pragma("clang loop unroll(full)")
      for (uint u = 0; u < 4u; ++u) {
        vec<sn_awq_acc_t, 4> wlo[COLS], whi[COLS];
        _Pragma("clang loop unroll(full)")
        for (uint col = 0; col < COLS; ++col) {
          const uint word = wq[col][u];
          const float scale = sb[col].x, bias = sb[col].y;
          wlo[col] = vec<sn_awq_acc_t, 4>(
              fma(float(word & 0xfu), scale, bias), fma(float((word >> 4) & 0xfu), scale, bias),
              fma(float((word >> 8) & 0xfu), scale, bias), fma(float((word >> 12) & 0xfu), scale, bias));
          whi[col] = vec<sn_awq_acc_t, 4>(
              fma(float((word >> 16) & 0xfu), scale, bias), fma(float((word >> 20) & 0xfu), scale, bias),
              fma(float((word >> 24) & 0xfu), scale, bias), fma(float((word >> 28) & 0xfu), scale, bias));
        }
        const size_t x_off = ((size_t)idx * 4u) + u;
        _Pragma("clang loop unroll(full)")
        for (uint r = 0; r < ROWS; ++r) {
          const uint rr = (int(r) < a.M) ? r : 0u;
          const uint4 xv = X4[(size_t)rr * (num_u4 * 4u) + x_off];
          const vec<sn_awq_acc_t, 4> xlo = vec<sn_awq_acc_t, 4>(as_type<bfloat4>(xv.xy));
          const vec<sn_awq_acc_t, 4> xhi = vec<sn_awq_acc_t, 4>(as_type<bfloat4>(xv.zw));
          _Pragma("clang loop unroll(full)")
          for (uint col = 0; col < COLS; ++col) acc[col][r] += dot(wlo[col], xlo) + dot(whi[col], xhi);
        }
      }
    }
  }

  if constexpr (SK == 1u) {
    _Pragma("clang loop unroll(full)")
    for (uint col = 0; col < COLS; ++col) {
      if (int(j0 + col) < a.N) {
        uint j_loc, ldy_eff;
        device bfloat* Y_dst = sn_awq_split_dest(Y, a, j0 + col, j_loc, ldy_eff);
        _Pragma("clang loop unroll(full)")
        for (uint r = 0; r < ROWS; ++r) {
          float v = float(simd_sum(acc[col][r]));
          if (lane == 0 && int(r) < a.M) {
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
      if (int(j0 + col) < a.N && lane == 0 && int(r) < a.M) {
        red[((g_idx * COLS + col) * SK + s) * ROWS + r] = v;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (s == 0) {
    _Pragma("clang loop unroll(full)")
    for (uint col = 0; col < COLS; ++col) {
      if (int(j0 + col) < a.N) {
        uint j_loc, ldy_eff;
        device bfloat* Y_dst = sn_awq_split_dest(Y, a, j0 + col, j_loc, ldy_eff);
        for (uint r = lane; int(r) < a.M; r += 32u) {
          sn_awq_acc_t total = sn_awq_acc_t(0);
          _Pragma("clang loop unroll(full)")
          for (uint k = 0; k < SK; ++k) total += sn_awq_acc_t(red[((g_idx * COLS + col) * SK + k) * ROWS + r]);
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

#define SN_AWQ_ROW_MV(ROWS, SK)                                                                                  \
  [[max_total_threads_per_threadgroup(256)]]                                                                     \
  [[kernel]] void sn_awq_row_mv_r##ROWS##_sk##SK(                                                                \
      const device bfloat* X [[buffer(0)]],                                                                      \
      const device uint* W [[buffer(1)]],                                                                        \
      const device float2* SB [[buffer(2)]],                                                                     \
      constant SnAwqArgs& a [[buffer(3)]],                                                                       \
      device bfloat* Y [[buffer(4)]],                                                                            \
      uint3 tg [[threadgroup_position_in_grid]],                                                                 \
      uint sg [[simdgroup_index_in_threadgroup]],                                                                \
      uint lane [[thread_index_in_simdgroup]]) {                                                                 \
    threadgroup float red[SK > 1 ? 8 * SN_AWQ_COLS(ROWS) * ROWS : 1];                                            \
    sn_awq_row_mv_impl<ROWS, SK>(X, W, SB, a, Y, tg, sg, lane, red);                                             \
  }

#define SN_AWQ_ROW_MV_R(ROWS) \
  SN_AWQ_ROW_MV(ROWS, 1)      \
  SN_AWQ_ROW_MV(ROWS, 2)      \
  SN_AWQ_ROW_MV(ROWS, 4)      \
  SN_AWQ_ROW_MV(ROWS, 8)

SN_AWQ_ROW_MV_R(1)
SN_AWQ_ROW_MV_R(2)
SN_AWQ_ROW_MV_R(4)
SN_AWQ_ROW_MV_R(8)
SN_AWQ_ROW_MV_R(16)
SN_AWQ_ROW_MV_R(32)

inline void sn_qmm_m3_load(
    const device uint* W, const device float2* SB, int n_row, int N, int KG, int kw, int half_k,
    int step, thread uint2& wq, thread float2& sb) {
  if (n_row < N) {
    const int g = step >> 2, sub = step & 3, k_base = (g << 7) + (sub << 5);
    sb = SB[(size_t)n_row * KG + g];
    wq = ((const device uint2*)(W + (size_t)n_row * kw + (k_base >> 3)))[half_k];
  } else {
    sb = float2(0.0f);
    wq = uint2(0u);
  }
}

inline void sn_qmm_m3_store(threadgroup float* Bs_buf, int n_row, int N, int r_loc, int half_k, uint2 wq, float2 sb) {
  threadgroup float* dst = Bs_buf + r_loc * 36 + half_k * 16;
  if (n_row < N) {
    TF_UNROLL
    for (int u = 0; u < 2; u++) {
      const uint word = (u == 0) ? wq.x : wq.y;
      TF_UNROLL
      for (int e = 0; e < 8; e++) {
        dst[u * 8 + e] = fma(float((word >> (4 * e)) & 0xfu), sb.x, sb.y);
      }
    }
  } else {
    TF_UNROLL
    for (int e = 0; e < 16; e++) dst[e] = 0.0f;
  }
}

// M3 prefill QMM (M > 32): double-buffered cooperative dequant of 64x32 weight tiles, mma_16x32 via nax.h.
[[max_total_threads_per_threadgroup(128)]]
[[kernel]] void sn_awq_qmm_m3(
    const device bfloat* X [[buffer(0)]],
    const device uint* W [[buffer(1)]],
    const device float2* SB [[buffer(2)]],
    constant SnAwqArgs& a [[buffer(3)]],
    device bfloat* Y [[buffer(4)]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 tg [[threadgroup_position_in_grid]]) {
#pragma clang fp contract(off)
  const int n0 = int(tg.x) * 64, m0 = int(tg.y) * 64;
  if (m0 >= a.M || n0 >= a.N) return;
#if TF_CTL == 2
  if (tg.x == 0 && tg.y == 0 && sg == 0 && lane == 0) ((device ushort*)X)[0] ^= 1u;
#endif
  threadgroup float Bs[2][64 * 36];
  const uint tid = sg * 32u + lane;
  const int r_loc = int(tid >> 1), half_k = int(tid & 1u);
  const int n_row = n0 + r_loc;
  const int KG = a.K / 128, kw = a.K / 8;
  const int sg_m = int(sg >> 1), sg_n = int(sg & 1u);
  const int row_sg = m0 + 32 * sg_m, col_sg = n0 + 32 * sg_n;
  const int rows = min(32, a.M - row_sg), cols = min(32, a.N - col_sg);
  const bool sg_live = (rows > 0 && cols > 0);
  const bool full = (rows == 32 && cols == 32);
  const short2 home = tfp::frag_home(ushort(lane));

  tfp::frag<float> acc[2][2];
  TF_UNROLL
  for (short i = 0; i < 2; i++)
    TF_UNROLL
    for (short j = 0; j < 2; j++) acc[i][j] = tfp::frag<float>(0);

  const int total_steps = (TF_CTL == 3 && KG > 0) ? (KG * 4 - 1) : (KG * 4);
  if (total_steps > 0) {
    uint2 wq0;
    float2 sb0;
    sn_qmm_m3_load(W, SB, n_row, a.N, KG, kw, half_k, 0, wq0, sb0);
    sn_qmm_m3_store(Bs[0], n_row, a.N, r_loc, half_k, wq0, sb0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (int step = 0; step < total_steps; step++) {
    uint2 wq_next;
    float2 sb_next;
    const bool has_next = (step + 1 < total_steps);
    if (has_next) sn_qmm_m3_load(W, SB, n_row, a.N, KG, kw, half_k, step + 1, wq_next, sb_next);
    if (sg_live) {
      const int k_base = step * 32;
      const threadgroup float* Bs_cur = Bs[step & 1];
      const device bfloat* Ap = X + (size_t)row_sg * a.K + k_base;
      tfp::frag<float> af[2][2], bf[2][2];
      TF_UNROLL
      for (short i = 0; i < 2; i++) {
        TF_UNROLL
        for (short j = 0; j < 2; j++) {
          if (full) {
            tfp::frag_get(af[i][j], Ap, a.K, 16 * i, 16 * j, home);
            tfp::frag_get_t(bf[i][j], Bs_cur, 36, 32 * sg_n + 16 * j, 16 * i, home);
          } else {
            tfp::frag_get_in(af[i][j], Ap, a.K, 16 * i, 16 * j, home, rows, a.K - k_base);
            tfp::frag_get_t(bf[i][j], Bs_cur, 36, 32 * sg_n + 16 * j, 16 * i, home);
          }
        }
      }
      TF_UNROLL
      for (short m = 0; m < 2; m++)
        TF_UNROLL
        for (short k2 = 0; k2 < 2; k2++) {
          tfp::mma_16x32<false, true>(acc[m][0], acc[m][1], af[m][k2], bf[k2][0], bf[k2][1]);
        }
#if TF_CTL == 4
      TF_UNROLL
      for (short m = 0; m < 2; m++)
        TF_UNROLL
        for (short n = 0; n < 2; n++)
          TF_UNROLL
          for (short e = 0; e < 8; e++) acc[m][n][e] = float(half(acc[m][n][e]));
#endif
    }
    if (has_next) sn_qmm_m3_store(Bs[(step + 1) & 1], n_row, a.N, r_loc, half_k, wq_next, sb_next);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (sg_live) {
#if TF_CTL == 1
    TF_UNROLL
    for (short m = 0; m < 2; m++)
      TF_UNROLL
      for (short n = 0; n < 2; n++)
        TF_UNROLL
        for (short e = 0; e < 8; e++) {
          if (isnan(acc[m][n][e]) || isinf(acc[m][n][e])) acc[m][n][e] = 0.0f;
        }
#endif
    uint col_loc, ldy_eff;
    device bfloat* Y_dst = sn_awq_split_dest(Y, a, uint(col_sg), col_loc, ldy_eff);
    device bfloat* Yp = Y_dst + (size_t)row_sg * ldy_eff + col_loc;
    TF_UNROLL
    for (short i = 0; i < 2; i++)
      TF_UNROLL
      for (short j = 0; j < 2; j++) {
        if (full) tfp::frag_put(acc[i][j], Yp, int(ldy_eff), 16 * i, 16 * j, home);
        else tfp::frag_put_in(acc[i][j], Yp, int(ldy_eff), 16 * i, 16 * j, home, rows, cols);
      }
  }
}

// Deterministic [KG][MP] FP32 group sums for GS=128, matching sn_awq_xsum_inline bit-for-bit.
[[max_total_threads_per_threadgroup(256)]]
[[kernel]] void sn_awq_xsum_128(
    const device bfloat* X [[buffer(0)]],
    constant int4& dims [[buffer(1)]],
    device float* XS [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  const int M = dims.x, MP = dims.y, K = dims.z;
  const int KG = K / 128;
  const int m = int(index) % MP, g = int(index) / MP;
  if (g >= KG) return;
  XS[index] = sn_awq_xsum_inline(X, m, g, M, K);
}

// Optional runtime input scaling for AWQ smooth input_scale: bf16(f32(x) / input_scale[k]).
[[max_total_threads_per_threadgroup(256)]]
[[kernel]] void sn_awq_scale_input(
    const device bfloat* X [[buffer(0)]],
    const device float* scale [[buffer(1)]],
    constant int2& dims [[buffer(2)]],
    device bfloat* Y [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (int(gid.x) >= dims.y || int(gid.y) >= dims.x) return;
  const size_t idx = (size_t)gid.y * dims.y + gid.x;
  Y[idx] = bfloat(float(X[idx]) / scale[gid.x]);
}

// Optional row-major int4 embedding lookup: W [vocab, K/8] sequential nibbles, SB [vocab, K/128] float2.
[[max_total_threads_per_threadgroup(256)]]
[[kernel]] void sn_awq_int4_embed(
    const device uint* ids [[buffer(0)]],
    const device uint* W [[buffer(1)]],
    const device float2* SB [[buffer(2)]],
    constant int4& dims [[buffer(3)]],
    device bfloat* Y [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  const int rows = dims.x, K = dims.y, vocab = dims.z;
  if (int(gid.x) >= K || int(gid.y) >= rows) return;
  const uint id = ids[gid.y];
  if (int(id) >= vocab) {
    Y[(size_t)gid.y * K + gid.x] = bfloat(0.0f);
    return;
  }
  const int k = int(gid.x);
  const uint word = W[(size_t)id * (K / 8) + (k >> 3)];
  const uint q = (word >> (4 * (k & 7))) & 0xfu;
  const float2 sb = SB[(size_t)id * (K / 128) + (k >> 7)];
  Y[(size_t)gid.y * K + k] = bfloat(fma(float(q), sb.x, sb.y));
}
