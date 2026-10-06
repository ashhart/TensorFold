// 4-bit projections of prompt chunks, bit-identical to MLX 0.32.3's NAX qmm, its sorted-expert gather and split-K qmm.
#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;
typedef half float16_t;
#include "../nax.h"

namespace tfp {

// BYTES packed bytes of one group's row as 2 BYTES weights: bf16(s q + b) per nibble, the high one as (s / 16)(q << 4).
template <int BYTES, typename T, typename O>
inline void qmm_dequant(const device uchar* w, T scale, T bias, threadgroup O* out) {
  const float s = float(scale), b = float(bias), s_hi = s / 16.0f;
  TF_UNROLL
  for (short i = 0; i < BYTES; i++) {
    const uchar q = w[i];
    out[2 * i] = O(static_cast<T>(s * (q & 0x0f) + b));
    out[2 * i + 1] = O(static_cast<T>(s_hi * (q & 0xf0) + b));
  }
}

// x (TM 16-row fragments, ld K) times a 4-bit [64 rows, K] block, 64 deep a step: thread t dequantizes row t / 2.
template <typename T, int TM>
inline void qmm_k_loop(thread frag<float> (&acc)[TM][2], const device T* x, int K, int live, bool inside,
                       const device uchar* wq, const device T* scales, const device T* biases, threadgroup T* tile,
                       int tn, uint t, short2 home) {
  constexpr int PAD = 64 + 16 / sizeof(T);
  threadgroup T* mine = tile + (t / 2) * PAD + 32 * (t % 2);
  TF_UNROLL
  for (short i = 0; i < TM; i++) {
    acc[i][0] = frag<float>(0);
    acc[i][1] = frag<float>(0);
  }
  for (int k = 0; k < K; k += 64) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qmm_dequant<16>(wq, *scales, *biases, mine);
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(disable)
    for (int kk = 0; kk < 64; kk += 32) {
      if (live > 0) {
        frag<T> a[TM][2], b[2][2];   // a[m][k] from x, b[k][n] from the block (stored [n][k])
        TF_UNROLL
        for (short i = 0; i < 2; i++) {
          TF_UNROLL
          for (short j = 0; j < 2; j++) {
            frag_get_t(b[j][i], (const threadgroup T*)tile, PAD, tn + 16 * i, kk + 16 * j, home);
          }
        }
        TF_UNROLL
        for (short i = 0; i < TM; i++) {
          TF_UNROLL
          for (short j = 0; j < 2; j++) {
            if (inside) {
              frag_get(a[i][j], x, K, 16 * i, kk + 16 * j, home);
            } else {
              frag_get_in(a[i][j], x, K, 16 * i, kk + 16 * j, home, live, kk + 32);
            }
          }
        }
        TF_UNROLL
        for (short m = 0; m < TM; m++) {
          TF_UNROLL
          for (short j = 0; j < 2; j++) {
            mma_16x32<false, true>(acc[m][0], acc[m][1], a[m][j], b[j][0], b[j][1]);
          }
        }
      }
    }
    x += 64;
    wq += 32;
    scales++;
    biases++;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// A simdgroup's TM x 2 fragments to y (ld N), rows below live only.
template <typename T, int TM>
inline void qmm_store(thread const frag<float> (&acc)[TM][2], device T* y, int N, int live, short2 home) {
  TF_UNROLL
  for (short i = 0; i < TM; i++) {
    TF_UNROLL
    for (short j = 0; j < 2; j++) {
      if (live == 16 * TM) {
        frag_put(acc[i][j], y, N, 16 * i, 16 * j, home);
      } else {
        frag_put_in(acc[i][j], y, N, 16 * i, 16 * j, home, live, 32);
      }
    }
  }
}

// y = x W^T, W 4-bit [N, K/8] in MLX's layout (group 64): affine_qmm_t_nax with 64x64 tiles. P: K N M.
template <typename T>
inline void qmm_t_nax(const device uint32_t* w, const device T* scales, const device T* biases, const device T* x,
                      device T* y, const device int* P, threadgroup T* tile, uint3 tg, uint sg, uint lane) {
  const int K = P[0], N = P[1], M = P[2];
  const int row = int(tg.y) * 64, col = int(tg.x) * 64, t = int(sg) * 32 + int(lane);
  const int tm = 32 * int(sg / 2), tn = 32 * int(sg % 2), live = min(32, M - (row + tm));
  const long wrow = long(col + t / 2);
  frag<float> acc[2][2];
  qmm_k_loop<T, 2>(acc, x + long(row + tm) * K, K, live, live == 32, (const device uchar*)w + wrow * (K / 2) + 16 * (t % 2),
                   scales + wrow * (K / 64), biases + wrow * (K / 64), tile, tn, uint(t), frag_home(ushort(lane)));
  qmm_store<T, 2>(acc, y + long(row + tm) * N + col + tn, N, live, frag_home(ushort(lane)));
}

// offsets[e] = the first row whose expert is not below e, rows sorted by expert (gather_mm_offsets). P: rows.
inline void expert_offsets(const device uint32_t* ids, device int32_t* offsets, const device int* P, uint e) {
  int first = 0, count = P[0];
  while (count > 0) {
    const int step = count >> 1;
    if (ids[first + step] < e) {
      first += step + 1;
      count -= step + 1;
    } else {
      count = step;
    }
  }
  offsets[e] = first;
}

// The tile'th BM-row tile over experts in order (MLX's schedule_row_tile): its expert, first row and row count.
template <int BM>
inline bool expert_tile(const device int32_t* offsets, int experts, int total, int tile, uint lane, thread int& expert,
                        thread int& row, thread int& rows) {
  int before = 0;
  for (int base = 0; base < experts; base += 32) {
    const int e = base + int(lane);
    const int first = e < experts ? offsets[e] : total, last = e + 1 < experts ? offsets[e + 1] : total;
    const int count = (last - first + BM - 1) / BM;
    const int upto = before + simd_prefix_inclusive_sum(count);
    const int owner = simd_sum(int(upto <= tile));
    if (owner < 32) {
      expert = base + owner;
      row = simd_shuffle(first, ushort(owner)) + (tile - simd_shuffle(upto - count, ushort(owner))) * BM;
      rows = min(BM, simd_shuffle(last, ushort(owner)) - row);
      return true;
    }
    before = simd_shuffle(upto, ushort(31));
  }
  return false;
}

// y[r] = x[r] W_e^T for rows sorted by expert e: affine_gather_qmm_rhs_nax (bn64 bk64), BM 32 or 64. P: M N K experts.
template <typename T, int BM>
inline void gather_qmm_rhs_nax(const device T* x, const device uint32_t* w, const device T* scales,
                               const device T* biases, const device int32_t* offsets, device T* y, const device int* P,
                               threadgroup T* tile, uint3 tg, uint sg, uint lane) {
  constexpr int SM = BM / 2;
  const int M = P[0], N = P[1], K = P[2];
  int expert, row, rows;
  if (!expert_tile<BM>(offsets, P[3], M, int(tg.y), lane, expert, row, rows)) {
    return;
  }
  const int col = int(tg.x) * 64, t = int(sg) * 32 + int(lane);
  const int tm = SM * int(sg / 2), tn = 32 * int(sg % 2), live = clamp(rows - tm, 0, SM);
  const long wrow = long(expert) * N + col + t / 2;
  frag<float> acc[SM / 16][2];
  qmm_k_loop<T, SM / 16>(acc, x + long(row + tm) * K, K, live, row + tm + SM <= M,
                         (const device uchar*)w + wrow * (K / 2) + 16 * (t % 2), scales + wrow * (K / 64),
                         biases + wrow * (K / 64), tile, tn, uint(t), frag_home(ushort(lane)));
  qmm_store<T, SM / 16>(acc, y + long(row + tm) * N + col + tn, N, live, frag_home(ushort(lane)));
}

// One K partition of x W^T on a 32x32 block (affine_qmm_t_splitk): fp32 8x8 MMAs, sums to bf16. P: K N M part stride.
template <typename T>
inline void qmm_splitk_part(const device uint32_t* w, const device T* scales, const device T* biases,
                            const device T* x, device T* y, const device int* P, threadgroup float* xs,
                            threadgroup float* ws, uint3 tg, uint t, uint sg) {
  constexpr int PAD = 40;
  const int K = P[0], N = P[1], M = P[2], part = P[3];
  const int k0 = int(tg.z) * part, row = int(tg.y) * 32, col = int(tg.x) * 32;
  const int r = int(t) / 4, c = (int(t) % 4) * 8, rows = min(32, M - row);
  x += long(row + r) * K + k0 + c;
  const device uchar* wq = (const device uchar*)w + long(col + r) * (K / 2) + (k0 + c) / 2;
  scales += long(col + r) * (K / 64) + k0 / 64;
  biases += long(col + r) * (K / 64) + k0 / 64;
  const int fm = 8 * (int(sg) / 2), fn = 8 * (int(sg) % 2);
  simdgroup_matrix<float, 8, 8> acc[2][2];
  TF_UNROLL
  for (short i = 0; i < 2; i++) {
    TF_UNROLL
    for (short j = 0; j < 2; j++) {
      acc[i][j] = simdgroup_matrix<float, 8, 8>(0);
    }
  }
  for (int k = 0; k < part; k += 32) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    TF_UNROLL
    for (short i = 0; i < 8; i++) {
      xs[r * PAD + c + i] = r < rows ? float(x[k + i]) : 0.0f;
    }
    qmm_dequant<4, T, float>(wq + k / 2, scales[k / 64], biases[k / 64], ws + r * PAD + c);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    TF_UNROLL
    for (short kk = 0; kk < 32; kk += 8) {
      simdgroup_matrix<float, 8, 8> a[2], b[2];
      TF_UNROLL
      for (short i = 0; i < 2; i++) {
        simdgroup_load(a[i], xs + (fm + 16 * i) * PAD + kk, PAD);
        simdgroup_load(b[i], ws + (fn + 16 * i) * PAD + kk, PAD, ulong2(0, 0), true);
      }
      TF_UNROLL
      for (short i = 0; i < 2; i++) {
        TF_UNROLL
        for (short j = 0; j < 2; j++) {
          simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  TF_UNROLL
  for (short i = 0; i < 2; i++) {
    TF_UNROLL
    for (short j = 0; j < 2; j++) {
      simdgroup_store(acc[i][j], xs + (fm + 16 * i) * PAD + fn + 16 * j, PAD);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  y += long(tg.z) * P[4] + long(row + r) * N + col + c;
  if (r < rows) {
    TF_UNROLL
    for (short i = 0; i < 8; i++) {
      y[i] = static_cast<T>(xs[r * PAD + c + i]);
    }
  }
}

// y = the bf16 partitions summed in bf16 as col_reduce_small runs min(8, parts) lanes a column. P: parts stride count.
template <typename T>
inline void qmm_splitk_sum(const device T* parts, device T* y, const device int* P, uint i) {
  if (int(i) >= P[2]) {
    return;
  }
  const int n = P[0], lanes = min(8, n);
  T total = T(0);
  for (int lane = 0; lane < lanes; lane++) {
    T t = T(0);
    for (int p = lane; p < n; p += lanes) {
      t = parts[long(p) * P[1] + i] + t;
    }
    total = lane == 0 ? t : t + total;
  }
  y[i] = total;
}

}  // namespace tfp
// ---- entry points: generated by tools/zig/check_prefill_ops.py --write ----
[[kernel]] void custom_kernel_tf_expert_offsets_uint32_t_int32_t_int32_t(
  const device uint32_t* I [[buffer(0)]],
  const device int32_t* P [[buffer(1)]],
  device int32_t* O [[buffer(2)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
  tfp::expert_offsets(I, O, P, thread_position_in_grid.x);

}
[[kernel]] void custom_kernel_tf_gather_qmm_rhs_nax_bf16_32_bfloat16_t_uint32_t_bfloat16_t_bfloat16_t_int32_t_int32_t_bfloat16_t(
  const device bfloat16_t* X [[buffer(0)]],
  const device uint32_t* W [[buffer(1)]],
  const device bfloat16_t* S [[buffer(2)]],
  const device bfloat16_t* B [[buffer(3)]],
  const device int32_t* O [[buffer(4)]],
  const device int32_t* P [[buffer(5)]],
  device bfloat16_t* Y [[buffer(6)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  threadgroup bfloat16_t tile[64 * (64 + 16 / sizeof(bfloat16_t))];
  tfp::gather_qmm_rhs_nax<bfloat16_t, 32>(X, W, S, B, O, Y, P, tile, threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);

}
[[kernel]] void custom_kernel_tf_gather_qmm_rhs_nax_bf16_64_bfloat16_t_uint32_t_bfloat16_t_bfloat16_t_int32_t_int32_t_bfloat16_t(
  const device bfloat16_t* X [[buffer(0)]],
  const device uint32_t* W [[buffer(1)]],
  const device bfloat16_t* S [[buffer(2)]],
  const device bfloat16_t* B [[buffer(3)]],
  const device int32_t* O [[buffer(4)]],
  const device int32_t* P [[buffer(5)]],
  device bfloat16_t* Y [[buffer(6)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  threadgroup bfloat16_t tile[64 * (64 + 16 / sizeof(bfloat16_t))];
  tfp::gather_qmm_rhs_nax<bfloat16_t, 64>(X, W, S, B, O, Y, P, tile, threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);

}
[[kernel]] void custom_kernel_tf_qmm_splitk_part_uint32_t_bfloat16_t_bfloat16_t_bfloat16_t_int32_t_bfloat16_t(
  const device uint32_t* W [[buffer(0)]],
  const device bfloat16_t* S [[buffer(1)]],
  const device bfloat16_t* B [[buffer(2)]],
  const device bfloat16_t* X [[buffer(3)]],
  const device int32_t* P [[buffer(4)]],
  device bfloat16_t* Y [[buffer(5)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_threadgroup [[thread_index_in_threadgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  threadgroup float xs[32 * 40];
  threadgroup float ws[32 * 40];
  tfp::qmm_splitk_part<bfloat16_t>(W, S, B, X, Y, P, xs, ws, threadgroup_position_in_grid, thread_index_in_threadgroup, simdgroup_index_in_threadgroup);

}
[[kernel]] void custom_kernel_tf_qmm_splitk_sum_bfloat16_t_int32_t_bfloat16_t(
  const device bfloat16_t* PARTS [[buffer(0)]],
  const device int32_t* P [[buffer(1)]],
  device bfloat16_t* Y [[buffer(2)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
  tfp::qmm_splitk_sum<bfloat16_t>(PARTS, Y, P, thread_position_in_grid.x);

}
[[kernel]] void custom_kernel_tf_qmm_t_nax_bf16_uint32_t_bfloat16_t_bfloat16_t_bfloat16_t_int32_t_bfloat16_t(
  const device uint32_t* W [[buffer(0)]],
  const device bfloat16_t* S [[buffer(1)]],
  const device bfloat16_t* B [[buffer(2)]],
  const device bfloat16_t* X [[buffer(3)]],
  const device int32_t* P [[buffer(4)]],
  device bfloat16_t* Y [[buffer(5)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  threadgroup bfloat16_t tile[64 * (64 + 16 / sizeof(bfloat16_t))];
  tfp::qmm_t_nax<bfloat16_t>(W, S, B, X, Y, P, tile, threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);

}
