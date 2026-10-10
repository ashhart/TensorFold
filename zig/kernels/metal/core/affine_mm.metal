// x W^T for MLX affine W on the tensor units: exact codes (4-bit ones plus 128) in the MMAs, fp32 group sums scaled, bias times fp32 row sums.
#include <metal_stdlib>
using namespace metal;
#include "../nax.h"
using namespace tfp;

// Defines: TF_BITS (4 or 8), TF_GROUP (32 or 64), TF_OUT_T (bfloat, or float for partial sums another pass adds).
#ifndef TF_OUT_T
#define TF_OUT_T bfloat
#endif
constant constexpr int STEP_BYTES = 8 * TF_BITS; // a weight row's bytes in one 64-deep step
constant constexpr int HALF_BYTES = 4 * TF_BITS; // a thread's half of them (32 codes)
constant constexpr int NG = 64 / TF_GROUP;       // groups a step
constant constexpr int PAD = 64 + 8;              // the code tile's row pitch (bf16)
constant constexpr float OFF = TF_BITS == 4 ? 128.0f : 0.0f; // bf16(128 + q) is q's bits ORed with 0x4300; bias - 128 scale undoes it

struct MmArgs {
  int rows, n, k, experts; // rows (sorted by expert for a gather), outputs a row, depth, experts (gather)
};

// A dense matmul's pitches, and each batch's start in x, y (elements), the sums and the weights (rows).
struct MmStrides {
  int x_row, y_row, sums_row, x_batch, y_batch, sums_batch, w_batch, pad;
};

// Each row's fp32 sum over each group of x (row pitch a.n, or a.k when 0), in order: the bias's operand.
[[kernel]] void tf_affine_row_sums(const device bfloat* X [[buffer(0)]], constant MmArgs& a [[buffer(5)]],
                                   device float* XS [[buffer(6)]], uint2 gid [[thread_position_in_grid]]) {
  const int g = int(gid.x), r = int(gid.y), groups = a.k / TF_GROUP;
  if (g >= groups || r >= a.rows) return;
  const device bfloat* x = X + long(r) * (a.n > 0 ? a.n : a.k) + g * TF_GROUP;
  float sum = 0.0f;
  for (int i = 0; i < TF_GROUP; i++) {
    sum += float(x[i]);
  }
  XS[long(r) * groups + g] = sum;
}

// (lo | hi) = a * (b0 | b1): a group's first 16-deep slice, its destination overwritten.
inline void am_first(thread frag<float>& lo, thread frag<float>& hi, thread const frag<bfloat>& a,
                     thread const frag<bfloat>& b0, thread const frag<bfloat>& b1) {
#ifndef TF_SIMD_FRAGS
  using namespace mpp::tensor_ops;
  constexpr auto shape = matmul2d_descriptor(16, 32, 16, false, true, true, matmul2d_descriptor::mode::multiply);
  matmul2d<shape, execution_simdgroup> op;
  auto left = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
  auto right = op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  auto acc = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,
                                                            metal::remove_addrspace_t<decltype(right)>, float>();
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    left[e] = a[e];
    right[e] = b0[e];
    right[8 + e] = b1[e];
  }
  op.run(left, right, acc);
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    lo[e] = acc[e];
    hi[e] = acc[8 + e];
  }
#else
  lo = frag<float>(0);
  hi = frag<float>(0);
  mma_16x32<false, true>(lo, hi, a, b0, b1);
#endif
}

// A thread's 32 codes of one weight row in a step (16-byte loads).
struct Codes {
  uint4 v[TF_BITS / 4];
};

inline Codes am_load(const device uchar* w) {
  Codes c;
  TF_UNROLL
  for (short i = 0; i < TF_BITS / 4; i++) {
    c.v[i] = ((const device uint4*)w)[i];
  }
  return c;
}

// The codes plus OFF, exactly, as bf16 into the tile row (16-byte stores).
inline void am_codes(thread const Codes& c, threadgroup bfloat* out) {
  threadgroup uint4* o = (threadgroup uint4*)out;
#if TF_BITS == 4
  TF_UNROLL
  for (short i = 0; i < 4; i++) {
    const uint w = c.v[0][i];
    const uint lo = w & 0x0f0f0f0fu, hi = (w >> 4) & 0x0f0f0f0fu;
    o[i] = uint4(0x43004300u | (lo & 0xffu) | ((hi & 0xffu) << 16),
                 0x43004300u | ((lo >> 8) & 0xffu) | (((hi >> 8) & 0xffu) << 16),
                 0x43004300u | ((lo >> 16) & 0xffu) | (((hi >> 16) & 0xffu) << 16),
                 0x43004300u | (lo >> 24) | ((hi >> 24) << 16));
  }
#else
  TF_UNROLL
  for (short i = 0; i < 8; i++) {
    const float4 f = float4(as_type<uchar4>(c.v[i / 4][i % 4]));
    o[i / 2][2 * (i % 2)] = as_type<uint>(ushort2(as_type<ushort>(bfloat(f.x)), as_type<ushort>(bfloat(f.y))));
    o[i / 2][2 * (i % 2) + 1] = as_type<uint>(ushort2(as_type<ushort>(bfloat(f.z)), as_type<ushort>(bfloat(f.w))));
  }
#endif
}

// x (TM 16-row fragments, `live` rows) times a [64, K] code block; each group's sums scaled plus its bias times `xs`.
template <int TM>
inline void am_k_loop(thread frag<float> (&acc)[TM][2], const device bfloat* x, int ldx, int K, int live, bool inside,
                      const device float* xs, int ldxs, const device uchar* wq, const device bfloat* scales,
                      const device bfloat* biases, threadgroup bfloat* tile, threadgroup float* sb, int tn, uint t,
                      short2 home) {
  threadgroup bfloat* mine = tile + (t / 2) * PAD + 32 * (t % 2);
  const int g_mine = TF_GROUP == 32 ? int(t % 2) : 0; // at g32 each thread of a pair takes one group's scale and bias
  TF_UNROLL
  for (short i = 0; i < TM; i++) {
    acc[i][0] = frag<float>(0);
    acc[i][1] = frag<float>(0);
  }
  for (int k = 0; k < K; k += 64) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    am_codes(am_load(wq), mine);
    if (TF_GROUP == 32 || t % 2 == 0) {
      const float s = float(scales[g_mine]);
      sb[g_mine * 128 + t / 2] = s;
      sb[g_mine * 128 + 64 + t / 2] = float(biases[g_mine]) - OFF * s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (live > 0) {
      frag<float> part[TM][2];
      TF_UNROLL
      for (short q = 0; q < 4; q++) { // the step's four 16-deep slices, every row fragment's chain interleaved
        frag<bfloat> b0, b1, a[TM];
        frag_get_t(b0, (const threadgroup bfloat*)tile, PAD, tn, 16 * q, home);
        frag_get_t(b1, (const threadgroup bfloat*)tile, PAD, tn + 16, 16 * q, home);
        TF_UNROLL
        for (short m = 0; m < TM; m++) {
          if (inside) {
            frag_get(a[m], x, ldx, 16 * m, 16 * q, home);
          } else {
            frag_get_in(a[m], x, ldx, 16 * m, 16 * q, home, live, 16 * q + 16);
          }
        }
        TF_UNROLL
        for (short m = 0; m < TM; m++) {
          if (q % (TF_GROUP / 16) == 0) {
            am_first(part[m][0], part[m][1], a[m], b0, b1);
          } else {
            mma_16x32<false, true>(part[m][0], part[m][1], a[m], b0, b1);
          }
        }
        if ((q + 1) % (TF_GROUP / 16) == 0) { // a group's sums done: its scale, its bias times the row sums
          const int g = q / (TF_GROUP / 16);
          const threadgroup float* sg = sb + g * 128;
          TF_UNROLL
          for (short m = 0; m < TM; m++) {
            float xr[2];
            TF_UNROLL
            for (short h = 0; h < 2; h++) {
              const int r = 16 * m + home.y + 8 * h;
              xr[h] = r < live ? xs[long(r) * ldxs + k / TF_GROUP + g] : 0.0f;
            }
            TF_UNROLL
            for (short j = 0; j < 2; j++) {
              TF_UNROLL
              for (short c = 0; c < 4; c++) {
                const int col = tn + 16 * j + home.x + TF_COL(c);
                const float s = sg[col], b = sg[64 + col];
                acc[m][j][c] += s * part[m][j][c] + b * xr[0];
                acc[m][j][4 + c] += s * part[m][j][4 + c] + b * xr[1];
              }
            }
          }
        }
      }
    }
    x += 64;
    wq += STEP_BYTES;
    scales += NG;
    biases += NG;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// A simdgroup's TM x 2 fragments to rows dst[i] of y (ld N), rows below live only: rows put back in another order.
template <int TM>
inline void am_scatter(thread const frag<float> (&acc)[TM][2], device TF_OUT_T* y, const device int32_t* dst, int N,
                       int col, int live, short2 home) {
  TF_UNROLL
  for (short i = 0; i < TM; i++) {
    TF_UNROLL
    for (short e = 0; e < 8; e++) {
      const int r = 16 * i + home.y + (e >> 2) * 8;
      if (r < live) {
        device TF_OUT_T* yr = y + long(dst[r]) * N + col + home.x + TF_COL(e);
        yr[0] = TF_OUT_T(acc[i][0][e]);
        yr[16] = TF_OUT_T(acc[i][1][e]);
      }
    }
  }
}

template <int TM>
inline void am_store(thread const frag<float> (&acc)[TM][2], device TF_OUT_T* out, int N, int live, short2 home) {
  TF_UNROLL
  for (short i = 0; i < TM; i++) {
    TF_UNROLL
    for (short j = 0; j < 2; j++) {
      if (live == 16 * TM) {
        frag_put(acc[i][j], out, N, 16 * i, 16 * j, home);
      } else {
        frag_put_in(acc[i][j], out, N, 16 * i, 16 * j, home, live, 32);
      }
    }
  }
}

// The tile'th BM-row tile over experts in order: its expert, first row and row count (false past the last tile).
template <int BM>
inline bool am_tile(const device int32_t* offsets, int experts, int total, int tile, uint lane, thread int& expert,
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

// Rows [row, row + rows) of x times weight rows [wrow0, wrow0 + 64) of a [*, K] matrix into y (ld N) at column col.
template <int BM>
inline void am_block(const device bfloat* x, int ldx, const device float* xs, int ldxs, int row, int rows, int M, int K,
                     int ldy, int col, long wrow0, const device uint32_t* w, const device bfloat* scales,
                     const device bfloat* biases, device TF_OUT_T* y, threadgroup bfloat* tile, threadgroup float* sb,
                     uint sg, uint lane, const device int32_t* dst = nullptr) {
  constexpr int SM = BM / 2;
  constexpr int TM = SM / 16;
  const int t = int(sg) * 32 + int(lane);
  const int tm = SM * int(sg / 2), tn = 32 * int(sg % 2), live = clamp(rows - tm, 0, SM);
  const long wrow = wrow0 + t / 2;
  const short2 home = frag_home(ushort(lane));
  frag<float> acc[TM][2];
  am_k_loop<TM>(acc, x + long(row + tm) * ldx, ldx, K, live, row + tm + SM <= M, xs + long(row + tm) * ldxs, ldxs,
                (const device uchar*)w + wrow * (K * TF_BITS / 8) + HALF_BYTES * (t % 2),
                scales + wrow * (K / TF_GROUP), biases + wrow * (K / TF_GROUP), tile, sb, tn, uint(t), home);
  if (dst != nullptr) {
    if (live > 0) am_scatter<TM>(acc, y, dst + row + tm, ldy, col + tn, live, home);
  } else {
    am_store<TM>(acc, y + long(row + tm) * ldy + col + tn, ldy, live, home);
  }
}

// y [rows, n] = x [rows, k] W^T, n a multiple of 64: 64x64 tiles, a batch along tg.z (MmStrides).
[[kernel]] void tf_affine_mm(const device bfloat* X [[buffer(0)]], const device uint32_t* W [[buffer(1)]],
                             const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]],
                             const device float* XS [[buffer(4)]], constant MmArgs& a [[buffer(5)]],
                             device TF_OUT_T* Y [[buffer(6)]], constant MmStrides& st [[buffer(7)]],
                             uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                             uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup bfloat tile[64 * PAD];
  threadgroup float sb[NG * 128];
  const int row = int(tg.y) * 64, col = int(tg.x) * 64, b = int(tg.z);
  am_block<64>(X + long(b) * st.x_batch, st.x_row, XS + long(b) * st.sums_batch * (a.k / TF_GROUP),
               st.sums_row * (a.k / TF_GROUP), row, min(64, a.rows - row), a.rows, a.k, st.y_row, col,
               long(b) * st.w_batch + col, W, S, B, Y + long(b) * st.y_batch, tile, sb, sg, lane);
}

#define TF_GATHER(BM)                                                                                                   \
  [[kernel]] void tf_affine_gather_##BM(                                                                                \
      const device bfloat* X [[buffer(0)]], const device uint32_t* W [[buffer(1)]],                                    \
      const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]],                                      \
      const device float* XS [[buffer(4)]], constant MmArgs& a [[buffer(5)]], device TF_OUT_T* Y [[buffer(6)]],       \
      const device int32_t* OFFS [[buffer(7)]], uint sg [[simdgroup_index_in_threadgroup]],                           \
      uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {                           \
    threadgroup bfloat tile[64 * PAD];                                                                                 \
    threadgroup float sb[NG * 128];                                                                                    \
    int expert, row, rows;                                                                                             \
    if (!am_tile<BM>(OFFS, a.experts, a.rows, int(tg.y), lane, expert, row, rows)) return;                            \
    const int col = int(tg.x) * 64;                                                                                    \
    am_block<BM>(X, a.k, XS, a.k / TF_GROUP, row, rows, a.rows, a.k, a.n, col, long(expert) * a.n + col, W, S, B, Y,   \
                 tile, sb, sg, lane);                                                                                  \
  }
TF_GATHER(32)
TF_GATHER(64)

// The gathers with each sorted row's output put back at row DST[row] (e.g. the (row, slot) pairs' own order).
#define TF_GATHER_SCATTER(BM)                                                                                           \
  [[kernel]] void tf_affine_gather_scatter_##BM(                                                                        \
      const device bfloat* X [[buffer(0)]], const device uint32_t* W [[buffer(1)]],                                    \
      const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]],                                      \
      const device float* XS [[buffer(4)]], constant MmArgs& a [[buffer(5)]], device TF_OUT_T* Y [[buffer(6)]],       \
      const device int32_t* OFFS [[buffer(7)]], const device int32_t* DST [[buffer(8)]],                              \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],                            \
      uint3 tg [[threadgroup_position_in_grid]]) {                                                                     \
    threadgroup bfloat tile[64 * PAD];                                                                                 \
    threadgroup float sb[NG * 128];                                                                                    \
    int expert, row, rows;                                                                                             \
    if (!am_tile<BM>(OFFS, a.experts, a.rows, int(tg.y), lane, expert, row, rows)) return;                            \
    const int col = int(tg.x) * 64;                                                                                    \
    am_block<BM>(X, a.k, XS, a.k / TF_GROUP, row, rows, a.rows, a.k, a.n, col, long(expert) * a.n + col, W, S, B, Y,   \
                 tile, sb, sg, lane, DST);                                                                             \
  }
TF_GATHER_SCATTER(32)
TF_GATHER_SCATTER(64)
