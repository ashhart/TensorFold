// x W^T for affine-quantized W (MLX's layout) on the tensor units, dense or over rows sorted by expert, with the row
// kernels' arithmetic: the MMAs take the integer codes exactly (bf16 holds 0..255), each group's fp32 sums get the
// group's scale, and its bias meets the row's sum over the group (tf_affine_row_sums: at 4 bits added four at a time
// in bf16, as the row kernels add them). No weight is rounded, so the sums differ from the row kernels' only in fp32
// order. A 64-column block a threadgroup (4 simdgroups); dense tiles of 64 rows, expert tiles of BM rows of one expert.
// Defines: TF_BITS (4 or 8), TF_GROUP (32 or 64), TF_OUT_T (bfloat, or float: partial sums another pass adds).
#include <metal_stdlib>
using namespace metal;
#include "../nax.h"
using namespace tfp;

#ifndef TF_OUT_T
#define TF_OUT_T bfloat
#endif
constant constexpr int STEP_BYTES = 8 * TF_BITS; // a weight row's bytes in one 64-deep step
constant constexpr int HALF_BYTES = 4 * TF_BITS; // a thread's half of them (32 codes)
constant constexpr int NG = 64 / TF_GROUP;       // groups a step
constant constexpr int PAD = 64 + 8;              // the code tile's row pitch (bf16)

struct MmArgs {
  int rows, n, k, experts; // rows (sorted by expert for a gather), outputs a row, depth, experts (gather)
};

// Each row's sum over each group [rows, k / TF_GROUP], as the row kernels take it: at 4 bits in chains of four, each
// add rounded to bf16, the chains added in fp32.
[[kernel]] void tf_affine_row_sums(const device bfloat* X [[buffer(0)]], constant MmArgs& a [[buffer(5)]],
                                   device float* XS [[buffer(6)]], uint2 gid [[thread_position_in_grid]]) {
  const int g = int(gid.x), r = int(gid.y), groups = a.k / TF_GROUP;
  if (g >= groups || r >= a.rows) return;
  const device bfloat* x = X + long(r) * a.k + g * TF_GROUP;
  float sum = 0.0f;
#if TF_BITS == 4
  for (int i = 0; i < TF_GROUP; i += 4) {
    const bfloat p = x[i], q = x[i + 1], u = x[i + 2], v = x[i + 3];
    sum += float(bfloat(float(bfloat(float(bfloat(float(p) + float(q))) + float(u))) + float(v)));
  }
#else
  for (int i = 0; i < TF_GROUP; i++) {
    sum += float(x[i]);
  }
#endif
  XS[long(r) * groups + g] = sum;
}

// A thread's 32 codes of one weight row in this step, exactly, as bf16.
inline void am_codes(const device uchar* w, threadgroup bfloat* out) {
#if TF_BITS == 4
  TF_UNROLL
  for (short i = 0; i < 16; i++) {
    const uchar q = w[i];
    out[2 * i] = bfloat(q & 0x0f);
    out[2 * i + 1] = bfloat(q >> 4);
  }
#else
  TF_UNROLL
  for (short i = 0; i < 32; i++) {
    out[i] = bfloat(w[i]);
  }
#endif
}

// x (TM 16-row fragments, ld K; `live` rows) times a [64 rows, K] code block: thread t holds half of weight row t / 2
// each step; each group's sums go into `acc` with the group's scale (sb), and its bias times the row sums `xs`.
template <int TM>
inline void am_k_loop(thread frag<float> (&acc)[TM][2], const device bfloat* x, int K, int live, bool inside,
                      const device float* xs, const device uchar* wq, const device bfloat* scales,
                      const device bfloat* biases, threadgroup bfloat* tile, threadgroup float* sb, int tn, uint t,
                      short2 home) {
  threadgroup bfloat* mine = tile + (t / 2) * PAD + 32 * (t % 2);
  const int g_mine = TF_GROUP == 32 ? int(t % 2) : 0;
  const int groups = K / TF_GROUP;
  TF_UNROLL
  for (short i = 0; i < TM; i++) {
    acc[i][0] = frag<float>(0);
    acc[i][1] = frag<float>(0);
  }
  for (int k = 0; k < K; k += 64) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    am_codes(wq, mine);
    sb[g_mine * 128 + t / 2] = float(*scales);
    sb[g_mine * 128 + 64 + t / 2] = float(*biases);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    frag<float> part[TM][2];
#pragma clang loop unroll(disable)
    for (int kk = 0; kk < 64; kk += 32) {
      if (TF_GROUP == 32 || kk == 0) {
        TF_UNROLL
        for (short i = 0; i < TM; i++) {
          part[i][0] = frag<float>(0);
          part[i][1] = frag<float>(0);
        }
      }
      if (live > 0) {
        frag<bfloat> a[TM][2], b[2][2]; // a[m][k] from x, b[k][n] from the codes (stored [n][k])
        TF_UNROLL
        for (short i = 0; i < 2; i++) {
          TF_UNROLL
          for (short j = 0; j < 2; j++) {
            frag_get_t(b[j][i], (const threadgroup bfloat*)tile, PAD, tn + 16 * i, kk + 16 * j, home);
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
            mma_16x32<false, true>(part[m][0], part[m][1], a[m][j], b[j][0], b[j][1]);
          }
        }
      }
      if (live > 0 && (TF_GROUP == 32 || kk == 32)) { // a group's sums done: its scale, its bias times the row sums
        const int g = TF_GROUP == 32 ? kk / 32 : 0;
        const int gi = k / TF_GROUP + g;
        TF_UNROLL
        for (short i = 0; i < TM; i++) {
          float xr[2];
          TF_UNROLL
          for (short h = 0; h < 2; h++) {
            const int r = 16 * i + home.y + 8 * h;
            xr[h] = r < live ? xs[long(r) * groups + gi] : 0.0f;
          }
          TF_UNROLL
          for (short j = 0; j < 2; j++) {
            TF_UNROLL
            for (short e = 0; e < 8; e++) {
              const int col = tn + 16 * j + home.x + TF_COL(e);
              acc[i][j][e] += sb[g * 128 + col] * part[i][j][e] + sb[g * 128 + 64 + col] * xr[e >> 2];
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
inline void am_block(const device bfloat* x, const device float* xs, int row, int rows, int M, int K, int N, int col,
                     long wrow0, const device uint32_t* w, const device bfloat* scales, const device bfloat* biases,
                     device TF_OUT_T* y, threadgroup bfloat* tile, threadgroup float* sb, uint sg, uint lane) {
  constexpr int SM = BM / 2;
  constexpr int TM = SM / 16;
  const int t = int(sg) * 32 + int(lane);
  const int tm = SM * int(sg / 2), tn = 32 * int(sg % 2), live = clamp(rows - tm, 0, SM);
  const long wrow = wrow0 + t / 2;
  const int g0 = TF_GROUP == 32 ? (t % 2) : 0;
  const short2 home = frag_home(ushort(lane));
  frag<float> acc[TM][2];
  am_k_loop<TM>(acc, x + long(row + tm) * K, K, live, row + tm + SM <= M, xs + long(row + tm) * (K / TF_GROUP),
                (const device uchar*)w + wrow * (K * TF_BITS / 8) + HALF_BYTES * (t % 2),
                scales + wrow * (K / TF_GROUP) + g0, biases + wrow * (K / TF_GROUP) + g0, tile, sb, tn, uint(t), home);
  am_store<TM>(acc, y + long(row + tm) * N + col + tn, N, live, home);
}

// y [rows, n] = x [rows, k] W^T, n a multiple of 64 (the weights' rows padded to it): 64x64 tiles.
[[kernel]] void tf_affine_mm(const device bfloat* X [[buffer(0)]], const device uint32_t* W [[buffer(1)]],
                             const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]],
                             const device float* XS [[buffer(4)]], constant MmArgs& a [[buffer(5)]],
                             device TF_OUT_T* Y [[buffer(6)]], uint sg [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup bfloat tile[64 * PAD];
  threadgroup float sb[NG * 128];
  const int row = int(tg.y) * 64, col = int(tg.x) * 64;
  am_block<64>(X, XS, row, min(64, a.rows - row), a.rows, a.k, a.n, col, long(col), W, S, B, Y, tile, sb, sg, lane);
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
    am_block<BM>(X, XS, row, rows, a.rows, a.k, a.n, col, long(expert) * a.n + col, W, S, B, Y, tile, sb, sg, lane);  \
  }
TF_GATHER(32)
TF_GATHER(64)
