// Rows sorted by expert times their expert's quantized W^T on the tensor units: affine weights in MLX's layout
// dequantized to bf16 a 64-deep step at a time, 16x32 MMAs with fp32 sums, a 64-column block a threadgroup over one
// expert's BM rows. At 4 bits, group 64 and bf16 out these are the prefill gather's bits (prefill/qmm_nax.metal).
// Defines: TF_BITS (4 or 8), TF_GROUP (32 or 64), TF_OUT_T (bfloat, or float: partial sums another pass adds).
#include <metal_stdlib>
using namespace metal;
#include "../nax.h"
using namespace tfp;

#ifndef TF_OUT_T
#define TF_OUT_T bfloat
#endif
constant constexpr int STEP_BYTES = 8 * TF_BITS; // a weight row's bytes in one 64-deep step
constant constexpr int HALF_BYTES = 4 * TF_BITS; // a thread's half of them (32 weights)
constant constexpr int SCALE_STEP = 64 / TF_GROUP; // scales a row in one step

struct GatherArgs {
  int rows, n, k, experts; // sorted rows, outputs a row, depth, experts
};

// A thread's 32 weights of one row in this step as bf16(s q + b); at 4 bits the high nibble as (s / 16)(q & 0xf0).
inline void eg_dequant(const device uchar* w, bfloat scale, bfloat bias, threadgroup bfloat* out) {
  const float s = float(scale), b = float(bias);
#if TF_BITS == 4
  const float s_hi = s / 16.0f;
  TF_UNROLL
  for (short i = 0; i < 16; i++) {
    const uchar q = w[i];
    out[2 * i] = static_cast<bfloat>(s * (q & 0x0f) + b);
    out[2 * i + 1] = static_cast<bfloat>(s_hi * (q & 0xf0) + b);
  }
#else
  TF_UNROLL
  for (short i = 0; i < 32; i++) {
    out[i] = static_cast<bfloat>(s * w[i] + b);
  }
#endif
}

// x (TM 16-row fragments, ld K) times a [64 rows, K] weight block: thread t dequantizes half of row t / 2 each step.
template <int TM>
inline void eg_k_loop(thread frag<float> (&acc)[TM][2], const device bfloat* x, int K, int live, bool inside,
                      const device uchar* wq, const device bfloat* scales, const device bfloat* biases,
                      threadgroup bfloat* tile, int tn, uint t, short2 home) {
  constexpr int PAD = 64 + 8;
  threadgroup bfloat* mine = tile + (t / 2) * PAD + 32 * (t % 2);
  TF_UNROLL
  for (short i = 0; i < TM; i++) {
    acc[i][0] = frag<float>(0);
    acc[i][1] = frag<float>(0);
  }
  for (int k = 0; k < K; k += 64) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    eg_dequant(wq, *scales, *biases, mine);
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(disable)
    for (int kk = 0; kk < 64; kk += 32) {
      if (live > 0) {
        frag<bfloat> a[TM][2], b[2][2]; // a[m][k] from x, b[k][n] from the block (stored [n][k])
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
            mma_16x32<false, true>(acc[m][0], acc[m][1], a[m][j], b[j][0], b[j][1]);
          }
        }
      }
    }
    x += 64;
    wq += STEP_BYTES;
    scales += SCALE_STEP;
    biases += SCALE_STEP;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// The tile'th BM-row tile over experts in order: its expert, first row and row count (false past the last tile).
template <int BM>
inline bool eg_tile(const device int32_t* offsets, int experts, int total, int tile, uint lane, thread int& expert,
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

template <int BM>
inline void eg_gather(const device bfloat* x, const device uint32_t* w, const device bfloat* scales,
                      const device bfloat* biases, const device int32_t* offsets, constant GatherArgs& a,
                      device TF_OUT_T* y, threadgroup bfloat* tile, uint3 tg, uint sg, uint lane) {
  constexpr int SM = BM / 2;
  constexpr int TM = SM / 16;
  const int M = a.rows, N = a.n, K = a.k;
  int expert, row, rows;
  if (!eg_tile<BM>(offsets, a.experts, M, int(tg.y), lane, expert, row, rows)) {
    return;
  }
  const int col = int(tg.x) * 64, t = int(sg) * 32 + int(lane);
  const int tm = SM * int(sg / 2), tn = 32 * int(sg % 2), live = clamp(rows - tm, 0, SM);
  const long wrow = long(expert) * N + col + t / 2;
  const int g0 = TF_GROUP == 32 ? int(t % 2) : 0; // a thread's first group in a step
  const short2 home = frag_home(ushort(lane));
  frag<float> acc[TM][2];
  eg_k_loop<TM>(acc, x + long(row + tm) * K, K, live, row + tm + SM <= M,
                (const device uchar*)w + wrow * (K * TF_BITS / 8) + HALF_BYTES * (t % 2),
                scales + wrow * (K / TF_GROUP) + g0, biases + wrow * (K / TF_GROUP) + g0, tile, tn, uint(t), home);
  device TF_OUT_T* out = y + long(row + tm) * N + col + tn;
  TF_UNROLL
  for (short i = 0; i < TM; i++) {
    TF_UNROLL
    for (short j = 0; j < 2; j++) {
      if (live == SM) {
        frag_put(acc[i][j], out, N, 16 * i, 16 * j, home);
      } else {
        frag_put_in(acc[i][j], out, N, 16 * i, 16 * j, home, live, 32);
      }
    }
  }
}

#define TF_GATHER(BM)                                                                                                   \
  [[kernel]] void tf_expert_gather_##BM(                                                                                \
      const device bfloat* X [[buffer(0)]], const device uint32_t* W [[buffer(1)]],                                    \
      const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]],                                      \
      const device int32_t* OFFS [[buffer(4)]], constant GatherArgs& a [[buffer(5)]],                                  \
      device TF_OUT_T* Y [[buffer(6)]], uint sg [[simdgroup_index_in_threadgroup]],                                   \
      uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {                           \
    threadgroup bfloat tile[64 * (64 + 8)];                                                                            \
    eg_gather<BM>(X, W, S, B, OFFS, a, Y, tile, tg, sg, lane);                                                          \
  }
TF_GATHER(32)
TF_GATHER(64)
