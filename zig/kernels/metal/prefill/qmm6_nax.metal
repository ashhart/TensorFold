// 6-bit (group 32) projections of prompt chunks on the tensor units: MLX-layout weights [N, K*6/32 words], one group's
// 32 weights dequantized a thread a step into bf16, 16x32 tensor-op MMAs with fp32 sums; dense and sorted-expert gather.
#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;
#include "../nax.h"
using namespace tfp;

namespace tfq6 {

// One group's 32 6-bit codes (6 words, little-endian bit stream) as bf16(s q + b) in out[0..32).
template <typename T, typename O>
inline void dequant6(const device uint* w, T scale, T bias, threadgroup O* out) {
  const float s = float(scale), b = float(bias);
  const uint2 a = *(const device uint2*)(w), c = *(const device uint2*)(w + 2), d = *(const device uint2*)(w + 4);
  const ulong lo = ulong(a.x) | (ulong(a.y) << 32), mid = ulong(c.x) | (ulong(c.y) << 32), hi = ulong(d.x) | (ulong(d.y) << 32);
  TF_UNROLL
  for (int i = 0; i < 32; i++) {
    const int bit = 6 * i;
    uint v;
    if (bit + 6 <= 64) v = uint(lo >> bit) & 63u;
    else if (bit < 64) v = (uint(lo >> bit) | uint(mid << (64 - bit))) & 63u;
    else if (bit + 6 <= 128) v = uint(mid >> (bit - 64)) & 63u;
    else if (bit < 128) v = (uint(mid >> (bit - 64)) | uint(hi << (128 - bit))) & 63u;
    else v = uint(hi >> (bit - 128)) & 63u;
    out[i] = O(static_cast<T>(s * float(v) + b));
  }
}

// One group's 32 codes from the words already in registers (dequant6's bits and order).
template <typename T, typename O>
inline void dequant6r(uint2 a, uint2 c, uint2 d, T scale, T bias, threadgroup O* out) {
  const float s = float(scale), b = float(bias);
  const ulong lo = ulong(a.x) | (ulong(a.y) << 32), mid = ulong(c.x) | (ulong(c.y) << 32), hi = ulong(d.x) | (ulong(d.y) << 32);
  TF_UNROLL
  for (int i = 0; i < 32; i++) {
    const int bit = 6 * i;
    uint v;
    if (bit + 6 <= 64) v = uint(lo >> bit) & 63u;
    else if (bit < 64) v = (uint(lo >> bit) | uint(mid << (64 - bit))) & 63u;
    else if (bit + 6 <= 128) v = uint(mid >> (bit - 64)) & 63u;
    else if (bit < 128) v = (uint(mid >> (bit - 64)) | uint(hi << (128 - bit))) & 63u;
    else v = uint(hi >> (bit - 128)) & 63u;
    out[i] = O(static_cast<T>(s * float(v) + b));
  }
}

// x (TM 16-row fragments, ld K) times a 6-bit [64 rows, K] block, 64 deep a step: thread t dequantizes row t / 2's
// group t % 2 of the step (wq, scales, biases point at the thread's first group); the next step's words and scale load
// while this step's MMAs run.
template <typename T, int TM>
inline void k_loop6(thread frag<float> (&acc)[TM][2], const device T* x, int K, int ldx, int live, bool inside,
                    const device uint* wq, const device T* scales, const device T* biases, threadgroup T* tile,
                    int tn, uint t, short2 home) {
  constexpr int PAD = 64 + 16 / sizeof(T);
  threadgroup T* mine = tile + (t / 2) * PAD + 32 * (t % 2);
  TF_UNROLL
  for (short i = 0; i < TM; i++) {
    acc[i][0] = frag<float>(0);
    acc[i][1] = frag<float>(0);
  }
  uint2 wa = *(const device uint2*)(wq), wb = *(const device uint2*)(wq + 2), wc = *(const device uint2*)(wq + 4);
  T sc = *scales, bi = *biases;
  for (int k = 0; k < K; k += 64) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    dequant6r<T>(wa, wb, wc, sc, bi, mine);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k + 64 < K) {
      wq += 12;
      scales += 2;
      biases += 2;
      wa = *(const device uint2*)(wq);
      wb = *(const device uint2*)(wq + 2);
      wc = *(const device uint2*)(wq + 4);
      sc = *scales;
      bi = *biases;
    }
#pragma clang loop unroll(disable)
    for (int kk = 0; kk < 64; kk += 32) {
      if (live > 0) {
        frag<T> a[TM][2], b[2][2];
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
              frag_get(a[i][j], x, ldx, 16 * i, kk + 16 * j, home);
            } else {
              frag_get_in(a[i][j], x, ldx, 16 * i, kk + 16 * j, home, live, kk + 32);
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
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// x times the gate's and the up projection's same 64 weight rows at once: each step's two tiles are dequantized side
// by side (thread t: row t / 2's group t % 2 of each), every x fragment feeds both products (k_loop6's sums, twice).
template <typename T, int TM>
inline void k_loop6x2(thread frag<float> (&ag)[TM][2], thread frag<float> (&au)[TM][2], const device T* x, int K,
                      int live, bool inside, const device uint* wg, const device T* sg, const device T* bg,
                      const device uint* wu, const device T* su, const device T* bu, threadgroup T* tg_,
                      threadgroup T* tu, int tn, uint t, short2 home) {
  constexpr int PAD = 64 + 16 / sizeof(T);
  threadgroup T* mg = tg_ + (t / 2) * PAD + 32 * (t % 2);
  threadgroup T* mu = tu + (t / 2) * PAD + 32 * (t % 2);
  TF_UNROLL
  for (short i = 0; i < TM; i++) {
    ag[i][0] = frag<float>(0);
    ag[i][1] = frag<float>(0);
    au[i][0] = frag<float>(0);
    au[i][1] = frag<float>(0);
  }
  uint2 ga = *(const device uint2*)(wg), gb = *(const device uint2*)(wg + 2), gc = *(const device uint2*)(wg + 4);
  uint2 ua = *(const device uint2*)(wu), ub = *(const device uint2*)(wu + 2), uc = *(const device uint2*)(wu + 4);
  T gs = *sg, gbi = *bg, us = *su, ubi = *bu;
  for (int k = 0; k < K; k += 64) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    dequant6r<T>(ga, gb, gc, gs, gbi, mg);
    dequant6r<T>(ua, ub, uc, us, ubi, mu);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k + 64 < K) {
      wg += 12;
      wu += 12;
      sg += 2;
      bg += 2;
      su += 2;
      bu += 2;
      ga = *(const device uint2*)(wg);
      gb = *(const device uint2*)(wg + 2);
      gc = *(const device uint2*)(wg + 4);
      ua = *(const device uint2*)(wu);
      ub = *(const device uint2*)(wu + 2);
      uc = *(const device uint2*)(wu + 4);
      gs = *sg;
      gbi = *bg;
      us = *su;
      ubi = *bu;
    }
#pragma clang loop unroll(disable)
    for (int kk = 0; kk < 64; kk += 32) {
      if (live > 0) {
        frag<T> a[TM][2], b[2][2], c[2][2];
        TF_UNROLL
        for (short i = 0; i < 2; i++) {
          TF_UNROLL
          for (short j = 0; j < 2; j++) {
            frag_get_t(b[j][i], (const threadgroup T*)tg_, PAD, tn + 16 * i, kk + 16 * j, home);
            frag_get_t(c[j][i], (const threadgroup T*)tu, PAD, tn + 16 * i, kk + 16 * j, home);
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
            mma_16x32<false, true>(ag[m][0], ag[m][1], a[m][j], b[j][0], b[j][1]);
            mma_16x32<false, true>(au[m][0], au[m][1], a[m][j], c[j][0], c[j][1]);
          }
        }
      }
    }
    x += 64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// The expert activation from the two products, as the unfused path rounds it: bf16(bf16(silu(bf16 g)) * bf16 u).
inline float act6(float g, float u) {
  const float gb = float(bfloat16_t(g)), ub = float(bfloat16_t(u));
  return float(bfloat16_t(float(bfloat16_t(gb / (1.0f + metal::exp(-gb)))) * ub));
}

template <int TM>
inline void store_act(thread const frag<float> (&ag)[TM][2], thread const frag<float> (&au)[TM][2], device bfloat16_t* y,
                      int ld, int live, int nc, short2 home) {
  TF_UNROLL
  for (short i = 0; i < TM; i++) {
    TF_UNROLL
    for (short j = 0; j < 2; j++) {
      frag<float> f;
      TF_UNROLL
      for (short e = 0; e < 8; e++) f[e] = act6(ag[i][j][e], au[i][j][e]);
      if (live == 16 * TM && nc >= 32) {
        frag_put(f, y, ld, 16 * i, 16 * j, home);
      } else {
        frag_put_in(f, y, ld, 16 * i, 16 * j, home, live, nc);
      }
    }
  }
}

// A simdgroup's TM x 2 fragments to y (row stride ld): rows below live, columns below nc.
template <typename T, int TM>
inline void store(thread const frag<float> (&acc)[TM][2], device T* y, int ld, int live, int nc, short2 home) {
  TF_UNROLL
  for (short i = 0; i < TM; i++) {
    TF_UNROLL
    for (short j = 0; j < 2; j++) {
      if (live == 16 * TM && nc >= 32) {
        frag_put(acc[i][j], y, ld, 16 * i, 16 * j, home);
      } else {
        frag_put_in(acc[i][j], y, ld, 16 * i, 16 * j, home, live, nc);
      }
    }
  }
}

// offsets[e] = the first row whose expert is not below e (rows sorted by expert). P: rows.
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

// The tile'th BM-row tile over experts in order: its expert, first row and row count.
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

}  // namespace tfq6

// y = x W^T for 6-bit g32 W [N, K]: BM x 64 output tiles, 4 simdgroups of BM/2 x 32 (each dequantized weight block
// serves BM rows). P: K N M and y's row stride (0: N).
template <int BM>
inline void qmm6_t(const device uint* W, const device bfloat16_t* S, const device bfloat16_t* B,
                   const device bfloat16_t* X, const device int* P, device bfloat16_t* Y, threadgroup bfloat16_t* tile,
                   uint sg, uint lane, uint3 tg) {
  constexpr int SM = BM / 2;
  const int K = P[0], N = P[1], M = P[2], LD = P[3] > 0 ? P[3] : N;
  const int row = int(tg.y) * BM, col = int(tg.x) * 64, t = int(sg) * 32 + int(lane);
  const int tm = SM * int(sg / 2), tn = 32 * int(sg % 2), live = min(SM, M - (row + tm));
  const long wrow = long(min(col + t / 2, N - 1));
  const int WPR = K * 6 / 32, KG = K / 32;
  frag<float> acc[SM / 16][2];
  tfq6::k_loop6<bfloat16_t, SM / 16>(acc, X + long(row + tm) * K, K, K, live, live == SM, W + wrow * WPR + 6 * (t % 2),
                                      S + wrow * KG + (t % 2), B + wrow * KG + (t % 2), tile, tn, uint(t),
                                      frag_home(ushort(lane)));
  if (col + tn < N && live > 0)
    tfq6::store<bfloat16_t, SM / 16>(acc, Y + long(row + tm) * LD + col + tn, LD, live, N - (col + tn), frag_home(ushort(lane)));
}

[[kernel]] void tf_qmm6_t_nax(const device uint* W [[buffer(0)]], const device bfloat16_t* S [[buffer(1)]],
    const device bfloat16_t* B [[buffer(2)]], const device bfloat16_t* X [[buffer(3)]], const device int* P [[buffer(4)]],
    device bfloat16_t* Y [[buffer(5)]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup bfloat16_t tile[64 * (64 + 16 / sizeof(bfloat16_t))];
  qmm6_t<64>(W, S, B, X, P, Y, tile, sg, lane, tg);
}

[[kernel]] void tf_qmm6_t_nax_128(const device uint* W [[buffer(0)]], const device bfloat16_t* S [[buffer(1)]],
    const device bfloat16_t* B [[buffer(2)]], const device bfloat16_t* X [[buffer(3)]], const device int* P [[buffer(4)]],
    device bfloat16_t* Y [[buffer(5)]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup bfloat16_t tile[64 * (64 + 16 / sizeof(bfloat16_t))];
  qmm6_t<128>(W, S, B, X, P, Y, tile, sg, lane, tg);
}

[[kernel]] void tf_expert_offsets6(const device uint32_t* I [[buffer(0)]], const device int32_t* P [[buffer(1)]],
    device int32_t* O [[buffer(2)]], uint3 pos [[thread_position_in_grid]]) {
  tfq6::expert_offsets(I, O, P, pos.x);
}

// y[r] = x[r] W_e^T for rows sorted by expert e, 6-bit g32 W [E, N, K]: BM-row tiles within an expert. P: M N K experts.
template <int BM>
inline void gather6(const device bfloat16_t* X, const device uint* W, const device bfloat16_t* S,
                    const device bfloat16_t* B, const device int32_t* O, device bfloat16_t* Y, const device int* P,
                    threadgroup bfloat16_t* tile, uint3 tg, uint sg, uint lane) {
  constexpr int SM = BM / 2;
  const int M = P[0], N = P[1], K = P[2];
  int expert, row, rows;
  if (!tfq6::expert_tile<BM>(O, P[3], M, int(tg.y), lane, expert, row, rows)) {
    return;
  }
  const int col = int(tg.x) * 64, t = int(sg) * 32 + int(lane);
  const int tm = SM * int(sg / 2), tn = 32 * int(sg % 2), live = clamp(rows - tm, 0, SM);
  const long wrow = long(expert) * N + min(col + t / 2, N - 1);
  const int WPR = K * 6 / 32, KG = K / 32;
  frag<float> acc[SM / 16][2];
  tfq6::k_loop6<bfloat16_t, SM / 16>(acc, X + long(row + tm) * K, K, K, live, row + tm + SM <= M,
                                      W + wrow * WPR + 6 * (t % 2), S + wrow * KG + (t % 2), B + wrow * KG + (t % 2),
                                      tile, tn, uint(t), frag_home(ushort(lane)));
  if (col + tn < N && live > 0)
    tfq6::store<bfloat16_t, SM / 16>(acc, Y + long(row + tm) * N + col + tn, N, live, N - (col + tn), frag_home(ushort(lane)));
}

// The experts' gate and up products and their activation in one pass: y[r] = act(x[r] Wg_e^T, x[r] Wu_e^T) for rows
// sorted by expert, BM-row tiles within an expert (gather6's tiles). P: M N K experts.
template <int BM>
inline void gather_gu6(const device bfloat16_t* X, const device uint* WG, const device bfloat16_t* SG,
                       const device bfloat16_t* BG, const device uint* WU, const device bfloat16_t* SU,
                       const device bfloat16_t* BU, const device int32_t* O, device bfloat16_t* Y, const device int* P,
                       threadgroup bfloat16_t* tile_g, threadgroup bfloat16_t* tile_u, uint3 tg, uint sg, uint lane) {
  constexpr int SM = BM / 2;
  const int M = P[0], N = P[1], K = P[2];
  int expert, row, rows;
  if (!tfq6::expert_tile<BM>(O, P[3], M, int(tg.y), lane, expert, row, rows)) {
    return;
  }
  const int col = int(tg.x) * 64, t = int(sg) * 32 + int(lane);
  const int tm = SM * int(sg / 2), tn = 32 * int(sg % 2), live = clamp(rows - tm, 0, SM);
  const long wrow = long(expert) * N + min(col + t / 2, N - 1);
  const int WPR = K * 6 / 32, KG = K / 32;
  frag<float> ag[SM / 16][2], au[SM / 16][2];
  tfq6::k_loop6x2<bfloat16_t, SM / 16>(ag, au, X + long(row + tm) * K, K, live, row + tm + SM <= M,
                                        WG + wrow * WPR + 6 * (t % 2), SG + wrow * KG + (t % 2), BG + wrow * KG + (t % 2),
                                        WU + wrow * WPR + 6 * (t % 2), SU + wrow * KG + (t % 2), BU + wrow * KG + (t % 2),
                                        tile_g, tile_u, tn, uint(t), frag_home(ushort(lane)));
  if (col + tn < N && live > 0)
    tfq6::store_act<SM / 16>(ag, au, Y + long(row + tm) * N + col + tn, N, live, N - (col + tn), frag_home(ushort(lane)));
}

[[kernel]] void tf_gather_gu6_nax_64(const device bfloat16_t* X [[buffer(0)]], const device uint* WG [[buffer(1)]],
    const device bfloat16_t* SG [[buffer(2)]], const device bfloat16_t* BG [[buffer(3)]], const device uint* WU [[buffer(4)]],
    const device bfloat16_t* SU [[buffer(5)]], const device bfloat16_t* BU [[buffer(6)]], const device int32_t* O [[buffer(7)]],
    const device int32_t* P [[buffer(8)]], device bfloat16_t* Y [[buffer(9)]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup bfloat16_t tile_g[64 * (64 + 16 / sizeof(bfloat16_t))], tile_u[64 * (64 + 16 / sizeof(bfloat16_t))];
  gather_gu6<64>(X, WG, SG, BG, WU, SU, BU, O, Y, P, tile_g, tile_u, tg, sg, lane);
}

[[kernel]] void tf_gather_gu6_nax_32(const device bfloat16_t* X [[buffer(0)]], const device uint* WG [[buffer(1)]],
    const device bfloat16_t* SG [[buffer(2)]], const device bfloat16_t* BG [[buffer(3)]], const device uint* WU [[buffer(4)]],
    const device bfloat16_t* SU [[buffer(5)]], const device bfloat16_t* BU [[buffer(6)]], const device int32_t* O [[buffer(7)]],
    const device int32_t* P [[buffer(8)]], device bfloat16_t* Y [[buffer(9)]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup bfloat16_t tile_g[64 * (64 + 16 / sizeof(bfloat16_t))], tile_u[64 * (64 + 16 / sizeof(bfloat16_t))];
  gather_gu6<32>(X, WG, SG, BG, WU, SU, BU, O, Y, P, tile_g, tile_u, tg, sg, lane);
}

[[kernel]] void tf_gather_qmm6_nax_64(const device bfloat16_t* X [[buffer(0)]], const device uint* W [[buffer(1)]],
    const device bfloat16_t* S [[buffer(2)]], const device bfloat16_t* B [[buffer(3)]], const device int32_t* O [[buffer(4)]],
    const device int32_t* P [[buffer(5)]], device bfloat16_t* Y [[buffer(6)]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup bfloat16_t tile[64 * (64 + 16 / sizeof(bfloat16_t))];
  gather6<64>(X, W, S, B, O, Y, P, tile, tg, sg, lane);
}

[[kernel]] void tf_gather_qmm6_nax_128(const device bfloat16_t* X [[buffer(0)]], const device uint* W [[buffer(1)]],
    const device bfloat16_t* S [[buffer(2)]], const device bfloat16_t* B [[buffer(3)]], const device int32_t* O [[buffer(4)]],
    const device int32_t* P [[buffer(5)]], device bfloat16_t* Y [[buffer(6)]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup bfloat16_t tile[64 * (64 + 16 / sizeof(bfloat16_t))];
  gather6<128>(X, W, S, B, O, Y, P, tile, tg, sg, lane);
}

[[kernel]] void tf_gather_qmm6_nax_32(const device bfloat16_t* X [[buffer(0)]], const device uint* W [[buffer(1)]],
    const device bfloat16_t* S [[buffer(2)]], const device bfloat16_t* B [[buffer(3)]], const device int32_t* O [[buffer(4)]],
    const device int32_t* P [[buffer(5)]], device bfloat16_t* Y [[buffer(6)]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup bfloat16_t tile[64 * (64 + 16 / sizeof(bfloat16_t))];
  gather6<32>(X, W, S, B, O, Y, P, tile, tg, sg, lane);
}

namespace tfq6 {

// x (TM 16-row fragments) times a bf16 [64 rows, K] block, 64 deep a step: thread t copies row t / 2's half t % 2.
template <typename T, int TM>
inline void k_loop_bf16(thread frag<float> (&acc)[TM][2], const device T* x, int K, int live, bool inside,
                        const device T* wr, threadgroup T* tile, int tn, uint t, short2 home) {
  constexpr int PAD = 64 + 16 / sizeof(T);
  threadgroup T* mine = tile + (t / 2) * PAD + 32 * (t % 2);
  TF_UNROLL
  for (short i = 0; i < TM; i++) {
    acc[i][0] = frag<float>(0);
    acc[i][1] = frag<float>(0);
  }
  for (int k = 0; k < K; k += 64) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    TF_UNROLL
    for (int i = 0; i < 32; i++) mine[i] = wr[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(disable)
    for (int kk = 0; kk < 64; kk += 32) {
      if (live > 0) {
        frag<T> a[TM][2], b[2][2];
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
    wr += 64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

}  // namespace tfq6

// y (fp32) = x W^T for bf16 W [N, K] (the router): 64x64 tiles, columns below N. P: K N M.
[[kernel]] void tf_mm_bf16_f32_t_nax(const device bfloat16_t* W [[buffer(0)]], const device bfloat16_t* X [[buffer(1)]],
    const device int* P [[buffer(2)]], device float* Y [[buffer(3)]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup bfloat16_t tile[64 * (64 + 16 / sizeof(bfloat16_t))];
  const int K = P[0], N = P[1], M = P[2];
  const int row = int(tg.y) * 64, col = int(tg.x) * 64, t = int(sg) * 32 + int(lane);
  const int tm = 32 * int(sg / 2), tn = 32 * int(sg % 2), live = min(32, M - (row + tm));
  const long wrow = long(min(col + t / 2, N - 1));
  frag<float> acc[2][2];
  tfq6::k_loop_bf16<bfloat16_t, 2>(acc, X + long(row + tm) * K, K, live, live == 32, W + wrow * K + 32 * (t % 2), tile,
                                    tn, uint(t), frag_home(ushort(lane)));
  if (col + tn < N && live > 0)
    tfq6::store<float, 2>(acc, Y + long(row + tm) * N + col + tn, N, live, N - (col + tn), frag_home(ushort(lane)));
}

// One K part of x W^T (6-bit g32): fp32 partials PART[z][M][N] for part z of P[3] parts. P: K N M parts.
[[kernel]] void tf_qmm6_splitk_nax(const device uint* W [[buffer(0)]], const device bfloat16_t* S [[buffer(1)]],
    const device bfloat16_t* B [[buffer(2)]], const device bfloat16_t* X [[buffer(3)]], const device int* P [[buffer(4)]],
    device float* PART [[buffer(5)]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup bfloat16_t tile[64 * (64 + 16 / sizeof(bfloat16_t))];
  const int K = P[0], N = P[1], M = P[2], parts = P[3], KP = K / parts, z = int(tg.z);
  const int row = int(tg.y) * 64, col = int(tg.x) * 64, t = int(sg) * 32 + int(lane);
  const int tm = 32 * int(sg / 2), tn = 32 * int(sg % 2), live = min(32, M - (row + tm));
  const long wrow = long(min(col + t / 2, N - 1));
  const int WPR = K * 6 / 32, KG = K / 32, k0 = z * KP;
  frag<float> acc[2][2];
  tfq6::k_loop6<bfloat16_t, 2>(acc, X + long(row + tm) * K + k0, KP, K, live, live == 32,
                                W + wrow * WPR + (k0 / 32) * 6 + 6 * (t % 2), S + wrow * KG + k0 / 32 + (t % 2),
                                B + wrow * KG + k0 / 32 + (t % 2), tile, tn, uint(t), frag_home(ushort(lane)));
  if (col + tn < N && live > 0)
    tfq6::store<float, 2>(acc, PART + (long(z) * M + row + tm) * N + col + tn, N, live, N - (col + tn), frag_home(ushort(lane)));
}

// y = bf16 of the parts summed in order. P: parts, M*N.
[[kernel]] void tf_parts_sum(const device float* PART [[buffer(0)]], const device int* P [[buffer(1)]],
    device bfloat16_t* Y [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (int(i) >= P[1]) return;
  float total = 0.0f;
  for (int z = 0; z < P[0]; z++) total += PART[long(z) * P[1] + i];
  Y[i] = bfloat16_t(total);
}

// Flash Next's causal prompt-chunk attention on the tensor units: 64 query rows of one head a threadgroup (16 a
// simdgroup), 32-key blocks of the cache, head size 256, 12 query heads a key head, fp32 online softmax; the output
// gate bf16(bf16(o) * bsig(gate)) on the way out. P: rows, keys (position + rows), position, cache rows.
constant constexpr float a_masked = -3.402823466e+38f;
inline float a_bsig(float x) { return float(bfloat(1.0f / (1.0f + metal::exp(-x)))); }
template <bool MAX>
inline void a_fold(thread const frag<float>& f, thread float (&r)[2]) {
  TF_UNROLL
  for (short h = 0; h < 2; h++) {
    const short b = 4 * h;
    float t = MAX ? max(max(f[b], f[b + 1]), max(f[b + 2], f[b + 3])) : (f[b] + f[b + 1]) + (f[b + 2] + f[b + 3]);
    const float u = simd_shuffle_xor(t, ushort(1));
    t = MAX ? max(t, u) : t + u;
    const float w = simd_shuffle_xor(t, ushort(8));
    t = MAX ? max(t, w) : t + w;
    r[h] = MAX ? max(r[h], t) : r[h] + t;
  }
}
[[kernel]] void tf_attn256_nax(const device bfloat16_t* Q [[buffer(0)]], const device bfloat16_t* K [[buffer(1)]],
    const device bfloat16_t* V [[buffer(2)]], const device bfloat16_t* GP [[buffer(3)]], const device int* P [[buffer(4)]],
    const device float* F [[buffer(5)]], device bfloat16_t* O [[buffer(6)]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr int BQ = 64, BK = 32, D = 256, H = 24, GQA = 12, PW = 13952;
  const int qL = P[0], kL = P[1], qoff = P[2], cap = P[3];
  const int h = int(tg.y), q0 = int(tg.x) * BQ;
  const short tm = 16 * short(sg);
  const device bfloat16_t* Qp = Q + long(q0 + tm) * (H * D) + h * D;
  const device bfloat16_t* Kp = K + long(h / GQA) * cap * D;
  const device bfloat16_t* Vp = V + long(h / GQA) * cap * D;
  const float scale2 = F[0] * 1.44269504089f;
  const short2 home = frag_home(ushort(lane));
  frag<float> acc[D / 16];
  TF_UNROLL
  for (short i = 0; i < D / 16; i++) acc[i] = frag<float>(0);
  float top[2] = {a_masked, a_masked}, total[2] = {0.0f, 0.0f};
  const int last_row = qoff + min(q0 + BQ, qL) - 1;
  const int blocks = (min(kL, last_row + 1) + BK - 1) / BK;
  for (int kb = 0; kb < blocks; kb++) {
    const int k0 = kb * BK;
    frag<float> s[2] = {frag<float>(0), frag<float>(0)};
#pragma clang loop unroll_count(4)
    for (short d = 0; d < D / 16; d++) {
      frag<bfloat16_t> q, ka, kb2;
      frag_get(q, Qp, H * D, 0, 16 * d, home);
      frag_get_t(ka, Kp + long(k0) * D, D, 0, 16 * d, home);
      frag_get_t(kb2, Kp + long(k0) * D, D, 16, 16 * d, home);
      mma_16x32<false, true>(s[0], s[1], q, ka, kb2);
    }
    const int r0 = qoff + q0 + tm + home.y;
    TF_UNROLL
    for (short f = 0; f < 2; f++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        const int col = k0 + 16 * f + home.x + TF_COL(e), row = r0 + (e >> 2) * 8;
        s[f][e] = (col >= kL || col > row) ? a_masked : s[f][e] * scale2;
      }
    }
    float top_new[2] = {top[0], top[1]};
    a_fold<true>(s[0], top_new);
    a_fold<true>(s[1], top_new);
    TF_UNROLL
    for (short f = 0; f < 2; f++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) s[f][e] = fast::exp2(s[f][e] - top_new[e >> 2]);
    }
    float scale_old[2];
    TF_UNROLL
    for (short hh = 0; hh < 2; hh++) {
      scale_old[hh] = fast::exp2(top[hh] - top_new[hh]);
      top[hh] = top_new[hh];
      total[hh] = total[hh] * scale_old[hh];
    }
    a_fold<false>(s[0], total);
    a_fold<false>(s[1], total);
    TF_UNROLL
    for (short i = 0; i < D / 16; i++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) acc[i][e] = acc[i][e] * scale_old[e >> 2];
    }
    TF_UNROLL
    for (short d = 0; d < D / 16; d += 2) {
      TF_UNROLL
      for (short k = 0; k < 2; k++) {
        frag<bfloat16_t> v0, v1;
        frag_get(v0, Vp + long(k0) * D, D, 16 * k, 16 * d, home);
        frag_get(v1, Vp + long(k0) * D, D, 16 * k, 16 * d + 16, home);
        mma_16x32<false, false>(acc[d], acc[d + 1], s[k], v0, v1);
      }
    }
  }
  float inv[2] = {1.0f / total[0], 1.0f / total[1]};
  TF_UNROLL
  for (short i = 0; i < D / 16; i++) {
    TF_UNROLL
    for (short e = 0; e < 8; e++) {
      const int row = q0 + tm + home.y + (e >> 2) * 8, col = 16 * i + home.x + TF_COL(e);
      if (row < qL) {
        const float o = float(bfloat(acc[i][e] * inv[e >> 2]));
        const float g = float(GP[long(row) * PW + h * 2 * D + D + col]);
        O[(long(row) * H + h) * D + col] = bfloat(o * a_bsig(g));
      }
    }
  }
}
