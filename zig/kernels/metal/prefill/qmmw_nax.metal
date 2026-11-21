// Prompt projections at any MLX width (WB bits, groups of WG, defined first): qmm6_nax's tiles and fp32 sums.
#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;
#include "../nax.h"
using namespace tfp;

namespace tfqw {

// One thread's 32 codes (WB words, little-endian bit stream) as bf16(s q + b) in out[0..32).
template <typename T, typename O>
inline void dequantw(thread const uint (&w)[WB + 1], T scale, T bias, threadgroup O* out) {
  const float s = float(scale), b = float(bias);
  TF_UNROLL
  for (int i = 0; i < 32; i++) {
    const int bit = WB * i, word = bit >> 5, sh = bit & 31;
    uint v = w[word] >> sh;
    if (sh + WB > 32) v |= w[word + 1] << (32 - sh);
    out[i] = O(static_cast<T>(s * float(v & ((1u << WB) - 1u)) + b));
  }
}

inline void loadw(thread uint (&w)[WB + 1], const device uint* src) {
  TF_UNROLL
  for (int i = 0; i < WB; i++) w[i] = src[i];
  w[WB] = 0;
}

// x times a [64 rows, K] block, 64 deep a step: thread t dequantizes row t / 2's codes k0 + k + 32 (t % 2) onward.
template <typename T, int TM>
inline void k_loopw(thread frag<float> (&acc)[TM][2], const device T* x, int K, int ldx, int live, bool inside,
                    const device uint* wq, const device T* srow, const device T* brow, int k0, threadgroup T* tile,
                    int tn, uint t, short2 home) {
  constexpr int PAD = 64 + 16 / sizeof(T);
  threadgroup T* mine = tile + (t / 2) * PAD + 32 * (t % 2);
  TF_UNROLL
  for (short i = 0; i < TM; i++) {
    acc[i][0] = frag<float>(0);
    acc[i][1] = frag<float>(0);
  }
  const int hk = k0 + 32 * int(t % 2);
  uint w[WB + 1];
  loadw(w, wq);
  T sc = srow[hk / WG], bi = brow[hk / WG];
  for (int k = 0; k < K; k += 64) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    dequantw<T>(w, sc, bi, mine);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k + 64 < K) {
      wq += 2 * WB;
      loadw(w, wq);
      sc = srow[(hk + k + 64) / WG];
      bi = brow[(hk + k + 64) / WG];
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

// Gate and up at once (k_loopw's sums, twice): both tiles dequantized side by side, each x fragment feeds both.
template <typename T, int TM>
inline void k_loopwx2(thread frag<float> (&ag)[TM][2], thread frag<float> (&au)[TM][2], const device T* x, int K,
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
  const int hk = 32 * int(t % 2);
  uint gw[WB + 1], uw[WB + 1];
  loadw(gw, wg);
  loadw(uw, wu);
  T gs = sg[hk / WG], gbi = bg[hk / WG], us = su[hk / WG], ubi = bu[hk / WG];
  for (int k = 0; k < K; k += 64) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    dequantw<T>(gw, gs, gbi, mg);
    dequantw<T>(uw, us, ubi, mu);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k + 64 < K) {
      wg += 2 * WB;
      wu += 2 * WB;
      loadw(gw, wg);
      loadw(uw, wu);
      const int g = (hk + k + 64) / WG;
      gs = sg[g];
      gbi = bg[g];
      us = su[g];
      ubi = bu[g];
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

// The expert activation from the two products, as qmm6_nax's act6 rounds it.
inline float actw(float g, float u) {
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
      for (short e = 0; e < 8; e++) f[e] = actw(ag[i][j][e], au[i][j][e]);
      if (live == 16 * TM && nc >= 32) {
        frag_put(f, y, ld, 16 * i, 16 * j, home);
      } else {
        frag_put_in(f, y, ld, 16 * i, 16 * j, home, live, nc);
      }
    }
  }
}

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

// The tile'th BM-row tile over experts in order: its expert, first row and row count (qmm6_nax's expert_tile).
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

}  // namespace tfqw

// y = x W^T for W [N, K] at WB/WG: 64 x 64 tiles, 4 simdgroups of 32 x 32. P: K N M, y's row stride (0: N).
[[kernel]] void tf_qmmw_t_nax(const device uint* W [[buffer(0)]], const device bfloat16_t* S [[buffer(1)]],
    const device bfloat16_t* B [[buffer(2)]], const device bfloat16_t* X [[buffer(3)]], const device int* P [[buffer(4)]],
    device bfloat16_t* Y [[buffer(5)]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup bfloat16_t tile[64 * (64 + 16 / sizeof(bfloat16_t))];
  const int K = P[0], N = P[1], M = P[2], LD = P[3] > 0 ? P[3] : N;
  const int row = int(tg.y) * 64, col = int(tg.x) * 64, t = int(sg) * 32 + int(lane);
  const int tm = 32 * int(sg / 2), tn = 32 * int(sg % 2), live = min(32, M - (row + tm));
  const long wrow = long(min(col + t / 2, N - 1));
  const int WPR = K * WB / 32, KG = K / WG;
  frag<float> acc[2][2];
  tfqw::k_loopw<bfloat16_t, 2>(acc, X + long(row + tm) * K, K, K, live, live == 32, W + wrow * WPR + WB * (t % 2),
                               S + wrow * KG, B + wrow * KG, 0, tile, tn, uint(t), frag_home(ushort(lane)));
  if (col + tn < N && live > 0)
    tfqw::store<bfloat16_t, 2>(acc, Y + long(row + tm) * LD + col + tn, LD, live, N - (col + tn), frag_home(ushort(lane)));
}

// One K part of x W^T: fp32 partials PART[z][M][N] for part z of P[3] parts. P: K N M parts.
[[kernel]] void tf_qmmw_splitk_nax(const device uint* W [[buffer(0)]], const device bfloat16_t* S [[buffer(1)]],
    const device bfloat16_t* B [[buffer(2)]], const device bfloat16_t* X [[buffer(3)]], const device int* P [[buffer(4)]],
    device float* PART [[buffer(5)]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup bfloat16_t tile[64 * (64 + 16 / sizeof(bfloat16_t))];
  const int K = P[0], N = P[1], M = P[2], parts = P[3], KP = K / parts, z = int(tg.z);
  const int row = int(tg.y) * 64, col = int(tg.x) * 64, t = int(sg) * 32 + int(lane);
  const int tm = 32 * int(sg / 2), tn = 32 * int(sg % 2), live = min(32, M - (row + tm));
  const long wrow = long(min(col + t / 2, N - 1));
  const int WPR = K * WB / 32, KG = K / WG, k0 = z * KP;
  frag<float> acc[2][2];
  tfqw::k_loopw<bfloat16_t, 2>(acc, X + long(row + tm) * K + k0, KP, K, live, live == 32,
                               W + wrow * WPR + (k0 / 32) * WB + WB * (t % 2), S + wrow * KG, B + wrow * KG, k0, tile, tn,
                               uint(t), frag_home(ushort(lane)));
  if (col + tn < N && live > 0)
    tfqw::store<float, 2>(acc, PART + (long(z) * M + row + tm) * N + col + tn, N, live, N - (col + tn), frag_home(ushort(lane)));
}

// y[r] = x[r] W_e^T for rows sorted by expert, W [E, N, K]: BM-row tiles within an expert. P: M N K experts.
template <int BM>
inline void gatherw(const device bfloat16_t* X, const device uint* W, const device bfloat16_t* S,
                    const device bfloat16_t* B, const device int32_t* O, device bfloat16_t* Y, const device int* P,
                    threadgroup bfloat16_t* tile, uint3 tg, uint sg, uint lane) {
  constexpr int SM = BM / 2;
  const int M = P[0], N = P[1], K = P[2];
  int expert, row, rows;
  if (!tfqw::expert_tile<BM>(O, P[3], M, int(tg.y), lane, expert, row, rows)) {
    return;
  }
  const int col = int(tg.x) * 64, t = int(sg) * 32 + int(lane);
  const int tm = SM * int(sg / 2), tn = 32 * int(sg % 2), live = clamp(rows - tm, 0, SM);
  const long wrow = long(expert) * N + min(col + t / 2, N - 1);
  const int WPR = K * WB / 32, KG = K / WG;
  frag<float> acc[SM / 16][2];
  tfqw::k_loopw<bfloat16_t, SM / 16>(acc, X + long(row + tm) * K, K, K, live, row + tm + SM <= M,
                                     W + wrow * WPR + WB * (t % 2), S + wrow * KG, B + wrow * KG, 0, tile, tn, uint(t),
                                     frag_home(ushort(lane)));
  if (col + tn < N && live > 0)
    tfqw::store<bfloat16_t, SM / 16>(acc, Y + long(row + tm) * N + col + tn, N, live, N - (col + tn), frag_home(ushort(lane)));
}

// Experts' gate and up products and activation in one pass, for rows sorted by expert. P: M N K experts.
template <int BM>
inline void gather_guw(const device bfloat16_t* X, const device uint* WG_, const device bfloat16_t* SG,
                       const device bfloat16_t* BG, const device uint* WU, const device bfloat16_t* SU,
                       const device bfloat16_t* BU, const device int32_t* O, device bfloat16_t* Y, const device int* P,
                       threadgroup bfloat16_t* tile_g, threadgroup bfloat16_t* tile_u, uint3 tg, uint sg, uint lane) {
  constexpr int SM = BM / 2;
  const int M = P[0], N = P[1], K = P[2];
  int expert, row, rows;
  if (!tfqw::expert_tile<BM>(O, P[3], M, int(tg.y), lane, expert, row, rows)) {
    return;
  }
  const int col = int(tg.x) * 64, t = int(sg) * 32 + int(lane);
  const int tm = SM * int(sg / 2), tn = 32 * int(sg % 2), live = clamp(rows - tm, 0, SM);
  const long wrow = long(expert) * N + min(col + t / 2, N - 1);
  const int WPR = K * WB / 32, KG = K / WG;
  frag<float> ag[SM / 16][2], au[SM / 16][2];
  tfqw::k_loopwx2<bfloat16_t, SM / 16>(ag, au, X + long(row + tm) * K, K, live, row + tm + SM <= M,
                                       WG_ + wrow * WPR + WB * (t % 2), SG + wrow * KG, BG + wrow * KG,
                                       WU + wrow * WPR + WB * (t % 2), SU + wrow * KG, BU + wrow * KG, tile_g, tile_u, tn,
                                       uint(t), frag_home(ushort(lane)));
  if (col + tn < N && live > 0)
    tfqw::store_act<SM / 16>(ag, au, Y + long(row + tm) * N + col + tn, N, live, N - (col + tn), frag_home(ushort(lane)));
}

#define TF_GATHERW(BM)                                                                                                  \
  [[kernel]] void tf_gather_qmmw_nax_##BM(const device bfloat16_t* X [[buffer(0)]], const device uint* W [[buffer(1)]], \
      const device bfloat16_t* S [[buffer(2)]], const device bfloat16_t* B [[buffer(3)]],                                \
      const device int32_t* O [[buffer(4)]], const device int32_t* P [[buffer(5)]], device bfloat16_t* Y [[buffer(6)]],  \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                                                                       \
    threadgroup bfloat16_t tile[64 * (64 + 16 / sizeof(bfloat16_t))];                                                   \
    gatherw<BM>(X, W, S, B, O, Y, P, tile, tg, sg, lane);                                                                \
  }
TF_GATHERW(32)
TF_GATHERW(64)
TF_GATHERW(128)

#define TF_GATHER_GUW(BM)                                                                                               \
  [[kernel]] void tf_gather_guw_nax_##BM(const device bfloat16_t* X [[buffer(0)]], const device uint* WG_ [[buffer(1)]], \
      const device bfloat16_t* SG [[buffer(2)]], const device bfloat16_t* BG [[buffer(3)]],                              \
      const device uint* WU [[buffer(4)]], const device bfloat16_t* SU [[buffer(5)]],                                    \
      const device bfloat16_t* BU [[buffer(6)]], const device int32_t* O [[buffer(7)]],                                  \
      const device int32_t* P [[buffer(8)]], device bfloat16_t* Y [[buffer(9)]],                                         \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],                              \
      uint3 tg [[threadgroup_position_in_grid]]) {                                                                       \
    threadgroup bfloat16_t tile_g[64 * (64 + 16 / sizeof(bfloat16_t))], tile_u[64 * (64 + 16 / sizeof(bfloat16_t))];   \
    gather_guw<BM>(X, WG_, SG, BG, WU, SU, BU, O, Y, P, tile_g, tile_u, tg, sg, lane);                                  \
  }
TF_GATHER_GUW(32)
TF_GATHER_GUW(64)
