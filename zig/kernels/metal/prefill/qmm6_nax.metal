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

// x (TM 16-row fragments, ld K) times a 6-bit [64 rows, K] block, 64 deep a step: thread t dequantizes row t / 2's
// group t % 2 of the step (wq, scales, biases point at the thread's first group).
template <typename T, int TM>
inline void k_loop6(thread frag<float> (&acc)[TM][2], const device T* x, int K, int live, bool inside,
                    const device uint* wq, const device T* scales, const device T* biases, threadgroup T* tile,
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
    dequant6<T>(wq, *scales, *biases, mine);
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(disable)
    for (int kk = 0; kk < 64; kk += 32) {
      if (live > 0) {
        frag<T> a[TM][2], b[2][2];
        TF_UNROLL
        for (short i = 0; i < 2; i++) {
          TF_UNROLL
          for (short j = 0; j < 2; j++) {
            frag_get(b[j][i], (const threadgroup T*)tile, PAD, tn + 16 * i, kk + 16 * j, home);
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
    wq += 12;
    scales += 2;
    biases += 2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
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

// y = x W^T for 6-bit g32 W [N, K]: 64x64 output tiles, 4 simdgroups of 32x32. P: K N M and y's row stride (0: N).
[[kernel]] void tf_qmm6_t_nax(const device uint* W [[buffer(0)]], const device bfloat16_t* S [[buffer(1)]],
    const device bfloat16_t* B [[buffer(2)]], const device bfloat16_t* X [[buffer(3)]], const device int* P [[buffer(4)]],
    device bfloat16_t* Y [[buffer(5)]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup bfloat16_t tile[64 * (64 + 16 / sizeof(bfloat16_t))];
  const int K = P[0], N = P[1], M = P[2], LD = P[3] > 0 ? P[3] : N;
  const int row = int(tg.y) * 64, col = int(tg.x) * 64, t = int(sg) * 32 + int(lane);
  const int tm = 32 * int(sg / 2), tn = 32 * int(sg % 2), live = min(32, M - (row + tm));
  const long wrow = long(min(col + t / 2, N - 1));
  const int WPR = K * 6 / 32, KG = K / 32;
  frag<float> acc[2][2];
  tfq6::k_loop6<bfloat16_t, 2>(acc, X + long(row + tm) * K, K, live, live == 32, W + wrow * WPR + 6 * (t % 2),
                                S + wrow * KG + (t % 2), B + wrow * KG + (t % 2), tile, tn, uint(t),
                                frag_home(ushort(lane)));
  if (col + tn < N && live > 0)
    tfq6::store<bfloat16_t, 2>(acc, Y + long(row + tm) * LD + col + tn, LD, live, N - (col + tn), frag_home(ushort(lane)));
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
  tfq6::k_loop6<bfloat16_t, SM / 16>(acc, X + long(row + tm) * K, K, live, row + tm + SM <= M,
                                      W + wrow * WPR + 6 * (t % 2), S + wrow * KG + (t % 2), B + wrow * KG + (t % 2),
                                      tile, tn, uint(t), frag_home(ushort(lane)));
  if (col + tn < N && live > 0)
    tfq6::store<bfloat16_t, SM / 16>(acc, Y + long(row + tm) * N + col + tn, N, live, N - (col + tn), frag_home(ushort(lane)));
}

[[kernel]] void tf_gather_qmm6_nax_64(const device bfloat16_t* X [[buffer(0)]], const device uint* W [[buffer(1)]],
    const device bfloat16_t* S [[buffer(2)]], const device bfloat16_t* B [[buffer(3)]], const device int32_t* O [[buffer(4)]],
    const device int32_t* P [[buffer(5)]], device bfloat16_t* Y [[buffer(6)]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup bfloat16_t tile[64 * (64 + 16 / sizeof(bfloat16_t))];
  gather6<64>(X, W, S, B, O, Y, P, tile, tg, sg, lane);
}

[[kernel]] void tf_gather_qmm6_nax_32(const device bfloat16_t* X [[buffer(0)]], const device uint* W [[buffer(1)]],
    const device bfloat16_t* S [[buffer(2)]], const device bfloat16_t* B [[buffer(3)]], const device int32_t* O [[buffer(4)]],
    const device int32_t* P [[buffer(5)]], device bfloat16_t* Y [[buffer(6)]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup bfloat16_t tile[64 * (64 + 16 / sizeof(bfloat16_t))];
  gather6<32>(X, W, S, B, O, Y, P, tile, tg, sg, lane);
}
