// x W^T, MLX affine W, tensor units: exact codes (4-bit + 128) in MMAs, fp32 group sums scaled, bias x fp32 row sums.
#include <metal_stdlib>
using namespace metal;
#include "../nax.h"
using namespace tfp;

// Defines: TF_BITS (2-6 or 8), TF_GROUP (32, 64 or 128), TF_OUT_T (bfloat, or float for sums another pass adds).
#ifndef TF_OUT_T
#define TF_OUT_T bfloat
#endif
constant constexpr int STEP_BYTES = 8 * TF_BITS; // a weight row's bytes in one 64-deep step
constant constexpr int HALF_BYTES = 4 * TF_BITS; // a thread's half of them (32 codes)
constant constexpr int NG = TF_GROUP > 64 ? 1 : 64 / TF_GROUP; // groups a step touches (g128: one, over two steps)
constant constexpr int PAD = 64 + 8;              // the code tile's row pitch (bf16)
constant constexpr float OFF = (TF_BITS == 4 || TF_BITS == 6) ? 128.0f : 0.0f; // bf16(128 + q) is q's bits ORed with 0x4300; bias - 128 scale undoes it

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

// A thread's 32 codes of one weight row in a step (16-byte loads; 8 bytes at 2 bits; words at 3, 5 and 6 bits).
#if TF_BITS == 2
struct Codes {
  uint2 v;
};

inline Codes am_load(const device uchar* w) {
  return Codes{*(const device uint2*)w};
}
#elif TF_BITS % 4 != 0
struct Codes {
  uint v[TF_BITS];
};

inline Codes am_load(const device uchar* w) {
  Codes c;
  TF_UNROLL
  for (short i = 0; i < TF_BITS; i++) {
    c.v[i] = ((const device uint*)w)[i];
  }
  return c;
}
#else
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
#endif

// The codes plus OFF, exactly, as bf16 into the tile row (16-byte stores).
inline void am_codes(thread const Codes& c, threadgroup bfloat* out) {
  threadgroup uint4* o = (threadgroup uint4*)out;
#if TF_BITS == 2
  TF_UNROLL
  for (short i = 0; i < 2; i++) { // a word's 16 codes, as themselves (no offset): byte j holds codes 4j..4j+3
    const uint w = c.v[i];
    const float4 f0 = float4(as_type<uchar4>(w & 0x03030303u)), f1 = float4(as_type<uchar4>((w >> 2) & 0x03030303u));
    const float4 f2 = float4(as_type<uchar4>((w >> 4) & 0x03030303u)), f3 = float4(as_type<uchar4>((w >> 6) & 0x03030303u));
    TF_UNROLL
    for (short j = 0; j < 4; j++) {
      o[2 * i + j / 2][2 * (j % 2)] = as_type<uint>(ushort2(as_type<ushort>(bfloat(f0[j])), as_type<ushort>(bfloat(f1[j]))));
      o[2 * i + j / 2][2 * (j % 2) + 1] = as_type<uint>(ushort2(as_type<ushort>(bfloat(f2[j])), as_type<ushort>(bfloat(f3[j]))));
    }
  }
#elif TF_BITS == 6
  const uint b0 = c.v[0];
  const uint b1 = (c.v[0] >> 24) | (c.v[1] << 8);
  const uint b2 = (c.v[1] >> 16) | (c.v[2] << 16);
  const uint b3 = c.v[2] >> 8;
  const uint b4 = c.v[3];
  const uint b5 = (c.v[3] >> 24) | (c.v[4] << 8);
  const uint b6 = (c.v[4] >> 16) | (c.v[5] << 16);
  const uint b7 = c.v[5] >> 8;
  o[0] = uint4(0x43004300u | (b0 & 0x3fu) | (((b0 >> 6) & 0x3fu) << 16),
               0x43004300u | ((b0 >> 12) & 0x3fu) | (((b0 >> 18) & 0x3fu) << 16),
               0x43004300u | (b1 & 0x3fu) | (((b1 >> 6) & 0x3fu) << 16),
               0x43004300u | ((b1 >> 12) & 0x3fu) | (((b1 >> 18) & 0x3fu) << 16));
  o[1] = uint4(0x43004300u | (b2 & 0x3fu) | (((b2 >> 6) & 0x3fu) << 16),
               0x43004300u | ((b2 >> 12) & 0x3fu) | (((b2 >> 18) & 0x3fu) << 16),
               0x43004300u | (b3 & 0x3fu) | (((b3 >> 6) & 0x3fu) << 16),
               0x43004300u | ((b3 >> 12) & 0x3fu) | (((b3 >> 18) & 0x3fu) << 16));
  o[2] = uint4(0x43004300u | (b4 & 0x3fu) | (((b4 >> 6) & 0x3fu) << 16),
               0x43004300u | ((b4 >> 12) & 0x3fu) | (((b4 >> 18) & 0x3fu) << 16),
               0x43004300u | (b5 & 0x3fu) | (((b5 >> 6) & 0x3fu) << 16),
               0x43004300u | ((b5 >> 12) & 0x3fu) | (((b5 >> 18) & 0x3fu) << 16));
  o[3] = uint4(0x43004300u | (b6 & 0x3fu) | (((b6 >> 6) & 0x3fu) << 16),
               0x43004300u | ((b6 >> 12) & 0x3fu) | (((b6 >> 18) & 0x3fu) << 16),
               0x43004300u | (b7 & 0x3fu) | (((b7 >> 6) & 0x3fu) << 16),
               0x43004300u | ((b7 >> 12) & 0x3fu) | (((b7 >> 18) & 0x3fu) << 16));
#elif TF_BITS % 4 != 0
  TF_UNROLL
  for (short i = 0; i < 8; i++) { // codes 4i .. 4i + 3 of the little-endian bit stream, as themselves
    float f[4];
    TF_UNROLL
    for (short j = 0; j < 4; j++) {
      const short at = TF_BITS * (4 * i + j), word = at / 32, shift = at % 32;
      uint v = c.v[word] >> shift;
      if (shift + TF_BITS > 32) v |= c.v[word + 1] << (32 - shift);
      f[j] = float(v & ((1u << TF_BITS) - 1u));
    }
    o[i / 2][2 * (i % 2)] = as_type<uint>(ushort2(as_type<ushort>(bfloat(f[0])), as_type<ushort>(bfloat(f[1]))));
    o[i / 2][2 * (i % 2) + 1] = as_type<uint>(ushort2(as_type<ushort>(bfloat(f[2])), as_type<ushort>(bfloat(f[3]))));
  }
#elif TF_BITS == 4
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

#if defined(TF_GLM_M3_FAST) && defined(TF_SIMD_FRAGS)
// GLM pre-M5: codes in [K=64][N=64] layout so frag_get reads transposed weights with zero bank conflict.
inline void am_codes_kn(thread const Codes& c, threadgroup bfloat* tile, int w_n, int w_kh) {
#if TF_BITS == 2
  TF_UNROLL
  for (short i = 0; i < 2; i++) {
    const uint w = c.v[i];
    TF_UNROLL
    for (short j = 0; j < 16; j++) tile[(w_kh * 32 + i * 16 + j) * PAD + w_n] = bfloat(float((w >> (2 * j)) & 3u));
  }
#elif TF_BITS == 6
  TF_UNROLL
  for (short j = 0; j < 32; j++) {
    const short at = 6 * j, word = at / 32, shift = at % 32;
    uint v = c.v[word] >> shift;
    if (shift + 6 > 32) v |= c.v[word + 1] << (32 - shift);
    tile[(w_kh * 32 + j) * PAD + w_n] = as_type<bfloat>(ushort(0x4300u | (v & 0x3fu)));
  }
#elif TF_BITS % 4 != 0
  TF_UNROLL
  for (short j = 0; j < 32; j++) {
    const short at = TF_BITS * j, word = at / 32, shift = at % 32;
    uint v = c.v[word] >> shift;
    if (shift + TF_BITS > 32) v |= c.v[word + 1] << (32 - shift);
    tile[(w_kh * 32 + j) * PAD + w_n] = bfloat(float(v & ((1u << TF_BITS) - 1u)));
  }
#elif TF_BITS == 4
  TF_UNROLL
  for (short i = 0; i < 4; i++) {
    const uint w = c.v[0][i];
    TF_UNROLL
    for (short j = 0; j < 8; j++) {
      tile[(w_kh * 32 + i * 8 + j) * PAD + w_n] = as_type<bfloat>(ushort(0x4300u | ((w >> (4 * j)) & 0xfu)));
    }
  }
#else
  TF_UNROLL
  for (short i = 0; i < 8; i++) {
    const uchar4 f = as_type<uchar4>(c.v[i / 4][i % 4]);
    const int b = w_kh * 32 + i * 4;
    tile[b * PAD + w_n] = bfloat(float(f.x));
    tile[(b + 1) * PAD + w_n] = bfloat(float(f.y));
    tile[(b + 2) * PAD + w_n] = bfloat(float(f.z));
    tile[(b + 3) * PAD + w_n] = bfloat(float(f.w));
  }
#endif
}
#endif

// A block of the code tile as the transposed right operand: 8-byte loads on M5 (PAD rows, 16-column blocks).
#ifndef TF_SIMD_FRAGS
#define am_tile_frag frag_get_tg
#else
#define am_tile_frag frag_get_t
#endif

// A lane's two rows (home.y and home.y + 8) of each of its TM fragments: their starts in x and in the row sums.
template <int TM>
struct AmRows {
  const device bfloat* x[TM][2];
  const device float* s[TM][2];
};
#include "affine_mm_nonfinite.h"

// One step's MMAs and group epilogues for the first TL of TM row fragments (TL < TM: a tile's tail, the rest dead).
template <int TM, int TL>
inline void am_step(thread frag<float> (&acc)[TM][2], thread const AmRows<TM>& rows, int k,
                    const threadgroup bfloat* tile, const threadgroup float* sb, int tn, short2 home) {
  frag<float> part[TL][2];
#ifdef TF_AM_EXACT_NONFINITE
  bool raw[TL][2];
  frag<float> exact[TL][2];
#endif
  TF_UNROLL
  for (short q = 0; q < 4; q++) { // the step's four 16-deep slices, every row fragment's chain interleaved
    frag<bfloat> b0, b1, a[TL];
#if defined(TF_GLM_M3_FAST) && defined(TF_SIMD_FRAGS)
    frag_get(b0, tile, PAD, 16 * q, tn, home);
    frag_get(b1, tile, PAD, 16 * q, tn + 16, home);
#else
    am_tile_frag(b0, tile, PAD, tn, 16 * q, home);
    am_tile_frag(b1, tile, PAD, tn + 16, 16 * q, home);
#endif
    TF_UNROLL
    for (short m = 0; m < TL; m++) {
      a[m] = am_frag(rows.x[m][0], rows.x[m][1], k + 16 * q, home);
    }
#ifdef TF_AM_EXACT_NONFINITE
    if (q % (TF_GROUP / 16) == 0) am_nonfinite<TM, TL>(rows, k + 16 * q, raw);
#endif
    TF_UNROLL
    for (short m = 0; m < TL; m++) {
      if (q % (TF_GROUP / 16) == 0) {
        am_first(part[m][0], part[m][1], a[m], b0, b1);
      } else {
        mma_16x32<false, true>(part[m][0], part[m][1], a[m], b0, b1);
      }
#ifdef TF_AM_EXACT_NONFINITE
      if (simd_any(raw[m][0] || raw[m][1])) {
        frag<bfloat> r0, r1;
        TF_UNROLL
        for (short e = 0; e < 8; e++) {
          r0[e] = bfloat(b0[e] - OFF);
          r1[e] = bfloat(b1[e] - OFF);
        }
        if (q % (TF_GROUP / 16) == 0) {
          am_first(exact[m][0], exact[m][1], a[m], r0, r1);
        } else {
          mma_16x32<false, true>(exact[m][0], exact[m][1], a[m], r0, r1);
        }
      }
#endif
    }
    if ((q + 1) % (TF_GROUP / 16) == 0) { // a group's sums done: its scale, its bias times the row sums
      const int g = q / (TF_GROUP / 16);
      const threadgroup float* sg = sb + g * 128;
      TF_UNROLL
      for (short m = 0; m < TL; m++) {
        const float xr0 = rows.s[m][0][k / TF_GROUP + g], xr1 = rows.s[m][1][k / TF_GROUP + g];
        TF_UNROLL
        for (short j = 0; j < 2; j++) {
          TF_UNROLL
          for (short c = 0; c < 4; c++) {
            const int col = tn + 16 * j + home.x + TF_COL(c);
            const float s = sg[col];
#ifdef TF_AM_EXACT_NONFINITE
            float b0 = sg[64 + col], b1 = b0;
            if (raw[m][0]) b0 += OFF * s;
            if (raw[m][1]) b1 += OFF * s;
            const float p0 = raw[m][0] ? exact[m][j][c] : part[m][j][c];
            const float p1 = raw[m][1] ? exact[m][j][4 + c] : part[m][j][4 + c];
            acc[m][j][c] += s * p0 + b0 * xr0;
            acc[m][j][4 + c] += s * p1 + b1 * xr1;
#else
            const float b = sg[64 + col];
            acc[m][j][c] += s * part[m][j][c] + b * xr0;
            acc[m][j][4 + c] += s * part[m][j][4 + c] + b * xr1;
#endif
          }
        }
      }
    }
  }
}


// TM 16-row fragments (`live` rows; a dead row's start is a live one's) times a [64, K] code block.
template <int TM>
inline void am_k_loop(thread frag<float> (&acc)[TM][2], thread const AmRows<TM>& rows, int K, int live,
                      const device uchar* wq, const device bfloat* scales, const device bfloat* biases,
                      threadgroup bfloat* tile, threadgroup float* sb, int tn, uint t, short2 home) {
#if defined(TF_GLM_M3_FAST) && defined(TF_SIMD_FRAGS)
  const int w_n = int(t / 2), w_kh = int(t % 2);
#else
  threadgroup bfloat* mine = tile + (t / 2) * PAD + 32 * (t % 2);
#endif
  const int g_mine = TF_GROUP == 32 ? int(t % 2) : 0; // at g32 each thread of a pair takes one group's scale and bias
#if TF_GROUP == 128
  frag<float> part[TM][2]; // a group's sums across its two steps
#ifdef TF_AM_EXACT_NONFINITE
  frag<float> exact[TM][2];
  bool raw[TM][2];
#endif
#endif
  TF_UNROLL
  for (short i = 0; i < TM; i++) {
    acc[i][0] = frag<float>(0);
    acc[i][1] = frag<float>(0);
  }
  for (int k = 0; k < K; k += 64) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
#if defined(TF_GLM_M3_FAST) && defined(TF_SIMD_FRAGS)
    am_codes_kn(am_load(wq), tile, w_n, w_kh);
#else
    am_codes(am_load(wq), mine);
#endif
    if (TF_GROUP == 32 || t % 2 == 0) {
      const float s = float(scales[g_mine]);
      sb[g_mine * 128 + t / 2] = s;
      sb[g_mine * 128 + 64 + t / 2] = float(biases[g_mine]) - OFF * s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#if TF_GROUP == 128
    if (live > 16 * (TM - 1)) {
      am_step128<TM, TM>(acc, part, rows, k, tile, sb,
#ifdef TF_AM_EXACT_NONFINITE
                         exact, raw,
#endif
                         tn, home);
    } else if (TM > 1 && live > 0) {
      am_step128<TM, 1>(acc, part, rows, k, tile, sb,
#ifdef TF_AM_EXACT_NONFINITE
                        exact, raw,
#endif
                        tn, home);
    }
    wq += STEP_BYTES;
    scales += k % 128 == 64 ? 1 : 0;
    biases += k % 128 == 64 ? 1 : 0;
#else
    if (live > 16 * (TM - 1)) {
      am_step<TM, TM>(acc, rows, k, tile, sb, tn, home);
    } else if (TM > 1 && live > 0) { // a tail with only the first fragment live
      am_step<TM, 1>(acc, rows, k, tile, sb, tn, home);
    }
    wq += STEP_BYTES;
    scales += NG;
    biases += NG;
#endif
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

// A simdgroup's row starts: x rows [row, row + rows) and their sums, or with `order` sorted token rows (order / topk).
template <int TM>
inline AmRows<TM> am_rows(const device bfloat* x, int ldx, const device float* xs, int ldxs, int row, int rows, int tm,
                          short2 home, const device int32_t* order, int topk) {
  AmRows<TM> r;
  TF_UNROLL
  for (short m = 0; m < TM; m++) {
    TF_UNROLL
    for (short h = 0; h < 2; h++) {
      const int i = tm + 16 * m + home.y + 8 * h;
      const int at = row + (i < rows ? i : 0); // a dead row reads a live one; its sums never reach y
      const long src = order != nullptr ? long(order[at] / topk) : long(at);
      r.x[m][h] = x + src * ldx;
      r.s[m][h] = xs + src * ldxs;
    }
  }
  return r;
}

// Rows [row, row + rows) of x times weight rows [wrow0, wrow0 + 64) of a [*, K] matrix into y (ld N) at column col.
template <int BM>
inline void am_block(const device bfloat* x, int ldx, const device float* xs, int ldxs, int row, int rows, int K,
                     int ldy, int col, long wrow0, const device uint32_t* w, const device bfloat* scales,
                     const device bfloat* biases, device TF_OUT_T* y, threadgroup bfloat* tile, threadgroup float* sb,
                     uint sg, uint lane, const device int32_t* dst = nullptr, const device int32_t* order = nullptr,
                     int topk = 1) {
  constexpr int SM = BM / 2;
  constexpr int TM = SM / 16;
  const int t = int(sg) * 32 + int(lane);
  const int tm = SM * int(sg / 2), tn = 32 * int(sg % 2), live = clamp(rows - tm, 0, SM);
  const long wrow = wrow0 + t / 2;
  const short2 home = frag_home(ushort(lane));
  const AmRows<TM> starts = am_rows<TM>(x, ldx, xs, ldxs, row, rows, tm, home, order, topk);
  frag<float> acc[TM][2];
  am_k_loop<TM>(acc, starts, K, live, (const device uchar*)w + wrow * (K * TF_BITS / 8) + HALF_BYTES * (t % 2),
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
#if defined(TF_GLM_M3_FAST) && defined(TF_SIMD_FRAGS)
  constexpr int BM = 32;
#else
  constexpr int BM = 64;
#endif
  const int row = int(tg.y) * BM, col = int(tg.x) * 64, b = int(tg.z);
  am_block<BM>(X + long(b) * st.x_batch, st.x_row, XS + long(b) * st.sums_batch * (a.k / TF_GROUP),
               st.sums_row * (a.k / TF_GROUP), row, min(BM, a.rows - row), a.k, st.y_row, col,
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
    am_block<BM>(X, a.k, XS, a.k / TF_GROUP, row, rows, a.k, a.n, col, long(expert) * a.n + col, W, S, B, Y,   \
                 tile, sb, sg, lane);                                                                                  \
  }
TF_GATHER(32)
TF_GATHER(64)

// The gathers with each sorted row's output put back at row DST[row] (e.g. the (row, slot) pairs' own order).
#ifdef TF_AM_MAPPED_EXPERTS
#define TF_AM_MAP_ARG const device int32_t* EA [[buffer(13)]],
#define TF_AM_EXPERT(e) EA[e]
#else
#define TF_AM_MAP_ARG
#define TF_AM_EXPERT(e) (e)
#endif

#define TF_GATHER_SCATTER(BM)                                                                                           \
  [[kernel]] void tf_affine_gather_scatter_##BM(                                                                        \
      const device bfloat* X [[buffer(0)]], const device uint32_t* W [[buffer(1)]],                                    \
      const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]],                                      \
      const device float* XS [[buffer(4)]], constant MmArgs& a [[buffer(5)]], device TF_OUT_T* Y [[buffer(6)]],       \
      const device int32_t* OFFS [[buffer(7)]], const device int32_t* DST [[buffer(8)]], TF_AM_MAP_ARG                \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],                            \
      uint3 tg [[threadgroup_position_in_grid]]) {                                                                     \
    threadgroup bfloat tile[64 * PAD];                                                                                 \
    threadgroup float sb[NG * 128];                                                                                    \
    int expert, row, rows;                                                                                             \
    if (!am_tile<BM>(OFFS, a.experts, a.rows, int(tg.y), lane, expert, row, rows)) return;                            \
    const int col = int(tg.x) * 64;                                                                                    \
    am_block<BM>(X, a.k, XS, a.k / TF_GROUP, row, rows, a.k, a.n, col, long(TF_AM_EXPERT(expert)) * a.n + col, W, S, B, Y, \
                 tile, sb, sg, lane, DST);                                                                             \
  }
TF_GATHER_SCATTER(32)
TF_GATHER_SCATTER(64)
