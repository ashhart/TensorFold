//! The trellis tile as int8 with its own per-tile scale, on the GPU: the numerical core of the gated int8
//! expert path, checked against the CPU int8 oracle (projection.applyF32TrellisInt8).
#include <metal_stdlib>
using namespace metal;
#include "../nax.h"

struct TfExl3Args {
  uint k, n, rows, halves;
  uint codebook, su_signs, sv_signs, has_bias;
  uint output_f32, input_ids, output_ids, routed;
  uint expert_id, repeat, indexed, experts, splits;
  ulong x_stride, y_stride, tile_k_stride, tile_n_stride;
  uint pair;
};

// A little-endian int16 pair as one stream word (the same read tf_exl3_word makes).
inline uint tf_i8_word(const device ushort* tile, uint i) {
  return uint(tile[2 * i]) | (uint(tile[2 * i + 1]) << 16);
}

// Where state p's window starts: the word index in the low 6 bits, the bit offset next, the next word after.
inline uint tf_i8_map(uint p, uint halves) {
  const uint ring = 128 * halves;
  const uint end = ((p + 1) * halves - ((halves & 1u) ? ((p + 1) & 1u) : 0u)) / 2;
  uint start = end + ring - 16;
  if (start >= ring) start -= ring;
  const uint wi = start >> 5, o = start & 31u, next = wi + 1 == 4 * halves ? 0 : wi + 1;
  return wi | (o << 6) | (next << 11);
}

// The state at a map: the two words it can span, shifted and carried into the top half.
inline ushort tf_i8_state(const device ushort* tile, uint map) {
  const uint lo = tf_i8_word(tile, map & 63u);
  const uint hi = tf_i8_word(tile, map >> 11);
  const uint off = (map >> 6) & 31u;
  const uint carry = (hi >> ((32 - off) & 31u)) & (0u - uint(off != 0));
  return ushort(((lo << off) | carry) >> 16);
}

// One 16x16 weight tile as int8 by its own peak, then a cell's dot with int8 x, scaled by both peaks.
// One threadgroup a (16-row, 16-column) output tile: 32 lanes decode the 256 states eight each, quantize, and
// each lane owns eight output cells.
kernel void tf_exl3_int8_tile(const device half* x [[buffer(0)]],
                              const device uchar* trellis [[buffer(1)]],
                              device float* sums [[buffer(2)]],
                              constant TfExl3Args& a [[buffer(3)]],
                              const device uchar* descriptors [[buffer(5)]],
                              const device half* codebook [[buffer(7)]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint3 tg [[threadgroup_position_in_grid]]) {
  const uint row0 = tg.y * 16u, col0 = tg.x * 16u;
  if (row0 >= a.rows || col0 >= a.n) return;
  threadgroup char wq[16 * 16];
  threadgroup char xq[16 * 16];
  threadgroup float wscale;
  threadgroup float xscale;
  const uint ks = a.tile_k_stride, ns = a.tile_n_stride, halves = a.halves, codebook_id = a.codebook;
  // x's 16 values a block, per live row: quantized by the block's peak over the live rows.
  const uint rows_here = min(16u, a.rows - row0);
  const uint cols_here = min(16u, a.n - col0);
  for (uint kb = 0; kb < a.k; kb += 16) {
    const device ushort* tile = reinterpret_cast<const device ushort*>(
        trellis + ulong(kb / 16) * ks + ulong(col0 / 16) * ns);
    // Eight states a lane: decode, then the tile's peak and its quantized copy.
    float mine[8];
    uint at[8];
    TF_UNROLL
    for (uint j = 0; j < 8; ++j) {
      const uint p = lane * 8u + j, l = p / 8u, e = p % 8u;
      const uint r = 2 * (l % 4) + (e & 1u) + 8 * ((e >> 1) & 1u);      // the state's k index
      const uint c = l / 4 + 8 * (e >> 2);                              // its column
      const ushort state = tf_i8_state(tile, tf_i8_map(p, halves));
      mine[j] = float(codebook[ulong(codebook_id) * 65536u + uint(state)]);
      at[j] = r * 16u + c;
    }
    float peak = 0;
    TF_UNROLL
    for (uint j = 0; j < 8; ++j) peak = max(peak, fabs(mine[j]));
    peak = simd_max(peak);
    const float ws = peak > 0 ? peak / 127.0f : 0.0f;
    TF_UNROLL
    for (uint j = 0; j < 8; ++j) wq[at[j]] = ws > 0 ? char(clamp(rint(mine[j] / ws), -127.0f, 127.0f)) : char(0);
    if (lane == 0) wscale = ws;
    // One lane loads and quantizes the 16 rows' 16 values (the tile is 256 values, 8 a lane).
    float xm[8];
    uint xat[8];
    TF_UNROLL
    for (uint j = 0; j < 8; ++j) {
      const uint p = lane * 8u + j;
      const uint r = p / 16u, c = p % 16u;
      xm[j] = (r < rows_here) ? float(x[ulong(row0 + r) * a.k + kb + c]) : 0.0f;
      xat[j] = r * 16u + c;
    }
    float xpeak = 0;
    TF_UNROLL
    for (uint j = 0; j < 8; ++j) xpeak = max(xpeak, fabs(xm[j]));
    xpeak = simd_max(xpeak);
    const float xs = xpeak > 0 ? xpeak / 127.0f : 1.0f;
    TF_UNROLL
    for (uint j = 0; j < 8; ++j) xq[xat[j]] = xs > 0 ? char(clamp(rint(xm[j] / xs), -127.0f, 127.0f)) : char(0);
    if (lane == 0) xscale = xs;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Eight output cells a lane, each the 16-deep int8 dot scaled by both peaks.
    TF_UNROLL
    for (uint j = 0; j < 8; ++j) {
      const uint p = lane * 8u + j;
      const uint r = p / 16u, c = p % 16u;
      if (r >= rows_here || c >= cols_here) continue;
      int acc = 0;
      for (uint kk = 0; kk < 16; ++kk) acc += int(xq[r * 16u + kk]) * int(wq[kk * 16u + c]);
      const ulong at_sum = ulong(row0 + r) * a.n + col0 + c;
      // The first k block writes, the rest add: the caller need not clear the sums.
      if (kb == 0) sums[at_sum] = float(acc) * wscale * xscale; else sums[at_sum] += float(acc) * wscale * xscale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
