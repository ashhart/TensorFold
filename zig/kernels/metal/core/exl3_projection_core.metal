// Native EXL3: stored trellis tiles, row-independent compensated dots, and normalized Sylvester rotations.

// Provenance: the model-facing order follows the release's inference/model.py (revision 314113f8).
#include <metal_stdlib>
using namespace metal;

struct TfExl3Args {
  uint k, n, rows, halves;
  uint codebook, su_signs, sv_signs, has_bias;
  uint output_f32, input_ids, output_ids, routed;
  uint expert_id, repeat, indexed, experts, splits;
  ulong x_stride, y_stride, tile_k_stride, tile_n_stride;
};

// Metal 3 / argument-buffer tier 2 descriptors are populated once from resident buffers' GPU addresses.
struct TfExl3Expert {
  const device uchar* trellis;
  const device ushort* su;
  const device ushort* sv;
  const device half* bias;
  ulong tile_k_stride, tile_n_stride;
  uint halves, codebook, su_signs, sv_signs, has_bias, reserved;
};

inline bool tf_exl3_active(const device uint* routes, constant TfExl3Args& a, uint row) {
  return row < a.rows && (a.indexed ? routes[row] < a.experts : (!a.routed || routes[row] == a.expert_id));
}

inline float tf_exl3_scale(const device ushort* p, uint i, uint signs) {
  if (signs) return ((p[i / 16] >> (i % 16)) & 1u) ? -1.0f : 1.0f;
  return float(as_type<half>(p[i]));
}

// Two little-endian int16 words form one MSB-first stream word, including unaligned borrowed tiles.
inline uint tf_exl3_word(const device ushort* tile, uint i) {
  return uint(tile[2 * i]) | (uint(tile[2 * i + 1]) << 16);
}

inline ushort tf_exl3_state(const device ushort* tile, uint halves, uint p) {
  const uint ring = 128 * halves;
  const uint end = ((p + 1) * halves - ((halves & 1u) ? ((p + 1) & 1u) : 0u)) / 2;
  uint start = end + ring - 16;
  if (start >= ring) start -= ring;
  const uint wi = start >> 5, off = start & 31u;
  const uint lo = tf_exl3_word(tile, wi);
  if (off <= 16) return ushort(lo >> (16 - off));
  const uint next = wi + 1 == 4 * halves ? 0 : wi + 1;
  const uint hi = tf_exl3_word(tile, next);
  return ushort((lo << (off - 16)) | (hi >> (48 - off)));
}

inline ushort tf_exl3_mapped_state(const device ushort* tile, uint map) {
  const uint wi = map & 63u, off = (map >> 6) & 31u;
  const uint lo = tf_exl3_word(tile, wi);
  const uint hi = tf_exl3_word(tile, map >> 11);
  const uint carry = (hi >> ((32 - off) & 31u)) & (0u - uint(off != 0));
  return ushort(((lo << off) | carry) >> 16);
}

// Both operations are exact in FP32 before the single FP16 round; do not split the mul1 FMA.
inline half tf_exl3_value(ushort state, uint codebook) {
  uint x = uint(state) * (codebook == 2 ? 0x83DCD12Du : codebook == 1 ? 0xCBAC1FEDu : 89226354u);
  if (codebook == 2) {
    const uint h = 1024 + (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + (x >> 24);
    return half(fma(float(h), float(as_type<half>(ushort(0x1EEE))), float(as_type<half>(ushort(0xC931)))));
  }
  if (codebook == 0) x += 64248484u;
  x = (x & 0x8FFF8FFFu) ^ 0x3B603B60u;
  return half(float(as_type<half>(ushort(x))) + float(as_type<half>(ushort(x >> 16))));
}

// A scalar lane's state window: the ring word, its bit offset and the next ring word.
struct TfExl3ScalarWindow {
  uint word, bit, next;
};

inline TfExl3ScalarWindow tf_exl3_scalar_window(uint p, uint halves) {
  const uint ring = 128 * halves;
  const uint end = ((p + 1) * halves - ((halves & 1u) ? ((p + 1) & 1u) : 0u)) / 2;
  uint start = end + ring - 16;
  if (start >= ring) start -= ring;
  const uint word = start >> 5, bit = start & 31u;
  TfExl3ScalarWindow w;
  w.word = word;
  w.bit = bit;
  w.next = word + 1 == 4 * halves ? 0 : word + 1;
  return w;
}

inline half tf_exl3_scalar_value(const device ushort* tile, TfExl3ScalarWindow w, uint codebook) {
  const uint lo = tf_exl3_word(tile, w.word);
  const uint hi = tf_exl3_word(tile, w.next);
  const uint carry = (hi >> ((32 - w.bit) & 31u)) & (0u - uint(w.bit != 0));
  return tf_exl3_value(ushort(((lo << w.bit) | carry) >> 16), codebook);
}


// Lane l holds indices l, l+32, l+64, l+96; all seven butterfly stages have a fixed FP32 order.
inline void tf_exl3_hadamard(thread float (&v)[4], uint lane) {
  for (uint d = 1; d < 32; d *= 2) {
    for (uint j = 0; j < 4; ++j) {
      const float peer = simd_shuffle_xor(v[j], ushort(d));
      v[j] = (lane & d) ? peer - v[j] : v[j] + peer;
    }
  }
  const float a = v[0], b = v[1], c = v[2], d = v[3];
  v[0] = a + b; v[1] = a - b; v[2] = c + d; v[3] = c - d;
  const float e = v[0], f = v[1], g = v[2], h = v[3];
  v[0] = e + g; v[1] = f + h; v[2] = e - g; v[3] = f - h;
}

kernel void tf_exl3_rotate_in(const device ushort* x [[buffer(0)]],
                             const device ushort* su [[buffer(1)]],
                             device half* rotated [[buffer(2)]],
                             constant TfExl3Args& a [[buffer(3)]],
                             const device uint* input_ids [[buffer(4)]],
                             const device uint* routes [[buffer(5)]],
                             const device TfExl3Expert* descriptors [[buffer(6)]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]],
                             uint3 tptg [[threads_per_threadgroup]],
                             uint3 tg [[threadgroup_position_in_grid]]) {
  // Four SIMDgroups a threadgroup, each a 128-element block of the same row.
  const uint row = tg.y;
  if (!tf_exl3_active(routes, a, row)) return;
  const device ushort* scales = a.indexed ? descriptors[routes[row]].su : su;
  const uint signs = a.indexed ? descriptors[routes[row]].su_signs : a.su_signs;
  const uint source = a.input_ids ? input_ids[row] : row / a.repeat;
  const uint base = (tg.x * (tptg.x / 32u) + sg) * 128;
  if (base >= a.k) return;
  float v[4];
  for (uint j = 0; j < 4; ++j) {
    const uint k = base + lane + 32 * j;
    v[j] = as_type<float>(uint(x[ulong(source) * a.x_stride + k]) << 16) * tf_exl3_scale(scales, k, signs);
  }
  tf_exl3_hadamard(v, lane);
  for (uint j = 0; j < 4; ++j)
    rotated[ulong(row) * a.k + base + lane + 32 * j] = half(v[j] * 0.08838834764831845f);
}

#ifndef TF_UNROLL
#define TF_UNROLL _Pragma("clang loop unroll(full)")
#endif

// The bit cursor for element p, as the prompt kernels' map: word, bit offset and wrap word in one integer.
inline uint tf_exl3_map_of(uint p, uint halves) {
  const uint ring = 128 * halves;
  const uint end = ((p + 1) * halves - ((halves & 1u) ? ((p + 1) & 1u) : 0u)) / 2;
  uint start = end + ring - 16;
  if (start >= ring) start -= ring;
  const uint wi = start >> 5, off = start & 31u, next = wi + 1 == 4 * halves ? 0 : wi + 1;
  return wi | (off << 6) | (next << 11);
}

// The same value as tf_exl3_value, read from the 3 x 65536-half table the engine already binds at buffer 7.
inline half tf_exl3_table_value(const device half* table, uint codebook, ushort state) {
  return table[codebook * 65536u + uint(state)];
}

// The mul1 value depends only on the state's byte sum.
inline uint tf_exl3_mul1_index(ushort state) {
  const uint x = uint(state) * 0x83DCD12Du;
  const uint bs = (x & 0x00FF00FFu) + ((x >> 8u) & 0x00FF00FFu);
  return (bs * 0x00010001u) >> 16u;
}

// The mul1 value for a byte sum: the same half the arithmetic path and the resident table hold.
inline half tf_exl3_mul1_value(uint bs) {
  const half hs = as_type<half>(ushort(bs + 0x6400u));
  return half(fma(float(hs), float(as_type<half>(ushort(0x1EEEu))), float(as_type<half>(ushort(0xC931u)))));
}

// The lane's eight windows at compile-time shifts.
template <uint S, uint J>
inline ushort tf_exl3_state_static(uint c_cur, uint c_next) {
  const uint bit = S * J;
  if (bit <= 16u) return ushort((c_cur >> (16u - bit)) & 0xFFFFu);
  if (bit <= 31u) return ushort(((c_cur << (bit - 16u)) | (c_next >> (48u - bit))) & 0xFFFFu);
  return ushort((c_next >> (48u - bit)) & 0xFFFFu);
}

// One SIMDgroup owns eight rows and sixteen columns. Only its current 16x16 weight tile is decoded.
template <bool Hoist, bool Table>
inline void tf_exl3_matmul_impl(const device half* x, const device uchar* trellis,
                                device float* sums, constant TfExl3Args& a,
                                const device uint* routes, const device TfExl3Expert* descriptors,
                                device float* partials, const device half* table,
                                uint lane, uint3 tg,
                                threadgroup float* wx, threadgroup float* xx) {
  const uint split = a.splits > 1 ? tg.y : 0;
  // A split dispatch carries a row group in tg.z (the tile already spans eight rows).
  const uint rb = a.splits > 1 ? (a.indexed == 1 ? tg.z : tg.z * 8) : (a.indexed == 1 ? tg.y : tg.y * 8), nb = tg.x * 16;
  if (!simd_any(tf_exl3_active(routes, a, rb + (a.indexed == 1 ? 0 : lane % 8)))) return;
  const uint id = a.indexed ? routes[rb] : 0;
  if (a.indexed && descriptors[id].reserved != 0 && nb >= descriptors[id].reserved) return;
  const device uchar* data = a.indexed ? descriptors[id].trellis : trellis;
  const ulong ks = a.indexed ? descriptors[id].tile_k_stride : a.tile_k_stride;
  const ulong ns = a.indexed ? descriptors[id].tile_n_stride : a.tile_n_stride;
  const uint halves = a.indexed ? descriptors[id].halves : a.halves;
  const uint codebook = a.indexed ? descriptors[id].codebook : a.codebook;
  uint maps[8];
  // Eight consecutive states a lane of the sixteen-column tile: the three stream words their windows can touch.
  uint at[8], off[8], sh[8], rel[8];
  const uint words = 4 * halves;
  const uint ring = 128 * halves;
  const uint wi0 = tf_exl3_map_of(lane * 8, halves) & 63u;
  if (Hoist) {
    const uint cur = (wi0 << 5) | ((tf_exl3_map_of(lane * 8, halves) >> 6) & 31u);
    TF_UNROLL
    for (uint j = 0; j < 8; ++j) {
      const uint p = lane * 8 + j, l = p / 8, e = p % 8;
      uint c8 = cur + j * (halves >> 1);
      if (c8 >= ring) c8 -= ring;
      off[j] = c8 & 31u;
      sh[j] = (32u - off[j]) & 31u;
      rel[j] = (c8 >> 5) >= wi0 ? (c8 >> 5) - wi0 : (c8 >> 5) + words - wi0;
      at[j] = (2 * (l % 4) + (e & 1u) + 8 * ((e >> 1) & 1u)) * 16u + (l / 4 + 8 * (e >> 2));
    }
  }
  {
    // The odd-width fallback still needs the per-state map.
    for (uint j = 0; j < 8; ++j) {
      const uint p = lane + 32 * j, ring2 = 128 * halves;
      const uint end = ((p + 1) * halves - ((halves & 1u) ? ((p + 1) & 1u) : 0u)) / 2;
      uint start = end + ring2 - 16;
      if (start >= ring2) start -= ring2;
      const uint wi = start >> 5, o = start & 31u, next = wi + 1 == 4 * halves ? 0 : wi + 1;
      maps[j] = wi | (o << 6) | (next << 11);
    }
  }
  float high[4]={0,0,0,0}, low[4]={0,0,0,0};
  const uint blocks = a.k / 16;
  const uint b0 = a.splits > 1 ? blocks * split / a.splits : 0;
  const uint b1 = a.splits > 1 ? blocks * (split + 1) / a.splits : blocks;
  uint prefetch[4];
  {
    uint a1 = wi0 + 1; if (a1 >= words) a1 -= words;
    uint a2 = wi0 + 2; if (a2 >= words) a2 -= words;
    uint a3 = wi0 + 3; if (a3 >= words) a3 -= words;
    const device ushort* t0 = reinterpret_cast<const device ushort*>(data + ulong(b0) * ks + ulong(nb / 16) * ns);
    prefetch[0] = tf_exl3_word(t0, wi0);
    prefetch[1] = tf_exl3_word(t0, a1);
    prefetch[2] = tf_exl3_word(t0, a2);
    prefetch[3] = tf_exl3_word(t0, a3);
  }
  for (uint b = b0; b < b1; ++b) {
    const uint kb = b * 16;
    const device ushort* tile = reinterpret_cast<const device ushort*>(
        data + ulong(kb / 16) * ks + ulong(nb / 16) * ns);
    if (Hoist and (halves & 1u) == 0 and halves <= 16) {
      // A block early, as the prompt tile does: the words' latency overlaps the multiply instead of stalling.
      uint at1 = wi0 + 1; if (at1 >= words) at1 -= words;
      uint at2 = wi0 + 2; if (at2 >= words) at2 -= words;
      uint at3 = wi0 + 3; if (at3 >= words) at3 -= words;
      const uint w0 = prefetch[0], w1 = prefetch[1], w2 = prefetch[2], w3 = prefetch[3];
      if (b + 1 < b1) {
        const device ushort* tn = reinterpret_cast<const device ushort*>(data + ulong(b + 1) * ks + ulong(nb / 16) * ns);
        prefetch[0] = tf_exl3_word(tn, wi0);
        prefetch[1] = tf_exl3_word(tn, at1);
        prefetch[2] = tf_exl3_word(tn, at2);
        prefetch[3] = tf_exl3_word(tn, at3);
      }
      TF_UNROLL
      for (uint j = 0; j < 8; ++j) {
        const uint lo = rel[j] == 0 ? w0 : rel[j] == 1 ? w1 : w2;
        const uint hi = rel[j] == 0 ? w1 : rel[j] == 1 ? w2 : w3;
        const uint carry = (hi >> sh[j]) & (0u - uint(off[j] != 0));
        const ushort s = ushort(((lo << off[j]) | carry) >> 16);
        wx[at[j]] = float(Table ? tf_exl3_table_value(table, codebook, s) : tf_exl3_value(s, codebook));
      }
    } else for (uint j = 0; j < 8; ++j) {
      const uint p = lane + 32 * j, l = p / 8, e = p % 8;
      const uint r = 2 * (l % 4) + (e & 1u) + 8 * ((e >> 1) & 1u);
      const uint c = l / 4 + 8 * (e >> 2);
      const ushort state = Hoist ? tf_exl3_mapped_state(tile, maps[j]) : tf_exl3_state(tile, halves, p);
      wx[r * 16 + c] = float(Table ? tf_exl3_table_value(table, codebook, state) : tf_exl3_value(state, codebook));
    }
    for (uint j = lane; j < 128; j += 32) {
      const uint row = rb + j / 16;
      xx[j] = (a.indexed != 1 || j / 16 == 0) && tf_exl3_active(routes, a, row)
          ? float(x[ulong(row) * a.k + kb + j % 16]) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i=0;i<4;++i) {
      const uint at=lane+32*i, row=rb+at/16, col=at%16;
      if ((a.indexed==1 && at/16!=0) || !tf_exl3_active(routes,a,row)) continue;
      for (uint k=0;k<16;++k) {
        high[i] += xx[(at/16)*16+k]*wx[k*16+col];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint i=0;i<4;++i) {
    const uint at=lane+32*i, row=rb+at/16;
    if ((a.indexed!=1 || at/16==0) && tf_exl3_active(routes,a,row)) {
      if (a.splits > 1) partials[(ulong(split) * a.rows + rb + at / 16) * a.n + nb + at % 16] = high[i] + low[i];
      else sums[ulong(row)*a.n+nb+at%16]=high[i]+low[i];
    }
  }
}

kernel void tf_exl3_matmul(const device half* x [[buffer(0)]],
                          const device uchar* trellis [[buffer(1)]],
                          device float* sums [[buffer(2)]],
                          constant TfExl3Args& a [[buffer(3)]],
                          const device uint* routes [[buffer(4)]],
                          const device TfExl3Expert* descriptors [[buffer(5)]],
                          device float* partials [[buffer(6)]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup float wx[256], xx[128];
  tf_exl3_matmul_impl<false, false>(x, trellis, sums, a, routes, descriptors, partials, nullptr, lane, tg, wx, xx);
}


kernel void tf_exl3_matmul_hoisted(const device half* x [[buffer(0)]],
                                  const device uchar* trellis [[buffer(1)]],
                                  device float* sums [[buffer(2)]],
                                  constant TfExl3Args& a [[buffer(3)]],
                                  const device uint* routes [[buffer(4)]],
                                  const device TfExl3Expert* descriptors [[buffer(5)]],
                                  device float* partials [[buffer(6)]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup float wx[256], xx[128];
  tf_exl3_matmul_impl<true, false>(x, trellis, sums, a, routes, descriptors, partials, nullptr, lane, tg, wx, xx);
}

// The one-row / narrow-window arm of the same body with the codebook as a table read (buffer 7).
kernel void tf_exl3_matmul_tab(const device half* x [[buffer(0)]],
                               const device uchar* trellis [[buffer(1)]],
                               device float* sums [[buffer(2)]],
                               constant TfExl3Args& a [[buffer(3)]],
                               const device uint* routes [[buffer(4)]],
                               const device TfExl3Expert* descriptors [[buffer(5)]],
                               device float* partials [[buffer(6)]],
                               const device half* table [[buffer(7)]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup float wx[256], xx[128];
  tf_exl3_matmul_impl<true, true>(x, trellis, sums, a, routes, descriptors, partials, table, lane, tg, wx, xx);
}

// Qualification only: compare every arithmetic state against the independent CPU-generated FP16 table.
kernel void tf_exl3_codebook_check(device half* arithmetic [[buffer(0)]],
                                  uint i [[thread_position_in_grid]]) {
  if (i < 3 * 65536) arithmetic[i] = tf_exl3_value(ushort(i % 65536), i / 65536);
}

// The rotate_out value of one output: the Hadamard result scaled, signed by sv, plus the bias; the caller rounds it.
inline float tf_exl3_out_value(float v, const device ushort* scales, const device half* biases, uint n, uint signs, uint has_bias) {
  float out = (v * 0.08838834764831845f) * tf_exl3_scale(scales, n, signs);
  if (has_bias) out += float(biases[n]);
  return out;
}

kernel void tf_exl3_rotate_out(const device float* sums [[buffer(0)]],
                              const device ushort* sv [[buffer(1)]],
                              const device half* bias [[buffer(2)]],
                              device uchar* y [[buffer(3)]],
                              constant TfExl3Args& a [[buffer(4)]],
                              const device uint* output_ids [[buffer(5)]],
                              const device uint* routes [[buffer(6)]],
                              const device TfExl3Expert* descriptors [[buffer(7)]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]],
                              uint3 tptg [[threads_per_threadgroup]],
                              uint3 tg [[threadgroup_position_in_grid]]) {
  const uint row = tg.y, base = (tg.x * (tptg.x / 32u) + sg) * 128;
  const bool active = tf_exl3_active(routes, a, row);
  const uint id = a.indexed ? routes[row] : 0;
  // An indexed batch may carry entries of different output width (descriptor.reserved, 0 = the call's n).
  const uint width = (a.indexed && active && descriptors[id].reserved != 0) ? descriptors[id].reserved : a.n;
  if (base >= width || !active) return;
  const device ushort* scales = a.indexed ? descriptors[id].sv : sv;
  const device half* biases = a.indexed ? descriptors[id].bias : bias;
  const uint signs = a.indexed ? descriptors[id].sv_signs : a.sv_signs;
  const uint has_bias = a.indexed ? descriptors[id].has_bias : a.has_bias;
  float v[4];
  for (uint j = 0; j < 4; ++j) v[j] = sums[ulong(row) * a.n + base + lane + 32 * j];
  tf_exl3_hadamard(v, lane);
  const uint dest = a.output_ids ? output_ids[row] : row;
  for (uint j = 0; j < 4; ++j) {
    const uint n = base + lane + 32 * j;
    const float out = tf_exl3_out_value(v[j], scales, biases, n, signs, has_bias);
    const ulong at = ulong(dest) * a.y_stride + n;
    if (a.output_f32) reinterpret_cast<device float*>(y)[at] = out;
    else reinterpret_cast<device bfloat*>(y)[at] = bfloat(out);
  }
}

// The pre-rotated int8 pack's epilogue.
kernel void tf_exl3_epilogue(const device float* sums [[buffer(0)]],
                             device uchar* y [[buffer(1)]],
                             constant TfExl3Args& a [[buffer(2)]],
                             uint2 gid [[thread_position_in_grid]]) {
  const uint r = gid.y, c = gid.x;
  if (r >= a.rows || c >= a.n) return;
  reinterpret_cast<device bfloat*>(y)[ulong(r) * a.y_stride + c] = bfloat(sums[ulong(r) * a.n + c]);
}

// Qualification only: compare mapped windows with independent CPU 64-bit concatenation.

kernel void tf_exl3_state_window_check(const device uint4* inputs [[buffer(0)]],
                              device ushort* outputs [[buffer(1)]],
                              constant uint& count [[buffer(2)]],
                              uint tid [[thread_position_in_grid]]) {
  if (tid >= count) return;
  const device ushort* tile = reinterpret_cast<const device ushort*>(inputs + tid);
  const uint off = inputs[tid].z;
  outputs[tid] = tf_exl3_mapped_state(tile, (off << 6) | (1u << 11));
}

// Kept-dense: rotated rows x bf16 weight, f32 in rotate_out sums; sequential k, one thread an output, in fp64 order.
kernel void tf_exl3_dense(const device half* rotated [[buffer(0)]],
                          const device bfloat* weight [[buffer(1)]],
                          device float* sums [[buffer(2)]],
                          constant TfExl3Args& a [[buffer(3)]],
                          const device uint* input_ids [[buffer(4)]],
                          uint2 g [[thread_position_in_grid]]) {
  const uint row = a.input_ids ? input_ids[g.y] : g.y;
  float acc = 0;
  for (uint k = 0; k < a.k; ++k)
    acc += float(rotated[ulong(g.y) * a.k + k]) * float(weight[ulong(g.x) * a.k + k]);
  sums[ulong(g.y) * a.n + g.x] = acc;
}

// Four SIMDgroups share decoded weights while each retains its original eight-row compensated dots.
kernel void tf_exl3_matmul_prefill(const device half* x [[buffer(0)]],
                                  const device uchar* trellis [[buffer(1)]],
                                  device float* sums [[buffer(2)]],
                                  constant TfExl3Args& a [[buffer(3)]],
                                  const device uint* routes [[buffer(4)]],
                                  const device TfExl3Expert* descriptors [[buffer(5)]],
                                  device float* partials [[buffer(6)]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint sg [[simdgroup_index_in_threadgroup]],
                                  uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup float wx[256], xx[512];
  const uint rb = tg.y * 32, nb = tg.x * 16;
  if (a.indexed || a.routed || rb >= a.rows) return;
  TfExl3ScalarWindow win[8];
  if (sg == 0) {
    for (uint j = 0; j < 8; ++j) win[j] = tf_exl3_scalar_window(lane + 32 * j, a.halves);
  }
  float high[4] = {0,0,0,0}, low[4] = {0,0,0,0};
  for (uint kb = 0; kb < a.k; kb += 16) {
    if (sg == 0) {
      const device ushort* tile = reinterpret_cast<const device ushort*>(
          trellis + ulong(kb / 16) * a.tile_k_stride + ulong(nb / 16) * a.tile_n_stride);
      for (uint j = 0; j < 8; ++j) {
        const uint p = lane + 32 * j, l = p / 8, e = p % 8;
        const uint r = 2 * (l % 4) + (e & 1u) + 8 * ((e >> 1) & 1u);
        const uint c = l / 4 + 8 * (e >> 2);
        wx[r * 16 + c] = float(tf_exl3_scalar_value(tile, win[j], a.codebook));
      }
    }
    for (uint j = lane; j < 128; j += 32) {
      const uint row = rb + sg * 8 + j / 16;
      xx[sg * 128 + j] = tf_exl3_active(routes, a, row)
          ? float(x[ulong(row) * a.k + kb + j % 16]) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 4; ++i) {
      const uint at = lane + 32 * i, row = rb + sg * 8 + at / 16, col = at % 16;
      if (!tf_exl3_active(routes, a, row)) continue;
      for (uint k = 0; k < 16; ++k) {
        const float product = xx[sg * 128 + (at / 16) * 16 + k] * wx[k * 16 + col];
        const float next = high[i] + product;
        const float borrowed = next - high[i];
        low[i] += (high[i] - (next - borrowed)) + (product - borrowed);
        high[i] = next;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint i = 0; i < 4; ++i) {
    const uint at = lane + 32 * i, row = rb + sg * 8 + at / 16;
    if (tf_exl3_active(routes, a, row)) sums[ulong(row) * a.n + nb + at % 16] = high[i] + low[i];
  }
}

// Split-K reduction: one row's partial dot products, summed in split order.
kernel void tf_exl3_splitk_reduce(const device float* partials [[buffer(0)]],
                                  device float* sums [[buffer(1)]],
                                  constant uint2& d [[buffer(2)]],
                                  uint i [[thread_position_in_grid]]) {
  if (i >= d.y) return;
  float s = 0;
  for (uint k = 0; k < d.x; ++k) s += partials[ulong(k) * d.y + i];
  sums[i] = s;
}
