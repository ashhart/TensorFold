// Native EXL3: stored trellis tiles, row-independent compensated dots, and normalized Sylvester rotations. The
// model-facing order (attention, MTP, dense/shared routing) follows the release's inference/model.py, revision
// 314113f8; the trellis arithmetic and the kernel shapes are the lane's own pack format.
#include <metal_stdlib>
using namespace metal;

struct TfExl3Args {
  uint k, n, rows, halves;
  uint codebook, su_signs, sv_signs, has_bias;
  uint output_f32, input_ids, output_ids, routed;
  uint expert_id, repeat, indexed, experts, splits;
  ulong x_stride, y_stride, tile_k_stride, tile_n_stride;
  uint pair;
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
  // Four SIMDgroups a threadgroup, each a 128-element block of the same row: the dispatch needs four times fewer
  // threadgroups, and at a prompt chunk these passes were a million threadgroups a layer.
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

// The same value as tf_exl3_value, read from the 3 x 65536-half table the engine already binds at buffer 7
// (built from format.codebookValue, the format's own decoder, and qualified entry-for-entry by
// tf_exl3_codebook_check). Bit-identical by construction; this arm only moves the work from ALU to a load.
inline half tf_exl3_table_value(const device half* table, uint codebook, ushort state) {
  return table[codebook * 65536u + uint(state)];
}

// The mul1 value depends only on the state's byte sum, so the 65,536-entry table holds 1,024
// distinct values and a threadgroup can hold them all: the table is filled once a threadgroup and the decode
// reads it in place of the 384 KB resident table's load.
inline uint tf_exl3_mul1_index(ushort state) {
  const uint x = uint(state) * 0x83DCD12Du;
  const uint bs = (x & 0x00FF00FFu) + ((x >> 8u) & 0x00FF00FFu);
  return (bs * 0x00010001u) >> 16u;
}

// The mul1 value for a byte sum: the same half the arithmetic path and the resident table hold (the +1024 bias
// rides in the mantissa, exactly as the format's own decoder puts it).
inline half tf_exl3_mul1_value(uint bs) {
  const half hs = as_type<half>(ushort(bs + 0x6400u));
  return half(fma(float(hs), float(as_type<half>(ushort(0x1EEEu))), float(as_type<half>(ushort(0xC931u)))));
}

// The lane's eight windows at compile-time shifts. C_cur is the 32-bit window at the lane's first
// state (its bit 31 is that window's first stream bit) and C_next the window 32 bits later; S is the states'
// stride in bits (halves / 2) and J the state's index, so every shift below folds at compile time. The
// structure: one funnel shift a block, then the windows at fixed shifts.
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
  // A split dispatch carries a row group in tg.z (the tile already spans eight rows), so a verify
  // window's row keeps the solo order instead of collapsing to row 0.
  const uint rb = a.splits > 1 ? tg.z * 8 : (a.indexed == 1 ? tg.y : tg.y * 8), nb = tg.x * 16;
  if (!simd_any(tf_exl3_active(routes, a, rb + (a.indexed == 1 ? 0 : lane % 8)))) return;
  const uint id = a.indexed ? routes[rb] : 0;
  const device uchar* data = a.indexed ? descriptors[id].trellis : trellis;
  const ulong ks = a.indexed ? descriptors[id].tile_k_stride : a.tile_k_stride;
  const ulong ns = a.indexed ? descriptors[id].tile_n_stride : a.tile_n_stride;
  const uint halves = a.indexed ? descriptors[id].halves : a.halves;
  const uint codebook = a.indexed ? descriptors[id].codebook : a.codebook;
  uint maps[8];
  // Eight consecutive states a lane of the sixteen-column tile: the three stream words their windows can touch,
  // the shift each window needs and its threadgroup slot all depend only on the lane, so they live in registers.
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
        // FP16 products are exact in FP32; the compensation TwoSum keeps is measured against the canon hashes.
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

// The one-row / narrow-window arm of the same body with the codebook as a table read (buffer 7). Same k order,
// same FP32 sums and the same FP16 values as tf_exl3_value, so its output is bit-identical.
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
  // An indexed batch may carry entries of different output width (descriptor.reserved, 0 = the
  // call's n). The entry's own width bounds the columns this row writes; single-width tables are unchanged.
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
    float out = (v[j] * 0.08838834764831845f) * tf_exl3_scale(scales, n, signs);
    if (has_bias) out += float(biases[n]);
    const ulong at = ulong(dest) * a.y_stride + n;
    if (a.output_f32) reinterpret_cast<device float*>(y)[at] = out;
    else reinterpret_cast<device bfloat*>(y)[at] = bfloat(out);
  }
}

// The pre-rotated int8 pack's epilogue. The byte body leaves f32 sums; the rotated pack has no
// Hadamard/svh left to apply, so this is one cast into the caller's bf16 output at its own row stride.
kernel void tf_exl3_epilogue(const device float* sums [[buffer(0)]],
                             device uchar* y [[buffer(1)]],
                             constant TfExl3Args& a [[buffer(2)]],
                             uint2 gid [[thread_position_in_grid]]) {
  const uint r = gid.y, c = gid.x;
  if (r >= a.rows || c >= a.n) return;
  reinterpret_cast<device bfloat*>(y)[ulong(r) * a.y_stride + c] = bfloat(sums[ulong(r) * a.n + c]);
}

// THE ONE-ROW TRANSFORM FOLD -- the projection's output transform, the RMS norm and the next
// projection's input transform in ONE dispatch (the chain wq_a -> q_norm -> wq_b). Section 3 designed
// it and section 3b named the blocker: the butterfly's lane layout (a lane holds lane+32j) against this
// family's STRIDED norm (thread t holds t, t+256, ...). Here the row goes through threadgroup memory, so the
// permutation is a shared-memory write/read, not a shuffle. The arithmetic is the three shipped kernels' own:
//   rotate_out: v = sums[..]; hadamard(v); out = (v * 0.08838834764831845) * sv; +bias; bfloat(out)
//   dsv41_rms:  s = sum over i = t, t+256, ... of bf(x)^2; simd_sum; the eight simdgroups' partials;
//               rsqrt(total / width + eps); y = rb((bf(x) * inv) * bf(w))
//   rotate_in:  v = bf(normed) * su; hadamard(v); half(v * 0.08838834764831845)
// so every value is the shipped value. The merged wq_a+wkv sums carry two rows of different width (1280, 512):
// threadgroup 1 does wkv's output transform alone and writes raw.
struct TfExl3NormTransform {
  uint n0, n1, k_next;
  uint su_signs, sv_signs0, sv_signs1;
  float eps;
};

// BF16 bits to float (the ops kernel's bf()).
inline float tf_exl3_nt_bf(ushort x) { return as_type<float>(uint(x) << 16); }

// The norm's own round-to-bfloat16 (ops.metal's rb, RNE with the NaN passthrough).
inline ushort tf_exl3_nt_rb(float x) {
  uint u = as_type<uint>(x);
  if ((u & 0x7fffffffu) > 0x7f800000u) return ushort((u >> 16) | 0x40u);
  return ushort((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

kernel void tf_exl3_norm_transform(const device float* sums [[buffer(0)]],
                                   const device ushort* sv0 [[buffer(1)]],
                                   const device ushort* sv1 [[buffer(2)]],
                                   const device ushort* weight [[buffer(3)]],
                                   const device ushort* su_next [[buffer(4)]],
                                   device ushort* qr [[buffer(5)]],
                                   device ushort* raw [[buffer(6)]],
                                   device half* rotated [[buffer(7)]],
                                   constant TfExl3NormTransform& a [[buffer(8)]],
                                   uint r [[threadgroup_position_in_grid]],
                                   uint t [[thread_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]],
                                   uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup ushort sh[1280];
  threadgroup float psum[8];
  threadgroup float inv[1];
  const uint n = r == 0 ? a.n0 : a.n1;
  if (r == 0) {
    // The wq_a output transform: 1280 = ten 128-blocks, one block a simdgroup.
    for (uint blk = sg; blk * 128 < n; blk += 8) {
      float v[4];
      for (uint j = 0; j < 4; ++j) v[j] = sums[blk * 128 + lane + 32 * j];
      tf_exl3_hadamard(v, lane);
      for (uint j = 0; j < 4; ++j) {
        const uint idx = blk * 128 + lane + 32 * j;
        const float out = (v[j] * 0.08838834764831845f) * tf_exl3_scale(sv0, idx, a.sv_signs0);
        sh[idx] = as_type<ushort>(bfloat(out));
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // The q_norm: dsv41_rms's own strided partition and order.
    float s = 0.0f;
    for (uint i = t; i < n; i += 256) { const float x = tf_exl3_nt_bf(sh[i]); s += x * x; }
    s = simd_sum(s);
    if (lane == 0) psum[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      const float total = simd_sum(lane < 8 ? psum[lane] : 0.0f);
      if (lane == 0) inv[0] = rsqrt(total * (1.0f / float(n)) + a.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = t; i < n; i += 256) {
      const ushort bits = tf_exl3_nt_rb((tf_exl3_nt_bf(sh[i]) * inv[0]) * tf_exl3_nt_bf(weight[i]));
      sh[i] = bits;
      qr[i] = bits;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // wq_b's input transform: the normalized row, one 128-block a simdgroup, f16 into the scratch.
    for (uint blk = sg; blk * 128 < a.k_next; blk += 8) {
      float v[4];
      for (uint j = 0; j < 4; ++j) {
        const uint idx = blk * 128 + lane + 32 * j;
        v[j] = tf_exl3_nt_bf(sh[idx]) * tf_exl3_scale(su_next, idx, a.su_signs);
      }
      tf_exl3_hadamard(v, lane);
      for (uint j = 0; j < 4; ++j)
        rotated[blk * 128 + lane + 32 * j] = half(v[j] * 0.08838834764831845f);
    }
  } else {
    // wkv's output transform alone: 512 = four 128-blocks.
    for (uint blk = sg; blk * 128 < n; blk += 8) {
      float v[4];
      for (uint j = 0; j < 4; ++j) v[j] = sums[a.n0 + blk * 128 + lane + 32 * j];
      tf_exl3_hadamard(v, lane);
      for (uint j = 0; j < 4; ++j) {
        const uint idx = blk * 128 + lane + 32 * j;
        const float out = (v[j] * 0.08838834764831845f) * tf_exl3_scale(sv1, idx, a.sv_signs1);
        raw[idx] = as_type<ushort>(bfloat(out));
      }
    }
  }
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
  uint maps[8];
  if (sg == 0) {
    for (uint j = 0; j < 8; ++j) {
      const uint p = lane + 32 * j, ring = 128 * a.halves;
      const uint end = ((p + 1) * a.halves - ((a.halves & 1u) ? ((p + 1) & 1u) : 0u)) / 2;
      const uint start = (end + ring - 16) % ring;
      const uint wi = start / 32, off = start % 32, next = wi + 1 == 4 * a.halves ? 0 : wi + 1;
      maps[j] = wi | (off << 6) | (next << 11);
    }
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
        wx[r * 16 + c] = float(tf_exl3_value(tf_exl3_mapped_state(tile, maps[j]), a.codebook));
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


// ------------------------------------------------------------------------------------------------
// SwiGLU + the down projection's rotate_in, fused (TF_DSV41_SWIGLU_ROT; default off until its differential and
// the canons are green). The down projection's rotated input is written in the SwiGLU pass instead of a separate
// rotate: the same bf16 round of silu(gate)*up, the
// same per-expert su scale and the same 128-point Sylvester rotation tf_exl3_rotate_in applies, so a row's
// rotated bits are the two-pass path's bits. Each SIMDgroup owns one 128-element block of one row, and the row
// is written to the destination slot its expert tile occupies.
inline float tf_exl3_bf(ushort x) { return as_type<float>(uint(x) << 16); }
inline ushort tf_exl3_round_bf16(float x) {
  uint u = as_type<uint>(x);
  if ((u & 0x7fffffffu) > 0x7f800000u) return ushort((u >> 16) | 0x40u);
  return ushort((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
struct TfExl3SwiGLURot { uint rows, k, input_stride, output_stride, weighted, scatter, experts, reserved; };
kernel void tf_exl3_swiglu_prerotate(const device ushort* gate [[buffer(0)]],
                                    const device ushort* up [[buffer(1)]],
                                    const device float* weights [[buffer(2)]],
                                    const device uint* routes [[buffer(3)]],
                                    const device uint* destinations [[buffer(4)]],
                                    const device TfExl3Expert* descriptors [[buffer(5)]],
                                    device half* rotated [[buffer(6)]],
                                    constant TfExl3SwiGLURot& d [[buffer(7)]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint sg [[simdgroup_index_in_threadgroup]],
                                    uint3 tptg [[threads_per_threadgroup]],
                                    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint row = tg.y;
  if (row >= d.rows) return;
  const uint expert = routes[row];
  if (expert >= d.experts) return;
  const uint base = (tg.x * (tptg.x / 32u) + sg) * 128u;
  if (base >= d.k) return;
  const device ushort* scales = descriptors[expert].su;
  const uint signs = descriptors[expert].su_signs;
  const float weight = d.weighted ? weights[row] : 1.0f;
  float v[4];
  for (uint j = 0; j < 4; ++j) {
    const uint k = base + lane + 32u * j;
    const ulong at = ulong(row) * d.input_stride + k;
    const float g = min(tf_exl3_bf(gate[at]), 10.0f);
    const float u = clamp(tf_exl3_bf(up[at]), -10.0f, 10.0f);
    float s = (g / (1.0f + exp(-g))) * u;
    if (d.weighted) s = weight * s;
    v[j] = tf_exl3_bf(tf_exl3_round_bf16(s)) * tf_exl3_scale(scales, k, signs);
  }
  tf_exl3_hadamard(v, lane);
  const uint dst = d.scatter ? destinations[row] : row;
  for (uint j = 0; j < 4; ++j)
    rotated[ulong(dst) * d.output_stride + base + lane + 32u * j] = half(v[j] * 0.08838834764831845f);
}

// Wide one-row / narrow-window tile: eight SIMDgroups each own one 16-column tile of the same 128-column group,
// so a threadgroup reads eight contiguous n tiles of a k block, and the shared activation tile is double
// buffered, so a block costs one threadgroup barrier instead of two. Each SIMDgroup runs the same decode, the
// same 16-product FP32 sum and the same k order as tf_exl3_matmul_tab, so the output is bit-identical.
// Bench arm for 's one-row tile geometry; not dispatched by the engine yet.
template <bool Table, bool Decode = true, bool Lut = false>
inline void tf_exl3_matmul_wide_impl(const device half* x, const device uchar* trellis,
                                     device float* sums, constant TfExl3Args& a,
                                     const device uint* routes, const device TfExl3Expert* descriptors,
                                     device float* partials, const device half* table,
                                     uint lane, uint sg, uint3 tg,
                                     threadgroup half* wx_all, threadgroup float* xx_all,
                                     threadgroup half* lut = nullptr) {
  const uint split = a.splits > 1 ? tg.y : 0;
  const uint rb = a.splits > 1 ? tg.z * 8 : tg.y * 8, nb = tg.x * 128 + sg * 16;
  if (!simd_any(tf_exl3_active(routes, a, rb + lane % 8))) return;
  const device uchar* data = trellis;
  const ulong ks = a.tile_k_stride;
  const ulong ns = a.tile_n_stride;
  const uint halves = a.halves;
  const uint codebook = a.codebook;
  threadgroup half* wx = wx_all + sg * 256;
  uint maps[8], at[8], off[8], sh[8], rel[8];
  const uint words = 4 * halves;
  const uint ring = 128 * halves;
  const uint wi0 = tf_exl3_map_of(lane * 8, halves) & 63u;
  const uint cur0 = (wi0 << 5) | ((tf_exl3_map_of(lane * 8, halves) >> 6) & 31u);
  TF_UNROLL
  for (uint j = 0; j < 8; ++j) {
    const uint p = lane * 8 + j, l = p / 8, e = p % 8;
    uint c8 = cur0 + j * (halves >> 1);
    if (c8 >= ring) c8 -= ring;
    off[j] = c8 & 31u;
    sh[j] = (32u - off[j]) & 31u;
    rel[j] = (c8 >> 5) >= wi0 ? (c8 >> 5) - wi0 : (c8 >> 5) + words - wi0;
    at[j] = (2 * (l % 4) + (e & 1u) + 8 * ((e >> 1) & 1u)) * 16u + (l / 4 + 8 * (e >> 2));
  }
  {
    for (uint j = 0; j < 8; ++j) {
      const uint p = lane + 32 * j, ring2 = 128 * halves;
      const uint end = ((p + 1) * halves - ((halves & 1u) ? ((p + 1) & 1u) : 0u)) / 2;
      uint start = end + ring2 - 16;
      if (start >= ring2) start -= ring2;
      const uint wi = start >> 5, o = start & 31u, next = wi + 1 == 4 * halves ? 0 : wi + 1;
      maps[j] = wi | (o << 6) | (next << 11);
    }
  }
  float high[4] = {0,0,0,0}, low[4] = {0,0,0,0};
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
    threadgroup float* xx = xx_all + ((b - b0) & 1u) * 128;
    // One warp loads the activation tile for all eight: 8 rows x 16 k, zeros past a.rows. The other seven warps
    // decode their weight tiles while it runs, so the load's latency is covered by work, not a second barrier.
    if (sg == 0) {
      for (uint j = lane; j < 128; j += 32) {
        const uint row = rb + j / 16;
        xx[j] = tf_exl3_active(routes, a, row) ? float(x[ulong(row) * a.k + b * 16 + j % 16]) : 0.0f;
      }
    }
    const device ushort* tile = reinterpret_cast<const device ushort*>(
        data + ulong(b) * ks + ulong(nb / 16) * ns);
    if ((halves & 1u) == 0 and halves <= 16) {
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
        wx[at[j]] = Decode ? (Lut ? lut[tf_exl3_mul1_index(s)]
                                 : (Table ? tf_exl3_table_value(table, codebook, s) : tf_exl3_value(s, codebook)))
                           : half(0.5f + float((lane + j) & 7u));
      }
    } else for (uint j = 0; j < 8; ++j) {
      const uint p = lane + 32 * j, l = p / 8, e = p % 8;
      const uint r = 2 * (l % 4) + (e & 1u) + 8 * ((e >> 1) & 1u);
      const uint c = l / 4 + 8 * (e >> 2);
      const ushort state = tf_exl3_mapped_state(tile, maps[j]);
      wx[r * 16 + c] = Decode ? (Lut ? lut[tf_exl3_mul1_index(state)]
                                   : (Table ? tf_exl3_table_value(table, codebook, state) : tf_exl3_value(state, codebook)))
                              : half(0.5f + float((lane + j) & 7u));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 4; ++i) {
      const uint at_i = lane + 32 * i, row = rb + at_i / 16, col = at_i % 16;
      if (!tf_exl3_active(routes, a, row)) continue;
      for (uint k = 0; k < 16; ++k) high[i] += xx[(at_i / 16) * 16 + k] * float(wx[k * 16 + col]);
    }
    // No second barrier: the next block writes the other activation buffer, and this one is read no more.
  }
  for (uint i = 0; i < 4; ++i) {
    const uint at_i = lane + 32 * i, row = rb + at_i / 16;
    if (tf_exl3_active(routes, a, row)) {
      if (a.splits > 1) partials[(ulong(split) * a.rows + row) * a.n + nb + at_i % 16] = high[i] + low[i];
      else sums[ulong(row) * a.n + nb + at_i % 16] = high[i] + low[i];
    }
  }
}

kernel void tf_exl3_matmul_wide_tab(const device half* x [[buffer(0)]],
                                    const device uchar* trellis [[buffer(1)]],
                                    device float* sums [[buffer(2)]],
                                    constant TfExl3Args& a [[buffer(3)]],
                                    const device uint* routes [[buffer(4)]],
                                    const device TfExl3Expert* descriptors [[buffer(5)]],
                                    device float* partials [[buffer(6)]],
                                    const device half* table [[buffer(7)]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint sg [[simdgroup_index_in_threadgroup]],
                                    uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half wx[8 * 256];
  threadgroup float xx[2 * 128];
  tf_exl3_matmul_wide_impl<true>(x, trellis, sums, a, routes, descriptors, partials, table, lane, sg, tg, wx, xx);
}

// The wide one-row body with the static state selection. The shipped body picks each state's
// word pair and bit offset at run time (the `rel[k]` cursor the record named as one of the body's two bounds);
// here one funnel shift a block puts the lane's window at a known place and the eight windows then sit at
// compile-time shifts, so the cursor leaves the inner loop and the lane reads three stream words, not four.
// The tile, the activation load, the 16-product FP32 sums and the k order are the shipped body's, so the output
// is bit-identical. S is the states' stride in bits (halves / 2) and must satisfy 7*S + 16 <= 64.
template <bool Lut, uint S>
inline void tf_exl3_matmul_wide_static_impl(const device half* x, const device uchar* trellis,
                                            device float* sums, constant TfExl3Args& a,
                                            const device uint* routes, const device TfExl3Expert* descriptors,
                                            device float* partials, const device half* table,
                                            uint lane, uint sg, uint3 tg,
                                            threadgroup half* wx_all, threadgroup float* xx_all,
                                            threadgroup half* lut) {
  const uint split = a.splits > 1 ? tg.y : 0;
  const uint rb = a.splits > 1 ? tg.z * 8 : tg.y * 8, nb = tg.x * 128 + sg * 16;
  if (!simd_any(tf_exl3_active(routes, a, rb + lane % 8))) return;
  const device uchar* data = trellis;
  const ulong ks = a.tile_k_stride;
  const ulong ns = a.tile_n_stride;
  const uint halves = a.halves;
  const uint codebook = a.codebook;
  threadgroup half* wx = wx_all + sg * 256;
  uint at[8];
  const uint words = 4 * halves;
  const uint wi0 = tf_exl3_map_of(lane * 8, halves) & 63u;
  const uint off0 = (tf_exl3_map_of(lane * 8, halves) >> 6) & 31u;
  TF_UNROLL
  for (uint j = 0; j < 8; ++j) {
    const uint p = lane * 8 + j, l = p / 8, e = p % 8;
    at[j] = (2 * (l % 4) + (e & 1u) + 8 * ((e >> 1) & 1u)) * 16u + (l / 4 + 8 * (e >> 2));
  }
  float high[4] = {0,0,0,0}, low[4] = {0,0,0,0};
  const uint blocks = a.k / 16;
  const uint b0 = a.splits > 1 ? blocks * split / a.splits : 0;
  const uint b1 = a.splits > 1 ? blocks * (split + 1) / a.splits : blocks;
  // Three stream words: the lane's window spans at most 7*S + 16 bits from its first state.
  uint a1 = wi0 + 1; if (a1 >= words) a1 -= words;
  uint a2 = wi0 + 2; if (a2 >= words) a2 -= words;
  uint prefetch[3];
  {
    const device ushort* t0 = reinterpret_cast<const device ushort*>(data + ulong(b0) * ks + ulong(nb / 16) * ns);
    prefetch[0] = tf_exl3_word(t0, wi0);
    prefetch[1] = tf_exl3_word(t0, a1);
    prefetch[2] = tf_exl3_word(t0, a2);
  }
  // The lane's bit offset inside its first word: the one runtime shift of the body.
  const uint sh0 = (32u - off0) & 31u;
  const uint m0 = 0u - uint(off0 != 0u);
  for (uint b = b0; b < b1; ++b) {
    threadgroup float* xx = xx_all + ((b - b0) & 1u) * 128;
    if (sg == 0) {
      for (uint j = lane; j < 128; j += 32) {
        const uint row = rb + j / 16;
        xx[j] = tf_exl3_active(routes, a, row) ? float(x[ulong(row) * a.k + b * 16 + j % 16]) : 0.0f;
      }
    }
    const uint w0 = prefetch[0], w1 = prefetch[1], w2 = prefetch[2];
    if (b + 1 < b1) {
      const device ushort* tn = reinterpret_cast<const device ushort*>(data + ulong(b + 1) * ks + ulong(nb / 16) * ns);
      prefetch[0] = tf_exl3_word(tn, wi0);
      prefetch[1] = tf_exl3_word(tn, a1);
      prefetch[2] = tf_exl3_word(tn, a2);
    }
    // The lane's window: C_cur is the 32-bit window at the lane's first state (its bit 31 is that window's
    // first stream bit) and C_next the window 32 bits later.
    const uint c_cur = (w0 << off0) | ((w1 >> sh0) & m0);
    const uint c_next = (w1 << off0) | ((w2 >> sh0) & m0);
    #define TF_EXL3_W(J) tf_exl3_state_static<S, J>(c_cur, c_next)
    #define TF_EXL3_V(J) (Lut ? lut[tf_exl3_mul1_index(TF_EXL3_W(J))] : tf_exl3_table_value(table, codebook, TF_EXL3_W(J)))
    wx[at[0]] = TF_EXL3_V(0);
    wx[at[1]] = TF_EXL3_V(1);
    wx[at[2]] = TF_EXL3_V(2);
    wx[at[3]] = TF_EXL3_V(3);
    wx[at[4]] = TF_EXL3_V(4);
    wx[at[5]] = TF_EXL3_V(5);
    wx[at[6]] = TF_EXL3_V(6);
    wx[at[7]] = TF_EXL3_V(7);
    #undef TF_EXL3_V
    #undef TF_EXL3_W
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 4; ++i) {
      const uint at_i = lane + 32 * i, row = rb + at_i / 16, col = at_i % 16;
      if (!tf_exl3_active(routes, a, row)) continue;
      for (uint k = 0; k < 16; ++k) high[i] += xx[(at_i / 16) * 16 + k] * float(wx[k * 16 + col]);
    }
  }
  for (uint i = 0; i < 4; ++i) {
    const uint at_i = lane + 32 * i, row = rb + at_i / 16;
    if (tf_exl3_active(routes, a, row)) {
      if (a.splits > 1) partials[(ulong(split) * a.rows + row) * a.n + nb + at_i % 16] = high[i] + low[i];
      else sums[ulong(row) * a.n + nb + at_i % 16] = high[i] + low[i];
    }
  }
}

// bench arm: the static state selection with the codebook read from the resident table.
kernel void tf_exl3_matmul_wide_static_tab(const device half* x [[buffer(0)]],
                                           const device uchar* trellis [[buffer(1)]],
                                           device float* sums [[buffer(2)]],
                                           constant TfExl3Args& a [[buffer(3)]],
                                           const device uint* routes [[buffer(4)]],
                                           const device TfExl3Expert* descriptors [[buffer(5)]],
                                           device float* partials [[buffer(6)]],
                                           const device half* table [[buffer(7)]],
                                           uint lane [[thread_index_in_simdgroup]],
                                           uint sg [[simdgroup_index_in_threadgroup]],
                                           uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half wx[8 * 256];
  threadgroup float xx[2 * 128];
  if (a.halves == 12u)
    tf_exl3_matmul_wide_static_impl<false, 6>(x, trellis, sums, a, routes, descriptors, partials, table, lane, sg, tg, wx, xx, nullptr);
  else if (a.halves == 10u)
    tf_exl3_matmul_wide_static_impl<false, 5>(x, trellis, sums, a, routes, descriptors, partials, table, lane, sg, tg, wx, xx, nullptr);
  else if (a.halves == 6u)
    tf_exl3_matmul_wide_static_impl<false, 3>(x, trellis, sums, a, routes, descriptors, partials, table, lane, sg, tg, wx, xx, nullptr);
  else
    // A width the static selection is not specialised for: the shipped body, which reads the states the same way.
    tf_exl3_matmul_wide_impl<true>(x, trellis, sums, a, routes, descriptors, partials, table, lane, sg, tg, wx, xx);
}

// bench arm: the static state selection with the codebook read from a 1024-entry threadgroup LUT.
kernel void tf_exl3_matmul_wide_static_lut(const device half* x [[buffer(0)]],
                                           const device uchar* trellis [[buffer(1)]],
                                           device float* sums [[buffer(2)]],
                                           constant TfExl3Args& a [[buffer(3)]],
                                           const device uint* routes [[buffer(4)]],
                                           const device TfExl3Expert* descriptors [[buffer(5)]],
                                           device float* partials [[buffer(6)]],
                                           const device half* table [[buffer(7)]],
                                           uint lane [[thread_index_in_simdgroup]],
                                           uint sg [[simdgroup_index_in_threadgroup]],
                                           uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half wx[8 * 256];
  threadgroup float xx[2 * 128];
  threadgroup half lut[1024];
  // The LUT holds the mul1 values (the value depends only on the state's byte sum), so it is only the right
  // table for that codebook; any other reads the resident table, and any width the static selection is not
  // specialised for reads the shipped body's states.
  if (a.codebook != 2u) {
    tf_exl3_matmul_wide_impl<true>(x, trellis, sums, a, routes, descriptors, partials, table, lane, sg, tg, wx, xx);
    return;
  }
  for (uint b = sg * 32u + lane; b < 1024u; b += 256u) lut[b] = tf_exl3_mul1_value(b);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (a.halves == 12u)
    tf_exl3_matmul_wide_static_impl<true, 6>(x, trellis, sums, a, routes, descriptors, partials, table, lane, sg, tg, wx, xx, lut);
  else if (a.halves == 10u)
    tf_exl3_matmul_wide_static_impl<true, 5>(x, trellis, sums, a, routes, descriptors, partials, table, lane, sg, tg, wx, xx, lut);
  else if (a.halves == 6u)
    tf_exl3_matmul_wide_static_impl<true, 3>(x, trellis, sums, a, routes, descriptors, partials, table, lane, sg, tg, wx, xx, lut);
  else
    tf_exl3_matmul_wide_impl<true>(x, trellis, sums, a, routes, descriptors, partials, table, lane, sg, tg, wx, xx);
}

// The static one-row body with a deeper weight prefetch (PF blocks of stream words in flight a
// lane). The shipped body issues the next block's three stream words at the top of the current block, so a
// block's DRAM latency is hidden by one block's compute only. The narrow one-row shapes (n = 2304, 1280, 512
// -- the shared expert's gate/up and the attention's wq_a) have one wave of threadgroups and nothing else to
// hide it with: the bench reads them at 70-240 GB/s against the box's 1,120 GB/s. The prefetch ring below
// keeps PF blocks of stream words in registers, so PF blocks' loads are in flight at once. Same states, same
// codebook values, same products, same k order, same partials: bit-identical.
template <bool Lut, uint S, uint PF>
inline void tf_exl3_matmul_wide_static_pf_impl(const device half* x, const device uchar* trellis,
                                               device float* sums, constant TfExl3Args& a,
                                               const device uint* routes, const device TfExl3Expert* descriptors,
                                               device float* partials, const device half* table,
                                               uint lane, uint sg, uint3 tg,
                                               threadgroup half* wx_all, threadgroup float* xx_all,
                                               threadgroup half* lut) {
  const uint split = a.splits > 1 ? tg.y : 0;
  const uint rb = a.splits > 1 ? tg.z * 8 : tg.y * 8, nb = tg.x * 128 + sg * 16;
  if (!simd_any(tf_exl3_active(routes, a, rb + lane % 8))) return;
  const device uchar* data = trellis;
  const ulong ks = a.tile_k_stride;
  const ulong ns = a.tile_n_stride;
  const uint halves = a.halves;
  const uint codebook = a.codebook;
  threadgroup half* wx = wx_all + sg * 256;
  uint at[8];
  const uint words = 4 * halves;
  const uint wi0 = tf_exl3_map_of(lane * 8, halves) & 63u;
  const uint off0 = (tf_exl3_map_of(lane * 8, halves) >> 6) & 31u;
  TF_UNROLL
  for (uint j = 0; j < 8; ++j) {
    const uint p = lane * 8 + j, l = p / 8, e = p % 8;
    at[j] = (2 * (l % 4) + (e & 1u) + 8 * ((e >> 1) & 1u)) * 16u + (l / 4 + 8 * (e >> 2));
  }
  float high[4] = {0,0,0,0}, low[4] = {0,0,0,0};
  const uint blocks = a.k / 16;
  const uint b0 = a.splits > 1 ? blocks * split / a.splits : 0;
  const uint b1 = a.splits > 1 ? blocks * (split + 1) / a.splits : blocks;
  uint a1 = wi0 + 1; if (a1 >= words) a1 -= words;
  uint a2 = wi0 + 2; if (a2 >= words) a2 -= words;
  const uint sh0 = (32u - off0) & 31u;
  const uint m0 = 0u - uint(off0 != 0u);
  // The prefetch ring: PF blocks of three stream words a lane, in registers (PF a compile-time constant, so
  // the array never lands in thread-local memory).
  uint pf[PF][3];
  TF_UNROLL
  for (uint p = 0; p < PF; ++p) {
    if (b0 + p < b1) {
      const device ushort* t = reinterpret_cast<const device ushort*>(data + ulong(b0 + p) * ks + ulong(nb / 16) * ns);
      pf[p][0] = tf_exl3_word(t, wi0);
      pf[p][1] = tf_exl3_word(t, a1);
      pf[p][2] = tf_exl3_word(t, a2);
    }
  }
  uint b = b0;
  const uint main_end = b0 + ((b1 - b0) / PF) * PF;
  while (b < main_end) {
    TF_UNROLL
    for (uint p = 0; p < PF; ++p) {
      const uint blk = b + p;
      threadgroup float* xx = xx_all + ((blk - b0) & 1u) * 128;
      if (sg == 0) {
        for (uint j = lane; j < 128; j += 32) {
          const uint row = rb + j / 16;
          xx[j] = tf_exl3_active(routes, a, row) ? float(x[ulong(row) * a.k + blk * 16 + j % 16]) : 0.0f;
        }
      }
      const uint w0 = pf[p][0], w1 = pf[p][1], w2 = pf[p][2];
      // The next PF blocks' words, issued now: PF loads in flight a lane instead of one.
      if (blk + PF < b1) {
        const device ushort* tn = reinterpret_cast<const device ushort*>(data + ulong(blk + PF) * ks + ulong(nb / 16) * ns);
        pf[p][0] = tf_exl3_word(tn, wi0);
        pf[p][1] = tf_exl3_word(tn, a1);
        pf[p][2] = tf_exl3_word(tn, a2);
      }
      const uint c_cur = (w0 << off0) | ((w1 >> sh0) & m0);
      const uint c_next = (w1 << off0) | ((w2 >> sh0) & m0);
      #define TF_EXL3_W(J) tf_exl3_state_static<S, J>(c_cur, c_next)
      #define TF_EXL3_V(J) (Lut ? lut[tf_exl3_mul1_index(TF_EXL3_W(J))] : tf_exl3_table_value(table, codebook, TF_EXL3_W(J)))
      wx[at[0]] = TF_EXL3_V(0);
      wx[at[1]] = TF_EXL3_V(1);
      wx[at[2]] = TF_EXL3_V(2);
      wx[at[3]] = TF_EXL3_V(3);
      wx[at[4]] = TF_EXL3_V(4);
      wx[at[5]] = TF_EXL3_V(5);
      wx[at[6]] = TF_EXL3_V(6);
      wx[at[7]] = TF_EXL3_V(7);
      #undef TF_EXL3_V
      #undef TF_EXL3_W
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint i = 0; i < 4; ++i) {
        const uint at_i = lane + 32 * i, row = rb + at_i / 16, col = at_i % 16;
        if (!tf_exl3_active(routes, a, row)) continue;
        for (uint k = 0; k < 16; ++k) high[i] += xx[(at_i / 16) * 16 + k] * float(wx[k * 16 + col]);
      }
    }
    b += PF;
  }
  // The tail: the blocks the pipelined loop did not cover (fewer than PF).
  for (; b < b1; ++b) {
    threadgroup float* xx = xx_all + ((b - b0) & 1u) * 128;
    if (sg == 0) {
      for (uint j = lane; j < 128; j += 32) {
        const uint row = rb + j / 16;
        xx[j] = tf_exl3_active(routes, a, row) ? float(x[ulong(row) * a.k + b * 16 + j % 16]) : 0.0f;
      }
    }
    const device ushort* t = reinterpret_cast<const device ushort*>(data + ulong(b) * ks + ulong(nb / 16) * ns);
    const uint w0 = tf_exl3_word(t, wi0), w1 = tf_exl3_word(t, a1), w2 = tf_exl3_word(t, a2);
    const uint c_cur = (w0 << off0) | ((w1 >> sh0) & m0);
    const uint c_next = (w1 << off0) | ((w2 >> sh0) & m0);
    #define TF_EXL3_W(J) tf_exl3_state_static<S, J>(c_cur, c_next)
    #define TF_EXL3_V(J) (Lut ? lut[tf_exl3_mul1_index(TF_EXL3_W(J))] : tf_exl3_table_value(table, codebook, TF_EXL3_W(J)))
    wx[at[0]] = TF_EXL3_V(0);
    wx[at[1]] = TF_EXL3_V(1);
    wx[at[2]] = TF_EXL3_V(2);
    wx[at[3]] = TF_EXL3_V(3);
    wx[at[4]] = TF_EXL3_V(4);
    wx[at[5]] = TF_EXL3_V(5);
    wx[at[6]] = TF_EXL3_V(6);
    wx[at[7]] = TF_EXL3_V(7);
    #undef TF_EXL3_V
    #undef TF_EXL3_W
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 4; ++i) {
      const uint at_i = lane + 32 * i, row = rb + at_i / 16, col = at_i % 16;
      if (!tf_exl3_active(routes, a, row)) continue;
      for (uint k = 0; k < 16; ++k) high[i] += xx[(at_i / 16) * 16 + k] * float(wx[k * 16 + col]);
    }
  }
  for (uint i = 0; i < 4; ++i) {
    const uint at_i = lane + 32 * i, row = rb + at_i / 16;
    if (tf_exl3_active(routes, a, row)) {
      if (a.splits > 1) partials[(ulong(split) * a.rows + row) * a.n + nb + at_i % 16] = high[i] + low[i];
      else sums[ulong(row) * a.n + nb + at_i % 16] = high[i] + low[i];
    }
  }
}

// bench arm: the static one-row body with the codebook LUT and a two-block prefetch.
kernel void tf_exl3_matmul_wide_static_lut_pf2(const device half* x [[buffer(0)]],
                                               const device uchar* trellis [[buffer(1)]],
                                               device float* sums [[buffer(2)]],
                                               constant TfExl3Args& a [[buffer(3)]],
                                               const device uint* routes [[buffer(4)]],
                                               const device TfExl3Expert* descriptors [[buffer(5)]],
                                               device float* partials [[buffer(6)]],
                                               const device half* table [[buffer(7)]],
                                               uint lane [[thread_index_in_simdgroup]],
                                               uint sg [[simdgroup_index_in_threadgroup]],
                                               uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half wx[8 * 256];
  threadgroup float xx[2 * 128];
  threadgroup half lut[1024];
  if (a.codebook != 2u) {
    tf_exl3_matmul_wide_impl<true>(x, trellis, sums, a, routes, descriptors, partials, table, lane, sg, tg, wx, xx);
    return;
  }
  for (uint b = sg * 32u + lane; b < 1024u; b += 256u) lut[b] = tf_exl3_mul1_value(b);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (a.halves == 12u)
    tf_exl3_matmul_wide_static_pf_impl<true, 6, 2>(x, trellis, sums, a, routes, descriptors, partials, table, lane, sg, tg, wx, xx, lut);
  else if (a.halves == 10u)
    tf_exl3_matmul_wide_static_pf_impl<true, 5, 2>(x, trellis, sums, a, routes, descriptors, partials, table, lane, sg, tg, wx, xx, lut);
  else if (a.halves == 6u)
    tf_exl3_matmul_wide_static_pf_impl<true, 3, 2>(x, trellis, sums, a, routes, descriptors, partials, table, lane, sg, tg, wx, xx, lut);
  else
    tf_exl3_matmul_wide_impl<true>(x, trellis, sums, a, routes, descriptors, partials, table, lane, sg, tg, wx, xx);
}

// bench arm: the static one-row body with the codebook LUT and a four-block prefetch.
kernel void tf_exl3_matmul_wide_static_lut_pf4(const device half* x [[buffer(0)]],
                                               const device uchar* trellis [[buffer(1)]],
                                               device float* sums [[buffer(2)]],
                                               constant TfExl3Args& a [[buffer(3)]],
                                               const device uint* routes [[buffer(4)]],
                                               const device TfExl3Expert* descriptors [[buffer(5)]],
                                               device float* partials [[buffer(6)]],
                                               const device half* table [[buffer(7)]],
                                               uint lane [[thread_index_in_simdgroup]],
                                               uint sg [[simdgroup_index_in_threadgroup]],
                                               uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half wx[8 * 256];
  threadgroup float xx[2 * 128];
  threadgroup half lut[1024];
  if (a.codebook != 2u) {
    tf_exl3_matmul_wide_impl<true>(x, trellis, sums, a, routes, descriptors, partials, table, lane, sg, tg, wx, xx);
    return;
  }
  for (uint b = sg * 32u + lane; b < 1024u; b += 256u) lut[b] = tf_exl3_mul1_value(b);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (a.halves == 12u)
    tf_exl3_matmul_wide_static_pf_impl<true, 6, 4>(x, trellis, sums, a, routes, descriptors, partials, table, lane, sg, tg, wx, xx, lut);
  else if (a.halves == 10u)
    tf_exl3_matmul_wide_static_pf_impl<true, 5, 4>(x, trellis, sums, a, routes, descriptors, partials, table, lane, sg, tg, wx, xx, lut);
  else if (a.halves == 6u)
    tf_exl3_matmul_wide_static_pf_impl<true, 3, 4>(x, trellis, sums, a, routes, descriptors, partials, table, lane, sg, tg, wx, xx, lut);
  else
    tf_exl3_matmul_wide_impl<true>(x, trellis, sums, a, routes, descriptors, partials, table, lane, sg, tg, wx, xx);
}

// bench arm: the shipped dynamic selection with the codebook read from a 1024-entry threadgroup LUT.
// The separation experiment's fourth cell: the table load removed, the cursor left in.
kernel void tf_exl3_matmul_wide_lut(const device half* x [[buffer(0)]],
                                    const device uchar* trellis [[buffer(1)]],
                                    device float* sums [[buffer(2)]],
                                    constant TfExl3Args& a [[buffer(3)]],
                                    const device uint* routes [[buffer(4)]],
                                    const device TfExl3Expert* descriptors [[buffer(5)]],
                                    device float* partials [[buffer(6)]],
                                    const device half* table [[buffer(7)]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint sg [[simdgroup_index_in_threadgroup]],
                                    uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half wx[8 * 256];
  threadgroup float xx[2 * 128];
  threadgroup half lut[1024];
  for (uint b = sg * 32u + lane; b < 1024u; b += 256u) lut[b] = tf_exl3_mul1_value(b);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  tf_exl3_matmul_wide_impl<true, true, true>(x, trellis, sums, a, routes, descriptors, partials, table, lane, sg, tg, wx, xx, lut);
}

// Diagnostic arm (): the wide body with the trellis load and the codebook value removed, so the difference
// against tf_exl3_matmul_wide_tab prices the decode (loads + bit extraction + table read) in the dispatch. Values
// are meaningless; timing only.
kernel void tf_exl3_matmul_wide_noload_tab(const device half* x [[buffer(0)]],
                                           const device uchar* trellis [[buffer(1)]],
                                           device float* sums [[buffer(2)]],
                                           constant TfExl3Args& a [[buffer(3)]],
                                           const device uint* routes [[buffer(4)]],
                                           const device TfExl3Expert* descriptors [[buffer(5)]],
                                           device float* partials [[buffer(6)]],
                                           const device half* table [[buffer(7)]],
                                           uint lane [[thread_index_in_simdgroup]],
                                           uint sg [[simdgroup_index_in_threadgroup]],
                                           uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half wx[8 * 256];
  threadgroup float xx[2 * 128];
  tf_exl3_matmul_wide_impl<true, false>(x, trellis, sums, a, routes, descriptors, partials, table, lane, sg, tg, wx, xx);
}

// 32-k continuation of the wide one-row tile (): eight SIMDgroups own 128 columns, and each block decodes
// two consecutive 16-k tiles before one barrier, so the barrier count halves and a threadgroup reads 1.5 KB of
// contiguous trellis a block. Each 16-k tile is decoded exactly as tf_exl3_matmul_wide_tab decodes it (same maps,
// same table value) and the accumulation still ascends k in order, so the output is bit-identical by construction.
inline void tf_exl3_wide2_decode16(const device ushort* tile, uint wi0, uint a1, uint a2, uint a3,
                                   threadgroup half* dst,
                                   const thread uint* at, const thread uint* off, const thread uint* sh,
                                   const thread uint* rel, const thread uint* maps, uint lane,
                                   const device half* table, uint codebook, bool table_on, bool fast) {
  if (fast) {
    const uint w0 = tf_exl3_word(tile, wi0), w1 = tf_exl3_word(tile, a1);
    const uint w2 = tf_exl3_word(tile, a2), w3 = tf_exl3_word(tile, a3);
    TF_UNROLL
    for (uint j = 0; j < 8; ++j) {
      const uint lo = rel[j] == 0 ? w0 : rel[j] == 1 ? w1 : w2;
      const uint hi = rel[j] == 0 ? w1 : rel[j] == 1 ? w2 : w3;
      const uint carry = (hi >> sh[j]) & (0u - uint(off[j] != 0));
      const ushort s = ushort(((lo << off[j]) | carry) >> 16);
      dst[at[j]] = table_on ? tf_exl3_table_value(table, codebook, s) : tf_exl3_value(s, codebook);
    }
  } else {
    for (uint j = 0; j < 8; ++j) {
      const uint p = lane + 32 * j, l = p / 8, e = p % 8;
      const uint r = 2 * (l % 4) + (e & 1u) + 8 * ((e >> 1) & 1u);
      const uint c = l / 4 + 8 * (e >> 2);
      const ushort s = tf_exl3_mapped_state(tile, maps[j]);
      dst[r * 16 + c] = table_on ? tf_exl3_table_value(table, codebook, s) : tf_exl3_value(s, codebook);
    }
  }
}

template <bool Table>
inline void tf_exl3_matmul_wide2_impl(const device half* x, const device uchar* trellis,
                                      device float* sums, constant TfExl3Args& a,
                                      const device uint* routes, const device TfExl3Expert* descriptors,
                                      device float* partials, const device half* table,
                                      uint lane, uint sg, uint3 tg,
                                      threadgroup half* wx_all, threadgroup float* xx_all) {
  const uint split = a.splits > 1 ? tg.y : 0;
  const uint rb = a.splits > 1 ? tg.z * 8 : tg.y * 8, nb = tg.x * 128 + sg * 16;
  if (!simd_any(tf_exl3_active(routes, a, rb + lane % 8))) return;
  const device uchar* data = trellis;
  const ulong ks = a.tile_k_stride;
  const ulong ns = a.tile_n_stride;
  const uint halves = a.halves;
  const uint codebook = a.codebook;
  threadgroup half* wx = wx_all + sg * 512;
  uint maps[8], at[8], off[8], sh[8], rel[8];
  const uint words = 4 * halves;
  const uint ring = 128 * halves;
  const uint wi0 = tf_exl3_map_of(lane * 8, halves) & 63u;
  const uint cur0 = (wi0 << 5) | ((tf_exl3_map_of(lane * 8, halves) >> 6) & 31u);
  TF_UNROLL
  for (uint j = 0; j < 8; ++j) {
    const uint p = lane * 8 + j, l = p / 8, e = p % 8;
    uint c8 = cur0 + j * (halves >> 1);
    if (c8 >= ring) c8 -= ring;
    off[j] = c8 & 31u;
    sh[j] = (32u - off[j]) & 31u;
    rel[j] = (c8 >> 5) >= wi0 ? (c8 >> 5) - wi0 : (c8 >> 5) + words - wi0;
    at[j] = (2 * (l % 4) + (e & 1u) + 8 * ((e >> 1) & 1u)) * 16u + (l / 4 + 8 * (e >> 2));
  }
  for (uint j = 0; j < 8; ++j) {
    const uint p = lane + 32 * j, ring2 = 128 * halves;
    const uint end = ((p + 1) * halves - ((halves & 1u) ? ((p + 1) & 1u) : 0u)) / 2;
    uint start = end + ring2 - 16;
    if (start >= ring2) start -= ring2;
    const uint wi = start >> 5, o = start & 31u, next = wi + 1 == 4 * halves ? 0 : wi + 1;
    maps[j] = wi | (o << 6) | (next << 11);
  }
  float high[4] = {0,0,0,0};
  const uint blocks = a.k / 16;
  const uint b0 = a.splits > 1 ? blocks * split / a.splits : 0;
  const uint b1 = a.splits > 1 ? blocks * (split + 1) / a.splits : blocks;
  const bool fast = (halves & 1u) == 0 and halves <= 16;
  const uint a1 = wi0 + 1 < words ? wi0 + 1 : wi0 + 1 - words;
  const uint a2 = wi0 + 2 < words ? wi0 + 2 : wi0 + 2 - words;
  const uint a3 = wi0 + 3 < words ? wi0 + 3 : wi0 + 3 - words;
  uint b = b0;
  for (; b + 1 < b1; b += 2) {
    threadgroup float* xx = xx_all + (((b - b0) >> 1) & 1u) * 256;
    // One warp loads the 8-row x 32-k activation tile for all eight; zeros past a.rows.
    if (sg == 0) {
      for (uint j = lane; j < 256; j += 32) {
        const uint row = rb + j / 32;
        const uint kk = b * 16 + j % 32;
        xx[j] = (kk < a.k and tf_exl3_active(routes, a, row)) ? float(x[ulong(row) * a.k + kk]) : 0.0f;
      }
    }
    const device ushort* tileA = reinterpret_cast<const device ushort*>(data + ulong(b) * ks + ulong(nb / 16) * ns);
    const device ushort* tileB = reinterpret_cast<const device ushort*>(data + ulong(b + 1) * ks + ulong(nb / 16) * ns);
    tf_exl3_wide2_decode16(tileA, wi0, a1, a2, a3, wx, at, off, sh, rel, maps, lane, table, codebook, Table, fast);
    tf_exl3_wide2_decode16(tileB, wi0, a1, a2, a3, wx + 256, at, off, sh, rel, maps, lane, table, codebook, Table, fast);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 4; ++i) {
      const uint at_i = lane + 32 * i, row = rb + at_i / 16, col = at_i % 16;
      if (!tf_exl3_active(routes, a, row)) continue;
      for (uint k = 0; k < 32; ++k) high[i] += xx[(at_i / 16) * 32 + k] * float(wx[k * 16 + col]);
    }
  }
  if (b < b1) {
    threadgroup float* xx = xx_all + (((b - b0) >> 1) & 1u) * 256;
    if (sg == 0) {
      for (uint j = lane; j < 128; j += 32) {
        const uint row = rb + j / 16;
        xx[(j / 16) * 32 + j % 16] = tf_exl3_active(routes, a, row) ? float(x[ulong(row) * a.k + b * 16 + j % 16]) : 0.0f;
      }
    }
    const device ushort* tileA = reinterpret_cast<const device ushort*>(data + ulong(b) * ks + ulong(nb / 16) * ns);
    tf_exl3_wide2_decode16(tileA, wi0, a1, a2, a3, wx, at, off, sh, rel, maps, lane, table, codebook, Table, fast);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 4; ++i) {
      const uint at_i = lane + 32 * i, row = rb + at_i / 16, col = at_i % 16;
      if (!tf_exl3_active(routes, a, row)) continue;
      for (uint k = 0; k < 16; ++k) high[i] += xx[(at_i / 16) * 32 + k] * float(wx[k * 16 + col]);
    }
  }
  for (uint i = 0; i < 4; ++i) {
    const uint at_i = lane + 32 * i, row = rb + at_i / 16;
    if (tf_exl3_active(routes, a, row)) {
      if (a.splits > 1) partials[(ulong(split) * a.rows + row) * a.n + nb + at_i % 16] = high[i];
      else sums[ulong(row) * a.n + nb + at_i % 16] = high[i];
    }
  }
}

kernel void tf_exl3_matmul_wide2_tab(const device half* x [[buffer(0)]],
                                     const device uchar* trellis [[buffer(1)]],
                                     device float* sums [[buffer(2)]],
                                     constant TfExl3Args& a [[buffer(3)]],
                                     const device uint* routes [[buffer(4)]],
                                     const device TfExl3Expert* descriptors [[buffer(5)]],
                                     device float* partials [[buffer(6)]],
                                     const device half* table [[buffer(7)]],
                                     uint lane [[thread_index_in_simdgroup]],
                                     uint sg [[simdgroup_index_in_threadgroup]],
                                     uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half wx[8 * 512];
  threadgroup float xx[2 * 256];
  tf_exl3_matmul_wide2_impl<true>(x, trellis, sums, a, routes, descriptors, partials, table, lane, sg, tg, wx, xx);
}

// Direct-register one-row decode (): a lane owns one output column, decodes that column's 16 states for
// a 16-k block into registers (eight stream words; the window is sized for halves <= 14), reads the codebook table and
// accumulates ascending k. No threadgroup weight tile, no barrier, no shared round trip; a warp covers 32 columns.
// The k order, the table entry and the FP32 FMA sequence are the wide body's, so the one-row output is
// bit-identical by construction (halves = 6, the shipped pack; guarded on that in the caller).
kernel void tf_exl3_matmul_direct_tab(const device half* x [[buffer(0)]],
                                      const device uchar* trellis [[buffer(1)]],
                                      device float* sums [[buffer(2)]],
                                      constant TfExl3Args& a [[buffer(3)]],
                                      const device uint* routes [[buffer(4)]],
                                      const device TfExl3Expert* descriptors [[buffer(5)]],
                                      device float* partials [[buffer(6)]],
                                      const device half* table [[buffer(7)]],
                                      uint lane [[thread_index_in_simdgroup]],
                                      uint sg [[simdgroup_index_in_threadgroup]],
                                      uint3 tptg [[threads_per_threadgroup]],
                                      uint3 tg [[threadgroup_position_in_grid]]) {
  const uint split = a.splits > 1 ? tg.y : 0;
  const uint span = tptg.x;
  const uint nb = tg.x * span + sg * 32 + lane;
  if (nb >= a.n) return;
  const device uchar* data = trellis;
  const ulong ks = a.tile_k_stride;
  const ulong ns = a.tile_n_stride;
  const uint halves = a.halves;
  const uint codebook = a.codebook;
  const uint words = 4 * halves;
  const uint c = lane & 15u;
  // The tile state of (k, c): p = base + 8*(k/2) + (k&1) - 30*(k/8), from the wide body's at[] map.
  const uint base = 32u * (c % 8u) + 4u * (c / 8u);
  uint rel[16], off[16], wnext[8];
  uint wbase = 0;
#if defined(TF_DIRECT_NW)
  uint maxrel = 0;
#endif
  for (uint k = 0; k < 16; ++k) {
    const uint p = base + 8u * (k >> 1) + (k & 1u) - 30u * ((k >> 3) & 1u);
    const uint map = tf_exl3_map_of(p, halves);
    const uint wi = map & 63u;
    if (k == 0) wbase = wi;
    rel[k] = (wi + words - wbase) % words;
#if defined(TF_DIRECT_NW)
    maxrel = max(maxrel, rel[k]);
#endif
    off[k] = (map >> 6) & 31u;
  }
  for (uint i = 0; i < 8; ++i) wnext[i] = (wbase + i) % words;
#if defined(TF_DIRECT_NW)
  // The sixteen states of a column span 27 ring positions; at halves = 10 the union is six words
  // (max(rel) = 4), at halves = 12 seven (max(rel) = 5), at halves = 6 five. The body's eight-word load is two
  // or three words more than the `rel[k] + 1` access ever reaches; load only what the map names.
  const uint nw = min(maxrel + 2u, words);
#endif
  float high = 0;
  const uint blocks = a.k / 16;
  const uint b0 = a.splits > 1 ? blocks * split / a.splits : 0;
  const uint b1 = a.splits > 1 ? blocks * (split + 1) / a.splits : blocks;
  for (uint b = b0; b < b1; ++b) {
    const device ushort* tile = reinterpret_cast<const device ushort*>(data + ulong(b) * ks + ulong(nb / 16) * ns);
    uint w[8];
#if defined(TF_DIRECT_NW)
    switch (nw) {
      case 3: { TF_UNROLL for (uint i = 0; i < 3; ++i) w[i] = tf_exl3_word(tile, wnext[i]); break; }
      case 4: { TF_UNROLL for (uint i = 0; i < 4; ++i) w[i] = tf_exl3_word(tile, wnext[i]); break; }
      case 5: { TF_UNROLL for (uint i = 0; i < 5; ++i) w[i] = tf_exl3_word(tile, wnext[i]); break; }
      case 6: { TF_UNROLL for (uint i = 0; i < 6; ++i) w[i] = tf_exl3_word(tile, wnext[i]); break; }
      case 7: { TF_UNROLL for (uint i = 0; i < 7; ++i) w[i] = tf_exl3_word(tile, wnext[i]); break; }
      default: { TF_UNROLL for (uint i = 0; i < 8; ++i) w[i] = tf_exl3_word(tile, wnext[i]); break; }
    }
#else
    TF_UNROLL
    for (uint i = 0; i < 8; ++i) w[i] = tf_exl3_word(tile, wnext[i]);
#endif
    TF_UNROLL
    for (uint k = 0; k < 16; ++k) {
      const uint lo = w[rel[k]], hi = w[rel[k] + 1];
      const uint carry = (hi >> ((32u - off[k]) & 31u)) & (0u - uint(off[k] != 0));
      const ushort s = ushort(((lo << off[k]) | carry) >> 16);
      high += float(x[ulong(b) * 16 + k]) * float(tf_exl3_table_value(table, codebook, s));
    }
  }
  if (a.splits > 1) partials[ulong(split) * a.rows * a.n + nb] = high;
  else sums[nb] = high;
}

// The routed decode's direct-register descriptor body. The routed stage reads its weights through
// descriptors[expert] (one row an expert), so the dense direct body's weight source does not apply; this is the
// same register body with the descriptor's trellis, strides, width and codebook in place of the dense arguments.
// A lane owns one output column of one row, decodes that column's sixteen states per 16-k block into registers
// (eight stream words), and accumulates ascending k in FP32: no threadgroup weight tile, no barrier, no MMA,
// and with splits == 1 no split reduce. The tile map is the dense body's, which is the inverse of the routed MMA
// body's at[] map, so the same (row, column) elements are decoded. Our own design; no ported code.
kernel void tf_exl3_matmul_direct_desc(const device half* x [[buffer(0)]],
                                       const device uchar* trellis [[buffer(1)]],
                                       device float* sums [[buffer(2)]],
                                       constant TfExl3Args& a [[buffer(3)]],
                                       const device uint* routes [[buffer(4)]],
                                       const device TfExl3Expert* descriptors [[buffer(5)]],
                                       device float* partials [[buffer(6)]],
                                       const device half* table [[buffer(7)]],
                                       uint lane [[thread_index_in_simdgroup]],
                                       uint sg [[simdgroup_index_in_threadgroup]],
                                       uint3 tptg [[threads_per_threadgroup]],
                                       uint3 tg [[threadgroup_position_in_grid]]) {
  const uint row = tg.y;
  if (row >= a.rows) return;
  const uint expert = routes[row];
  if (expert >= a.experts) return;
  const uint split = a.splits > 1 ? tg.z : 0;
  const uint span = tptg.x;
  const uint nb = tg.x * span + sg * 32u + lane;
  if (nb >= a.n) return;
  const device uchar* data = descriptors[expert].trellis;
  const ulong ks = descriptors[expert].tile_k_stride;
  const ulong ns = descriptors[expert].tile_n_stride;
  const uint halves = descriptors[expert].halves;
  const uint codebook = descriptors[expert].codebook;
  const uint words = 4 * halves;
  const uint c = nb & 15u;
  // The tile state of (k, c): p = base + 8*(k/2) + (k&1) - 30*(k/8), the dense body's map.
  const uint base = 32u * (c % 8u) + 4u * (c / 8u);
  uint rel[16], off[16], wnext[8];
  uint wbase = 0;
  for (uint k = 0; k < 16; ++k) {
    const uint p = base + 8u * (k >> 1) + (k & 1u) - 30u * ((k >> 3) & 1u);
    const uint map = tf_exl3_map_of(p, halves);
    const uint wi = map & 63u;
    if (k == 0) wbase = wi;
    rel[k] = (wi + words - wbase) % words;
    off[k] = (map >> 6) & 31u;
  }
  for (uint i = 0; i < 8; ++i) wnext[i] = (wbase + i) % words;
  float high = 0;
  const uint blocks = a.k / 16;
  const uint b0 = a.splits > 1 ? blocks * split / a.splits : 0;
  const uint b1 = a.splits > 1 ? blocks * (split + 1) / a.splits : blocks;
  const device half* xr = x + ulong(row) * a.k;
  uint w[8];
  if (b0 < b1) {
    const device ushort* tile = reinterpret_cast<const device ushort*>(data + ulong(b0) * ks + ulong(nb / 16) * ns);
    TF_UNROLL
    for (uint i = 0; i < 8; ++i) w[i] = tf_exl3_word(tile, wnext[i]);
  }
  for (uint b = b0; b < b1; ++b) {
    // The next block's words are in flight while this block multiplies: the routed stage's weights come from the
    // expert stream, so the load latency is the wall.
    uint wn[8];
    if (b + 1 < b1) {
      const device ushort* tn = reinterpret_cast<const device ushort*>(data + ulong(b + 1) * ks + ulong(nb / 16) * ns);
      TF_UNROLL
      for (uint i = 0; i < 8; ++i) wn[i] = tf_exl3_word(tn, wnext[i]);
    }
    TF_UNROLL
    for (uint k = 0; k < 16; ++k) {
      const uint lo = w[rel[k]], hi = w[rel[k] + 1];
      const uint carry = (hi >> ((32u - off[k]) & 31u)) & (0u - uint(off[k] != 0));
      const ushort s = ushort(((lo << off[k]) | carry) >> 16);
#if defined(TF_DIRECT_DESC_ARITH)
      high += float(xr[ulong(b) * 16 + k]) * float(tf_exl3_value(s, codebook));
#else
      high += float(xr[ulong(b) * 16 + k]) * float(tf_exl3_table_value(table, codebook, s));
#endif
    }
    TF_UNROLL
    for (uint i = 0; i < 8; ++i) w[i] = wn[i];
  }
  if (a.splits > 1) partials[ulong(split) * a.rows * a.n + ulong(row) * a.n + nb] = high;
  else sums[ulong(row) * a.n + nb] = high;
}

// The direct-register one-row body with a wider load width (): the trellis stream word is read as one
// 32-bit load instead of two 16-bit loads, and the activation row is read as four half4 loads instead of sixteen
// 16-bit loads. The bytes read and their order are unchanged, the codebook entry is the same table entry and the
// accumulation still ascends k in the same FP32 order, so the output is bit-identical by construction.
// The caller must guarantee the alignment the wide loads need: the trellis base 4-byte aligned and the tile
// strides a multiple of four (the EXL3 tile stride is halves*16 bytes, always a multiple of four), and the
// activation row 8-byte aligned. The narrow body (tf_exl3_matmul_direct_tab) is the fallback for the rest.
kernel void tf_exl3_matmul_direct_tab_v(const device half* x [[buffer(0)]],
                                        const device uchar* trellis [[buffer(1)]],
                                        device float* sums [[buffer(2)]],
                                        constant TfExl3Args& a [[buffer(3)]],
                                        const device uint* routes [[buffer(4)]],
                                        const device TfExl3Expert* descriptors [[buffer(5)]],
                                        device float* partials [[buffer(6)]],
                                        const device half* table [[buffer(7)]],
                                        uint lane [[thread_index_in_simdgroup]],
                                        uint sg [[simdgroup_index_in_threadgroup]],
                                        uint3 tptg [[threads_per_threadgroup]],
                                        uint3 tg [[threadgroup_position_in_grid]]) {
  const uint split = a.splits > 1 ? tg.y : 0;
  const uint span = tptg.x;
  const uint nb = tg.x * span + sg * 32 + lane;
  if (nb >= a.n) return;
  const device uchar* data = trellis;
  const ulong ks = a.tile_k_stride;
  const ulong ns = a.tile_n_stride;
  const uint halves = a.halves;
  const uint codebook = a.codebook;
  const uint words = 4 * halves;
  const uint c = lane & 15u;
  // The tile state of (k, c): p = base + 8*(k/2) + (k&1) - 30*(k/8), from the wide body's at[] map.
  const uint base = 32u * (c % 8u) + 4u * (c / 8u);
  uint rel[16], off[16], wnext[8];
  uint wbase = 0;
  for (uint k = 0; k < 16; ++k) {
    const uint p = base + 8u * (k >> 1) + (k & 1u) - 30u * ((k >> 3) & 1u);
    const uint map = tf_exl3_map_of(p, halves);
    const uint wi = map & 63u;
    if (k == 0) wbase = wi;
    rel[k] = (wi + words - wbase) % words;
    off[k] = (map >> 6) & 31u;
  }
  for (uint i = 0; i < 8; ++i) wnext[i] = (wbase + i) % words;
  float high = 0;
  const uint blocks = a.k / 16;
  const uint b0 = a.splits > 1 ? blocks * split / a.splits : 0;
  const uint b1 = a.splits > 1 ? blocks * (split + 1) / a.splits : blocks;
  for (uint b = b0; b < b1; ++b) {
    const device ushort* tile = reinterpret_cast<const device ushort*>(data + ulong(b) * ks + ulong(nb / 16) * ns);
    uint w[8];
    TF_UNROLL
    for (uint i = 0; i < 8; ++i) w[i] = *reinterpret_cast<const device uint*>(tile + 2 * wnext[i]);
    float xv[16];
    const device half4* x4 = reinterpret_cast<const device half4*>(x + ulong(b) * 16);
    TF_UNROLL
    for (uint j = 0; j < 4; ++j) {
      const half4 h = x4[j];
      xv[4 * j] = float(h.x);
      xv[4 * j + 1] = float(h.y);
      xv[4 * j + 2] = float(h.z);
      xv[4 * j + 3] = float(h.w);
    }
    TF_UNROLL
    for (uint k = 0; k < 16; ++k) {
      const uint lo = w[rel[k]], hi = w[rel[k] + 1];
      const uint carry = (hi >> ((32u - off[k]) & 31u)) & (0u - uint(off[k] != 0));
      const ushort s = ushort(((lo << off[k]) | carry) >> 16);
      high += xv[k] * float(tf_exl3_table_value(table, codebook, s));
    }
  }
  if (a.splits > 1) partials[ulong(split) * a.rows * a.n + nb] = high;
  else sums[nb] = high;
}

// The direct-register one-row body with its stream words named (): the eight stream words of a 16-k
// block are read into named scalars and selected with a comparison chain, instead of an array indexed by the
// runtime cursor `rel[k]`. An array with a runtime index cannot live in registers on this target, so the
// narrow body was reading and writing thread-local memory sixteen times a block. The words read, the windows
// extracted, the codebook entry and the ascending-k FP32 accumulation are all unchanged, so the output is
// bit-identical by construction. The ninth word is the carry word of the last window: the narrow body indexed
// `w[rel[k] + 1]` past its eight entries when rel[k] was 7, which this one reads correctly.
kernel void tf_exl3_matmul_direct_tab_r(const device half* x [[buffer(0)]],
                                       const device uchar* trellis [[buffer(1)]],
                                       device float* sums [[buffer(2)]],
                                       constant TfExl3Args& a [[buffer(3)]],
                                       const device uint* routes [[buffer(4)]],
                                       const device TfExl3Expert* descriptors [[buffer(5)]],
                                       device float* partials [[buffer(6)]],
                                       const device half* table [[buffer(7)]],
                                       uint lane [[thread_index_in_simdgroup]],
                                       uint sg [[simdgroup_index_in_threadgroup]],
                                       uint3 tptg [[threads_per_threadgroup]],
                                       uint3 tg [[threadgroup_position_in_grid]]) {
  const uint split = a.splits > 1 ? tg.y : 0;
  const uint span = tptg.x;
  const uint nb = tg.x * span + sg * 32 + lane;
  if (nb >= a.n) return;
  const device uchar* data = trellis;
  const ulong ks = a.tile_k_stride;
  const ulong ns = a.tile_n_stride;
  const uint halves = a.halves;
  const uint codebook = a.codebook;
  const uint words = 4 * halves;
  const uint c = lane & 15u;
  // The tile state of (k, c): p = base + 8*(k/2) + (k&1) - 30*(k/8), from the wide body's at[] map.
  const uint base = 32u * (c % 8u) + 4u * (c / 8u);
  uint rel[16], off[16], wnext[9];
  uint wbase = 0;
  for (uint k = 0; k < 16; ++k) {
    const uint p = base + 8u * (k >> 1) + (k & 1u) - 30u * ((k >> 3) & 1u);
    const uint map = tf_exl3_map_of(p, halves);
    const uint wi = map & 63u;
    if (k == 0) wbase = wi;
    rel[k] = (wi + words - wbase) % words;
    off[k] = (map >> 6) & 31u;
  }
  for (uint i = 0; i < 9; ++i) wnext[i] = (wbase + i) % words;
  float high = 0;
  const uint blocks = a.k / 16;
  const uint b0 = a.splits > 1 ? blocks * split / a.splits : 0;
  const uint b1 = a.splits > 1 ? blocks * (split + 1) / a.splits : blocks;
  for (uint b = b0; b < b1; ++b) {
    const device ushort* tile = reinterpret_cast<const device ushort*>(data + ulong(b) * ks + ulong(nb / 16) * ns);
    uint w0, w1, w2, w3, w4, w5, w6, w7, w8;
    TF_UNROLL
    for (uint i = 0; i < 9; ++i) {
      const uint v = tf_exl3_word(tile, wnext[i]);
      if (i == 0) w0 = v;
      if (i == 1) w1 = v;
      if (i == 2) w2 = v;
      if (i == 3) w3 = v;
      if (i == 4) w4 = v;
      if (i == 5) w5 = v;
      if (i == 6) w6 = v;
      if (i == 7) w7 = v;
      if (i == 8) w8 = v;
    }
    TF_UNROLL
    for (uint k = 0; k < 16; ++k) {
      const uint r = rel[k];
      const uint lo = r == 0 ? w0 : r == 1 ? w1 : r == 2 ? w2 : r == 3 ? w3 : r == 4 ? w4 : r == 5 ? w5 : r == 6 ? w6 : r == 7 ? w7 : w8;
      const uint hi = r == 0 ? w1 : r == 1 ? w2 : r == 2 ? w3 : r == 3 ? w4 : r == 4 ? w5 : r == 5 ? w6 : r == 6 ? w7 : w8;
      const uint carry = (hi >> ((32u - off[k]) & 31u)) & (0u - uint(off[k] != 0));
      const ushort s = ushort(((lo << off[k]) | carry) >> 16);
      high += float(x[ulong(b) * 16 + k]) * float(tf_exl3_table_value(table, codebook, s));
    }
  }
  if (a.splits > 1) partials[ulong(split) * a.rows * a.n + nb] = high;
  else sums[nb] = high;
}
