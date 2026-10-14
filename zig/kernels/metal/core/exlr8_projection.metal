// EXLR8 lane projections: trellis tiles decode to exact E4M3 weights in registers; every row sums in one fixed order.
#include <metal_stdlib>
using namespace metal;

constant constexpr uint TF_NONE = 0xFFFFFFFFu;

// One threadgroup's work: 128 output columns over `blocks` input blocks of 128, each a run of 64 tiles.
struct TfExlr8Job {
  uint k, blocks, base, in_scale;
  uint out_scale, out_sign, out_block, x_col;
  uint y_col, raw, members, count;
};
struct TfExlr8Rows {
  uint rows, x_stride, y_stride, y_f32;
};
struct TfExlr8Rotate {
  uint rows, x_stride, x_col, x_f32;
  uint r_stride, r_col, sign, scale;
  uint slot_cols, scales, row_slots, pad1;
};
struct TfExlr8Combine {
  uint rows, slots, z_stride, y_stride;
  uint y_f32, sign, z_slot, per_row;
};

// The unnormalized Sylvester transform of 128 values held as lane + 32 q, in one fixed order of fp32 adds.
inline void tf_exlr8_fwht(thread float (&v)[4], uint lane) {
  for (ushort d = 1; d < 32; d <<= 1) {
    for (uint q = 0; q < 4; ++q) {
      const float other = simd_shuffle_xor(v[q], d);
      v[q] = (lane & d) ? other - v[q] : v[q] + other;
    }
  }
  const float a0 = v[0] + v[1], a1 = v[0] - v[1], a2 = v[2] + v[3], a3 = v[2] - v[3];
  v[0] = a0 + a2;
  v[1] = a1 + a3;
  v[2] = a0 - a2;
  v[3] = a1 - a3;
}

// The mcg codebook's fp16 sum times 2^-8 (xor 0x1B60, not 0x3B60: each half 8 binades lower, still normal; exact).
inline ushort tf_exlr8_sum8(uint state) {
  uint x = state * 0xCBAC1FEDu;
  x = (x & 0x8FFF8FFFu) ^ 0x1B601B60u;
  const half2 h = as_type<half2>(x);
  return as_type<ushort>(half(h.x + h.y));
}

// Two states' weights: at 2^-8 E4M3 nearest even is the fp16 pattern's nearest even at bit 7, subnormals included.
inline half2 tf_exlr8_pair(uint s0, uint s1) {
  const uint b = uint(tf_exlr8_sum8(s0)) | (uint(tf_exlr8_sum8(s1)) << 16);
  return as_type<half2>((b + 0x003F003Fu + ((b >> 7) & 0x00010001u)) & 0xFF80FF80u) * half2(256.0h);
}

// Lane l's stream words for its eight values 8 l .. 8 l + 7, MSB first per u32 word.
template <uint K>
inline uint3 tf_exlr8_words(const device uint* tile, uint lane) {
  constexpr uint ring = 256 * K, words = 8 * K;
  const uint i0 = ((8 * lane * K + K + ring - 16) % ring) / 32;
  return uint3(tile[i0], tile[(i0 + 1) % words], tile[(i0 + 2) % words]);
}

// Each state is the 16 stream bits that end at its own value: one variable shift a tile, then constant ones.
template <uint K>
inline void tf_exlr8_unpack(uint3 t, uint lane, thread half (&w)[8]) {
  const uint off = (8 * lane * K + K + 256 * K - 16) % 32;
  const ulong hi = (ulong(t.x) << 32) | t.y;
  const ulong bits = off ? (hi << off) | (ulong(t.z) << off >> 32) : hi;
  for (uint n = 0; n < 8; n += 2) {
    const half2 v = tf_exlr8_pair(uint(bits >> (48 - n * K)) & 0xFFFF, uint(bits >> (48 - n * K - K)) & 0xFFFF);
    w[n] = v.x;
    w[n + 1] = v.y;
  }
}

template <uint K>
inline void tf_exlr8_decode(const device uint* tile, uint lane, thread half (&w)[8]) {
  tf_exlr8_unpack<K>(tf_exlr8_words<K>(tile, lane), lane, w);
}

inline void tf_exlr8_store(device uchar* y, ulong at, float v, uint f32) {
  if (f32) reinterpret_cast<device float*>(y)[at] = v;
  else reinterpret_cast<device bfloat*>(y)[at] = bfloat(v);
}

// A simdgroup owns a 16-column tile (lane 4 c + q: cols c, c + 8, rows 2 q, +1, +8, +9); members map rows x << 16 | y.
template <uint K, uint R>
inline void tf_exlr8_job(const device float* X, const device uchar* W, const device ulong* bases,
                         const device half* halves, const device char* signs, constant TfExlr8Rows& a,
                         TfExlr8Job j, device uchar* Y, const device uint* members, uint sg, uint lane, uint tid,
                         threadgroup float* xs, threadgroup float* zs) {
  constexpr uint TB = 32 * K;
  constexpr bool alternate = R <= 4;
  const uint rows = min(j.count ? j.count : a.rows, R), r0 = 2 * (lane % 4), c0 = lane / 4;
  float total[2][R], part[2][R];
  for (uint i = 0; i < R; ++i) total[0][i] = total[1][i] = 0.0f;
  // Each tile's words load a tile ahead of its decode.
  const device uchar* run = W + bases[j.base];
  uint3 next = tf_exlr8_words<K>(reinterpret_cast<const device uint*>(run + sg * TB), lane);
  // Small capacities alternate tiles; wider capacities keep the original load and end barriers.
  for (uint ib = 0; ib < j.blocks; ++ib) {
    threadgroup float* tile = alternate && (ib & 1) ? zs : xs;
    for (uint i = tid; i < R * 128; i += 256) {
      const uint rr = i / 128;
      tile[i] = rr < rows ? X[ulong(j.count ? members[j.members + rr] >> 16 : rr) * a.x_stride + j.x_col + ib * 128 + i % 128] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < R; ++i) part[0][i] = part[1][i] = 0.0f;
    const device uchar* later = ib + 1 < j.blocks ? W + bases[j.base + ib + 1] : run;
    for (uint r = 0; r < 8; ++r) {
      const uint3 now = next;
      next = tf_exlr8_words<K>(reinterpret_cast<const device uint*>((r < 7 ? run + (8 * r + 8) * TB : later) + sg * TB), lane);
      half w[8];
      tf_exlr8_unpack<K>(now, lane, w);
      for (uint i = 0; i < R; ++i) {
        const threadgroup float* x = tile + i * 128 + 16 * r + r0;
        const float2 lo = *reinterpret_cast<const threadgroup float2*>(x);
        const float2 hi = *reinterpret_cast<const threadgroup float2*>(x + 8);
        part[0][i] = fma(float(w[3]), hi.y, fma(float(w[2]), hi.x, fma(float(w[1]), lo.y, fma(float(w[0]), lo.x, part[0][i]))));
        part[1][i] = fma(float(w[7]), hi.y, fma(float(w[6]), hi.x, fma(float(w[5]), lo.y, fma(float(w[4]), lo.x, part[1][i]))));
      }
    }
    run = later;
    // The input block's scale goes on its partial sums, between 128-wide stages.
    const float s = j.in_scale == TF_NONE ? 1.0f : float(halves[j.in_scale + ib]);
    for (uint i = 0; i < R; ++i) {
      total[0][i] = fma(s, part[0][i], total[0][i]);
      total[1][i] = fma(s, part[1][i], total[1][i]);
    }
    if (!alternate) threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (alternate) zs = (j.blocks & 1) ? zs : xs;
  for (uint i = 0; i < R; ++i) {
    for (uint c = 0; c < 2; ++c) {
      float v = total[c][i];
      v += simd_shuffle_xor(v, ushort(1));
      v += simd_shuffle_xor(v, ushort(2));
      if (lane % 4 == 0) zs[i * 128 + 16 * sg + c0 + 8 * c] = v;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = sg; i < rows; i += 8) {
    float v[4];
    for (uint q = 0; q < 4; ++q) v[q] = zs[i * 128 + lane + 32 * q];
    const ulong row = ulong(j.count ? members[j.members + i] & 0xFFFFu : i) * a.y_stride + j.y_col;
    if (j.raw) {
      for (uint q = 0; q < 4; ++q) reinterpret_cast<device float*>(Y)[row + lane + 32 * q] = v[q];
      continue;
    }
    tf_exlr8_fwht(v, lane);
    const float block = j.out_block == TF_NONE ? 1.0f : float(halves[j.out_block]);
    for (uint q = 0; q < 4; ++q) {
      const uint c = lane + 32 * q;
      // svh, the sign and the block scale times 2^-7 multiply exactly; one rounding remains.
      float scale = block * 0.0078125f;
      if (j.out_scale != TF_NONE) scale *= float(halves[j.out_scale + c]);
      if (j.out_sign != TF_NONE) scale *= float(signs[j.out_sign + c]);
      tf_exlr8_store(Y, row + c, v[q] * scale, a.y_f32);
    }
  }
}

// S sets of 8 simdgroups take a job's blocks round-robin; set 0 folds every block's sums in block order, as one set.
template <uint K, uint R, uint S>
inline void tf_exlr8_split(const device float* X, const device uchar* W, const device ulong* bases,
                           const device half* halves, const device char* signs, constant TfExlr8Rows& a,
                           TfExlr8Job j, device uchar* Y, const device uint* members, uint sg, uint lane, uint tid,
                           threadgroup float* xs, threadgroup float* ps, threadgroup float* zs) {
  constexpr uint TB = 32 * K;
  const uint set = sg / 8, ct = sg % 8;
  const uint rows = min(j.count ? j.count : a.rows, R), r0 = 2 * (lane % 4), c0 = lane / 4;
  float total[2][R], part[2][R];
  for (uint i = 0; i < R; ++i) total[0][i] = total[1][i] = 0.0f;
  // A set past the last block decodes the last block again and its sums are never folded.
  const device uchar* run = W + bases[j.base + min(set, j.blocks - 1)];
  uint3 next = tf_exlr8_words<K>(reinterpret_cast<const device uint*>(run + ct * TB), lane);
  for (uint ib0 = 0; ib0 < j.blocks; ib0 += S) {
    for (uint i = tid; i < S * R * 128; i += S * 256) {
      const uint b = min(ib0 + i / (R * 128), j.blocks - 1), rr = (i / 128) % R;
      xs[i] = rr < rows ? X[ulong(j.count ? members[j.members + rr] >> 16 : rr) * a.x_stride + j.x_col + b * 128 + i % 128] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < R; ++i) part[0][i] = part[1][i] = 0.0f;
    const uint nb = min(ib0 + set + S, j.blocks - 1);
    const device uchar* later = W + bases[j.base + nb];
    for (uint r = 0; r < 8; ++r) {
      const uint3 now = next;
      next = tf_exlr8_words<K>(reinterpret_cast<const device uint*>((r < 7 ? run + (8 * r + 8) * TB : later) + ct * TB), lane);
      half w[8];
      tf_exlr8_unpack<K>(now, lane, w);
      for (uint i = 0; i < R; ++i) {
        const threadgroup float* x = xs + (set * R + i) * 128 + 16 * r + r0;
        const float2 lo = *reinterpret_cast<const threadgroup float2*>(x);
        const float2 hi = *reinterpret_cast<const threadgroup float2*>(x + 8);
        part[0][i] = fma(float(w[3]), hi.y, fma(float(w[2]), hi.x, fma(float(w[1]), lo.y, fma(float(w[0]), lo.x, part[0][i]))));
        part[1][i] = fma(float(w[7]), hi.y, fma(float(w[6]), hi.x, fma(float(w[5]), lo.y, fma(float(w[4]), lo.x, part[1][i]))));
      }
    }
    run = later;
    if (set > 0)
      for (uint i = 0; i < R; ++i)
        for (uint c = 0; c < 2; ++c) ps[(((set - 1) * R + i) * 8 + ct) * 64 + 2 * lane + c] = part[c][i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (set == 0) {
      // The input block's scale goes on its partial sums, between 128-wide stages, block after block.
      for (uint s = 0; s < S && ib0 + s < j.blocks; ++s) {
        const float sc = j.in_scale == TF_NONE ? 1.0f : float(halves[j.in_scale + ib0 + s]);
        for (uint i = 0; i < R; ++i)
          for (uint c = 0; c < 2; ++c) total[c][i] = fma(sc, s ? ps[(((s - 1) * R + i) * 8 + ct) * 64 + 2 * lane + c] : part[c][i], total[c][i]);
      }
    }
  }
  if (set == 0)
    for (uint i = 0; i < R; ++i) {
      for (uint c = 0; c < 2; ++c) {
        float v = total[c][i];
        v += simd_shuffle_xor(v, ushort(1));
        v += simd_shuffle_xor(v, ushort(2));
        if (lane % 4 == 0) zs[i * 128 + 16 * ct + c0 + 8 * c] = v;
      }
    }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = sg; i < rows; i += 8 * S) {
    float v[4];
    for (uint q = 0; q < 4; ++q) v[q] = zs[i * 128 + lane + 32 * q];
    const ulong row = ulong(j.count ? members[j.members + i] & 0xFFFFu : i) * a.y_stride + j.y_col;
    if (j.raw) {
      for (uint q = 0; q < 4; ++q) reinterpret_cast<device float*>(Y)[row + lane + 32 * q] = v[q];
      continue;
    }
    tf_exlr8_fwht(v, lane);
    const float block = j.out_block == TF_NONE ? 1.0f : float(halves[j.out_block]);
    for (uint q = 0; q < 4; ++q) {
      const uint c = lane + 32 * q;
      // svh, the sign and the block scale times 2^-7 multiply exactly; one rounding remains.
      float scale = block * 0.0078125f;
      if (j.out_scale != TF_NONE) scale *= float(halves[j.out_scale + c]);
      if (j.out_sign != TF_NONE) scale *= float(signs[j.out_sign + c]);
      tf_exlr8_store(Y, row + c, v[q] * scale, a.y_f32);
    }
  }
}

// Capacities 1-4 reuse the inactive input tile for final sums; capacities 8 and 16 keep separate sums.
#define TF_EXLR8_PROJECT(R)                                                                                          \
  kernel void tf_exlr8_project_##R(const device float* X [[buffer(0)]], const device uchar* W [[buffer(1)]],         \
                                   const device ulong* bases [[buffer(2)]], const device half* halves [[buffer(3)]], \
                                   const device char* signs [[buffer(4)]], const device TfExlr8Job* jobs [[buffer(5)]], \
                                   constant TfExlr8Rows& a [[buffer(6)]], device uchar* Y [[buffer(7)]],             \
                                   const device uint* members [[buffer(8)]],                                         \
                                   uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], \
                                   uint tid [[thread_index_in_threadgroup]], uint job [[threadgroup_position_in_grid]]) { \
    threadgroup float xs[R * 128], zs[R * 128];                                                                     \
    const TfExlr8Job j = jobs[job];                                                                                  \
    switch (j.k) {                                                                                                   \
      case 2: tf_exlr8_job<2, R>(X, W, bases, halves, signs, a, j, Y, members, sg, lane, tid, xs, zs); break;       \
      case 3: tf_exlr8_job<3, R>(X, W, bases, halves, signs, a, j, Y, members, sg, lane, tid, xs, zs); break;       \
      case 4: tf_exlr8_job<4, R>(X, W, bases, halves, signs, a, j, Y, members, sg, lane, tid, xs, zs); break;       \
      default: break;                                                                                                \
    }                                                                                                                \
  }
TF_EXLR8_PROJECT(1)
TF_EXLR8_PROJECT(2)
TF_EXLR8_PROJECT(4)
TF_EXLR8_PROJECT(8)
TF_EXLR8_PROJECT(16)
// A job on 256 S threads for few long jobs (core/exlr8_job.zig's splits); its bits are the one-set kernel's.
#define TF_EXLR8_SPLIT(R, S)                                                                                         \
  kernel void tf_exlr8_split_##R##_##S(const device float* X [[buffer(0)]], const device uchar* W [[buffer(1)]],     \
                                       const device ulong* bases [[buffer(2)]], const device half* halves [[buffer(3)]], \
                                       const device char* signs [[buffer(4)]], const device TfExlr8Job* jobs [[buffer(5)]], \
                                       constant TfExlr8Rows& a [[buffer(6)]], device uchar* Y [[buffer(7)]],         \
                                       const device uint* members [[buffer(8)]],                                     \
                                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], \
                                       uint tid [[thread_index_in_threadgroup]], uint job [[threadgroup_position_in_grid]]) { \
    threadgroup float xs[S * R * 128], ps[(S - 1) * R * 512], zs[R * 128];                                         \
    const TfExlr8Job j = jobs[job];                                                                                  \
    switch (j.k) {                                                                                                   \
      case 2: tf_exlr8_split<2, R, S>(X, W, bases, halves, signs, a, j, Y, members, sg, lane, tid, xs, ps, zs); break; \
      case 3: tf_exlr8_split<3, R, S>(X, W, bases, halves, signs, a, j, Y, members, sg, lane, tid, xs, ps, zs); break; \
      case 4: tf_exlr8_split<4, R, S>(X, W, bases, halves, signs, a, j, Y, members, sg, lane, tid, xs, ps, zs); break; \
      default: break;                                                                                                \
    }                                                                                                                \
  }
TF_EXLR8_SPLIT(1, 2)
TF_EXLR8_SPLIT(1, 4)
TF_EXLR8_SPLIT(2, 2)
TF_EXLR8_SPLIT(2, 4)
TF_EXLR8_SPLIT(4, 2)

// r = (x * s) H0 for one row's 128-block of one slot, s = signs times an optional fp16 per-channel scale.
kernel void tf_exlr8_rotate(const device uchar* X [[buffer(0)]], const device char* signs [[buffer(1)]],
                            const device half* halves [[buffer(2)]], constant TfExlr8Rotate& a [[buffer(3)]],
                            device float* R [[buffer(4)]], const device uint* slot_scale [[buffer(5)]],
                            uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  const uint row = tg.y, base = tg.x * 128, slot = tg.z, col = slot * a.slot_cols;
  const uint scale = a.scales ? slot_scale[a.row_slots ? row * a.row_slots + slot : slot] : a.scale;
  float v[4];
  for (uint q = 0; q < 4; ++q) {
    const uint c = base + lane + 32 * q;
    const ulong at = ulong(row) * a.x_stride + a.x_col + col + c;
    float x = a.x_f32 ? reinterpret_cast<const device float*>(X)[at] : float(reinterpret_cast<const device bfloat*>(X)[at]);
    if (a.sign != TF_NONE) x *= float(signs[a.sign + c]);
    if (scale != TF_NONE) x *= float(halves[scale + c]);
    v[q] = x;
  }
  tf_exlr8_fwht(v, lane);
  for (uint q = 0; q < 4; ++q) R[ulong(row) * a.r_stride + a.r_col + col + base + lane + 32 * q] = v[q];
}

// The down projection's fan-in: sum_e w_e sv_e z_e per row and 128-block, then one transform, the signs and 2^-7.
kernel void tf_exlr8_combine(const device float* Z [[buffer(0)]], const device float* weights [[buffer(1)]],
                             const device uint* sv [[buffer(2)]], const device half* halves [[buffer(3)]],
                             const device char* signs [[buffer(4)]], constant TfExlr8Combine& a [[buffer(5)]],
                             device uchar* Y [[buffer(6)]], uint lane [[thread_index_in_simdgroup]],
                             uint2 tg [[threadgroup_position_in_grid]]) {
  const uint row = tg.y, b = tg.x;
  float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (uint e = 0; e < a.slots; ++e) {
    const float s = weights[ulong(row) * a.slots + e] * float(halves[sv[a.per_row ? row * a.slots + e : e] + b]);
    const device float* z = Z + ulong(row) * a.z_stride + ulong(e) * a.z_slot + b * 128;
    for (uint q = 0; q < 4; ++q) v[q] = fma(s, z[lane + 32 * q], v[q]);
  }
  tf_exlr8_fwht(v, lane);
  for (uint q = 0; q < 4; ++q) {
    const uint c = b * 128 + lane + 32 * q;
    const float scale = a.sign == TF_NONE ? 0.0078125f : float(signs[a.sign + c]) * 0.0078125f;
    tf_exlr8_store(Y, ulong(row) * a.y_stride + c, v[q] * scale, a.y_f32);
  }
}

// Qualification: every state's weight, for the host to compare with its own codebook bit for bit.
kernel void tf_exlr8_weights(device float* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  if (i >= 65536) return;
  const half2 v = tf_exlr8_pair(i & ~1u, i | 1u);
  out[i] = float(i & 1 ? v.y : v.x);
}

// Qualification for trellis matrices: run `run`'s 64 tiles as E4M3 codes, a 128 x 128 row-major block per run.
kernel void tf_exlr8_tile_codes(const device uchar* W [[buffer(0)]], const device ulong* bases [[buffer(1)]],
                                constant uint& k [[buffer(2)]], device uchar* out [[buffer(3)]],
                                uint lane [[thread_index_in_simdgroup]], uint2 tg [[threadgroup_position_in_grid]]) {
  const uint t = tg.x, run = tg.y;
  const device uint* tile = reinterpret_cast<const device uint*>(W + bases[run] + ulong(t) * 32 * k);
  half w[8];
  switch (k) {
    case 2: tf_exlr8_decode<2>(tile, lane, w); break;
    case 3: tf_exlr8_decode<3>(tile, lane, w); break;
    default: tf_exlr8_decode<4>(tile, lane, w); break;
  }
  for (uint n = 0; n < 8; ++n) {
    // an E4M3 value times 2^-8 is an fp16 whose bits are the code's 7 low bits shifted up by 7
    const uint code = (uint(as_type<ushort>(half(fabs(w[n]) * 0.00390625f))) >> 7) | (signbit(w[n]) ? 0x80u : 0u);
    const uint row = 16 * (t / 8) + 2 * (lane % 4) + (n & 1) + 8 * ((n >> 1) & 1);
    const uint col = 16 * (t % 8) + lane / 4 + 8 * (n >> 2);
    out[ulong(run) * 16384 + row * 128 + col] = uchar(code);
  }
}
