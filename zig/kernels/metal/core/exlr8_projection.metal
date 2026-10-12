// EXLR8 lane projections: trellis tiles decode to exact E4M3 weights in registers; every row sums in one fixed order.
#include <metal_stdlib>
using namespace metal;

constant constexpr uint TF_NONE = 0xFFFFFFFFu;

// One threadgroup's work: 128 output columns over `blocks` input blocks of 128, each a run of 64 tiles.
struct TfExlr8Job {
  uint k, blocks, base, in_scale;
  uint out_scale, out_sign, out_block, x_col;
  uint y_col, raw, pad0, pad1;
};
struct TfExlr8Rows {
  uint rows, x_stride, y_stride, y_f32;
};
struct TfExlr8Rotate {
  uint rows, x_stride, x_col, x_f32;
  uint r_stride, r_col, sign, scale;
  uint slot_cols, scales, pad0, pad1;
};
struct TfExlr8Combine {
  uint rows, slots, z_stride, y_stride;
  uint y_f32, sign, z_slot, pad;
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

// The mcg codebook's fp16 sum, then E4M3 nearest even: adding 1.5 2^(e+7) puts the fp16 ulp on the E4M3 grid.
inline float tf_exlr8_weight(uint state) {
  uint x = state * 0xCBAC1FEDu;
  x = (x & 0x8FFF8FFFu) ^ 0x3B603B60u;
  const half2 h = as_type<half2>(x);
  const ushort bits = as_type<ushort>(half(h.x + h.y));
  const ushort mag = bits & 0x7FFF;
  const half magic = as_type<half>(ushort((max(mag, ushort(0x2400)) & 0x7C00) + 0x1E00));
  const float w = float((as_type<half>(mag) + magic) - magic);
  return (bits & 0x8000) ? -w : w;
}

// Lane l's eight values 8 l .. 8 l + 7: each state is the 16 stream bits that end at its own, MSB first per u32 word.
template <uint K>
inline void tf_exlr8_decode(const device uint* tile, uint lane, thread float (&w)[8]) {
  constexpr uint ring = 256 * K, words = 8 * K;
  const uint first = (8 * lane * K + K + ring - 16) % ring;
  const uint i0 = first / 32, off = first % 32;
  const ulong hi = (ulong(tile[i0]) << 32) | tile[(i0 + 1) % words];
  // The lane's 7 K + 16 bits, first bit at bit 63: one variable shift, then constant ones.
  const ulong bits = off ? (hi << off) | (ulong(tile[(i0 + 2) % words]) << off >> 32) : hi;
  for (uint n = 0; n < 8; ++n) w[n] = tf_exlr8_weight(uint(bits >> (48 - n * K)) & 0xFFFF);
}

inline void tf_exlr8_store(device uchar* y, ulong at, float v, uint f32) {
  if (f32) reinterpret_cast<device float*>(y)[at] = v;
  else reinterpret_cast<device bfloat*>(y)[at] = bfloat(v);
}

// A simdgroup owns a 16-column tile; lane 4 c + q holds columns c, c + 8 at tile rows 2 q, 2 q + 1, 2 q + 8, 2 q + 9.
template <uint K, uint R>
inline void tf_exlr8_job(const device float* X, const device uchar* W, const device ulong* bases,
                         const device half* halves, const device char* signs, constant TfExlr8Rows& a,
                         TfExlr8Job j, device uchar* Y, uint sg, uint lane, uint tid,
                         threadgroup float* xs, threadgroup float* zs) {
  constexpr uint TB = 32 * K;
  const uint rows = min(a.rows, R), r0 = 2 * (lane % 4), c0 = lane / 4;
  float total[2][R], part[2][R];
  for (uint i = 0; i < R; ++i) total[0][i] = total[1][i] = 0.0f;
  for (uint ib = 0; ib < j.blocks; ++ib) {
    for (uint i = tid; i < R * 128; i += 256)
      xs[i] = i / 128 < rows ? X[ulong(i / 128) * a.x_stride + j.x_col + ib * 128 + i % 128] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < R; ++i) part[0][i] = part[1][i] = 0.0f;
    const device uchar* run = W + bases[j.base + ib];
    for (uint r = 0; r < 8; ++r) {
      float w[8];
      tf_exlr8_decode<K>(reinterpret_cast<const device uint*>(run + (8 * r + sg) * TB), lane, w);
      for (uint i = 0; i < R; ++i) {
        const threadgroup float* x = xs + i * 128 + 16 * r + r0;
        const float2 lo = *reinterpret_cast<const threadgroup float2*>(x);
        const float2 hi = *reinterpret_cast<const threadgroup float2*>(x + 8);
        part[0][i] = fma(w[3], hi.y, fma(w[2], hi.x, fma(w[1], lo.y, fma(w[0], lo.x, part[0][i]))));
        part[1][i] = fma(w[7], hi.y, fma(w[6], hi.x, fma(w[5], lo.y, fma(w[4], lo.x, part[1][i]))));
      }
    }
    // The input block's scale goes on its partial sums, between 128-wide stages.
    const float s = j.in_scale == TF_NONE ? 1.0f : float(halves[j.in_scale + ib]);
    for (uint i = 0; i < R; ++i) {
      total[0][i] = fma(s, part[0][i], total[0][i]);
      total[1][i] = fma(s, part[1][i], total[1][i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
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
    const ulong row = ulong(i) * a.y_stride + j.y_col;
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

// One kernel a row capacity R (1, 2, 4, 8, 16): the accumulators stay in registers; a row's sums never depend on R.
#define TF_EXLR8_PROJECT(R)                                                                                          \
  kernel void tf_exlr8_project_##R(const device float* X [[buffer(0)]], const device uchar* W [[buffer(1)]],         \
                                   const device ulong* bases [[buffer(2)]], const device half* halves [[buffer(3)]], \
                                   const device char* signs [[buffer(4)]], const device TfExlr8Job* jobs [[buffer(5)]], \
                                   constant TfExlr8Rows& a [[buffer(6)]], device uchar* Y [[buffer(7)]],             \
                                   uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], \
                                   uint tid [[thread_index_in_threadgroup]], uint job [[threadgroup_position_in_grid]]) { \
    threadgroup float xs[R * 128], zs[R * 128];                                                                     \
    const TfExlr8Job j = jobs[job];                                                                                  \
    switch (j.k) {                                                                                                   \
      case 2: tf_exlr8_job<2, R>(X, W, bases, halves, signs, a, j, Y, sg, lane, tid, xs, zs); break;                \
      case 3: tf_exlr8_job<3, R>(X, W, bases, halves, signs, a, j, Y, sg, lane, tid, xs, zs); break;                \
      case 4: tf_exlr8_job<4, R>(X, W, bases, halves, signs, a, j, Y, sg, lane, tid, xs, zs); break;                \
      default: break;                                                                                                \
    }                                                                                                                \
  }
TF_EXLR8_PROJECT(1)
TF_EXLR8_PROJECT(2)
TF_EXLR8_PROJECT(4)
TF_EXLR8_PROJECT(8)
TF_EXLR8_PROJECT(16)

// r = (x * s) H0 for one row's 128-block of one slot, s = signs times an optional fp16 per-channel scale.
kernel void tf_exlr8_rotate(const device uchar* X [[buffer(0)]], const device char* signs [[buffer(1)]],
                            const device half* halves [[buffer(2)]], constant TfExlr8Rotate& a [[buffer(3)]],
                            device float* R [[buffer(4)]], const device uint* slot_scale [[buffer(5)]],
                            uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  const uint row = tg.y, base = tg.x * 128, slot = tg.z, col = slot * a.slot_cols;
  const uint scale = a.scales ? slot_scale[slot] : a.scale;
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
    const float s = weights[ulong(row) * a.slots + e] * float(halves[sv[e] + b]);
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
  if (i < 65536) out[i] = tf_exlr8_weight(i);
}

// Qualification for trellis matrices: run `run`'s 64 tiles as E4M3 codes, a 128 x 128 row-major block per run.
kernel void tf_exlr8_tile_codes(const device uchar* W [[buffer(0)]], const device ulong* bases [[buffer(1)]],
                                constant uint& k [[buffer(2)]], device uchar* out [[buffer(3)]],
                                uint lane [[thread_index_in_simdgroup]], uint2 tg [[threadgroup_position_in_grid]]) {
  const uint t = tg.x, run = tg.y;
  const device uint* tile = reinterpret_cast<const device uint*>(W + bases[run] + ulong(t) * 32 * k);
  float w[8];
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
