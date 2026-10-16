// Native EXL3: stored trellis tiles, row-independent compensated dots, and normalized Sylvester rotations.
#include <metal_stdlib>
using namespace metal;

struct TfExl3Args {
  uint k, n, rows, halves;
  uint codebook, su_signs, sv_signs, has_bias;
  uint output_f32, input_ids, output_ids, routed;
  uint expert_id, repeat, indexed, experts;
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
  const uint start = (end + ring - 16) % ring;
  const uint wi = start / 32, off = start % 32;
  const uint lo = tf_exl3_word(tile, wi);
  if (off <= 16) return ushort(lo >> (16 - off));
  const uint hi = tf_exl3_word(tile, (wi + 1) % (ring / 32));
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
                             uint3 tg [[threadgroup_position_in_grid]]) {
  const uint row = tg.y;
  if (!tf_exl3_active(routes, a, row)) return;
  const device ushort* scales = a.indexed ? descriptors[routes[row]].su : su;
  const uint signs = a.indexed ? descriptors[routes[row]].su_signs : a.su_signs;
  const uint source = a.input_ids ? input_ids[row] : row / a.repeat;
  const uint base = tg.x * 128;
  float v[4];
  for (uint j = 0; j < 4; ++j) {
    const uint k = base + lane + 32 * j;
    v[j] = as_type<float>(uint(x[ulong(source) * a.x_stride + k]) << 16) * tf_exl3_scale(scales, k, signs);
  }
  tf_exl3_hadamard(v, lane);
  for (uint j = 0; j < 4; ++j)
    rotated[ulong(row) * a.k + base + lane + 32 * j] = half(v[j] * 0.08838834764831845f);
}

// One SIMDgroup owns eight rows and sixteen columns. Only its current 16x16 weight tile is decoded.
template <bool Hoist>
inline void tf_exl3_matmul_impl(const device half* x, const device uchar* trellis,
                                device float* sums, constant TfExl3Args& a,
                                const device uint* routes, const device TfExl3Expert* descriptors,
                                uint lane, uint3 tg,
                                threadgroup float* wx, threadgroup float* xx) {
  const uint rb = a.indexed == 1 ? tg.y : tg.y * 8, nb = tg.x * 16;
  if (!simd_any(tf_exl3_active(routes, a, rb + (a.indexed == 1 ? 0 : lane % 8)))) return;
  const uint id = a.indexed ? routes[rb] : 0;
  const device uchar* data = a.indexed ? descriptors[id].trellis : trellis;
  const ulong ks = a.indexed ? descriptors[id].tile_k_stride : a.tile_k_stride;
  const ulong ns = a.indexed ? descriptors[id].tile_n_stride : a.tile_n_stride;
  const uint halves = a.indexed ? descriptors[id].halves : a.halves;
  const uint codebook = a.indexed ? descriptors[id].codebook : a.codebook;
  uint maps[8];
  if (Hoist) {
    for (uint j = 0; j < 8; ++j) {
      const uint p = lane + 32 * j, ring = 128 * halves;
      const uint end = ((p + 1) * halves - ((halves & 1u) ? ((p + 1) & 1u) : 0u)) / 2;
      const uint start = (end + ring - 16) % ring;
      const uint wi = start / 32, off = start % 32, next = wi + 1 == 4 * halves ? 0 : wi + 1;
      maps[j] = wi | (off << 6) | (next << 11);
    }
  }
  float high[4]={0,0,0,0}, low[4]={0,0,0,0};
  for (uint kb = 0; kb < a.k; kb += 16) {
    const device ushort* tile = reinterpret_cast<const device ushort*>(
        data + ulong(kb / 16) * ks + ulong(nb / 16) * ns);
    for (uint j = 0; j < 8; ++j) {
      const uint p = lane + 32 * j, l = p / 8, e = p % 8;
      const uint r = 2 * (l % 4) + (e & 1u) + 8 * ((e >> 1) & 1u);
      const uint c = l / 4 + 8 * (e >> 2);
      const ushort state = Hoist ? tf_exl3_mapped_state(tile, maps[j]) : tf_exl3_state(tile, halves, p);
      wx[r * 16 + c] = float(tf_exl3_value(state, codebook));
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
        // FP16 products are exact in FP32. TwoSum retains each addition's rounding error.
        const float product=xx[(at/16)*16+k]*wx[k*16+col];
        const float next=high[i]+product;
        const float borrowed=next-high[i];
        low[i]+=(high[i]-(next-borrowed))+(product-borrowed);
        high[i]=next;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint i=0;i<4;++i) {
    const uint at=lane+32*i, row=rb+at/16;
    if ((a.indexed!=1 || at/16==0) && tf_exl3_active(routes,a,row))
      sums[ulong(row)*a.n+nb+at%16]=high[i]+low[i];
  }
}

kernel void tf_exl3_matmul(const device half* x [[buffer(0)]],
                          const device uchar* trellis [[buffer(1)]],
                          device float* sums [[buffer(2)]],
                          constant TfExl3Args& a [[buffer(3)]],
                          const device uint* routes [[buffer(4)]],
                          const device TfExl3Expert* descriptors [[buffer(5)]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup float wx[256], xx[128];
  tf_exl3_matmul_impl<false>(x, trellis, sums, a, routes, descriptors, lane, tg, wx, xx);
}


kernel void tf_exl3_matmul_hoisted(const device half* x [[buffer(0)]],
                                  const device uchar* trellis [[buffer(1)]],
                                  device float* sums [[buffer(2)]],
                                  constant TfExl3Args& a [[buffer(3)]],
                                  const device uint* routes [[buffer(4)]],
                                  const device TfExl3Expert* descriptors [[buffer(5)]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup float wx[256], xx[128];
  tf_exl3_matmul_impl<true>(x, trellis, sums, a, routes, descriptors, lane, tg, wx, xx);
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
                              uint3 tg [[threadgroup_position_in_grid]]) {
  const uint row = tg.y, base = tg.x * 128;
  if (!tf_exl3_active(routes, a, row)) return;
  const uint id = a.indexed ? routes[row] : 0;
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
