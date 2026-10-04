// Quantized embedding lookups and MLX's RMS norm, bit-identical to MLX 0.32.3's gathers, dequantize and rmsbfloat16.
#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;
// tf:header

// One packed unit of table row id (a byte at 4 and 8 bits, three at 6) as bf16 scale * q + bias, as MLX writes it.
template <int BITS>
inline void tf_dequant_unit(const device uint8_t* table, const device bfloat* scales, const device bfloat* biases,
                            long id, int width, int unit, device bfloat* out) {
  constexpr int PER = BITS == 6 ? 4 : 8 / BITS;
  const int first = unit * PER;
  const bfloat scale = scales[id * (width / 64) + first / 64];
  const bfloat bias = biases[id * (width / 64) + first / 64];
  const device uint8_t* w = table + id * (long(width) * BITS / 8) + long(unit) * (BITS == 6 ? 3 : 1);
  if (BITS == 6) {
    out[0] = (w[0] & 0x3f) * scale + bias;
    out[1] = (((w[0] >> 6) & 0x03) + ((w[1] & 0x0f) << 2)) * scale + bias;
    out[2] = (((w[1] >> 4) & 0x0f) + ((w[2] & 0x03) << 4)) * scale + bias;
    out[3] = ((w[2] >> 2) & 0x3f) * scale + bias;
  } else {
    const uint byte = w[0];
    for (int i = 0; i < PER; ++i) {
      const uint8_t q = BITS == 8 ? uint8_t(byte) : uint8_t((byte >> (BITS * i)) & ((1u << BITS) - 1u));
      out[i] = scale * q + bias;
    }
  }
}

// MLX's one-row RMS norm: thread t's squares of 4t..4t+3 in order, simd_sum, a zero-padded 32-slot simd_sum.
inline void tf_rms_row(const device bfloat* src, const device bfloat* gain, device bfloat* dst, float eps, uint width,
                       uint row, uint t, uint sg, uint lane, threadgroup float* sg_sums, threadgroup float* inv_rms) {
  const uint lo = 4 * t;
  src += size_t(row) * width + lo;
  dst += size_t(row) * width + lo;
  gain += lo;
  const bool whole = lo + 4 <= width;
  float v[4];
  float sq = 0.0f;
  if (whole) {
    for (int j = 0; j < 4; ++j) {
      v[j] = src[j];
      sq += v[j] * v[j];
    }
  } else {
    for (int j = 0; j < 4; ++j) {
      v[j] = lo + j < width ? float(src[j]) : 0.0f;
      sq += v[j] * v[j];
    }
  }
  sq = simd_sum(sq);
  if (sg == 0) sg_sums[lane] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) sg_sums[sg] = sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const float all = simd_sum(sg_sums[lane]);
    if (lane == 0) inv_rms[0] = metal::precise::rsqrt(all / float(width) + eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (whole) {
    for (int j = 0; j < 4; ++j) dst[j] = gain[j] * static_cast<bfloat>(v[j] * inv_rms[0]);
  } else {
    for (int j = 0; j < 4; ++j)
      if (lo + j < width) dst[j] = gain[j] * static_cast<bfloat>(v[j] * inv_rms[0]);
  }
}

// tf:kernel tf_embed_b4_g64 inputs=IDS,TABLE,S,B,P outputs=OUT
[[kernel]] void tf_embed_b4_g64(
  const device uint32_t* IDS [[buffer(0)]],
  const device uint8_t* TABLE [[buffer(1)]],
  const device bfloat16_t* S [[buffer(2)]],
  const device bfloat16_t* B [[buffer(3)]],
  const constant int32_t* P [[buffer(4)]],
  device bfloat16_t* OUT [[buffer(5)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
  const int unit = int(thread_position_in_grid.x), r = int(thread_position_in_grid.y);
  if (unit < P[0] / 2) tf_dequant_unit<4>(TABLE, S, B, long(IDS[r]), P[0], unit, OUT + long(r) * P[0] + 2 * unit);
}

// tf:kernel tf_embed_b6_g64 inputs=IDS,TABLE,S,B,P outputs=OUT
[[kernel]] void tf_embed_b6_g64(
  const device uint32_t* IDS [[buffer(0)]],
  const device uint8_t* TABLE [[buffer(1)]],
  const device bfloat16_t* S [[buffer(2)]],
  const device bfloat16_t* B [[buffer(3)]],
  const constant int32_t* P [[buffer(4)]],
  device bfloat16_t* OUT [[buffer(5)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
  const int unit = int(thread_position_in_grid.x), r = int(thread_position_in_grid.y);
  if (unit < P[0] / 4) tf_dequant_unit<6>(TABLE, S, B, long(IDS[r]), P[0], unit, OUT + long(r) * P[0] + 4 * unit);
}

// tf:kernel tf_embed_b8_g64 inputs=IDS,TABLE,S,B,P outputs=OUT
[[kernel]] void tf_embed_b8_g64(
  const device uint32_t* IDS [[buffer(0)]],
  const device uint8_t* TABLE [[buffer(1)]],
  const device bfloat16_t* S [[buffer(2)]],
  const device bfloat16_t* B [[buffer(3)]],
  const constant int32_t* P [[buffer(4)]],
  device bfloat16_t* OUT [[buffer(5)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
  const int unit = int(thread_position_in_grid.x), r = int(thread_position_in_grid.y);
  if (unit < P[0]) tf_dequant_unit<8>(TABLE, S, B, long(IDS[r]), P[0], unit, OUT + long(r) * P[0] + unit);
}

// tf:kernel tf_rms_norm_bf16 inputs=X,W,EPS,DIM outputs=OUT
[[kernel]] void tf_rms_norm_bf16(
  const device bfloat16_t* X [[buffer(0)]],
  const device bfloat16_t* W [[buffer(1)]],
  const constant float* EPS [[buffer(2)]],
  const constant int32_t* DIM [[buffer(3)]],
  device bfloat16_t* OUT [[buffer(4)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  threadgroup float sg_sums[32];
  threadgroup float inv_rms[1];
  tf_rms_row(X, W, OUT, EPS[0], DIM[0], threadgroup_position_in_grid.x, thread_position_in_threadgroup.x,
             simdgroup_index_in_threadgroup, thread_index_in_simdgroup, sg_sums, inv_rms);
}

// tf:kernel tf_embed_rms_b4_g64 inputs=IDS,TABLE,S,B,W,EPS,DIM outputs=H,OUT
[[kernel]] void tf_embed_rms_b4_g64(
  const device uint32_t* IDS [[buffer(0)]],
  const device uint8_t* TABLE [[buffer(1)]],
  const device bfloat16_t* S [[buffer(2)]],
  const device bfloat16_t* B [[buffer(3)]],
  const device bfloat16_t* W [[buffer(4)]],
  const constant float* EPS [[buffer(5)]],
  const constant int32_t* DIM [[buffer(6)]],
  device bfloat16_t* H [[buffer(7)]],
  device bfloat16_t* OUT [[buffer(8)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  threadgroup float sg_sums[32];
  threadgroup float inv_rms[1];
  const uint t = thread_position_in_threadgroup.x, row = threadgroup_position_in_grid.x;
  const int width = DIM[0];
  if (int(4 * t) < width)
    for (int u = 0; u < 2; ++u)
      tf_dequant_unit<4>(TABLE, S, B, long(IDS[row]), width, int(2 * t) + u, H + long(row) * width + 4 * t + 2 * u);
  tf_rms_row(H, W, OUT, EPS[0], uint(width), row, t, simdgroup_index_in_threadgroup, thread_index_in_simdgroup,
             sg_sums, inv_rms);
}
