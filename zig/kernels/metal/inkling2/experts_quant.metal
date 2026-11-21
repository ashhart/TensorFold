// Inkling 2 Mini checkpoint activation quantization. The finite-input arithmetic follows SGLang routed_quant.cuh
// (41419ca9), with the same quantizers on w13 input and BF16 SwiGLU output. No CPU reformatting or expert sorting.
// Non-finite extension: any NaN or infinity makes the entire row's amax NaN, so its downstream dots are NaN.
// NVFP4 uses one amax per row, E4M3 scales for adjacent 16 values, and E2M1 low/even nibble first.
// It selects the 6 or 4 candidate by the smaller pairwise sum of absolute errors; ties select 6.
// MXFP8 uses E4M3 codes and a UE8M0 exponent for each 32 values, ceil(log2(amax/448)) clamped to [-127,127].
#define I2X_ROW_THREADS 256u

// Row maximum and whether any element is not finite, over 256 threads. Every thread returns the same pair.
inline float2 i2x_row_amax(const device bfloat* x, uint k, uint tid, uint lane, uint sg, threadgroup float* shared) {
  float m = 0.0f, bad = 0.0f;
  for (uint i = tid; i < k; i += I2X_ROW_THREADS) {
    const float v = float(x[i]);
    if (!isfinite(v)) bad = 1.0f;
    m = max(m, abs(v));
  }
  m = simd_max(m);
  bad = simd_max(bad);
  if (lane == 0u) {
    shared[sg] = m;
    shared[8u + sg] = bad;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float amax = 0.0f, any_bad = 0.0f;
  for (uint s = 0; s < 8u; ++s) {
    amax = max(amax, shared[s]);
    any_bad = max(any_bad, shared[8u + s]);
  }
  return float2(amax, any_bad);
}

[[kernel, max_total_threads_per_threadgroup(256)]] void inkling2_exp_quant_nvfp4(const device bfloat* src [[buffer(0)]],
                                device uchar* codes [[buffer(1)]],
                                device uchar* scales [[buffer(2)]],
                                device float* row_amax [[buffer(3)]],
                                constant Inkling2ExpQuantArgs& a [[buffer(4)]],
                                uint row [[threadgroup_position_in_grid]],
                                uint tid [[thread_position_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]]) {
  if (row >= a.rows) return;
  threadgroup float shared[16];
  const device bfloat* x = src + ulong(row) * a.k;
  const float2 stat = i2x_row_amax(x, a.k, tid, lane, sg, shared);
  const float amax = stat.x;
  const bool bad = stat.y != 0.0f;
  const float t = min(precise::divide(1536.0f, amax), kI2xF32Max);
  if (tid == 0u) row_amax[row] = bad ? NAN : amax;
  const uint blocks = a.k / 16u;
  device uchar* out = codes + ulong(row) * (a.k / 2u);
  device uchar* sc = scales + ulong(row) * blocks;
  for (uint b = tid; b < blocks; b += I2X_ROW_THREADS) {
    if (bad) {
      for (uint j = 0; j < 8u; ++j) out[b * 8u + j] = 0;
      sc[b] = 0;
      continue;
    }
    float xv[16];
    float bm = 0.0f;
    for (uint i = 0; i < 16u; ++i) {
      xv[i] = float(x[b * 16u + i]);
      bm = max(bm, abs(xv[i]));
    }
    const float dec = precise::divide(1.0f, t);
    const float outer = amax * (1.0f / 1536.0f);
    const float hp6 = bm * (t * (1.0f / 6.0f));
    const float hp4 = bm * (t * 0.25f);
    const uint s6 = i2x_e4m3_encode(hp6), s4 = i2x_e4m3_encode(hp4);
    const float f6 = i2x_e4m3(s6), f4 = i2x_e4m3(s4);
    const float inv6 = min(precise::divide(1.0f, dec * f6), kI2xF32Max);
    const float inv4 = min(precise::divide(1.0f, dec * f4), kI2xF32Max);
    uint c6[16], c4[16];
    float e6[16], e4[16];
    for (uint i = 0; i < 16u; ++i) {
      const float r6 = i2x_e2m1_round(xv[i] * inv6), r4 = i2x_e2m1_round(xv[i] * inv4);
      c6[i] = i2x_e2m1_code(r6);
      c4[i] = i2x_e2m1_code(r4);
      e6[i] = abs((r6 * f6) * outer - xv[i]);
      e4[i] = abs((r4 * f4) * outer - xv[i]);
    }
    for (uint width = 8u; width != 0u; width >>= 1u) {
      for (uint i = 0; i < width; ++i) {
        e6[i] = e6[2u * i] + e6[2u * i + 1u];
        e4[i] = e4[2u * i] + e4[2u * i + 1u];
      }
    }
    const bool use4 = e4[0] < e6[0];
    for (uint j = 0; j < 8u; ++j) {
      const uint lo = use4 ? c4[2u * j] : c6[2u * j];
      const uint hi = use4 ? c4[2u * j + 1u] : c6[2u * j + 1u];
      out[b * 8u + j] = uchar(lo | (hi << 4));
    }
    sc[b] = uchar(use4 ? s4 : s6);
  }
}

[[kernel, max_total_threads_per_threadgroup(256)]] void inkling2_exp_quant_mxfp8(const device bfloat* src [[buffer(0)]],
                                device uchar* codes [[buffer(1)]],
                                device uchar* scales [[buffer(2)]],
                                device float* row_amax [[buffer(3)]],
                                constant Inkling2ExpQuantArgs& a [[buffer(4)]],
                                uint row [[threadgroup_position_in_grid]],
                                uint tid [[thread_position_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]]) {
  if (row >= a.rows) return;
  threadgroup float shared[16];
  const device bfloat* x = src + ulong(row) * a.k;
  const float2 stat = i2x_row_amax(x, a.k, tid, lane, sg, shared);
  const bool bad = stat.y != 0.0f;
  if (tid == 0u) row_amax[row] = bad ? NAN : 1.0f;
  const uint blocks = a.k / 32u;
  device uchar* out = codes + ulong(row) * a.k;
  device uchar* sc = scales + ulong(row) * blocks;
  for (uint b = tid; b < blocks; b += I2X_ROW_THREADS) {
    float bm = 0.0f;
    if (!bad) {
      for (uint i = 0; i < 32u; ++i) bm = max(bm, abs(float(x[b * 32u + i])));
    }
    if (bad) {
      for (uint i = 0; i < 32u; ++i) out[b * 32u + i] = 0;
      sc[b] = 255;
      continue;
    }
    const float raw = bm * (1.0f / 448.0f);
    const uint bits = as_type<uint>(raw);
    const uint exponent = bits <= 0x00400000u ? 0u : min(((bits + 0x007fffffu) >> 23) & 255u, 254u);
    sc[b] = uchar(exponent);
    for (uint i = 0; i < 32u; ++i) {
      const bfloat input = x[b * 32u + i];
      const uint word = as_type<ushort>(input);
      // Reconstruct a BF16 subnormal from its significand before exponent scaling. This does not rely on preserving
      // a subnormal FP32 intermediate when the code-0 inverse scale lifts the magnitude into the normal range.
      const float magnitude = (word & 0x7f80u) == 0u
          ? ldexp(float(word & 0x7fu), -6 - int(exponent))
          : ldexp(abs(float(input)), 127 - int(exponent));
      out[b * 32u + i] = uchar(i2x_e4m3_encode(magnitude) | ((word >> 8) & 0x80u));
    }
  }
}
