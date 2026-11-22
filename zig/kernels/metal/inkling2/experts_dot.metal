// Each SIMDgroup decodes four adjacent packed weight rows to BF16 and reduces K in a fixed FP32 tree.
#define I2X_DOT_THREADS 256u
#define I2X_COMBINE_THREADS 256u

// Exact FP32 bit patterns for E4M3FN magnitudes 0x00..0x7f (0x7f = qNaN); sign bit is OR'd at bit 31.
constant uint kI2xE4m3AbsBits[128] = {
  0x00000000u, 0x3b000000u, 0x3b800000u, 0x3bc00000u, 0x3c000000u, 0x3c200000u, 0x3c400000u, 0x3c600000u,
  0x3c800000u, 0x3c900000u, 0x3ca00000u, 0x3cb00000u, 0x3cc00000u, 0x3cd00000u, 0x3ce00000u, 0x3cf00000u,
  0x3d000000u, 0x3d100000u, 0x3d200000u, 0x3d300000u, 0x3d400000u, 0x3d500000u, 0x3d600000u, 0x3d700000u,
  0x3d800000u, 0x3d900000u, 0x3da00000u, 0x3db00000u, 0x3dc00000u, 0x3dd00000u, 0x3de00000u, 0x3df00000u,
  0x3e000000u, 0x3e100000u, 0x3e200000u, 0x3e300000u, 0x3e400000u, 0x3e500000u, 0x3e600000u, 0x3e700000u,
  0x3e800000u, 0x3e900000u, 0x3ea00000u, 0x3eb00000u, 0x3ec00000u, 0x3ed00000u, 0x3ee00000u, 0x3ef00000u,
  0x3f000000u, 0x3f100000u, 0x3f200000u, 0x3f300000u, 0x3f400000u, 0x3f500000u, 0x3f600000u, 0x3f700000u,
  0x3f800000u, 0x3f900000u, 0x3fa00000u, 0x3fb00000u, 0x3fc00000u, 0x3fd00000u, 0x3fe00000u, 0x3ff00000u,
  0x40000000u, 0x40100000u, 0x40200000u, 0x40300000u, 0x40400000u, 0x40500000u, 0x40600000u, 0x40700000u,
  0x40800000u, 0x40900000u, 0x40a00000u, 0x40b00000u, 0x40c00000u, 0x40d00000u, 0x40e00000u, 0x40f00000u,
  0x41000000u, 0x41100000u, 0x41200000u, 0x41300000u, 0x41400000u, 0x41500000u, 0x41600000u, 0x41700000u,
  0x41800000u, 0x41900000u, 0x41a00000u, 0x41b00000u, 0x41c00000u, 0x41d00000u, 0x41e00000u, 0x41f00000u,
  0x42000000u, 0x42100000u, 0x42200000u, 0x42300000u, 0x42400000u, 0x42500000u, 0x42600000u, 0x42700000u,
  0x42800000u, 0x42900000u, 0x42a00000u, 0x42b00000u, 0x42c00000u, 0x42d00000u, 0x42e00000u, 0x42f00000u,
  0x43000000u, 0x43100000u, 0x43200000u, 0x43300000u, 0x43400000u, 0x43500000u, 0x43600000u, 0x43700000u,
  0x43800000u, 0x43900000u, 0x43a00000u, 0x43b00000u, 0x43c00000u, 0x43d00000u, 0x43e00000u, 0x7fc00000u
};

inline float i2x_e4m3_lut(uchar b) {
  return as_type<float>(kI2xE4m3AbsBits[b & 0x7fu] | ((uint(b) & 0x80u) << 24u));
}

inline float4 i2x_bf4(float4 x) {
  return float4(float(bfloat(x.x)), float(bfloat(x.y)), float(bfloat(x.z)), float(bfloat(x.w)));
}

inline float i2x_swiglu(float gate, float up, float gate_cap, float up_cap) {
  const float sigmoid = precise::divide(1.0f, 1.0f + precise::exp(-gate));
  gate = precise::tanh(precise::divide(gate, gate_cap)) * gate_cap;
  up = precise::tanh(precise::divide(up, up_cap)) * up_cap;
  return (gate * sigmoid) * up;
}

template <bool NV, bool GATE_UP>
inline void i2x_dot(const device uchar* weight, const device uchar* w_scales,
                    const device float* w_factor, const device bfloat* x,
                    const device uint* ids, device bfloat* out,
                    constant Inkling2ExpDotArgs& a, uint2 at, uint lane, uint sg) {
  if (at.y >= a.pairs) return;
  const uint col = (at.x * 8u + sg) * (GATE_UP ? 2u : 4u);
  if (col >= a.out_width) return;
  const uint expert = ids[at.y];
  device bfloat* dst = out + ulong(at.y) * a.out_width + col;
  if (expert >= a.experts) {
    if (lane == 0u) {
      dst[0] = bfloat(0.0f);
      dst[1] = bfloat(0.0f);
      if (!GATE_UP) { dst[2] = bfloat(0.0f); dst[3] = bfloat(0.0f); }
    }
    return;
  }
  const uint input_row = at.y / a.src_div;
  const uint pitch = NV ? (a.k >> 1u) : a.k;
  const uint blocks = a.k / (NV ? 16u : 32u);
  const ulong w_row = ulong(expert) * a.n + col * (GATE_UP ? 2u : 1u);
  x += ulong(input_row) * a.k;
  const device uchar* w_r0 = weight + w_row * pitch;
  const device uchar* w_r1 = w_r0 + pitch;
  const device uchar* w_r2 = w_r1 + pitch;
  const device uchar* w_r3 = w_r2 + pitch;
  const device uchar* s_r0 = w_scales + w_row * blocks;
  const device uchar* s_r1 = s_r0 + blocks;
  const device uchar* s_r2 = s_r1 + blocks;
  const device uchar* s_r3 = s_r2 + blocks;
  const float factor = NV ? w_factor[expert] : 1.0f;

  float4 accum = 0.0f;
  for (uint block = lane; block < blocks; block += 32u) {
    float4 part = 0.0f;
    const uint base = block * (NV ? 8u : 32u);
    const uchar4 sw = uchar4(s_r0[block], s_r1[block], s_r2[block], s_r3[block]);
    const float4 scale = NV ? float4(i2x_e4m3_lut(sw.x), i2x_e4m3_lut(sw.y), i2x_e4m3_lut(sw.z), i2x_e4m3_lut(sw.w))
                            : float4(i2x_mx_scale(sw.x), i2x_mx_scale(sw.y), i2x_mx_scale(sw.z), i2x_mx_scale(sw.w));
    if (NV) {
      _Pragma("clang loop unroll(full)")
      for (uint g = 0u; g < 2u; ++g) {
        const uint p4 = base + g * 4u;
        const uchar4 w0 = *(const device uchar4*)(w_r0 + p4);
        const uchar4 w1 = *(const device uchar4*)(w_r1 + p4);
        const uchar4 w2 = *(const device uchar4*)(w_r2 + p4);
        const uchar4 w3 = *(const device uchar4*)(w_r3 + p4);
        const device bfloat* xp = x + 2u * p4;
        const float4 xa = float4(float(xp[0]), float(xp[1]), float(xp[2]), float(xp[3]));
        const float4 xb = float4(float(xp[4]), float(xp[5]), float(xp[6]), float(xp[7]));
        _Pragma("clang loop unroll(full)")
        for (uint i = 0u; i < 4u; ++i) {
          const uint4 w = uint4(w0[i], w1[i], w2[i], w3[i]);
          const float4 low = float4(kI2xE2m1[w.x & 15u], kI2xE2m1[w.y & 15u],
                                     kI2xE2m1[w.z & 15u], kI2xE2m1[w.w & 15u]);
          const float4 high = float4(kI2xE2m1[w.x >> 4u], kI2xE2m1[w.y >> 4u],
                                      kI2xE2m1[w.z >> 4u], kI2xE2m1[w.w >> 4u]);
          const float4 w_low = i2x_bf4((low * scale) * factor);
          const float4 w_high = i2x_bf4((high * scale) * factor);
          const float x_lo = (i < 2u) ? xa[2u * i] : xb[2u * (i - 2u)];
          const float x_hi = (i < 2u) ? xa[2u * i + 1u] : xb[2u * (i - 2u) + 1u];
          part += w_low * x_lo;
          part += w_high * x_hi;
        }
      }
    } else {
      const float4 mantissa = select(float4(1.0f), float4(0.5f), sw == uchar4(0));
      for (uint g = 0u; g < 8u; ++g) {
        const uint p4 = base + g * 4u;
        const uchar4 w0 = *(const device uchar4*)(w_r0 + p4);
        const uchar4 w1 = *(const device uchar4*)(w_r1 + p4);
        const uchar4 w2 = *(const device uchar4*)(w_r2 + p4);
        const uchar4 w3 = *(const device uchar4*)(w_r3 + p4);
        const device bfloat* xp = x + p4;
        const float4 xv = float4(float(xp[0]), float(xp[1]), float(xp[2]), float(xp[3]));
        _Pragma("clang loop unroll(full)")
        for (uint i = 0u; i < 4u; ++i) {
          const float4 w = float4(i2x_e4m3_lut(w0[i]), i2x_e4m3_lut(w1[i]), i2x_e4m3_lut(w2[i]), i2x_e4m3_lut(w3[i]));
          part += i2x_bf4((w * mantissa) * scale) * xv[i];
        }
      }
    }
    accum += part;
  }
  for (uint shift = 16u; shift != 0u; shift >>= 1u) accum += simd_shuffle_xor(accum, shift);
  if (lane != 0u) return;
  if (GATE_UP) {
    // The gate_up rows interleave gate and up: (x, y) and (z, w) are each one output row's (gate, up) for SwiGLU.
    const float4 v = i2x_bf4(accum);
    dst[0] = bfloat(i2x_swiglu(v.x, v.y, a.gate_cap, a.up_cap));
    dst[1] = bfloat(i2x_swiglu(v.z, v.w, a.gate_cap, a.up_cap));
  } else {
    dst[0] = bfloat(accum.x);
    dst[1] = bfloat(accum.y);
    dst[2] = bfloat(accum.z);
    dst[3] = bfloat(accum.w);
  }
}

#define I2X_DOT(symbol, nv, gate_up) \
  [[kernel, max_total_threads_per_threadgroup(I2X_DOT_THREADS)]] \
  void symbol(const device uchar* weight [[buffer(0)]], const device uchar* w_scales [[buffer(1)]], \
              const device float* w_factor [[buffer(2)]], const device bfloat* x [[buffer(3)]], \
              const device uint* ids [[buffer(4)]], device bfloat* out [[buffer(5)]], \
              constant Inkling2ExpDotArgs& a [[buffer(6)]], uint2 at [[threadgroup_position_in_grid]], \
              uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) { \
    i2x_dot<nv, gate_up>(weight, w_scales, w_factor, x, ids, out, a, at, lane, sg); \
  }

I2X_DOT(inkling2_exp_gate_up_nvfp4, true, true)
I2X_DOT(inkling2_exp_gate_up_mxfp8, false, true)
I2X_DOT(inkling2_exp_down_nvfp4, true, false)
I2X_DOT(inkling2_exp_down_mxfp8, false, false)
#undef I2X_DOT

inline uint i2x_route_order(const device uint* ids, uint topk, uint experts) {
  uint order = 0xffffffffu;
  uint remaining = (1u << topk) - 1u;
  uint shift = 0u;
  // Pack active slots in ascending expert order with stable ties; 0xf marks the end.
  while (remaining != 0u) {
    uint expert = experts;
    uint slot = 0u;
    for (uint slots = remaining; slots != 0u; slots &= slots - 1u) {
      const uint j = ctz(slots);
      const uint id = ids[j];
      if (id < expert) { expert = id; slot = j; }
    }
    if (expert == experts) break;
    order ^= (0xfu ^ slot) << shift;
    shift += 4u;
    remaining &= ~(1u << slot);
  }
  return order;
}

[[kernel, max_total_threads_per_threadgroup(I2X_COMBINE_THREADS)]]
void inkling2_exp_combine(const device bfloat* down [[buffer(0)]], const device uint* ids [[buffer(1)]],
                          const device float* gammas [[buffer(2)]], device bfloat* out [[buffer(3)]],
                          constant Inkling2ExpCombineArgs& a [[buffer(4)]], uint2 at [[thread_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]]) {
  if (at.x >= a.hidden || at.y >= a.rows) return;
  const ulong routes = ulong(at.y) * a.topk;
  const ulong first = routes * a.hidden + at.x;
  // The hidden width is a multiple of 32, so a SIMDgroup shares one token's order.
  const uint order = simd_broadcast(lane == 0u ? i2x_route_order(ids + routes, a.topk, a.experts) : 0u, 0);
  float sum = 0.0f;
  for (uint rank = 0u; rank < a.topk; ++rank) {
    const uint slot = (order >> (4u * rank)) & 0xfu;
    if (slot == 0xfu) break;
    sum += float(down[first + slot * a.hidden]) * gammas[routes + slot];
  }
  out[ulong(at.y) * a.hidden + at.x] = bfloat(sum);
}
