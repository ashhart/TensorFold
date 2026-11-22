// Each SIMDgroup decodes two adjacent packed weight rows to BF16 and reduces K in a fixed FP32 tree.
#define I2X_DOT_THREADS 256u
#define I2X_COMBINE_THREADS 256u

inline float2 i2x_bf2(float2 x) {
  return float2(float(bfloat(x.x)), float(bfloat(x.y)));
}

inline float i2x_swiglu(float gate, float up, float gate_cap, float up_cap) {
  const float sigmoid = precise::divide(1.0f, 1.0f + precise::exp(-gate));
  gate = precise::tanh(precise::divide(gate, gate_cap)) * gate_cap;
  up = precise::tanh(precise::divide(up, up_cap)) * up_cap;
  return (gate * sigmoid) * up;
}

// Two E2M1 codes of one byte as FP16 bit pairs; the product's 2^14 is carried by the NVFP4 block scale.
inline float2 i2x_e2m1_pair(uint code) {
  return float2(as_type<half2>(((code * 0x00200200u) & 0x0e000e00u) | ((code * 0x01001000u) & 0x80008000u)));
}

template <bool NV, bool GATE_UP>
inline void i2x_dot(const device uchar* weight, const device uchar* w_scales,
                    const device float* w_factor, const device bfloat* x,
                    const device uint* ids, device bfloat* out,
                    constant Inkling2ExpDotArgs& a, uint2 at, uint lane, uint sg) {
  if (at.y >= a.pairs) return;
  const uint col = (at.x * 8u + sg) * (GATE_UP ? 1u : 2u);
  if (col >= a.out_width) return;
  const uint expert = ids[at.y];
  device bfloat* dst = out + ulong(at.y) * a.out_width + col;
  if (expert >= a.experts) {
    if (lane == 0u) {
      dst[0] = bfloat(0.0f);
      if (!GATE_UP) dst[1] = bfloat(0.0f);
    }
    return;
  }
  const uint input_row = at.y / a.src_div;
  const uint pitch = NV ? a.k / 2u : a.k;
  const uint stride = pitch / 4u;
  const uint blocks = a.k / (NV ? 16u : 32u);
  const ulong w_row = ulong(expert) * a.n + col * (GATE_UP ? 2u : 1u);
  x += ulong(input_row) * a.k;
  weight += w_row * pitch;
  w_scales += w_row * blocks;
  const float factor = NV ? w_factor[expert] : 1.0f;

  float2 accum = 0.0f;
  for (uint block = lane; block < blocks; block += 32u) {
    float2 part = 0.0f;
    const uint sx = w_scales[block], sy = w_scales[blocks + block];
    const device uint* row = (const device uint*)(weight + block * (NV ? 8u : 32u));
    if (NV) {
      const float2 scale = float2(i2x_e4m3(sx), i2x_e4m3(sy)) * 16384.0f;
      const device bfloat* xb = x + block * 16u;
      for (uint i = 0u; i < 8u; ++i) {
        const uint shift = 8u * (i & 3u);
        const uint2 code = uint2((row[i >> 2u] >> shift) & 255u, (row[stride + (i >> 2u)] >> shift) & 255u);
        const float2 lo = i2x_e2m1_pair(code.x), hi = i2x_e2m1_pair(code.y);
        const float2 low = float2(lo.x, hi.x);
        const float2 high = float2(lo.y, hi.y);
        const float2 w_low = i2x_bf2((low * scale) * factor);
        const float2 w_high = i2x_bf2((high * scale) * factor);
        part += w_low * float(xb[2u * i]);
        part += w_high * float(xb[2u * i + 1u]);
      }
    } else {
      const float2 scale = float2(i2x_mx_scale(sx), i2x_mx_scale(sy));
      const float2 mantissa = select(float2(1.0f), float2(0.5f), uint2(sx, sy) == uint2(0));
      const device bfloat* xb = x + block * 32u;
      for (uint i = 0u; i < 32u; ++i) {
        const uint shift = 8u * (i & 3u);
        const uint2 code = uint2((row[i >> 2u] >> shift) & 255u, (row[stride + (i >> 2u)] >> shift) & 255u);
        const float2 w = float2(i2x_e4m3(code.x), i2x_e4m3(code.y));
        part += i2x_bf2((w * mantissa) * scale) * float(xb[i]);
      }
    }
    accum += part;
  }
  for (uint shift = 16u; shift != 0u; shift >>= 1u) accum += simd_shuffle_xor(accum, shift);
  if (lane != 0u) return;
  if (GATE_UP) {
    // The gate_up rows interleave gate and up: (x, y) are one output row's (gate, up) for SwiGLU.
    const float2 v = i2x_bf2(accum);
    dst[0] = bfloat(i2x_swiglu(v.x, v.y, a.gate_cap, a.up_cap));
  } else {
    dst[0] = bfloat(accum.x);
    dst[1] = bfloat(accum.y);
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
