// Each SIMDgroup decodes four adjacent packed weight rows to BF16 and reduces K in a fixed FP32 tree.
#define I2X_DOT_THREADS 256u
#define I2X_COMBINE_THREADS 256u

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
  const uint pitch = NV ? a.k / 2u : a.k;
  const uint blocks = a.k / (NV ? 16u : 32u);
  const ulong w_row = ulong(expert) * a.n + col * (GATE_UP ? 2u : 1u);
  x += ulong(input_row) * a.k;
  weight += w_row * pitch;
  w_scales += w_row * blocks;
  const float factor = NV ? w_factor[expert] : 1.0f;

  float4 accum = 0.0f;
  for (uint block = lane; block < blocks; block += 32u) {
    float4 part = 0.0f;
    const uint base = block * (NV ? 8u : 32u);
    const uchar4 sw = uchar4(w_scales[block], w_scales[blocks + block],
                             w_scales[2u * blocks + block], w_scales[3u * blocks + block]);
    const float4 scale = NV ? float4(i2x_e4m3(sw.x), i2x_e4m3(sw.y), i2x_e4m3(sw.z), i2x_e4m3(sw.w))
                            : float4(i2x_mx_scale(sw.x), i2x_mx_scale(sw.y), i2x_mx_scale(sw.z), i2x_mx_scale(sw.w));
    if (NV) {
      for (uint i = 0u; i < 8u; ++i) {
        const uint p = base + i;
        const uint4 w = uint4(weight[p], weight[pitch + p], weight[2u * pitch + p], weight[3u * pitch + p]);
        const float4 low = float4(kI2xE2m1[w.x & 15u], kI2xE2m1[w.y & 15u],
                                   kI2xE2m1[w.z & 15u], kI2xE2m1[w.w & 15u]);
        const float4 high = float4(kI2xE2m1[w.x >> 4u], kI2xE2m1[w.y >> 4u],
                                    kI2xE2m1[w.z >> 4u], kI2xE2m1[w.w >> 4u]);
        const float4 w_low = i2x_bf4((low * scale) * factor);
        const float4 w_high = i2x_bf4((high * scale) * factor);
        part += w_low * float(x[2u * p]);
        part += w_high * float(x[2u * p + 1u]);
      }
    } else {
      const float4 mantissa = select(float4(1.0f), float4(0.5f), sw == uchar4(0));
      for (uint i = 0u; i < 32u; ++i) {
        const uint p = base + i;
        const float4 w = float4(i2x_e4m3(weight[p]), i2x_e4m3(weight[pitch + p]),
                                  i2x_e4m3(weight[2u * pitch + p]), i2x_e4m3(weight[3u * pitch + p]));
        part += i2x_bf4((w * mantissa) * scale) * float(x[p]);
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
