// Inkling 2 Mini native routed expert dots. Each SIMDgroup handles four adjacent weight rows of one (token, route)
// pair. Lanes own successive K blocks, folding their FP32 partials in a fixed tree regardless of batch size.
//
// Rounding follows Mini SGLang PR #51926's SM120 path. NVFP4 raw sums have no outer scales yet; they round to BF16,
// then multiply by (x_amax * w_amax) / (1536 * weight_denominator) and round again. MXFP8 rounds its scaled dot once.
// Gate/up rows are interleaved. Gamma is applied to the capped SwiGLU before BF16 rounding and the second quantizer;
// the final reduction is an unweighted FP32 sum of the BF16 down rows in route-slot order, then BF16.
#define I2X_DOT_THREADS 256u
#define I2X_COMBINE_THREADS 256u

inline float4 i2x_bf4(float4 x) {
  return float4(float(bfloat(x.x)), float(bfloat(x.y)), float(bfloat(x.z)), float(bfloat(x.w)));
}

inline float i2x_swiglu(float gate, float up, float gamma, float gate_cap, float up_cap) {
  const float sigmoid = precise::divide(1.0f, 1.0f + precise::exp(-gate));
  if (gate_cap > 0.0f) gate = gate_cap * precise::tanh(precise::divide(gate, gate_cap));
  if (up_cap > 0.0f) up = up_cap * precise::tanh(precise::divide(up, up_cap));
  return ((gate * sigmoid) * up) * gamma;
}

template <bool NV, bool GATE_UP>
inline void i2x_dot(const device uchar* weight, const device uchar* w_scales,
                    const device float* w_amax, const device float* w_denominator,
                    const device uchar* x, const device uchar* x_scales, const device float* x_amax,
                    const device uint* ids, const device float* gammas, device bfloat* out,
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
  x += ulong(input_row) * pitch;
  x_scales += ulong(input_row) * blocks;
  weight += w_row * pitch;
  w_scales += w_row * blocks;

  float4 accum = 0.0f;
  for (uint block = lane; block < blocks; block += 32u) {
    float4 part = 0.0f;
    const uint base = block * (NV ? 8u : 32u);
    const uchar4 sw = uchar4(w_scales[block], w_scales[blocks + block],
                             w_scales[2u * blocks + block], w_scales[3u * blocks + block]);
    const uint sx = x_scales[block];
    if (NV) {
      for (uint i = 0u; i < 8u; ++i) {
        const uint p = base + i;
        const uint v = x[p];
        const uint4 w = uint4(weight[p], weight[pitch + p], weight[2u * pitch + p], weight[3u * pitch + p]);
        const float4 low = float4(kI2xE2m1[w.x & 15u], kI2xE2m1[w.y & 15u],
                                   kI2xE2m1[w.z & 15u], kI2xE2m1[w.w & 15u]);
        const float4 high = float4(kI2xE2m1[w.x >> 4u], kI2xE2m1[w.y >> 4u],
                                    kI2xE2m1[w.z >> 4u], kI2xE2m1[w.w >> 4u]);
        part += low * kI2xE2m1[v & 15u];
        part += high * kI2xE2m1[v >> 4u];
      }
      const float4 scale = float4(i2x_e4m3(sw.x), i2x_e4m3(sw.y), i2x_e4m3(sw.z), i2x_e4m3(sw.w));
      part *= scale * i2x_e4m3(sx);
    } else {
      for (uint i = 0u; i < 32u; ++i) {
        const uint p = base + i;
        const float v = i2x_e4m3(x[p]);
        const float4 w = float4(i2x_e4m3(weight[p]), i2x_e4m3(weight[pitch + p]),
                                  i2x_e4m3(weight[2u * pitch + p]), i2x_e4m3(weight[3u * pitch + p]));
        part += w * v;
      }
      part = float4(i2x_mx_scale(part.x, sx, sw.x), i2x_mx_scale(part.y, sx, sw.y),
                    i2x_mx_scale(part.z, sx, sw.z), i2x_mx_scale(part.w, sx, sw.w));
    }
    accum += part;
  }
  for (uint shift = 16u; shift != 0u; shift >>= 1u) accum += simd_shuffle_xor(accum, shift);
  if (lane != 0u) return;
  float4 v = i2x_bf4(accum);
  if (NV) {
    const float outer = (x_amax[input_row] * w_amax[expert]) * precise::divide(1.0f, 1536.0f * w_denominator[0]);
    v = i2x_bf4(v * outer);
  }
  if (GATE_UP) {
    const float gamma = gammas[at.y];
    dst[0] = bfloat(i2x_swiglu(v.x, v.y, gamma, a.gate_cap, a.up_cap));
    dst[1] = bfloat(i2x_swiglu(v.z, v.w, gamma, a.gate_cap, a.up_cap));
  } else {
    dst[0] = bfloat(v.x);
    dst[1] = bfloat(v.y);
    dst[2] = bfloat(v.z);
    dst[3] = bfloat(v.w);
  }
}

#define I2X_DOT(symbol, nv, gate_up) \
  [[kernel, max_total_threads_per_threadgroup(I2X_DOT_THREADS)]] \
  void symbol(const device uchar* weight [[buffer(0)]], const device uchar* w_scales [[buffer(1)]], \
              const device float* w_amax [[buffer(2)]], const device float* w_denominator [[buffer(3)]], \
              const device uchar* x [[buffer(4)]], const device uchar* x_scales [[buffer(5)]], \
              const device float* x_amax [[buffer(6)]], const device uint* ids [[buffer(7)]], \
              const device float* gammas [[buffer(8)]], device bfloat* out [[buffer(9)]], \
              constant Inkling2ExpDotArgs& a [[buffer(10)]], uint2 at [[threadgroup_position_in_grid]], \
              uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) { \
    i2x_dot<nv, gate_up>(weight, w_scales, w_amax, w_denominator, x, x_scales, x_amax, ids, gammas, out, a, at, lane, sg); \
  }

I2X_DOT(inkling2_exp_gate_up_nvfp4, true, true)
I2X_DOT(inkling2_exp_gate_up_mxfp8, false, true)
I2X_DOT(inkling2_exp_down_nvfp4, true, false)
I2X_DOT(inkling2_exp_down_mxfp8, false, false)
#undef I2X_DOT

[[kernel, max_total_threads_per_threadgroup(I2X_COMBINE_THREADS)]]
void inkling2_exp_combine(const device bfloat* down [[buffer(0)]], device bfloat* out [[buffer(1)]],
                          constant Inkling2ExpCombineArgs& a [[buffer(2)]], uint2 at [[thread_position_in_grid]]) {
  if (at.x >= a.hidden || at.y >= a.rows) return;
  const ulong first = ulong(at.y) * a.topk * a.hidden + at.x;
  float sum = 0.0f;
  for (uint slot = 0u; slot < a.topk; ++slot) sum += float(down[first + slot * a.hidden]);
  out[ulong(at.y) * a.hidden + at.x] = bfloat(sum);
}
