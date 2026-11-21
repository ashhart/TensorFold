// Chain kernels: the epilogue of one EXL3 projection fused with the block-local work that follows it.

// A bf16 value as a float: the load tf_exl3_rotate_in does of a stored activation.
inline float tf_exl3_chain_bf(ushort bits) { return as_type<float>(uint(bits) << 16); }

// The next projection's rotate_in applied to a block whose bf16 values the lanes hold (lane l: l, l + 32, ...).
inline void tf_exl3_chain_rotate_in(thread float (&v)[4], uint lane, const device ushort* su, uint su_signs,
                                    uint first, device half* rotated, ulong dest) {
  for (uint j = 0; j < 4; ++j) v[j] = v[j] * tf_exl3_scale(su, first + 32 * j, su_signs);
  tf_exl3_hadamard(v, lane);
  for (uint j = 0; j < 4; ++j) rotated[dest + 32 * j] = half(v[j] * 0.08838834764831845f);
}

struct TfExl3Next {
  uint k, su_signs, reserved0, reserved1;
};

// rotate_out of a projection, then the next projection's rotate_in on the bf16 values it stores (y keeps them).
kernel void tf_exl3_rotate_out_in(const device float* sums [[buffer(0)]],
                                  const device ushort* sv [[buffer(1)]],
                                  const device half* bias [[buffer(2)]],
                                  device uchar* y [[buffer(3)]],
                                  constant TfExl3Args& a [[buffer(4)]],
                                  const device uint* output_ids [[buffer(5)]],
                                  const device uint* routes [[buffer(6)]],
                                  const device TfExl3Expert* descriptors [[buffer(7)]],
                                  const device ushort* su_next [[buffer(8)]],
                                  constant TfExl3Next& next [[buffer(9)]],
                                  device half* rotated [[buffer(10)]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint sg [[simdgroup_index_in_threadgroup]],
                                  uint3 tptg [[threads_per_threadgroup]],
                                  uint3 tg [[threadgroup_position_in_grid]]) {
  const uint row = tg.y, base = (tg.x * (tptg.x / 32u) + sg) * 128;
  const bool active = tf_exl3_active(routes, a, row);
  const uint id = a.indexed ? routes[row] : 0;
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
  const ulong first = ulong(dest) * a.y_stride + base;
  for (uint j = 0; j < 4; ++j) {
    const uint n = base + lane + 32 * j;
    float out = (v[j] * 0.08838834764831845f) * tf_exl3_scale(scales, n, signs);
    if (has_bias) out += float(biases[n]);
    const bfloat stored = bfloat(out);
    reinterpret_cast<device bfloat*>(y)[ulong(dest) * a.y_stride + n] = stored;
    v[j] = tf_exl3_chain_bf(as_type<ushort>(stored));
  }
  tf_exl3_chain_rotate_in(v, lane, su_next, next.su_signs, uint(first % next.k) + lane, rotated, first + lane);
}

struct TfExl3RmsRot {
  uint rows, width, x_stride, y_stride;
  float eps;
  uint weight_f32, su_signs, reserved;
};

inline float tf_exl3_chain_weight(const device uchar* w, ulong i, uint f32) {
  if (f32) return reinterpret_cast<const device float*>(w)[i];
  return tf_exl3_chain_bf(reinterpret_cast<const device ushort*>(w)[i]);
}

// dsv41_rms over a row, then the next projection's rotate_in on the rounded row; contraction off as in dsv41_rms.
#pragma clang fp contract(off)
kernel void tf_exl3_rms_rotate_in(const device ushort* x [[buffer(0)]],
                                  const device uchar* w [[buffer(1)]],
                                  device ushort* y [[buffer(2)]],
                                  constant TfExl3RmsRot& d [[buffer(3)]],
                                  const device ushort* su [[buffer(4)]],
                                  device half* rotated [[buffer(5)]],
                                  uint row [[threadgroup_position_in_grid]],
                                  uint t [[thread_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint sg [[simdgroup_index_in_threadgroup]],
                                  uint threads [[threads_per_threadgroup]]) {
  threadgroup float sums[8];
  if (row >= d.rows) return;
  if (t < 256) {
    float s = 0;
    for (uint i = t; i < d.width; i += 256) { float a = tf_exl3_chain_bf(x[ulong(row) * d.x_stride + i]); s += a * a; }
    s = simd_sum(s);
    if (lane == 0) sums[sg] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float total = simd_sum(lane < 8 ? sums[lane] : 0.0f);
    const float factor = 1.0f / float(d.width);
    if (lane == 0) sums[0] = rsqrt(total * factor + d.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float scale = sums[0];
  for (uint block = sg; block < d.width / 128u; block += threads / 32u) {
    const uint first = block * 128u + lane;
    float v[4];
    for (uint j = 0; j < 4; ++j) {
      const uint i = first + 32 * j;
      const ushort bits = tf_exl3_round_bf16((tf_exl3_chain_bf(x[ulong(row) * d.x_stride + i]) * scale) * tf_exl3_chain_weight(w, i, d.weight_f32));
      y[ulong(row) * d.y_stride + i] = bits;
      v[j] = tf_exl3_chain_bf(bits);
    }
    tf_exl3_chain_rotate_in(v, lane, su, d.su_signs, first, rotated, ulong(row) * d.width + first);
  }
}
#pragma clang fp contract(on)

struct TfExl3Rope {
  uint width, rotary, position, step, inverse, reserved0, reserved1, reserved2;
};

// rotate_out, then dsv41_rope over each head's last rotary columns (a multiple of 32) on the stored bf16 values.
kernel void tf_exl3_rotate_out_rope(const device float* sums [[buffer(0)]],
                                    const device ushort* sv [[buffer(1)]],
                                    const device half* bias [[buffer(2)]],
                                    device uchar* y [[buffer(3)]],
                                    constant TfExl3Args& a [[buffer(4)]],
                                    const device uint* output_ids [[buffer(5)]],
                                    const device uint* routes [[buffer(6)]],
                                    const device TfExl3Expert* descriptors [[buffer(7)]],
                                    const device float2* factors [[buffer(8)]],
                                    constant TfExl3Rope& rp [[buffer(9)]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint sg [[simdgroup_index_in_threadgroup]],
                                    uint3 tptg [[threads_per_threadgroup]],
                                    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint row = tg.y, base = (tg.x * (tptg.x / 32u) + sg) * 128;
  const bool active = tf_exl3_active(routes, a, row);
  const uint id = a.indexed ? routes[row] : 0;
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
    ushort bits = as_type<ushort>(bfloat(out));
    // The pair's other element, fetched by every lane (the rotary columns are SIMDgroup uniform).
    const float other = tf_exl3_chain_bf(ushort(simd_shuffle_xor(uint(bits), 1)));
    const uint d = n % rp.width;
    if (d >= rp.width - rp.rotary) {
      const float2 rotation = factors[(ulong(rp.position) + ulong(dest) * rp.step) * (rp.rotary / 2) + (d - (rp.width - rp.rotary)) / 2];
      const float c = rotation.x, s = rp.inverse ? -rotation.y : rotation.y;
      const float own = tf_exl3_chain_bf(bits);
      uint u;
      if ((n & 1u) == 0u) u = as_type<uint>(own * c - other * s);
      else u = as_type<uint>(other * s + own * c);
      bits = ushort((u + 0x7fff + ((u >> 16) & 1)) >> 16);
    }
    reinterpret_cast<device ushort*>(y)[ulong(dest) * a.y_stride + n] = bits;
  }
}

struct TfExl3RopeIn {
  uint width, rotary, stride, position, step, inverse, reserved0, reserved1;
};

// dsv41_rope over the heads of x in place, then rotate_in of the projection that reads x.
kernel void tf_exl3_rope_rotate_in(device ushort* x [[buffer(0)]],
                                   const device ushort* su [[buffer(1)]],
                                   device half* rotated [[buffer(2)]],
                                   constant TfExl3Args& a [[buffer(3)]],
                                   const device uint* input_ids [[buffer(4)]],
                                   const device uint* routes [[buffer(5)]],
                                   const device TfExl3Expert* descriptors [[buffer(6)]],
                                   const device float2* factors [[buffer(7)]],
                                   constant TfExl3RopeIn& rp [[buffer(8)]],
                                   uint lane [[thread_index_in_simdgroup]],
                                   uint sg [[simdgroup_index_in_threadgroup]],
                                   uint3 tptg [[threads_per_threadgroup]],
                                   uint3 tg [[threadgroup_position_in_grid]]) {
  const uint row = tg.y;
  if (!tf_exl3_active(routes, a, row)) return;
  const device ushort* scales = a.indexed ? descriptors[routes[row]].su : su;
  const uint signs = a.indexed ? descriptors[routes[row]].su_signs : a.su_signs;
  const uint source = a.input_ids ? input_ids[row] : row / a.repeat;
  const uint base = (tg.x * (tptg.x / 32u) + sg) * 128;
  if (base >= a.k) return;
  float v[4];
  ushort bits[4];
  for (uint j = 0; j < 4; ++j) {
    const uint k = base + lane + 32 * j;
    const ulong at = ulong(source) * a.x_stride + k;
    bits[j] = x[at];
    // The pair's other element, fetched by every lane (the rotary columns are SIMDgroup uniform).
    const float other = tf_exl3_chain_bf(ushort(simd_shuffle_xor(uint(bits[j]), 1)));
    const uint d = uint(at % rp.stride) % rp.width;
    if (d >= rp.width - rp.rotary) {
      const ulong z = at / rp.stride;
      const float2 rotation = factors[(ulong(rp.position) + z * rp.step) * (rp.rotary / 2) + (d - (rp.width - rp.rotary)) / 2];
      const float c = rotation.x, s = rp.inverse ? -rotation.y : rotation.y;
      const float own = tf_exl3_chain_bf(bits[j]);
      uint u;
      if ((k & 1u) == 0u) u = as_type<uint>(own * c - other * s);
      else u = as_type<uint>(other * s + own * c);
      bits[j] = ushort((u + 0x7fff + ((u >> 16) & 1)) >> 16);
      x[at] = bits[j];
    }
    v[j] = tf_exl3_chain_bf(bits[j]) * tf_exl3_scale(scales, k, signs);
  }
  tf_exl3_hadamard(v, lane);
  for (uint j = 0; j < 4; ++j)
    rotated[ulong(row) * a.k + base + lane + 32 * j] = half(v[j] * 0.08838834764831845f);
}

struct TfExl3Qkv {
  uint q_width, q_su_signs, q_weight_f32, kv_width;
  uint position, rotary, packed_stride, block;
  float eps;
  uint reserved0, reserved1, reserved2;
};

// dsv41_rms's scalars (contraction off): a thread's sum of squares over i = t, t + 256, ..., and the factor.
#pragma clang fp contract(off)
inline float tf_exl3_chain_square_sum(threadgroup const ushort* row, uint width, uint t) {
  float s = 0;
  for (uint i = t; i < width; i += 256) { float a = tf_exl3_chain_bf(row[i]); s += a * a; }
  return s;
}
inline float tf_exl3_chain_rms_factor(float total, uint width, float eps) {
  const float factor = 1.0f / float(width);
  return rsqrt(total * factor + eps);
}
#pragma clang fp contract(on)

inline ushort tf_exl3_chain_round_plain(float x) {
  const uint u = as_type<uint>(x);
  return ushort((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

inline uchar tf_exl3_chain_e4code(float x) {
  float a = min(abs(x), 448.0f);
  if (a == 0) return signbit(x) ? 128 : 0;
  int e = max(-6, int((as_type<uint>(a) >> 23) & 255) - 127);
  int q = int(rint(ldexp(a, 3 - e)));
  uint c = e == -6 && q < 8 ? uint(q) : uint((e + 7) * 8 + q - 8);
  return uchar(min(c, 126u) | (signbit(x) ? 128u : 0u));
}

// The merged wq_a + wkv epilogue: row 0 rms + wq_b rotate_in, row 1 raw key + kv_glue pack; a threadgroup a row.
kernel void tf_exl3_qkv_epilogue(const device float* sums [[buffer(0)]],
                                 device uchar* y [[buffer(1)]],
                                 constant TfExl3Args& a [[buffer(2)]],
                                 const device uint* output_ids [[buffer(3)]],
                                 const device uint* routes [[buffer(4)]],
                                 const device TfExl3Expert* descriptors [[buffer(5)]],
                                 const device uchar* q_weight [[buffer(6)]],
                                 const device ushort* q_su [[buffer(7)]],
                                 device half* rotated [[buffer(8)]],
                                 const device ushort* kv_weight [[buffer(9)]],
                                 const device float2* factors [[buffer(10)]],
                                 device uchar* packed [[buffer(11)]],
                                 constant TfExl3Qkv& d [[buffer(12)]],
                                 uint row [[threadgroup_position_in_grid]],
                                 uint t [[thread_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]],
                                 uint threads [[threads_per_threadgroup]]) {
  threadgroup ushort bits[1280];
  threadgroup float part[8];
  if (!tf_exl3_active(routes, a, row)) return;
  const uint id = routes[row];
  const uint width = descriptors[id].reserved != 0 ? descriptors[id].reserved : a.n;
  const device ushort* scales = descriptors[id].sv;
  const device half* biases = descriptors[id].bias;
  const uint signs = descriptors[id].sv_signs;
  const uint has_bias = descriptors[id].has_bias;
  const uint dest = a.output_ids ? output_ids[row] : row;
  const uint groups = threads / 32u;
  for (uint base = sg * 128; base < width; base += groups * 128) {
    float v[4];
    for (uint j = 0; j < 4; ++j) v[j] = sums[ulong(row) * a.n + base + lane + 32 * j];
    tf_exl3_hadamard(v, lane);
    for (uint j = 0; j < 4; ++j) {
      const uint n = base + lane + 32 * j;
      float out = (v[j] * 0.08838834764831845f) * tf_exl3_scale(scales, n, signs);
      if (has_bias) out += float(biases[n]);
      const bfloat stored = bfloat(out);
      bits[n] = as_type<ushort>(stored);
      if (row != 0) reinterpret_cast<device bfloat*>(y)[ulong(dest) * a.y_stride + n] = stored;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (row == 0) {
    if (t < 256) {
      float s = simd_sum(tf_exl3_chain_square_sum(bits, width, t));
      if (lane == 0) part[sg] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      float total = simd_sum(lane < 8 ? part[lane] : 0.0f);
      if (lane == 0) part[0] = tf_exl3_chain_rms_factor(total, width, d.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = part[0];
    for (uint base = sg * 128; base < width; base += groups * 128) {
      float v[4];
      for (uint j = 0; j < 4; ++j) {
        const uint i = base + lane + 32 * j;
        const ushort out = tf_exl3_round_bf16((tf_exl3_chain_bf(bits[i]) * scale) * tf_exl3_chain_weight(q_weight, i, d.q_weight_f32));
        reinterpret_cast<device ushort*>(y)[ulong(dest) * a.y_stride + i] = out;
        v[j] = tf_exl3_chain_bf(out);
      }
      tf_exl3_chain_rotate_in(v, lane, q_su, d.q_su_signs, base + lane, rotated, base + lane);
    }
    return;
  }
  // The key: dsv41_kv_glue_body over the stored bits (its norm sums columns t, t + 256 with fused multiply-adds).
  if (t < 256) {
    const float v0 = tf_exl3_chain_bf(bits[t]), v1 = tf_exl3_chain_bf(bits[t + 256u]);
    float s = fma(v0, v0, 0.0f);
    s = fma(v1, v1, s);
    for (uint m = 16; m > 0; m >>= 1) s += simd_shuffle_xor(s, m);
    if (lane == 0) part[sg] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = part[0];
  for (uint j = 1; j < 8; ++j) total += part[j];
  const float r = precise::rsqrt(precise::divide(total, float(d.kv_width)) + d.eps);
  for (uint blk = sg; blk < d.kv_width / d.block; blk += groups) {
    const uint col = blk * d.block + lane;
    ushort yb = tf_exl3_chain_round_plain(tf_exl3_chain_bf(kv_weight[col]) * (tf_exl3_chain_bf(bits[col]) * r));
    if (col >= d.kv_width - d.rotary) {
      const float a0 = tf_exl3_chain_bf(yb);
      const float other = tf_exl3_chain_bf(ushort(simd_shuffle_xor(uint(yb), 1)));
      const uint pr = (col - (d.kv_width - d.rotary)) / 2u;
      const float2 cs = factors[ulong(d.position) * (d.rotary / 2u) + pr];
      uint u;
      if ((col & 1u) == 0u) u = as_type<uint>(a0 * cs.x - other * cs.y);
      else u = as_type<uint>(other * cs.y + a0 * cs.x);
      yb = ushort((u + 0x7fff + ((u >> 16) & 1)) >> 16);
    }
    const float val = tf_exl3_chain_bf(yb);
    const float amax = simd_max(abs(val));
    const float z = max(amax, 1e-4f) / 448.0f;
    const uint zbits = as_type<uint>(z);
    const int exponent = int((zbits >> 23) & 255) - 127 + ((zbits & 0x7fffff) != 0);
    const float scale = ldexp(1.0f, exponent);
    packed[col] = tf_exl3_chain_e4code(val / scale);
    if (lane == 0) packed[d.kv_width + blk] = uchar(exponent + 127);
  }
}
