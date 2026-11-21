// Inkling 2 inputs: scaled L2 clip of q and rel states (log scaling); key clip / value norm into the KV cache.

constant constexpr uint I2_HEAD_LANE_ELEMS = 8; // head vectors up to 256 wide

struct I2ClipArgs {
  uint rows, heads, dim, pos0;
  uint log_scaled, floor_positions, pad0, pad1;
  float threshold, alpha;
};

// ScaledL2Clip: bf16(v * min(threshold / (|v| + 1e-6), 1)); full-attention times tau = 1 + alpha * log((pos+1)/floor).
[[kernel, max_total_threads_per_threadgroup(32)]] void inkling2_scaled_l2_clip(
    const device bfloat* x [[buffer(0)]], const device bfloat* scale [[buffer(1)]], device bfloat* out [[buffer(2)]],
    constant I2ClipArgs& a [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint head = tg.x, row = tg.y;
  const ulong base = (ulong(row) * a.heads + head) * a.dim;
  float v[I2_HEAD_LANE_ELEMS];
  float ss = 0.0f;
  for (uint e = 0; e < I2_HEAD_LANE_ELEMS; ++e) {
    const uint i = lane + 32 * e;
    v[e] = 0.0f;
    if (i < a.dim) {
      v[e] = float(x[base + i]) * float(scale[ulong(head) * a.dim + i]);
      ss = fma(v[e], v[e], ss);
    }
  }
  const float factor = i2_clip_factor(simd_sum(ss), a.threshold);
  float tau = 1.0f;
  if (a.log_scaled != 0) {
    const float ratio = max(float(a.pos0 + row + 1) / float(a.floor_positions), 1.0f);
    tau = 1.0f + a.alpha * metal::precise::log(ratio);
  }
  for (uint e = 0; e < I2_HEAD_LANE_ELEMS; ++e) {
    const uint i = lane + 32 * e;
    if (i >= a.dim) continue;
    float r = float(bfloat(v[e] * factor));
    if (a.log_scaled != 0) r = float(bfloat(r * tau));
    out[base + i] = bfloat(r);
  }
}

struct I2KvArgs {
  uint rows, kv_heads, dim, pos0;
  uint ring;
  float threshold, eps;
  uint pad;
};

// Key: L2 clip of conv into kcache; value: unweighted RMS norm into vcache; slot (pos0 + t) % ring else pos0 + t.
[[kernel, max_total_threads_per_threadgroup(32)]] void inkling2_kv_prep(
    const device bfloat* k [[buffer(0)]], const device bfloat* v [[buffer(1)]], device bfloat* kcache [[buffer(2)]],
    device bfloat* vcache [[buffer(3)]], constant I2KvArgs& a [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint head = tg.x, row = tg.y;
  if (a.ring != 0 && row + a.ring < a.rows) return;  // a later row of this call owns the slot: no concurrent writers
  const ulong src = (ulong(row) * a.kv_heads + head) * a.dim;
  const ulong position = ulong(a.pos0) + row;
  const ulong slot = a.ring == 0 ? position : position % a.ring;
  const ulong dst = (slot * a.kv_heads + head) * a.dim;
  float kv[I2_HEAD_LANE_ELEMS], vv[I2_HEAD_LANE_ELEMS];
  float kss = 0.0f, vss = 0.0f;
  for (uint e = 0; e < I2_HEAD_LANE_ELEMS; ++e) {
    const uint i = lane + 32 * e;
    kv[e] = 0.0f;
    vv[e] = 0.0f;
    if (i < a.dim) {
      kv[e] = float(k[src + i]);
      vv[e] = float(v[src + i]);
      kss = fma(kv[e], kv[e], kss);
      vss = fma(vv[e], vv[e], vss);
    }
  }
  const float kfactor = i2_clip_factor(simd_sum(kss), a.threshold);
  const float vinv = i2_rsqrt_mean(simd_sum(vss), a.dim, a.eps);
  for (uint e = 0; e < I2_HEAD_LANE_ELEMS; ++e) {
    const uint i = lane + 32 * e;
    if (i >= a.dim) continue;
    kcache[dst + i] = bfloat(kv[e] * kfactor);
    vcache[dst + i] = bfloat(vv[e] * vinv);
  }
}
