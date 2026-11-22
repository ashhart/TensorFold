// Router: score = sigmoid(logit), top-k of score+bias (NaN top, ties low index), weights score/(sum+1e-20)*route_scale.

constant constexpr uint I2_ROUTE_PER_LANE = 16; // experts up to 512
constant constexpr uint I2_ROUTE_MAX_TOPK = 16;

struct I2RouteArgs {
  uint rows, experts, topk, pad;
  float scale;
};

inline uint i2_order_key(float v) {
  if (isnan(v)) return 0xFFFFFFFFu;
  if (v == 0.0f) v = 0.0f;
  const uint b = as_type<uint>(v);
  return (b & 0x80000000u) != 0u ? ~b : (b | 0x80000000u);
}

[[kernel, max_total_threads_per_threadgroup(32)]] void inkling2_route_select(
    const device float* logits [[buffer(0)]], const device float* bias [[buffer(1)]], device uint* ids [[buffer(2)]],
    device float* weights [[buffer(3)]], constant I2RouteArgs& a [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  float score[I2_ROUTE_PER_LANE];
  uint key[I2_ROUTE_PER_LANE];
  uint taken = 0;
  for (uint t = 0; t < I2_ROUTE_PER_LANE; ++t) {
    const uint e = lane + 32 * t;
    score[t] = 0.0f;
    key[t] = 0;
    if (e < a.experts) {
      score[t] = i2_sigmoid(logits[ulong(row) * a.experts + e]);
      key[t] = i2_order_key(score[t] + bias[e]);
    } else {
      taken |= 1u << t;
    }
  }
  float picked[I2_ROUTE_MAX_TOPK];
  for (uint r = 0; r < I2_ROUTE_MAX_TOPK; ++r) {
    if (r >= a.topk) break;
    uint best = 0, at = 0xFFFFFFFFu;
    for (uint t = 0; t < I2_ROUTE_PER_LANE; ++t) {
      if ((taken >> t) & 1u) continue;
      if (key[t] > best) {
        best = key[t];
        at = lane + 32 * t;
      }
    }
    const uint top = simd_max(best);
    const uint winner = simd_min(best == top ? at : 0xFFFFFFFFu);
    float s = 0.0f;
    if ((winner & 31u) == lane) {
      const uint t = winner >> 5;
      taken |= 1u << t;
      for (uint u = 0; u < I2_ROUTE_PER_LANE; ++u)
        if (u == t) s = score[u];
    }
    picked[r] = simd_shuffle(s, ushort(winner & 31u));
    if (lane == 0) ids[ulong(row) * a.topk + r] = winner;
  }
  float sum = 0.0f;
  for (uint r = 0; r < I2_ROUTE_MAX_TOPK; ++r) {
    if (r >= a.topk) break;
    sum += picked[r];
  }
  if (lane == 0)
    for (uint r = 0; r < I2_ROUTE_MAX_TOPK; ++r) {
      if (r >= a.topk) break;
      weights[ulong(row) * a.topk + r] = (picked[r] / (sum + 1e-20f)) * a.scale;
    }
}
