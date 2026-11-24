// Main's router bits in two launches, after glue.metal: a thread a score with main's serial dot, then picks.

kernel void tf_inkling2_route_scores(
    const device bfloat* x [[buffer(0)]], const device bfloat* weight [[buffer(1)]], device float* scores [[buffer(2)]],
    constant InkRouteArgs& a [[buffer(3)]], uint t [[thread_position_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
  const uint row = group.y, e = group.x * 32 + t;
  if (row >= a.rows || e >= 128) return;
  const device bfloat* w = weight + size_t(e) * 2304;
  const device bfloat* v = x + size_t(row) * 2304;
  float dot = 0.0f;
  for (uint d = 0; d < 2304; ++d) dot += float(w[d]) * float(v[d]);
  scores[size_t(row) * 128 + e] = ink_sigmoid(dot);
}

// One thread a row: main's top-8 and ascending-index combine over the stored scores.
kernel void tf_inkling2_route_select(
    const device float* scores [[buffer(0)]], const device float* bias [[buffer(1)]], device uint* ids [[buffer(2)]],
    device float* weights [[buffer(3)]], constant InkRouteArgs& a [[buffer(4)]], uint row [[thread_position_in_grid]]) {
  if (row >= a.rows) return;
  const device float* s = scores + size_t(row) * 128;
  float chosen[8];
  uint picked[8];
  bool used[128] = {};
  for (uint j = 0; j < 8; ++j) {
    uint best = 128;
    for (uint e = 0; e < 128; ++e) if (!used[e]) {
      if (best == 128 || s[e] + bias[e] > s[best] + bias[best]) best = e;
    }
    used[best] = true; picked[j] = best; chosen[j] = s[best];
  }
  // Combining, including the denominator, is in ascending expert index.
  for (uint j = 1; j < 8; ++j) {
    uint id = picked[j]; float score = chosen[j]; uint k = j;
    while (k > 0 && picked[k - 1] > id) { picked[k] = picked[k - 1]; chosen[k] = chosen[k - 1]; --k; }
    picked[k] = id; chosen[k] = score;
  }
  float den = 1e-20f;
  for (uint j = 0; j < 8; ++j) den += chosen[j];
  for (uint j = 0; j < 8; ++j) {
    ids[row * 8 + j] = picked[j];
    weights[row * 8 + j] = precise::divide(chosen[j], den) * 8.0f;
  }
}
