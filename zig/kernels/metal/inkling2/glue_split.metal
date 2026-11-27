// Main's router bits in two launches, after glue.metal: a thread a score with main's serial dot, then picks.

[[max_total_threads_per_threadgroup(32)]]
kernel void tf_inkling2_route_scores(
    const device bfloat* x [[buffer(0)]], const device bfloat* weight [[buffer(1)]], device float* scores [[buffer(2)]],
    constant InkRouteArgs& a [[buffer(3)]], uint2 tid [[thread_position_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
  const uint row = group.y, e = group.x * 32 + tid.x;
  if (row >= a.rows || e >= 128) return;
  const device bfloat* w = weight + size_t(e) * 2304;
  const device bfloat* v = x + size_t(row) * 2304;
  float dot = 0.0f;
  for (uint d = 0; d < 2304; ++d) dot += float(w[d]) * float(v[d]);
  scores[size_t(row) * 128 + e] = ink_sigmoid(dot);
}

// One thread a row: main's top-8 and ascending-index combine over the stored scores.
[[max_total_threads_per_threadgroup(32)]]
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

struct InkArgmaxArgs { uint vocab; };
inline bool ink2_argmax_better(float va, uint ia, float vb, uint ib) {
  const bool na = isnan(va), nb = isnan(vb);
  if (na || nb) {
    if (na && ia == 0u) return !(nb && ib == 0u);
    if (nb && ib == 0u) return false;
    return (na != nb) ? !na : (ia < ib);
  }
  return va > vb || (va == vb && ia < ib);
}
[[max_total_threads_per_threadgroup(1024)]]
kernel void tf_ink2_argmax(
    const device float* logits [[buffer(0)]], device uint* out [[buffer(1)]],
    constant InkArgmaxArgs& a [[buffer(2)]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint3 gpos [[threadgroup_position_in_grid]]) {
  threadgroup float sg_val[32]; threadgroup uint sg_idx[32];
  const device float* row = logits + size_t(gpos.x) * a.vocab;
  float bv = NAN; uint bi = 0xFFFFFFFFu;
  for (uint i = tid; i < a.vocab; i += 1024u) {
    const float v = row[i];
    if (ink2_argmax_better(v, i, bv, bi)) { bv = v; bi = i; }
  }
  for (ushort off = 16; off > 0; off >>= 1) {
    const float ov = simd_shuffle_xor(bv, off); const uint oi = simd_shuffle_xor(bi, off);
    if (ink2_argmax_better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
  }
  if (lane == 0u) { sg_val[sg] = bv; sg_idx[sg] = bi; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid != 0u) return;
  for (uint g = 0u; g < 32u; ++g)
    if (ink2_argmax_better(sg_val[g], sg_idx[g], bv, bi)) { bv = sg_val[g]; bi = sg_idx[g]; }
  out[gpos.x] = bi;
}
