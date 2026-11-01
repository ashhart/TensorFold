// Key-parallel attention for decode windows and short tails: FP32 online softmax; nonfinite rows replay the exact path.
constant constexpr uint L3S_PARTS = 64;
constant constexpr uint L3S_MIN = 128;

inline uint l3s_cutoff(constant lfm3_attn_args& p, uint count) { return p.window == 0 ? 0 : count > p.window ? count - p.window : 0; }
inline uint l3s_length(uint n) { return max(L3S_MIN, (((n + L3S_PARTS - 1) / L3S_PARTS) + L3S_MIN - 1) & ~(L3S_MIN - 1)); }

inline float l3s_dot(const threadgroup float* q, const device bfloat* k) {
  const device uint4* words = reinterpret_cast<const device uint4*>(k);
  float sum = 0.0f;
  for (uint i = 0; i < 16; ++i) {
    const uint4 w = words[i];
    for (uint c = 0; c < 4; ++c) {
      sum += q[8 * i + 2 * c] * as_type<float>(w[c] << 16);
      sum += q[8 * i + 2 * c + 1] * as_type<float>(w[c] & 0xffff0000u);
    }
  }
  return sum;
}

inline void l3s_partial(const device bfloat* queries, const device bfloat* keys, const device bfloat* values,
    constant lfm3_attn_args& p, device float* pm, device float* pl, device float* po, uint head, uint row, uint part,
    uint sg, uint lane, threadgroup float* q, threadgroup float* maxima, threadgroup float* totals,
    threadgroup float* outputs) {
  const uint count = p.len + row + 1, cutoff = l3s_cutoff(p, count), length = l3s_length(count - cutoff);
  const uint first = cutoff + part * length;
  if (first >= count) return;
  const uint end = min(first + length, count), kh = head / 4;
  q[sg * 32 + lane] = float(queries[(size_t(row) * 16 + head) * 128 + sg * 32 + lane]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint begin = first + sg * (length / 4), stop = min(begin + length / 4, end);
  float maximum = -INFINITY, total = 0.0f, acc[4] = {0, 0, 0, 0};
  for (uint block = begin; block < stop; block += 32) {
    const uint key = block + lane;
    const float score = key < stop ? l3s_dot(q, keys + (size_t(kh) * p.cap + key) * 128) * p.scale : -INFINITY;
    const float next = max(maximum, simd_max(score));
    if (next == -INFINITY) continue;
    const float alpha = maximum == -INFINITY ? 0.0f : metal::precise::exp(maximum - next);
    const float probability = score == -INFINITY ? 0.0f : metal::precise::exp(score - next);
    total = total * alpha + simd_sum(probability);
    for (uint j = 0; j < 4; ++j) acc[j] *= alpha;
    const uint live = min(32u, stop - block);
    for (uint k = 0; k < live; ++k) {
      const float weight = simd_broadcast(probability, k);
      const size_t base = (size_t(kh) * p.cap + block + k) * 128 + lane;
      for (uint j = 0; j < 4; ++j) acc[j] += weight * float(values[base + 32 * j]);
    }
    maximum = next;
  }
  if (lane == 0) { maxima[sg] = maximum; totals[sg] = total; }
  for (uint j = 0; j < 4; ++j) outputs[sg * 128 + lane + 32 * j] = acc[j];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;
  float merged = -INFINITY;
  for (uint s = 0; s < 4; ++s) merged = max(merged, maxima[s]);
  float sum = 0.0f, out[4] = {0, 0, 0, 0};
  for (uint s = 0; s < 4; ++s) {
    if (totals[s] == 0.0f) continue;
    const float w = metal::precise::exp(maxima[s] - merged);
    sum += totals[s] * w;
    for (uint j = 0; j < 4; ++j) out[j] += outputs[s * 128 + lane + 32 * j] * w;
  }
  const size_t target = (size_t(row) * 16 + head) * L3S_PARTS + part;
  if (lane == 0) { pm[target] = merged; pl[target] = sum; }
  for (uint j = 0; j < 4; ++j) po[target * 128 + lane + 32 * j] = out[j];
}

inline void l3s_merge(const device float* pm, const device float* pl, const device float* po, constant lfm3_attn_args& p,
    device bfloat* y, const device bfloat* queries, const device bfloat* keys, const device bfloat* values, uint head,
    uint row, uint lane) {
  const uint count = p.len + row + 1, cutoff = l3s_cutoff(p, count), n = count - cutoff, length = l3s_length(n);
  const uint parts = (n + length - 1) / length;
  const size_t base = (size_t(row) * 16 + head) * L3S_PARTS;
  float maximum = -INFINITY;
  for (uint i = 0; i < parts; ++i) maximum = max(maximum, pm[base + i]);
  float total = 0.0f, out[4] = {0, 0, 0, 0};
  for (uint i = 0; i < parts; ++i) {
    if (pl[base + i] == 0.0f) continue;
    const float w = metal::precise::exp(pm[base + i] - maximum);
    total += pl[base + i] * w;
    for (uint j = 0; j < 4; ++j) out[j] += po[(base + i) * 128 + lane + 32 * j] * w;
  }
  bool bad = !isfinite(total) || total == 0.0f;
  for (uint j = 0; j < 4; ++j) bad |= !isfinite(out[j]);
  if (simd_any(bad)) {
    l3_x repaired[4];
    l3_attention_replay(queries, keys, values, p, head, row, lane, repaired);
    for (uint j = 0; j < 4; ++j)
      reinterpret_cast<device ushort*>(y)[(size_t(row) * 16 + head) * 128 + lane + 32 * j] = l3_x_pack(repaired[j]);
    return;
  }
  for (uint j = 0; j < 4; ++j) y[(size_t(row) * 16 + head) * 128 + lane + 32 * j] = bfloat(out[j] / total);
}

[[kernel, max_total_threads_per_threadgroup(128)]] void lfm3_attn_split(const device bfloat* queries [[buffer(0)]],
    const device bfloat* keys [[buffer(1)]], const device bfloat* values [[buffer(2)]],
    constant lfm3_attn_args& p [[buffer(3)]], device float* pm [[buffer(4)]], device float* pl [[buffer(5)]],
    device float* po [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float q[128], maxima[4], totals[4], outputs[4 * 128];
  l3s_partial(queries, keys, values, p, pm, pl, po, tg.x, tg.y, tg.z, sg, lane, q, maxima, totals, outputs);
}

[[kernel, max_total_threads_per_threadgroup(32)]] void lfm3_attn_split_merge(const device float* pm [[buffer(0)]],
    const device float* pl [[buffer(1)]], const device float* po [[buffer(2)]], constant lfm3_attn_args& p [[buffer(3)]],
    device bfloat* y [[buffer(4)]], const device bfloat* queries [[buffer(5)]], const device bfloat* keys [[buffer(6)]],
    const device bfloat* values [[buffer(7)]], uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  l3s_merge(pm, pl, po, p, y, queries, keys, values, tg.x, tg.y, lane);
}

[[kernel, max_total_threads_per_threadgroup(128)]] void lfm3_attn_split_segments(
    const device bfloat* queries [[buffer(0)]], constant lfm3_attn_segment* segments [[buffer(1)]],
    const device uint* row_segments [[buffer(2)]], device float* pm [[buffer(3)]], device float* pl [[buffer(4)]],
    device float* po [[buffer(5)]], uint3 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constant lfm3_attn_segment& seg = segments[row_segments[tg.y]];
  const size_t offset = size_t(seg.base) * 16 * L3S_PARTS;
  threadgroup float q[128], maxima[4], totals[4], outputs[4 * 128];
  l3s_partial(queries + size_t(seg.base) * 16 * 128, reinterpret_cast<const device bfloat*>(seg.keys),
      reinterpret_cast<const device bfloat*>(seg.values), seg.args, pm + offset, pl + offset, po + offset * 128, tg.x,
      tg.y - seg.base, tg.z, sg, lane, q, maxima, totals, outputs);
}

[[kernel, max_total_threads_per_threadgroup(32)]] void lfm3_attn_split_segments_merge(
    const device float* pm [[buffer(0)]], const device float* pl [[buffer(1)]], const device float* po [[buffer(2)]],
    constant lfm3_attn_segment* segments [[buffer(3)]], device bfloat* y [[buffer(4)]],
    const device uint* row_segments [[buffer(5)]], const device bfloat* queries [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  constant lfm3_attn_segment& seg = segments[row_segments[tg.y]];
  const size_t offset = size_t(seg.base) * 16 * L3S_PARTS;
  l3s_merge(pm + offset, pl + offset, po + offset * 128, seg.args, y + size_t(seg.base) * 16 * 128,
      queries + size_t(seg.base) * 16 * 128, reinterpret_cast<const device bfloat*>(seg.keys),
      reinterpret_cast<const device bfloat*>(seg.values), tg.x, tg.y - seg.base, lane);
}
