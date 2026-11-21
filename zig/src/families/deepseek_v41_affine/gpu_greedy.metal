#include <metal_stdlib>
using namespace metal;

// Order the unmodified BF16 bits, including subnormals. The largest key

// reports a nonfinite logit, and both signed zeros compare equally.
static uint tf_greedy_key(uint bits) {
  if ((bits & 0x7f80u) == 0x7f80u) return 0xffffu;
  if (bits == 0x8000u) bits = 0;
  return (bits & 0x8000u) ? 0xffffu - bits : bits | 0x8000u;
}

static uint2 tf_greedy_simd(uint2 pair) {
  const uint max_key = simd_max(pair.x);
  const uint min_id = simd_min(pair.x == max_key ? pair.y : 0xffffffffu);
  return uint2(max_key, min_id);
}

static uint2 tf_greedy_group(uint2 pair, uint lane, uint sg, threadgroup uint2* waves, uint count) {
  const uint2 winner = tf_greedy_simd(pair);
  if (lane == 0) waves[sg] = winner;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return uint2(0, 0xffffffffu);
  return tf_greedy_simd(lane < count ? waves[lane] : uint2(0, 0xffffffffu));
}

[[kernel]] void tf_greedy_tiles(const device ushort* logits [[buffer(0)]],
                                 device uint2* winners [[buffer(1)]],
                                 constant uint2& dims [[buffer(2)]],
                                 uint2 group [[threadgroup_position_in_grid]],
                                 uint2 position [[thread_position_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup uint2 waves[8];
  const uint t = position.x;
  const device ushort* row = logits + size_t(group.y) * dims.x;
  uint2 best(0, 0xffffffffu);
  for (uint x = t; x < 1024; x += 256) {
    const uint id = group.x * 1024 + x;
    if (id < dims.x) {
      const uint key = tf_greedy_key(row[id]);
      if (key > best.x || (key == best.x && id < best.y)) best = uint2(key, id);
    }
  }
  const uint2 winner = tf_greedy_group(best, lane, sg, waves, 8);
  if (t == 0) winners[size_t(group.y) * dims.y + group.x] = winner;
}

[[kernel]] void tf_greedy_finish(const device uint2* winners [[buffer(0)]],
                                  device uint* best [[buffer(1)]],
                                  constant uint& tiles [[buffer(2)]],
                                  uint row [[threadgroup_position_in_grid]],
                                  uint t [[thread_position_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup uint2 waves[4];
  const uint2 local = t < tiles ? winners[size_t(row) * tiles + t] : uint2(0, 0xffffffffu);
  const uint2 winner = tf_greedy_group(local, lane, sg, waves, 4);
  if (t == 0) best[row] = winner.x == 0xffffu ? 0xffffffffu : winner.y;
}
