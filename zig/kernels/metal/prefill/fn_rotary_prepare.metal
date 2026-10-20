
[[kernel]] void tf_rotary_prepare(const constant float* INPUT [[buffer(0)]], device float* OUT [[buffer(1)]],
    const device int* POS [[buffer(2)]], const constant uint2& range [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sgi [[simdgroup_index_in_threadgroup]]) {
  device atomic_int* tags = reinterpret_cast<device atomic_int*>(OUT + TF_ROPE_KEYS);
  threadgroup uint reset;
  if (tid == 0) reset = range.y != 0 || OUT[TF_ROPE_ENABLE] == 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (reset && tid < TF_ROPE_SLOTS) atomic_store_explicit(tags + tid, -1, memory_order_relaxed);
  if (reset && tid == 0) OUT[TF_ROPE_ENABLE] = 1.0f;
  threadgroup_barrier(mem_flags::mem_device);
  const uint count = range.x;
  for (uint candidate = sgi; candidate < 2 * count; candidate += 4) {
    int position = POS[candidate % count];
    if (position < 0) continue;
    if (candidate >= count) position = position / 4 * 4;
    int chosen = -1;
    if (lane == 0) {
      uint hash = uint(position);
      hash = (hash ^ (hash >> 16)) * 0x7feb352du;
      hash = (hash ^ (hash >> 15)) * 0x846ca68bu;
      hash ^= hash >> 16;
      for (uint probe = 0; probe < TF_ROPE_SLOTS; ++probe) {
        const uint slot = (hash + probe) & (TF_ROPE_SLOTS - 1);
        int expected = -1;
        bool inserted;
        do {
          expected = -1;
          inserted = atomic_compare_exchange_weak_explicit(tags + slot, &expected, position,
              memory_order_relaxed, memory_order_relaxed);
        } while (!inserted && expected == -1);
        if (inserted) {
          chosen = int(slot);
          break;
        }
        if (expected == position) break;
      }
    }
    chosen = simd_broadcast(chosen, 0);
    if (chosen >= 0) {
      const TF_RotaryScalar value = tf_rotary_scalar(int(lane), position, INPUT);
      device float4* values = reinterpret_cast<device float4*>(OUT + TF_ROPE_DATA);
      values[uint(chosen) * 32 + lane] = float4(value.frequency, value.angle, value.cosine, value.sine);
    }
  }
}
