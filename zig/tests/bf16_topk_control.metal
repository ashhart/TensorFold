// Bounded BF16 top-k with integer ordering and lowest-token-ID ties.
#include <metal_stdlib>
using namespace metal;
struct Params { uint rows, vocab, k, stride; };
struct Entry { uint token, score_bits; };
struct Row { uint nonfinite, count; Entry entries[16]; };
inline uint tf_bf16_key(ushort raw) {
  const uint bits = (raw & 0x7fffu) == 0 ? 0u : uint(raw);
  return (bits & 0x8000u) != 0 ? (~bits & 0xffffu) : (bits | 0x8000u);
}
kernel void tf_bf16_topk(const device ushort* logits [[buffer(0)]],
                         device Row* output [[buffer(1)]],
                         constant Params& p [[buffer(2)]],
                         uint row [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_threadgroup]]) {
  threadgroup uint keys[256], ids[256], score_bits[256], masks[256];
  threadgroup uint previous_key, previous_id;
  if (lane == 0) {
    output[row].nonfinite = 0;
    output[row].count = 0;
    for (uint i = 0; i < 16; i++) output[row].entries[i] = { 0xffffffffu, 0u };
  }
  for (uint rank = 0; rank < p.k; rank++) {
    uint best_key = 0, best_id = 0xffffffffu, best_bits = 0, mask = 0;
    for (ulong token = lane; token < p.vocab; token += 256) {
      const ushort raw = logits[ulong(row) * p.stride + token];
      if ((raw & 0x7f80u) == 0x7f80u) {
        mask |= (raw & 0x7fu) != 0 ? 1u : (raw & 0x8000u) == 0 ? 2u : 4u;
        continue;
      }
      const uint key = tf_bf16_key(raw);
      if (rank > 0 && (key > previous_key || (key == previous_key && token <= previous_id))) continue;
      if (key > best_key || (key == best_key && token < best_id)) {
        best_key = key;
        best_id = uint(token);
        best_bits = uint(raw) << 16;
      }
    }
    keys[lane] = best_key;
    ids[lane] = best_id;
    score_bits[lane] = best_bits;
    masks[lane] = mask;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 128; step > 0; step >>= 1) {
      if (lane < step) {
        const uint other_key = keys[lane + step], other_id = ids[lane + step];
        if (other_key > keys[lane] || (other_key == keys[lane] && other_id < ids[lane])) {
          keys[lane] = other_key;
          ids[lane] = other_id;
          score_bits[lane] = score_bits[lane + step];
        }
        masks[lane] |= masks[lane + step];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (masks[0] != 0) {
      if (lane == 0) output[row].nonfinite = masks[0];
      return;
    }
    if (lane == 0) {
      output[row].entries[rank] = { ids[0], score_bits[0] };
      output[row].count = rank + 1;
      previous_key = keys[0];
      previous_id = ids[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
