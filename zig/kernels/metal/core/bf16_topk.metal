// One input scan builds local min-heaps, sorted once before exact score/lowest-token-ID merges.
#include <metal_stdlib>
using namespace metal;
struct Params { uint rows, vocab, k, stride; };
struct Entry { uint token, score_bits; };
struct Row { uint nonfinite, count; Entry entries[16]; };
inline uint tf_bf16_key(ushort raw) {
  const uint bits = (raw & 0x7fffu) == 0 ? 0u : uint(raw);
  return (bits & 0x8000u) != 0 ? (~bits & 0xffffu) : (bits | 0x8000u);
}
inline bool tf_topk_before(uint key, uint id, uint other_key, uint other_id) {
  return key > other_key || (key == other_key && id < other_id);
}
inline void tf_topk_swap(thread uint* key, thread uint* id, thread uint* bits, uint a, uint b) {
  const uint k = key[a], i = id[a], s = bits[a]; key[a] = key[b]; id[a] = id[b]; bits[a] = bits[b];
  key[b] = k; id[b] = i; bits[b] = s;
}
inline void tf_topk_sift(thread uint* key, thread uint* id, thread uint* bits, uint count) {
  uint at = 0;
  while (2 * at + 1 < count) {
    uint child = 2 * at + 1;
    if (child + 1 < count && tf_topk_before(key[child], id[child], key[child + 1], id[child + 1])) child++;
    if (!tf_topk_before(key[at], id[at], key[child], id[child])) break;
    tf_topk_swap(key, id, bits, at, child); at = child;
  }
}
kernel void tf_bf16_topk(const device ushort* logits [[buffer(0)]],
                        device Row* output [[buffer(1)]], constant Params& p [[buffer(2)]],
                        uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
  threadgroup uint keys[256], ids[256], score_bits[256], masks[256];
  uint local_key[16], local_id[16], local_bits[16];
  for (uint i = 0; i < 16; i++) { local_key[i] = 0; local_id[i] = 0xffffffffu; local_bits[i] = 0; }
  if (lane == 0) {
    output[row].nonfinite = 0; output[row].count = 0;
    for (uint i = 0; i < 16; i++) output[row].entries[i] = { 0xffffffffu, 0u };
  }
  uint mask = 0, count = 0;
  for (ulong token = lane; token < p.vocab; token += 256) {
    const ushort raw = logits[ulong(row) * p.stride + token];
    if ((raw & 0x7f80u) == 0x7f80u) {
      mask |= (raw & 0x7fu) != 0 ? 1u : (raw & 0x8000u) == 0 ? 2u : 4u;
      continue;
    }
    const uint key = tf_bf16_key(raw), id = uint(token), bits = uint(raw) << 16;
    if (count < p.k) {
      uint at = count++;
      local_key[at] = key; local_id[at] = id; local_bits[at] = bits;
      while (at > 0) {
        const uint parent = (at - 1) / 2;
        if (!tf_topk_before(local_key[parent], local_id[parent], local_key[at], local_id[at])) break;
        tf_topk_swap(local_key, local_id, local_bits, at, parent); at = parent;
      }
    } else if (tf_topk_before(key, id, local_key[0], local_id[0])) {
      local_key[0] = key; local_id[0] = id; local_bits[0] = bits;
      tf_topk_sift(local_key, local_id, local_bits, count);
    }
  }
  for (uint end = count; end > 1; end--) {
    tf_topk_swap(local_key, local_id, local_bits, 0, end - 1);
    tf_topk_sift(local_key, local_id, local_bits, end - 1);
  }
  uint cursor = 0;
  for (uint rank = 0; rank < p.k; rank++) {
    keys[lane] = cursor < p.k ? local_key[cursor] : 0;
    ids[lane] = cursor < p.k ? local_id[cursor] : 0xffffffffu;
    score_bits[lane] = cursor < p.k ? local_bits[cursor] : 0;
    masks[lane] = mask;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 128; step > 0; step >>= 1) {
      if (lane < step) {
        if (tf_topk_before(keys[lane + step], ids[lane + step], keys[lane], ids[lane])) {
          keys[lane] = keys[lane + step]; ids[lane] = ids[lane + step]; score_bits[lane] = score_bits[lane + step];
        }
        masks[lane] |= masks[lane + step];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (masks[0] != 0) { if (lane == 0) output[row].nonfinite = masks[0]; return; }
    const uint winner = ids[0];
    if (lane == 0) { output[row].entries[rank] = { winner, score_bits[0] }; output[row].count = rank + 1; }
    if (cursor < p.k && local_id[cursor] == winner) cursor++;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

kernel void tf_bf16_topk_max(const device ushort* logits [[buffer(0)]],
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
