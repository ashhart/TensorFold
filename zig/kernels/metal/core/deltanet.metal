// DeltaNet: bf16 conv and gates, fp32 state, fixed reductions; steps and accepted chains share one delta update.
#include <metal_stdlib>
using namespace metal;

struct TfDeltaParams {
  uint rows, nk, nv, dk, dv, taps, slots, weight_flags;
  float eps;
  uint zba_stride;  // 0: z, a and b in their own rows; else one fused row of this many values (z, b, a)
};
struct TfDeltaSegment { uint first, rows, state_slot, next_slot; };
struct TfDeltaKeep { uint first, rows, state_slot, next_slot; };

inline float tf_delta_weight(device const void* data, uint at, uint flags, uint bit) {
  if ((flags & (1u << bit)) != 0) return ((device const float*)data)[at];
  return float(((device const bfloat*)data)[at]);
}

// Every position uses the same per-lane decay, key-dot and delta update before a SIMD reduction.
inline void tf_delta_delta_update(thread float* state, device const bfloat* key,
                              float forget, float incoming, float mix, uint lane, uint count) {
  float remembered = 0.0f;
  for (uint part = 0; part < count; ++part) {
    const uint channel = lane * count + part;
    state[part] = state[part] * forget;
    remembered += state[part] * float(key[channel]);
  }
  remembered = simd_sum(remembered);
  const float correction = (incoming - remembered) * mix;
  for (uint part = 0; part < count; ++part) state[part] = state[part] + float(key[lane * count + part]) * correction;
}

inline float tf_delta_delta_read(thread const float* state, device const bfloat* query, uint lane, uint count) {
  float result = 0.0f;
  for (uint part = 0; part < count; ++part) result += state[part] * float(query[lane * count + part]);
  return simd_sum(result);
}
// Convolution, activation and head normalization retain every declared BF16 rounding point.

// C = 4 (128-wide heads) gives every per-lane loop a constant trip count (arrays in registers); C = 0 reads p.
template <uint C>
inline void tf_delta_gdn_pre_body(device const bfloat* projection,
    device const bfloat* a,
    device const bfloat* b,
    device const bfloat* history,
    device const void* weights,
    device const void* decay_weight,
    device const void* time_bias,
    device const uint* windows,
    device const uint* row_slot,
    constant TfDeltaParams& p,
    device bfloat* query,
    device bfloat* key,
    device bfloat* value,
    device float* decay,
    device bfloat* mixing,
    device bfloat* tails,
    uint3 group,
    uint lane) {
  const uint row = group.z, head = group.y;
  const uint channels = 2 * p.nk * p.dk + p.nv * p.dv;
  const uint keep = p.taps - 1, elements = C ? C : p.dk / 32;
  const uint channel0 = head < 2 * p.nk ? head * p.dk : 2 * p.nk * p.dk + (head - 2 * p.nk) * p.dv;
  float activated[C ? C : 8];
  for (uint part = 0; part < elements; ++part) {
    const uint channel = channel0 + lane * elements + part;
    float result = 0.0f;
    for (uint tap = 0; tap < p.taps; ++tap) {
      const uint source = windows[row * p.taps + tap];
      const float input = source < keep ? float(history[(row_slot[row] * keep + source) * channels + channel]) : float(projection[(source - keep) * channels + channel]);
      result += tf_delta_weight(weights, channel * p.taps + tap, p.weight_flags, 0) * input;
    }
    const float rounded = float(bfloat(result));
    const float sigmoid = float(bfloat(1.0f / (1.0f + metal::exp(-rounded))));
    activated[part] = float(bfloat(rounded * sigmoid));
    for (uint tap = 0; tap < keep; ++tap) {
      const uint source = windows[row * p.taps + tap + 1];
      tails[(row * keep + tap) * channels + channel] = source < keep ? history[(row_slot[row] * keep + source) * channels + channel] : projection[(source - keep) * channels + channel];
    }
  }
  if (head < 2 * p.nk) {
    float square = 0.0f;
    for (uint part = 0; part < elements; ++part) square += activated[part] * activated[part];
    square = simd_sum(square);
    const float inverse = metal::rsqrt(square / float(p.dk) + 1e-6f);
    const float scale = head < p.nk ? float(bfloat(1.0f / float(p.dk))) : float(bfloat(metal::rsqrt(float(p.dk))));
    const uint output_head = head % p.nk;
    for (uint part = 0; part < elements; ++part) {
      const uint position = (row * p.nk + output_head) * p.dk + lane * elements + part;
      const bfloat normalized = bfloat(scale * float(bfloat(activated[part] * inverse)));
      if (head < p.nk) query[position] = normalized;
      else key[position] = normalized;
    }
    return;
  }
  const uint output_head = head - 2 * p.nk;
  for (uint part = 0; part < elements; ++part) value[(row * p.nv + output_head) * p.dv + lane * elements + part] = bfloat(activated[part]);
  if (lane == 0) {
    const uint at = row * p.nv + output_head, gate_at = p.zba_stride ? row * p.zba_stride + output_head : at;
    const float shifted = float(bfloat(float(a[gate_at]) + tf_delta_weight(time_bias, output_head, p.weight_flags, 2)));
    const float softplus = float(bfloat(metal::max(shifted, 0.0f) + metal::log(1.0f + metal::exp(-metal::abs(shifted)))));
    decay[at] = metal::exp(-metal::exp(tf_delta_weight(decay_weight, output_head, p.weight_flags, 1)) * softplus);
    mixing[at] = bfloat(1.0f / (1.0f + metal::exp(-float(b[gate_at]))));
  }
}

kernel void tf_delta_gdn_pre(device const bfloat* projection [[buffer(0)]],
                        device const bfloat* a [[buffer(1)]],
                        device const bfloat* b [[buffer(2)]],
                        device const bfloat* history [[buffer(3)]],
                        device const void* weights [[buffer(4)]],
                        device const void* decay_weight [[buffer(5)]],
                        device const void* time_bias [[buffer(6)]],
                        device const uint* windows [[buffer(7)]],
                        device const uint* row_slot [[buffer(8)]],
                        constant TfDeltaParams& p [[buffer(9)]],
                        device bfloat* query [[buffer(10)]],
                        device bfloat* key [[buffer(11)]],
                        device bfloat* value [[buffer(12)]],
                        device float* decay [[buffer(13)]],
                        device bfloat* mixing [[buffer(14)]],
                        device bfloat* tails [[buffer(15)]],
                        uint3 group [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]]) {
  if (p.dk == 128 && p.dv == 128) tf_delta_gdn_pre_body<4>(projection, a, b, history, weights, decay_weight, time_bias, windows, row_slot, p, query, key, value, decay, mixing, tails, group, lane);
  else tf_delta_gdn_pre_body<0>(projection, a, b, history, weights, decay_weight, time_bias, windows, row_slot, p, query, key, value, decay, mixing, tails, group, lane);
}
// One tree node from its parent's state: tf_delta_delta_update and tf_delta_delta_read as the snapshot path runs them.
inline float4 tf_delta_tree_node(float4 parent_state, device const bfloat* query, device const bfloat* key, device const bfloat* value,
                                 device const float* decay, device const bfloat* mixing, constant TfDeltaParams& p, device bfloat* output,
                                 uint row, uint head, uint key_head, uint dimension, uint lane) {
  float state[4] = {parent_state[0], parent_state[1], parent_state[2], parent_state[3]};
  const uint at = row * p.nv + head;
  tf_delta_delta_update(state, key + (row * p.nk + key_head) * p.dk, decay[at], float(value[at * p.dv + dimension]), float(mixing[at]), lane, 4);
  const float result = tf_delta_delta_read(state, query + (row * p.nk + key_head) * p.dk, lane, 4);
  if (lane == 0) output[at * p.dv + dimension] = bfloat(result);
  return float4(state[0], state[1], state[2], state[3]);
}

// Parent states (registers to 16 nodes, else snapshots) and the kept-path replay share one fp32 state update.
kernel void tf_delta_gdn_tree(device const bfloat* query [[buffer(0)]],
                         device const bfloat* key [[buffer(1)]],
                         device const bfloat* value [[buffer(2)]],
                         device const float* decay [[buffer(3)]],
                         device const bfloat* mixing [[buffer(4)]],
                         device const float* committed [[buffer(5)]],
                         device const int* parents [[buffer(6)]],
                         device const TfDeltaSegment* segments [[buffer(7)]],
                         constant TfDeltaParams& p [[buffer(8)]],
                         device bfloat* output [[buffer(9)]],
                         device float* snapshots [[buffer(10)]],
                         uint3 position [[thread_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]]) {
  const uint segment_index = position.z / p.nv, head = position.z % p.nv;
  const uint dimension = position.y, key_head = head / (p.nv / p.nk), count = p.dk / 32;
  const TfDeltaSegment span = segments[segment_index];
  const uint state_stride = p.nv * p.dv * p.dk;
  const uint column = (head * p.dv + dimension) * p.dk + lane * count;
  float state[8];
  if (count == 4 && span.rows <= 16) {
    // Up to 16 nodes hold states in named registers: a child takes its parent's through a uniform switch.
    float4 root;
    for (uint part = 0; part < 4; ++part) root[part] = committed[span.state_slot * state_stride + column + part];
    float4 kept0 = 0.0f, kept1 = 0.0f, kept2 = 0.0f, kept3 = 0.0f, kept4 = 0.0f, kept5 = 0.0f, kept6 = 0.0f, kept7 = 0.0f;
    float4 kept8 = 0.0f, kept9 = 0.0f, kept10 = 0.0f, kept11 = 0.0f, kept12 = 0.0f, kept13 = 0.0f, kept14 = 0.0f, kept15 = 0.0f;
#define TF_PICK(j) case j: s = kept##j; break;
#define TF_TREE_NODE(i) \
    if (i < span.rows) { \
      float4 s = root; \
      switch (parents[span.first + i]) { TF_PICK(0) TF_PICK(1) TF_PICK(2) TF_PICK(3) TF_PICK(4) TF_PICK(5) TF_PICK(6) TF_PICK(7) TF_PICK(8) TF_PICK(9) TF_PICK(10) TF_PICK(11) TF_PICK(12) TF_PICK(13) TF_PICK(14) default: break; } \
      kept##i = tf_delta_tree_node(s, query, key, value, decay, mixing, p, output, span.first + i, head, key_head, dimension, lane); \
    }
    TF_TREE_NODE(0) TF_TREE_NODE(1) TF_TREE_NODE(2) TF_TREE_NODE(3) TF_TREE_NODE(4) TF_TREE_NODE(5) TF_TREE_NODE(6) TF_TREE_NODE(7)
    TF_TREE_NODE(8) TF_TREE_NODE(9) TF_TREE_NODE(10) TF_TREE_NODE(11) TF_TREE_NODE(12) TF_TREE_NODE(13) TF_TREE_NODE(14) TF_TREE_NODE(15)
#undef TF_TREE_NODE
#undef TF_PICK
    return;
  }
  for (uint node = 0; node < span.rows; ++node) {
    const uint row = span.first + node;
    const int parent = parents[row];
    const uint source = parent < 0 ? span.state_slot * state_stride : (span.first + uint(parent)) * state_stride;
    for (uint part = 0; part < count; ++part) state[part] = parent < 0 ? committed[source + column + part] : snapshots[source + column + part];
    const uint at = row * p.nv + head;
    device const bfloat* k = key + (row * p.nk + key_head) * p.dk;
    device const bfloat* q = query + (row * p.nk + key_head) * p.dk;
    tf_delta_delta_update(state, k, decay[at], float(value[at * p.dv + dimension]), float(mixing[at]), lane, count);
    const float result = tf_delta_delta_read(state, q, lane, count);
    if (lane == 0) output[at * p.dv + dimension] = bfloat(result);
    for (uint part = 0; part < count; ++part) snapshots[row * state_stride + column + part] = state[part];
  }
}

// C = 4 (128-wide heads) gives every per-lane loop a constant trip count (arrays in registers); C = 0 reads p.
template <uint C>
inline void tf_delta_gdn_replay_body(device const bfloat* key,
    device const bfloat* value,
    device const float* decay,
    device const bfloat* mixing,
    device const float* committed,
    device const uint* kept_rows,
    device const TfDeltaKeep* keeps,
    constant TfDeltaParams& p,
    device float* next,
    uint3 position,
    uint lane) {
  const uint segment_index = position.z / p.nv, head = position.z % p.nv;
  const uint dimension = position.y, key_head = head / (p.nv / p.nk), count = C ? C : p.dk / 32;
  const uint state_stride = p.nv * p.dv * p.dk;
  const uint column = (head * p.dv + dimension) * p.dk + lane * count;
  const TfDeltaKeep keep = keeps[segment_index];
  float state[C ? C : 8];
  for (uint part = 0; part < count; ++part) state[part] = committed[keep.state_slot * state_stride + column + part];
  for (uint step = 0; step < keep.rows; ++step) {
    const uint row = kept_rows[keep.first + step], at = row * p.nv + head;
    device const bfloat* k = key + (row * p.nk + key_head) * p.dk;
    tf_delta_delta_update(state, k, decay[at], float(value[at * p.dv + dimension]), float(mixing[at]), lane, count);
  }
  for (uint part = 0; part < count; ++part) next[keep.next_slot * state_stride + column + part] = state[part];
}

kernel void tf_delta_gdn_replay(device const bfloat* key [[buffer(0)]],
                           device const bfloat* value [[buffer(1)]],
                           device const float* decay [[buffer(2)]],
                           device const bfloat* mixing [[buffer(3)]],
                           device const float* committed [[buffer(4)]],
                           device const uint* kept_rows [[buffer(5)]],
                           device const TfDeltaKeep* keeps [[buffer(6)]],
                           constant TfDeltaParams& p [[buffer(7)]],
                           device float* next [[buffer(8)]],
                           uint3 position [[thread_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]]) {
  if (p.dk == 128) tf_delta_gdn_replay_body<4>(key, value, decay, mixing, committed, kept_rows, keeps, p, next, position, lane);
  else tf_delta_gdn_replay_body<0>(key, value, decay, mixing, committed, kept_rows, keeps, p, next, position, lane);
}

// Accepted chains keep the tree path's delta update and sum order; a prompt chunk differs only in trip count.
template <uint C>
inline void tf_delta_gdn_chain_body(device const bfloat* query,
    device const bfloat* key,
    device const bfloat* value,
    device const float* decay,
    device const bfloat* mixing,
    device const float* committed,
    device const TfDeltaSegment* segments,
    constant TfDeltaParams& p,
    device bfloat* output,
    device float* next_state,
    uint3 position,
    uint lane) {
  const uint segment_index = position.z / p.nv, head = position.z % p.nv;
  const uint dimension = position.y, key_head = head / (p.nv / p.nk), count = C ? C : p.dk / 32;
  const TfDeltaSegment span = segments[segment_index];
  const uint state_stride = p.nv * p.dv * p.dk;
  const uint column = (head * p.dv + dimension) * p.dk + lane * count;
  float state[C ? C : 8];
  for (uint part = 0; part < count; ++part) state[part] = committed[span.state_slot * state_stride + column + part];
  for (uint node = 0; node < span.rows; ++node) {
    const uint row = span.first + node, at = row * p.nv + head;
    device const bfloat* k = key + (row * p.nk + key_head) * p.dk;
    device const bfloat* q = query + (row * p.nk + key_head) * p.dk;
    tf_delta_delta_update(state, k, decay[at], float(value[at * p.dv + dimension]), float(mixing[at]), lane, count);
    const float result = tf_delta_delta_read(state, q, lane, count);
    if (lane == 0) output[at * p.dv + dimension] = bfloat(result);
  }
  for (uint part = 0; part < count; ++part) next_state[span.next_slot * state_stride + column + part] = state[part];
}

kernel void tf_delta_gdn_chain(device const bfloat* query [[buffer(0)]],
                          device const bfloat* key [[buffer(1)]],
                          device const bfloat* value [[buffer(2)]],
                          device const float* decay [[buffer(3)]],
                          device const bfloat* mixing [[buffer(4)]],
                          device const float* committed [[buffer(5)]],
                          device const TfDeltaSegment* segments [[buffer(6)]],
                          constant TfDeltaParams& p [[buffer(7)]],
                          device bfloat* output [[buffer(8)]],
                          device float* next_state [[buffer(9)]],
                          uint3 position [[thread_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]]) {
  if (p.dk == 128) tf_delta_gdn_chain_body<4>(query, key, value, decay, mixing, committed, segments, p, output, next_state, position, lane);
  else tf_delta_gdn_chain_body<0>(query, key, value, decay, mixing, committed, segments, p, output, next_state, position, lane);
}
// Prompt chains (may differ in bits from decode): 8 lanes a value column, 16 key channels each; dk = 128.
kernel void tf_delta_gdn_chain_wide(device const bfloat* query [[buffer(0)]],
                          device const bfloat* key [[buffer(1)]],
                          device const bfloat* value [[buffer(2)]],
                          device const float* decay [[buffer(3)]],
                          device const bfloat* mixing [[buffer(4)]],
                          device const float* committed [[buffer(5)]],
                          device const TfDeltaSegment* segments [[buffer(6)]],
                          constant TfDeltaParams& p [[buffer(7)]],
                          device bfloat* output [[buffer(8)]],
                          device float* next_state [[buffer(9)]],
                          uint3 position [[thread_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]]) {
  const uint segment_index = position.z / p.nv, head = position.z % p.nv;
  const uint sub = lane & 7, dimension = position.y * 4 + (lane >> 3), key_head = head / (p.nv / p.nk);
  const TfDeltaSegment span = segments[segment_index];
  const uint state_stride = p.nv * p.dv * p.dk;
  const uint column = (head * p.dv + dimension) * p.dk + sub * 16;
  float state[16];
  for (uint i = 0; i < 16; ++i) state[i] = committed[span.state_slot * state_stride + column + i];
  for (uint node = 0; node < span.rows; ++node) {
    const uint row = span.first + node, at = row * p.nv + head;
    device const bfloat* k = key + (row * p.nk + key_head) * p.dk + sub * 16;
    device const bfloat* q = query + (row * p.nk + key_head) * p.dk + sub * 16;
    const float forget = decay[at];
    float remembered = 0.0f;
    for (uint i = 0; i < 16; ++i) {
      state[i] = state[i] * forget;
      remembered += state[i] * float(k[i]);
    }
    remembered += simd_shuffle_xor(remembered, 1);
    remembered += simd_shuffle_xor(remembered, 2);
    remembered += simd_shuffle_xor(remembered, 4);
    const float correction = (float(value[at * p.dv + dimension]) - remembered) * float(mixing[at]);
    float result = 0.0f;
    for (uint i = 0; i < 16; ++i) {
      state[i] = state[i] + float(k[i]) * correction;
      result += state[i] * float(q[i]);
    }
    result += simd_shuffle_xor(result, 1);
    result += simd_shuffle_xor(result, 2);
    result += simd_shuffle_xor(result, 4);
    if (sub == 0) output[at * p.dv + dimension] = bfloat(result);
  }
  for (uint i = 0; i < 16; ++i) next_state[span.next_slot * state_stride + column + i] = state[i];
}
// The normalized value rounds to BF16 before the SiLU gate multiplies it.

// C = 4 (128-wide heads) gives every per-lane loop a constant trip count (arrays in registers); C = 0 reads p.
template <uint C>
inline void tf_delta_gdn_post_body(device const bfloat* recurrence,
    device const bfloat* gate,
    device const void* norm,
    constant TfDeltaParams& p,
    device bfloat* output,
    uint3 group,
    uint lane) {
  const uint row = group.z, head = group.y, elements = C ? C : p.dv / 32;
  float values[C ? C : 8];
  float square = 0.0f;
  const uint at = (row * p.nv + head) * p.dv + lane * elements, z_at = p.zba_stride ? row * p.zba_stride + head * p.dv + lane * elements : at;
  for (uint part = 0; part < elements; ++part) {
    values[part] = float(recurrence[at + part]);
    square += values[part] * values[part];
  }
  square = simd_sum(square);
  const float inverse = metal::rsqrt(square / float(p.dv) + p.eps);
  for (uint part = 0; part < elements; ++part) {
    const uint channel = lane * elements + part;
    const float normalized = float(bfloat(tf_delta_weight(norm, channel, p.weight_flags, 3) * (values[part] * inverse)));
    const float z = float(gate[z_at + part]);
    output[at + part] = bfloat(z / (1.0f + metal::exp(-z)) * normalized);
  }
}

kernel void tf_delta_gdn_post(device const bfloat* recurrence [[buffer(0)]],
                         device const bfloat* gate [[buffer(1)]],
                         device const void* norm [[buffer(2)]],
                         constant TfDeltaParams& p [[buffer(3)]],
                         device bfloat* output [[buffer(4)]],
                         uint3 group [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]]) {
  if (p.dv == 128) tf_delta_gdn_post_body<4>(recurrence, gate, norm, p, output, group, lane);
  else tf_delta_gdn_post_body<0>(recurrence, gate, norm, p, output, group, lane);
}
// Projected rows and ancestor indices reconstruct the accepted tail without persisting every node's conv window.

kernel void tf_delta_gdn_conv_commit(device const bfloat* committed [[buffer(0)]],
                                device const bfloat* projection [[buffer(1)]],
                                device const uint* kept_rows [[buffer(2)]],
                                device const TfDeltaKeep* keeps [[buffer(3)]],
                                device const uint* windows [[buffer(4)]],
                                constant TfDeltaParams& p [[buffer(5)]],
                                device bfloat* next [[buffer(6)]],
                                uint3 position [[thread_position_in_grid]]) {
  const uint channels = 2 * p.nk * p.dk + p.nv * p.dv;
  const uint width = (p.taps - 1) * channels, item = position.x;
  const TfDeltaKeep keep = keeps[position.y];
  if (item >= width) return;
  if (keep.rows == 0) {
    next[keep.next_slot * width + item] = committed[keep.state_slot * width + item];
    return;
  }
  const uint row = kept_rows[keep.first + keep.rows - 1];
  const uint source = windows[row * p.taps + item / channels + 1];
  next[keep.next_slot * width + item] = source < p.taps - 1 ? committed[(keep.state_slot * (p.taps - 1) + source) * channels + item % channels] : projection[(source - (p.taps - 1)) * channels + item % channels];
}
// Replay finishes before publication, which copies storage words without another floating-point operation.

kernel void tf_delta_gdn_publish(device const uint* next_state [[buffer(0)]],
                            device const ushort* next_conv [[buffer(1)]],
                            device const TfDeltaKeep* keeps [[buffer(2)]],
                            constant TfDeltaParams& p [[buffer(3)]],
                            device uint* state [[buffer(4)]],
                            device ushort* conv [[buffer(5)]],
                            uint2 at [[thread_position_in_grid]]) {
  const uint state_width = p.nv * p.dv * p.dk;
  const uint conv_width = (p.taps - 1) * (2 * p.nk * p.dk + p.nv * p.dv);
  const uint slot = keeps[at.y].next_slot;
  if (at.x < state_width) state[slot * state_width + at.x] = next_state[slot * state_width + at.x];
  if (at.x < conv_width) conv[slot * conv_width + at.x] = next_conv[slot * conv_width + at.x];
}

// The conv window alone, for commits whose replay wrote the state in place.
kernel void tf_delta_gdn_publish_conv(device const ushort* next_conv [[buffer(0)]],
                                 device const TfDeltaKeep* keeps [[buffer(1)]],
                                 constant TfDeltaParams& p [[buffer(2)]],
                                 device ushort* conv [[buffer(3)]],
                                 uint2 at [[thread_position_in_grid]]) {
  const uint conv_width = (p.taps - 1) * (2 * p.nk * p.dk + p.nv * p.dv);
  const uint slot = keeps[at.y].next_slot;
  if (at.x < conv_width) conv[slot * conv_width + at.x] = next_conv[slot * conv_width + at.x];
}

kernel void tf_delta_gdn_clear(device uint* state [[buffer(0)]],
                          device ushort* conv [[buffer(1)]],
                          constant TfDeltaParams& p [[buffer(2)]],
                          constant uint& slot [[buffer(3)]],
                          uint2 at [[thread_position_in_grid]]) {
  const uint state_width = p.nv * p.dv * p.dk;
  const uint conv_width = (p.taps - 1) * (2 * p.nk * p.dk + p.nv * p.dv);
  if (at.x < state_width) state[(at.y * p.slots + slot) * state_width + at.x] = 0u;
  if (at.x < conv_width) conv[(at.y * p.slots + slot) * conv_width + at.x] = 0u;
}
