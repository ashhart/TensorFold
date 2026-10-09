// Prompt state advances through the core tap order and one fixed per-position reduction topology.
#include <metal_stdlib>
using namespace metal;

struct Q27PromptConv { uint rows, channels, taps, streams; };
struct Q27PromptRecur { uint rows, nk, nv, dv, streams, pad0, pad1, pad2; };

kernel void q27_prompt_conv(device const bfloat* projected [[buffer(0)]],
                            device const bfloat* history [[buffer(1)]],
                            device const bfloat* weight [[buffer(2)]],
                            constant Q27PromptConv& p [[buffer(3)]],
                            device bfloat* output [[buffer(4)]],
                            device bfloat* next_history [[buffer(5)]],
                            uint3 at [[thread_position_in_grid]]) {
  if (at.x >= p.channels || at.y >= p.rows || at.z >= p.streams) return;
  const uint keep = p.taps - 1;
  const uint old_base = at.z * keep * p.channels;
  const uint new_base = at.z * p.rows * p.channels;
  float sum = 0.0f;
  for (uint tap = 0; tap < p.taps; ++tap) {
    const uint row = at.y + tap;
    const float value = row < keep ? float(history[old_base + row * p.channels + at.x]) : float(projected[new_base + (row - keep) * p.channels + at.x]);
    sum += value * float(weight[at.x * p.taps + tap]);
  }
  output[new_base + at.y * p.channels + at.x] = bfloat(sum);
  if (at.y == 0) {
    for (uint tail = 0; tail < keep; ++tail) {
      const uint row = p.rows + tail;
      next_history[old_base + tail * p.channels + at.x] = row < keep ? history[old_base + row * p.channels + at.x] : projected[new_base + (row - keep) * p.channels + at.x];
    }
  }
}

inline float q27_prompt_fold(thread float* partial) {
  for (uint stride = 1; stride < 8; stride *= 2) {
    for (uint at = 0; at < 8; at += 2 * stride) partial[at] = partial[at] + partial[at + stride];
  }
  float total = partial[0];
  for (uint neighbor = 1; neighbor < 4; neighbor *= 2) total = total + simd_shuffle_xor(total, neighbor);
  return total;
}

kernel void q27_prompt_recur(device const bfloat* query [[buffer(0)]],
                             device const bfloat* key [[buffer(1)]],
                             device const bfloat* value [[buffer(2)]],
                             device const float* decay [[buffer(3)]],
                             device const bfloat* beta [[buffer(4)]],
                             device const float* committed [[buffer(5)]],
                             constant Q27PromptRecur& p [[buffer(6)]],
                             device bfloat* result [[buffer(7)]],
                             device float* next_state [[buffer(8)]],
                             uint3 at [[thread_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]]) {
  const uint stream = at.z / p.nv;
  const uint head = at.z % p.nv;
  const uint dimension = at.y * 8 + lane / 4;
  const uint first = (lane % 4) * 32;
  const uint key_head = head / (p.nv / p.nk);
  const uint state_base = ((stream * p.nv + head) * p.dv + dimension) * 128 + first;
  float state[32];
  for (uint channel = 0; channel < 32; ++channel) state[channel] = committed[state_base + channel];
  for (uint row = 0; row < p.rows; ++row) {
    const uint key_base = ((stream * p.rows + row) * p.nk + key_head) * 128 + first;
    const uint head_at = (stream * p.rows + row) * p.nv + head;
    float partial[8];
    for (uint i = 0; i < 8; ++i) partial[i] = 0.0f;
    for (uint channel = 0; channel < 32; ++channel) {
      state[channel] = state[channel] * decay[head_at];
      partial[channel / 4] += state[channel] * float(key[key_base + channel]);
    }
    const float remembered = q27_prompt_fold(partial);
    const float correction = (float(value[head_at * p.dv + dimension]) - remembered) * float(beta[head_at]);
    for (uint i = 0; i < 8; ++i) partial[i] = 0.0f;
    for (uint channel = 0; channel < 32; ++channel) {
      state[channel] = state[channel] + float(key[key_base + channel]) * correction;
      partial[channel / 4] += state[channel] * float(query[key_base + channel]);
    }
    const float output = q27_prompt_fold(partial);
    if (lane % 4 == 0) result[head_at * p.dv + dimension] = bfloat(output);
  }
  for (uint channel = 0; channel < 32; ++channel) next_state[state_base + channel] = state[channel];
}
