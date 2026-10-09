// Prompt intermediates round where the declared BF16 graph rounds, rather than at the lane fusion's final store.
#include <metal_stdlib>
using namespace metal;

struct Q27PromptProduct { uint width, rows, mode, pad; };
struct Q27PromptQk { uint rows, heads, width, stride, offset; float eps, scale; uint pad; };
struct Q27PromptDecay { uint rows, heads, weight_flags, pad; };

inline float q27_prompt_probability(float value) {
  const float magnitude = metal::abs(value);
  const float denominator = precise::exp(magnitude) + 1.0f;
  const float tail = 1.0f / denominator;
  if (value < 0.0f) return tail;
  return 1.0f - tail;
}

inline float q27_prompt_softplus(float value) {
  const float small = metal::exp(-metal::abs(value));
  const float shifted = 1.0f + small;
  float logarithm = small;
  if (shifted != 1.0f) {
    const float difference = shifted - 1.0f;
    const float ratio = metal::log(shifted) / difference;
    logarithm = ratio * small;
  }
  return metal::max(value, 0.0f) + logarithm;
}

kernel void q27_prompt_product(device const bfloat* gate [[buffer(0)]],
                               device const bfloat* factor [[buffer(1)]],
                               constant Q27PromptProduct& p [[buffer(2)]],
                               device bfloat* result [[buffer(3)]],
                               uint at [[thread_position_in_grid]]) {
  if (at >= p.rows * p.width) return;
  const float x = float(gate[at]);
  const float probability = q27_prompt_probability(x);
  float activated;
  if (p.mode == 0) activated = float(bfloat(x * float(bfloat(probability))));
  else if (p.mode == 1) activated = x * probability;
  else activated = x / (1.0f + metal::exp(-x));
  result[at] = bfloat(activated * float(factor[at]));
}

kernel void q27_prompt_silu(device const bfloat* input [[buffer(0)]],
                            constant Q27PromptProduct& p [[buffer(1)]],
                            device bfloat* result [[buffer(2)]],
                            uint at [[thread_position_in_grid]]) {
  if (at >= p.rows * p.width) return;
  const float value = float(input[at]);
  const bfloat probability = bfloat(q27_prompt_probability(value));
  result[at] = bfloat(value * float(probability));
}

kernel void q27_prompt_decay(device const bfloat* a [[buffer(0)]],
                             device const bfloat* b [[buffer(1)]],
                             device const void* a_log [[buffer(2)]],
                             device const void* time_bias [[buffer(3)]],
                             constant Q27PromptDecay& p [[buffer(4)]],
                             device float* decay [[buffer(5)]],
                             device bfloat* beta [[buffer(6)]],
                             uint at [[thread_position_in_grid]]) {
  if (at >= p.rows * p.heads) return;
  const uint head = at % p.heads;
  const bool wide_bias = (p.weight_flags & 4u) != 0;
  const float bias = wide_bias ? ((device const float*)time_bias)[head] : float(((device const bfloat*)time_bias)[head]);
  const float logarithm = (p.weight_flags & 2u) != 0 ? ((device const float*)a_log)[head] : float(((device const bfloat*)a_log)[head]);
  const float sum = float(a[at]) + bias;
  const float shifted = wide_bias ? sum : float(bfloat(sum));
  const float raw = q27_prompt_softplus(shifted);
  const float smooth = wide_bias ? raw : float(bfloat(raw));
  const float speed = precise::exp(logarithm);
  decay[at] = precise::exp(-speed * smooth);
  beta[at] = bfloat(q27_prompt_probability(float(b[at])));
}

kernel void q27_prompt_qk(device const bfloat* activation [[buffer(0)]],
                          constant Q27PromptQk& p [[buffer(1)]],
                          device bfloat* result [[buffer(2)]],
                          uint2 group [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]]) {
  const uint begin = group.y * p.stride + p.offset + group.x * p.width + lane * 4;
  float values[4];
  float square = 0.0f;
  for (uint part = 0; part < 4; ++part) {
    values[part] = float(activation[begin + part]);
    square += values[part] * values[part];
  }
  const float inverse = precise::rsqrt(simd_sum(square) / float(p.width) + p.eps);
  const float scale = float(bfloat(p.scale));
  const uint output = (group.y * p.heads + group.x) * p.width + lane * 4;
  for (uint part = 0; part < 4; ++part) result[output + part] = bfloat(scale * float(bfloat(values[part] * inverse)));
}
