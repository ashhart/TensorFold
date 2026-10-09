#include <metal_stdlib>
using namespace metal;

// bf16 token-major Q/K/V; each row and head runs a fixed-order online fp32 softmax; the mask decides visibility.
struct tf_context_args { uint rows, keys, mask_stride, heads, scale_bits; };

template <bool masked>
inline void tf_context_row(const device bfloat* Q, const device bfloat* K, const device bfloat* V,
    const device uchar* mask, device TF_OUT* O, uint row, uint head, uint keys, uint mask_stride,
    float scale, uint lane) {
  const uint kh = head / (TF_QH / TF_KVH);
  const size_t qi = ((size_t)row * TF_QH + head) * TF_D;
  float out[TF_D / 32];
  for (uint d = 0; d < TF_D / 32; ++d) out[d] = 0.0f;
  float maximum = -INFINITY, total = 0.0f;
  for (uint key = 0; key < keys; ++key) {
    if (masked && mask[(size_t)row * mask_stride + key] == 0) continue;
    const size_t ki = ((size_t)key * TF_KVH + kh) * TF_D;
    float dot = 0.0f;
    for (uint d = lane; d < TF_D; d += 32) dot = fma(float(Q[qi+d]), float(K[ki+d]), dot);
    const float score = simd_sum(dot) * scale;
    const float next = max(maximum, score);
    const float old_weight = maximum == -INFINITY ? 0.0f : precise::exp(maximum-next);
    const float weight = precise::exp(score-next);
    total = fma(total, old_weight, weight);
    for (uint d = 0; d < TF_D / 32; ++d)
      out[d] = fma(out[d], old_weight, weight * float(V[ki + lane + d*32]));
    maximum = next;
  }
  for (uint d = 0; d < TF_D / 32; ++d)
    O[qi + lane + d*32] = TF_OUT(total == 0.0f ? 0.0f : out[d] / total);
}

[[kernel]] void tf_shared_lane_attention(const device bfloat* Q [[buffer(0)]],
    const device bfloat* K [[buffer(1)]], const device bfloat* V [[buffer(2)]],
    const device uchar* mask [[buffer(3)]], constant tf_context_args& args [[buffer(4)]],
    device TF_OUT* O [[buffer(5)]], uint lane [[thread_index_in_simdgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  if (tg.y < args.rows && tg.x < TF_QH)
    tf_context_row<true>(Q, K, V, mask, O, tg.y, tg.x, args.keys, args.mask_stride, as_type<float>(args.scale_bits), lane);
}

// Ordinary one-sequence decode over all keys: the same fp32 order, with no mask or lane indexing.
[[kernel]] void tf_core_decode_attention(const device bfloat* Q [[buffer(0)]],
    const device bfloat* K [[buffer(1)]], const device bfloat* V [[buffer(2)]],
    constant tf_context_args& args [[buffer(3)]], device TF_OUT* O [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]], uint head [[threadgroup_position_in_grid]]) {
  if (head < TF_QH)
    tf_context_row<false>(Q, K, V, (const device uchar*)nullptr, O, 0, head, args.keys, 0, as_type<float>(args.scale_bits), lane);
}

// Position is supplied per row. Caller-supplied inverse frequencies preserve the model's RoPE convention.
[[kernel]] void tf_lane_rope(const device bfloat* X [[buffer(0)]], const device uint* positions [[buffer(1)]],
    const device float* frequencies [[buffer(2)]], constant tf_context_args& args [[buffer(3)]],
    device bfloat* Y [[buffer(4)]], uint index [[thread_position_in_grid]]) {
  if ((size_t)index >= (size_t)args.rows * args.heads * TF_D) return;
  const uint d = index % TF_D;
#if TF_RD > 0
  if (d >= TF_RD) { Y[index] = X[index]; return; }
#if TF_INTERLEAVED
  const uint pair = d / 2;
  const uint first = pair * 2, second = first + 1;
  const bool high = (d & 1) != 0;
#else
  const uint pair = d % (TF_RD / 2);
  const uint first = pair, second = first + TF_RD / 2;
  const bool high = d >= TF_RD / 2;
#endif
  const uint row = index / (args.heads * TF_D);
  const size_t base = index - d;
  const float angle = float(positions[row]) * frequencies[pair];
  const float c = precise::cos(angle), s = precise::sin(angle);
  const float x = float(X[base + first]), y = float(X[base + second]);
  Y[index] = bfloat(high ? fma(x, s, y*c) : fma(x, c, -y*s));
#else
  Y[index] = X[index];
#endif
}

// Only newly produced rows are written. Prompt and previous-round cache slots are never copied or rewritten.
[[kernel]] void tf_lane_append_kv(const device bfloat* K [[buffer(0)]], const device bfloat* V [[buffer(1)]],
    constant uint* slots [[buffer(2)]], constant tf_context_args& args [[buffer(3)]],
    device bfloat* cacheK [[buffer(4)]], device bfloat* cacheV [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
  const uint width = TF_KVH * TF_D;
  if ((size_t)index >= (size_t)args.rows * width) return;
  const uint row = index / width, column = index % width;
  const size_t target = (size_t)slots[row] * width + column;
  cacheK[target] = K[index]; cacheV[target] = V[index];
}
