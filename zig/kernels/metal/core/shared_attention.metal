#include <metal_stdlib>
using namespace metal;

// bf16 token-major Q/K/V; each row and head runs a fixed-order online fp32 softmax; the mask decides visibility.
struct tf_context_args { uint rows, keys, mask_stride, heads, scale_bits; };

// Keys per tile: a tile's scores and values load together, then its weights and sums apply in key order.
#define TF_TILE 4

template <bool masked>
inline void tf_context_row(const device bfloat* Q, const device bfloat* K, const device bfloat* V,
    const device uchar* mask, device TF_OUT* O, uint row, uint head, uint keys, uint mask_stride,
    float scale, uint lane) {
  const uint kh = head / (TF_QH / TF_KVH);
  const size_t qi = ((size_t)row * TF_QH + head) * TF_D;
  float q[TF_D / 32];
  for (uint d = 0; d < TF_D / 32; ++d) q[d] = float(Q[qi + lane + d*32]);
  float out[TF_D / 32];
  for (uint d = 0; d < TF_D / 32; ++d) out[d] = 0.0f;
  float maximum = -INFINITY, total = 0.0f;
  for (uint first = 0; first < keys; first += TF_TILE) {
    bool live[TF_TILE];
    float score[TF_TILE];
    float value[TF_TILE][TF_D / 32];
    for (uint t = 0; t < TF_TILE; ++t) {
      const uint key = first + t;
      live[t] = key < keys;
      if (masked && live[t]) live[t] = mask[(size_t)row * mask_stride + key] != 0;
      score[t] = 0.0f;
      if (live[t]) {
        const size_t ki = ((size_t)key * TF_KVH + kh) * TF_D;
        float dot = 0.0f;
        for (uint d = 0; d < TF_D / 32; ++d) dot = fma(q[d], float(K[ki + lane + d*32]), dot);
        score[t] = simd_sum(dot) * scale;
        for (uint d = 0; d < TF_D / 32; ++d) value[t][d] = float(V[ki + lane + d*32]);
      }
    }
    // the maximum after each key and both weights come from the scores alone
    float old_weight[TF_TILE], weight[TF_TILE];
    float next = maximum;
    for (uint t = 0; t < TF_TILE; ++t) {
      old_weight[t] = 0.0f;
      weight[t] = 0.0f;
      if (live[t]) {
        const float before = next;
        next = max(next, score[t]);
        old_weight[t] = before == -INFINITY ? 0.0f : precise::exp(before - next);
        weight[t] = precise::exp(score[t] - next);
      }
    }
    maximum = next;
    for (uint t = 0; t < TF_TILE; ++t) {
      if (live[t]) {
        total = fma(total, old_weight[t], weight[t]);
        for (uint d = 0; d < TF_D / 32; ++d)
          out[d] = fma(out[d], old_weight[t], weight[t] * value[t][d]);
      }
    }
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

#define TF_CHUNK 256
#define TF_G (TF_QH / TF_KVH)

// Split over 256-key chunks: one simdgroup takes a KV head's query heads for one row, reading each key once for them.
[[kernel]] void tf_shared_lane_attention_split(const device bfloat* Q [[buffer(0)]],
    const device bfloat* K [[buffer(1)]], const device bfloat* V [[buffer(2)]],
    const device uchar* mask [[buffer(3)]], constant tf_context_args& args [[buffer(4)]],
    device float* part [[buffer(5)]], uint lane [[thread_index_in_simdgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint kh = tg.x, row = tg.y, chunk = tg.z, chunks = (args.keys + TF_CHUNK - 1) / TF_CHUNK;
  if (kh >= TF_KVH || row >= args.rows || chunk >= chunks) return;
  const float scale = as_type<float>(args.scale_bits);
  float q[TF_G][TF_D / 32], out[TF_G][TF_D / 32], maximum[TF_G], total[TF_G];
  for (uint g = 0; g < TF_G; ++g) {
    const size_t qi = ((size_t)row * TF_QH + kh * TF_G + g) * TF_D;
    for (uint d = 0; d < TF_D / 32; ++d) { q[g][d] = float(Q[qi + lane + d * 32]); out[g][d] = 0.0f; }
    maximum[g] = -INFINITY;
    total[g] = 0.0f;
  }
  const uint last = min(args.keys, (chunk + 1) * TF_CHUNK);
  for (uint key = chunk * TF_CHUNK; key < last; ++key) {
    if (mask[(size_t)row * args.mask_stride + key] == 0) continue;
    const size_t ki = ((size_t)key * TF_KVH + kh) * TF_D;
    float k[TF_D / 32], v[TF_D / 32];
    for (uint d = 0; d < TF_D / 32; ++d) { k[d] = float(K[ki + lane + d * 32]); v[d] = float(V[ki + lane + d * 32]); }
    for (uint g = 0; g < TF_G; ++g) {
      float dot = 0.0f;
      for (uint d = 0; d < TF_D / 32; ++d) dot = fma(q[g][d], k[d], dot);
      const float score = simd_sum(dot) * scale;
      const float next = max(maximum[g], score);
      const float old_weight = maximum[g] == -INFINITY ? 0.0f : exp(maximum[g] - next);
      const float weight = exp(score - next);
      total[g] = fma(total[g], old_weight, weight);
      for (uint d = 0; d < TF_D / 32; ++d) out[g][d] = fma(out[g][d], old_weight, weight * v[d]);
      maximum[g] = next;
    }
  }
  // partials [row][kv head][chunk][query head][D + 2]: outputs, then the chunk's maximum and total
  for (uint g = 0; g < TF_G; ++g) {
    device float* p = part + ((((size_t)row * TF_KVH + kh) * chunks + chunk) * TF_G + g) * (TF_D + 2);
    for (uint d = 0; d < TF_D / 32; ++d) p[lane + d * 32] = out[g][d];
    if (lane == 0) { p[TF_D] = maximum[g]; p[TF_D + 1] = total[g]; }
  }
}

// One query head and row: its chunks' partials rescaled to the largest maximum and added in chunk order.
[[kernel]] void tf_shared_lane_attention_merge(const device float* part [[buffer(0)]],
    constant tf_context_args& args [[buffer(1)]], device TF_OUT* O [[buffer(2)]],
    uint lane [[thread_index_in_simdgroup]], uint2 tg [[threadgroup_position_in_grid]]) {
  const uint head = tg.x, row = tg.y, chunks = (args.keys + TF_CHUNK - 1) / TF_CHUNK;
  if (head >= TF_QH || row >= args.rows) return;
  const uint kh = head / TF_G, g = head % TF_G;
  const device float* first = part + (((size_t)row * TF_KVH + kh) * chunks * TF_G + g) * (TF_D + 2);
  const size_t step = (size_t)TF_G * (TF_D + 2);
  float maximum = -INFINITY;
  for (uint c = 0; c < chunks; ++c) maximum = max(maximum, first[c * step + TF_D]);
  float out[TF_D / 32];
  for (uint d = 0; d < TF_D / 32; ++d) out[d] = 0.0f;
  float total = 0.0f;
  for (uint c = 0; c < chunks; ++c) {
    const device float* p = first + c * step;
    if (p[TF_D] == -INFINITY) continue;
    const float w = exp(p[TF_D] - maximum);
    total = fma(p[TF_D + 1], w, total);
    for (uint d = 0; d < TF_D / 32; ++d) out[d] = fma(p[lane + d * 32], w, out[d]);
  }
  const size_t qi = ((size_t)row * TF_QH + head) * TF_D;
  for (uint d = 0; d < TF_D / 32; ++d) O[qi + lane + d * 32] = TF_OUT(total == 0.0f ? 0.0f : out[d] / total);
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
