// Cosmos3-Edge's reasoner beside the core row projections: the text model's glue (M-RoPE, relu^2, attention) and the
// SigLIP2 image tower's (layer norms, biases, GELU, bidirectional attention over one image's patches).
#include <metal_stdlib>
using namespace metal;

// The text model's pinned geometry.
constant constexpr uint D = 2048;
constant constexpr uint QH = 16;
constant constexpr uint KVH = 8;
constant constexpr uint HD = 128;
// Rotary pairs on each M-RoPE axis (t, h, w), interleaved as Qwen3-VL's: h at j % 3 == 1, w at j % 3 == 2.
constant constexpr uint SEC_H = 20;
constant constexpr uint SEC_W = 20;

// Table row ids[r] as bf16 scale * q + bias (MLX's dequantize), 8-bit codes, one byte a thread.
[[kernel]] void cos3_embed_b8(const device uint* ids [[buffer(0)]], const device uint8_t* table [[buffer(1)]],
                              const device bfloat* scales [[buffer(2)]], const device bfloat* biases [[buffer(3)]],
                              device bfloat* out [[buffer(4)]], uint2 at [[thread_position_in_grid]]) {
  if (at.x >= D) return;
  const long id = long(ids[at.y]);
  const bfloat scale = scales[id * (D / 64) + at.x / 64];
  const bfloat bias = biases[id * (D / 64) + at.x / 64];
  out[size_t(at.y) * D + at.x] = scale * table[id * D + at.x] + bias;
}

// Image rows over their placeholders: row r takes rows[map[r]] when map[r] >= 0.
[[kernel]] void cos3_place(device bfloat* h [[buffer(0)]], const device bfloat* rows [[buffer(1)]], const device int* map [[buffer(2)]],
                           uint2 at [[thread_position_in_grid]]) {
  if (at.x >= D) return;
  const int src = map[at.y];
  if (src >= 0) h[size_t(at.y) * D + at.x] = rows[size_t(src) * D + at.x];
}

// h += r (bf16) when `add`, then x = w * bf16(h * rsqrt(mean(h^2) + eps)), 4 values a thread (Llama's RMSNorm order).
[[kernel]] void cos3_norm(device bfloat* h [[buffer(0)]], const device bfloat* r [[buffer(1)]], const device bfloat* w [[buffer(2)]],
                          constant float& eps [[buffer(3)]], constant uint& add [[buffer(4)]], device bfloat* x [[buffer(5)]],
                          uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const size_t base = size_t(row) * D + 4 * t;
  float v[4];
  float sq = 0.0f;
  for (int j = 0; j < 4; ++j) {
    bfloat hv = h[base + j];
    if (add != 0) {
      hv = hv + r[base + j];
      h[base + j] = hv;
    }
    v[j] = hv;
    sq += v[j] * v[j];
  }
  threadgroup float sums[32];
  threadgroup float inv[1];
  sq = simd_sum(sq);
  if (sg == 0) sums[lane] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) sums[sg] = sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const float all = simd_sum(sums[lane]);
    if (lane == 0) inv[0] = metal::precise::rsqrt(all / float(D) + eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int j = 0; j < 4; ++j) x[base + j] = w[4 * t + j] * static_cast<bfloat>(v[j] * inv[0]);
}

// The axis of rotary pair j (0 t, 1 h, 2 w).
inline uint cos3_axis(uint j) {
  if (j % 3 == 1 && j < 3 * SEC_H) return 1;
  if (j % 3 == 2 && j < 3 * SEC_W) return 2;
  return 0;
}

// One head of one row rotated (rotate_half pairs j and j + HD/2) at its three positions; lane l: pairs l and l + 32.
inline void cos3_rotate(const device bfloat* src, const thread int* pos, float log2_theta, uint lane, device bfloat* dst) {
  for (uint i = 0; i < 2; ++i) {
    const uint j = lane + 32 * i;
    const float inv_freq = metal::exp2(-(float(2 * j) / float(HD)) * log2_theta);
    const float angle = float(pos[cos3_axis(j)]) * inv_freq;
    const float c = metal::precise::cos(angle), s = metal::precise::sin(angle);
    const float a = src[j], b = src[j + HD / 2];
    dst[j] = static_cast<bfloat>(a * c - b * s);
    dst[j + HD / 2] = static_cast<bfloat>(b * c + a * s);
  }
}

// A simdgroup a (head, row): queries rotated into `queries`; keys rotated and values copied into the [KVH, cap, HD]
// caches at slot slots[r] (the row's place in its stream), positions pos[r] = (t, h, w).
struct cos3_qk_args { uint cap; float log2_theta; };
[[kernel]] void cos3_qk(const device bfloat* q [[buffer(0)]], const device bfloat* k [[buffer(1)]], const device bfloat* v [[buffer(2)]],
                        const device int* pos [[buffer(3)]], const device uint* slots [[buffer(4)]], constant cos3_qk_args& p [[buffer(5)]],
                        device bfloat* queries [[buffer(6)]], device bfloat* keys [[buffer(7)]], device bfloat* values [[buffer(8)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint head = tg.x, r = tg.y;
  const int at[3] = {pos[3 * r], pos[3 * r + 1], pos[3 * r + 2]};
  if (head < QH) {
    cos3_rotate(q + (size_t(r) * QH + head) * HD, at, p.log2_theta, lane, queries + (size_t(r) * QH + head) * HD);
    return;
  }
  const uint kh = head - QH;
  const size_t dst = (size_t(kh) * p.cap + slots[r]) * HD;
  cos3_rotate(k + (size_t(r) * KVH + kh) * HD, at, p.log2_theta, lane, keys + dst);
  for (uint d = lane; d < HD; d += 32) values[dst + d] = v[(size_t(r) * KVH + kh) * HD + d];
}

// x = bf16(relu(x)^2) in place.
[[kernel]] void cos3_relu2(device bfloat* x [[buffer(0)]], constant uint& count [[buffer(1)]], uint i [[thread_position_in_grid]]) {
  if (i >= count) return;
  const float v = max(float(x[i]), 0.0f);
  x[i] = static_cast<bfloat>(v * v);
}

// Attention, one simdgroup a (head, row): queries [rows, heads, HDT] (rows q_row apart), keys and values at
// key j of kv head kh: base + kh * k_head + j * k_row; a causal row r sees keys [0, first + r], else all `keys`.
// Each 32-key block: a lane's key scored, the block's softmax folded in, then every lane's dims take the block's values.
struct cos3_attn_args { uint first, keys, causal, group, q_row, k_row, k_head, o_row; float scale; };
template <int HDT>
[[kernel]] void cos3_attn(const device bfloat* q [[buffer(0)]], const device bfloat* k [[buffer(1)]], const device bfloat* v [[buffer(2)]],
                          constant cos3_attn_args& p [[buffer(3)]], device bfloat* o [[buffer(4)]],
                          uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr int NS = (HDT + 31) / 32;
  const uint head = tg.x, r = tg.y, kh = head / p.group;
  threadgroup float qs[HDT];
  for (uint d = lane; d < uint(HDT); d += 32) qs[d] = float(q[size_t(r) * p.q_row + head * HDT + d]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint n = p.causal != 0 ? p.first + r + 1 : p.keys;
  const device bfloat* kb = k + size_t(kh) * p.k_head;
  const device bfloat* vb = v + size_t(kh) * p.k_head;
  float m = -INFINITY, l = 0.0f;
  float acc[NS];
  for (int i = 0; i < NS; ++i) acc[i] = 0.0f;
  for (uint j0 = 0; j0 < n; j0 += 32) {
    const uint j = j0 + lane;
    float s = -INFINITY;
    if (j < n) {
      const device bfloat* kr = kb + size_t(j) * p.k_row;
      float dot = 0.0f;
      for (int d = 0; d < HDT; d += 4) {
        const float4 kv4 = float4(*(const device vec<bfloat, 4>*)(kr + d));
        dot += qs[d] * kv4.x + qs[d + 1] * kv4.y + qs[d + 2] * kv4.z + qs[d + 3] * kv4.w;
      }
      s = dot * p.scale;
    }
    const float mn = max(m, simd_max(s));
    const float pj = j < n ? metal::exp(s - mn) : 0.0f;
    const float alpha = m == -INFINITY ? 0.0f : metal::exp(m - mn);
    l = l * alpha + simd_sum(pj);
    for (int i = 0; i < NS; ++i) acc[i] *= alpha;
    const uint count = min(32u, n - j0);
    for (uint t = 0; t < count; ++t) {
      const float pt = simd_shuffle(pj, ushort(t));
      const device bfloat* vr = vb + size_t(j0 + t) * p.k_row;
      for (int i = 0; i < NS; ++i) {
        const uint d = lane + 32 * i;
        if (d < uint(HDT)) acc[i] += pt * float(vr[d]);
      }
    }
    m = mn;
  }
  for (int i = 0; i < NS; ++i) {
    const uint d = lane + 32 * i;
    if (d < uint(HDT)) o[size_t(r) * p.o_row + head * HDT + d] = static_cast<bfloat>(acc[i] / l);
  }
}
template [[host_name("cos3_attn_128")]] [[kernel]] decltype(cos3_attn<128>) cos3_attn<128>;
template [[host_name("cos3_attn_72")]] [[kernel]] decltype(cos3_attn<72>) cos3_attn<72>;

// h += r (bf16) when `add`, then x = LayerNorm(h) * w + b over `width` values (fp32 statistics), a threadgroup a row.
struct cos3_ln_args { uint width; float eps; uint add; };
[[kernel]] void cos3_layer_norm(device bfloat* h [[buffer(0)]], const device bfloat* r [[buffer(1)]], const device bfloat* w [[buffer(2)]],
                                const device bfloat* b [[buffer(3)]], constant cos3_ln_args& p [[buffer(4)]], device bfloat* x [[buffer(5)]],
                                uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                uint threads [[threads_per_threadgroup]]) {
  device bfloat* hr = h + size_t(row) * p.width;
  float s = 0.0f;
  for (uint i = t; i < p.width; i += threads) {
    bfloat hv = hr[i];
    if (p.add != 0) {
      hv = hv + r[size_t(row) * p.width + i];
      hr[i] = hv;
    }
    s += float(hv);
  }
  threadgroup float part[32];
  threadgroup float stat[2];
  s = simd_sum(s);
  if (lane == 0) part[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const float all = simd_sum(lane < (threads + 31) / 32 ? part[lane] : 0.0f);
    if (lane == 0) stat[0] = all / float(p.width);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float mean = stat[0];
  float q = 0.0f;
  for (uint i = t; i < p.width; i += threads) {
    const float d = float(hr[i]) - mean;
    q += d * d;
  }
  q = simd_sum(q);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) part[sg] = q;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const float all = simd_sum(lane < (threads + 31) / 32 ? part[lane] : 0.0f);
    if (lane == 0) stat[1] = metal::precise::rsqrt(all / float(p.width) + p.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = stat[1];
  for (uint i = t; i < p.width; i += threads)
    x[size_t(row) * p.width + i] = static_cast<bfloat>((float(hr[i]) - mean) * inv * float(w[i]) + float(b[i]));
}

// erf to ~1.5e-7 (Abramowitz and Stegun 7.1.26): Metal has none.
inline float cos3_erf(float x) {
  const float s = x < 0.0f ? -1.0f : 1.0f;
  const float a = fabs(x);
  const float t = 1.0f / (1.0f + 0.3275911f * a);
  const float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t * metal::exp(-a * a);
  return s * y;
}

// y[r, c] = act(bf16(y + bias[c])) in place over [rows, width]: act 0 none, 1 GELU (tanh), 2 GELU (erf).
struct cos3_bias_args { uint width, count, act; };
[[kernel]] void cos3_bias(device bfloat* y [[buffer(0)]], const device bfloat* bias [[buffer(1)]], constant cos3_bias_args& p [[buffer(2)]],
                          uint i [[thread_position_in_grid]]) {
  if (i >= p.count) return;
  const float v = float(static_cast<bfloat>(float(y[i]) + float(bias[i % p.width])));
  float out = v;
  if (p.act == 1) {
    const float inner = 0.7978845608028654f * (v + 0.044715f * v * v * v);
    out = 0.5f * v * (1.0f + metal::precise::tanh(inner));
  } else if (p.act == 2) {
    out = 0.5f * v * (1.0f + cos3_erf(v * 0.7071067811865476f));
  }
  y[i] = static_cast<bfloat>(out);
}

// a += b (bf16) over `count` values.
[[kernel]] void cos3_add(device bfloat* a [[buffer(0)]], const device bfloat* b [[buffer(1)]], constant uint& count [[buffer(2)]],
                         uint i [[thread_position_in_grid]]) {
  if (i < count) a[i] = a[i] + b[i];
}
