// Experimental K3.1 indexer, adapted from our GLM DSA; its assumed semantics still need a model reference.
#include <metal_stdlib>
using namespace metal;

constant constexpr uint K31_IH = 32, K31_ID = 128, K31_TOPK = 2048;
struct K31IndexRow { uint slot, pos, list, nkeys; };
struct K31IndexArgs {
  uint rows, slot_keys, q_stride, k_stride, w_stride, score_stride;
  float eps, key_scale, head_scale;
};

inline float k31_index_sum(float v) {
  for (ushort d = 16; d > 0; d >>= 1) v += simd_shuffle_xor(v, d);
  return v;
}

// Assumption: BF16 projection, biased population LayerNorm in FP32, one BF16 store, and no index RoPE.
[[kernel, max_total_threads_per_threadgroup(32)]] void k3_index_key(const device bfloat* proj [[buffer(0)]], const device bfloat* w [[buffer(1)]],
                         const device bfloat* b [[buffer(2)]], const device K31IndexRow* rows [[buffer(3)]],
                         constant K31IndexArgs& a [[buffer(4)]], device bfloat* cache [[buffer(5)]],
                         uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  float v[4], sum = 0.0f;
  for (uint j = 0; j < 4; ++j) v[j] = float(proj[ulong(row) * a.k_stride + lane + 32 * j]), sum += v[j];
  const float mean = k31_index_sum(sum) / float(K31_ID);
  float variance = 0.0f;
  for (uint j = 0; j < 4; ++j) variance = fma(v[j] - mean, v[j] - mean, variance);
  const float inv = precise::rsqrt(k31_index_sum(variance) / float(K31_ID) + a.eps);
  const K31IndexRow r = rows[row];
  device bfloat* out = cache + (ulong(r.slot) * a.slot_keys + r.pos) * K31_ID;
  for (uint j = 0; j < 4; ++j) {
    const uint col = lane + 32 * j;
    out[col] = bfloat(fma((v[j] - mean) * inv, float(w[col]), float(b[col])));
  }
}

// Assumption: heads in order, sequential FMA dots, inclusive causal visibility, the GLM score's two scales.
[[kernel, max_total_threads_per_threadgroup(128)]] void k3_index_scores(const device bfloat* q [[buffer(0)]], const device bfloat* weights [[buffer(1)]],
                            const device bfloat* cache [[buffer(2)]], const device K31IndexRow* rows [[buffer(3)]],
                            constant K31IndexArgs& a [[buffer(4)]], device float* scores [[buffer(5)]],
                            uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  threadgroup float query[K31_IH * K31_ID];
  threadgroup float head[K31_IH];
  const uint r = tg.y;
  const K31IndexRow row = rows[r];
  for (uint i = t; i < K31_IH * K31_ID; i += 128) query[i] = float(q[ulong(r) * a.q_stride + i]);
  if (t < K31_IH) head[t] = float(weights[ulong(r) * a.w_stride + t]) * a.head_scale;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint k = tg.x * 128 + t;
  if (k > row.pos) return;
  const device bfloat* key = cache + (ulong(row.slot) * a.slot_keys + k) * K31_ID;
  float score = 0.0f;
  for (uint h = 0; h < K31_IH; ++h) {
    float dot = 0.0f;
    for (uint col = 0; col < K31_ID; ++col) dot = fma(query[h * K31_ID + col], float(key[col]), dot);
    const float scaled = dot * a.key_scale;
    const float relu = isnan(scaled) ? scaled : max(scaled, 0.0f);
    score = fma(head[h], relu, score);
  }
  scores[ulong(r) * a.score_stride + k] = score;
}

// Experimental total order: NaNs last, equal signed zeros tied, other numeric ties to the lower position.
inline uint k31_index_order(float v) {
  if (isnan(v)) return 0u;
  const uint u = v == 0.0f ? 0u : as_type<uint>(v);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Our GLM radix selection with an explicit experimental exceptional-value order; selected keys remain chronological.
[[kernel, max_total_threads_per_threadgroup(1024)]] void k3_index_topk(const device float* score [[buffer(0)]], device K31IndexRow* rows [[buffer(1)]],
                          constant K31IndexArgs& a [[buffer(2)]], device uint* list [[buffer(3)]],
                          uint r [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup atomic_uint hist[256];
  threadgroup uint found[2], sums[2][32];
  const K31IndexRow row = rows[r];
  const uint n = row.pos + 1;
  device uint* out = list + row.list;
  const device float* sc = score + ulong(r) * a.score_stride;
  if (n <= K31_TOPK) {
    for (uint i = t; i < n; i += 1024) out[i] = i;
    if (t == 0) rows[r].nkeys = n;
    return;
  }
  uint prefix = 0, mask = 0, need = K31_TOPK;
  for (int shift = 24; shift >= 0; shift -= 8) {
    if (t < 256) atomic_store_explicit(&hist[t], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = t; i < n; i += 1024) {
      const uint k = k31_index_order(sc[i]);
      if ((k & mask) == prefix) atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
      uint above = 0, b = 255;
      for (;; --b) {
        const uint count = atomic_load_explicit(&hist[b], memory_order_relaxed);
        if (above + count >= need || b == 0) break;
        above += count;
      }
      found[0] = b;
      found[1] = need - above;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    prefix |= found[0] << shift;
    mask |= 255u << shift;
    need = found[1];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  uint gt_seen = 0, eq_seen = 0;
  for (uint base = 0; base < n; base += 1024) {
    const uint i = base + t, k = i < n ? k31_index_order(sc[i]) : 0u;
    const uint gt = i < n && k > prefix, eq = i < n && k == prefix;
    const uint gx = simd_prefix_exclusive_sum(gt), ex = simd_prefix_exclusive_sum(eq);
    if (lane == 31) sums[0][sg] = gx + gt, sums[1][sg] = ex + eq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint gb = 0, eb = 0, gall = 0, eall = 0;
    for (uint g = 0; g < 32; ++g) {
      if (g < sg) gb += sums[0][g], eb += sums[1][g];
      gall += sums[0][g], eall += sums[1][g];
    }
    const uint eq_before = eq_seen + eb + ex;
    if (gt || (eq && eq_before < need)) out[gt_seen + gb + gx + min(eq_before, need)] = i;
    gt_seen += gall;
    eq_seen += eall;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (t == 0) rows[r].nkeys = K31_TOPK;
}
