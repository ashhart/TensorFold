// Inkling 2 Mini DFlash 2 drafter kernels (incoai/Inkling-2-Mini-DFlash2, Qwen3 + dynamic conv + selector).
#include <metal_stdlib>
using namespace metal;
#pragma METAL fp contract(off)

constant uint DF2_QH = 32;
constant uint DF2_KVH = 8;
constant uint DF2_G = 4;
constant uint DF2_D = 128;
constant uint DF2_CHUNK = 128;
constant uint DF2_PART = 132;
constant uint DF2_MAX_K = 16;
constant uint DF2_TOPK_THREADS = 512;
constant uint DF2_TOPK_SIMDS = 16;
constant uint DF2_TOPK_PICKS = 64;

struct Df2EmbedArgs { uint rows, width, use_norm; float eps; };
struct Df2NormArgs { uint rows, width; float eps; uint add; };
struct Df2ConvArgs { uint rows, width, block, groups, side; };
struct Df2CtxArgs { uint rows, pos0, ring; float eps, theta; };
struct Df2BlockQkvArgs { uint rows, block, pos0; float eps, theta; };
struct Df2AttendArgs { uint rows, anchor, from, lo, retained, keys, window, ring, chunks; float scale; };
struct Df2ActArgs { uint count; };
struct Df2TopkArgs { uint rows, vocab, k, stride; };
struct Df2WalkArgs { uint slots, rank, top, vocab; float softcap; };

// Embed block token IDs into FP32 residual rows, optionally applying target embed_norm.
[[kernel, max_total_threads_per_threadgroup(256)]] void tf_ink2_df2_embed(
    const device uint* ids [[buffer(0)]], const device bfloat* table [[buffer(1)]],
    const device bfloat* weight [[buffer(2)]], constant Df2EmbedArgs& a [[buffer(3)]],
    device float* out [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  if (row >= a.rows) return;
  const device bfloat* src = table + (size_t)ids[row] * a.width;
  device float* dst = out + (size_t)row * a.width;
  if (!a.use_norm) {
    for (uint i = t; i < a.width; i += 256) dst[i] = float(src[i]);
    return;
  }
  threadgroup float part[8];
  float sum = 0.0f;
  for (uint i = t; i < a.width; i += 256) {
    const float v = float(src[i]);
    sum = fma(v, v, sum);
  }
  const float sg_sum = simd_sum(sum);
  if (lane == 0) part[sg] = sg_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint s = 0; s < 8; ++s) total += part[s];
  const float inv = precise::rsqrt(total / float(a.width) + a.eps);
  for (uint i = t; i < a.width; i += 256) dst[i] = float(bfloat(float(src[i]) * inv * float(weight[i])));
}

// Qwen3 RMSNorm over FP32 residual rows, with optional BF16-rounded residual addition into R.
[[kernel, max_total_threads_per_threadgroup(256)]] void tf_ink2_df2_norm(
    device float* R [[buffer(0)]], const device bfloat* X [[buffer(1)]],
    const device bfloat* w [[buffer(2)]], constant Df2NormArgs& a [[buffer(3)]],
    device bfloat* out [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  if (row >= a.rows) return;
  threadgroup float part[8];
  device float* r = R + (size_t)row * a.width;
  const device bfloat* x = X + (size_t)row * a.width;
  float sum = 0.0f;
  for (uint i = t; i < a.width; i += 256) {
    float v = r[i];
    if (a.add) {
      v = float(bfloat(v + float(x[i])));
      r[i] = v;
    }
    sum = fma(v, v, sum);
  }
  const float sg_sum = simd_sum(sum);
  if (lane == 0) part[sg] = sg_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint s = 0; s < 8; ++s) total += part[s];
  const float inv = precise::rsqrt(total / float(a.width) + a.eps);
  device bfloat* y = out + (size_t)row * a.width;
  for (uint i = t; i < a.width; i += 256) y[i] = bfloat(float(w[i]) * float(bfloat(r[i] * inv)));
}

// Qwen3 RMSNorm over BF16 rows (fc output into hidden_norm).
[[kernel, max_total_threads_per_threadgroup(256)]] void tf_ink2_df2_norm16(
    const device bfloat* X [[buffer(0)]], const device bfloat* w [[buffer(1)]],
    constant Df2NormArgs& a [[buffer(2)]], device bfloat* out [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  if (row >= a.rows) return;
  threadgroup float part[8];
  const device bfloat* x = X + (size_t)row * a.width;
  float sum = 0.0f;
  for (uint i = t; i < a.width; i += 256) {
    const float v = float(x[i]);
    sum = fma(v, v, sum);
  }
  const float sg_sum = simd_sum(sum);
  if (lane == 0) part[sg] = sg_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint s = 0; s < 8; ++s) total += part[s];
  const float inv = precise::rsqrt(total / float(a.width) + a.eps);
  device bfloat* y = out + (size_t)row * a.width;
  for (uint i = t; i < a.width; i += 256) y[i] = bfloat(float(w[i]) * float(bfloat(float(x[i]) * inv)));
}

// Two-tap grouped dynamic block convolution: side 0 prepare, side 1 finish.
[[kernel, max_total_threads_per_threadgroup(256)]] void tf_ink2_df2_conv(
    const device bfloat* X [[buffer(0)]], const device bfloat* D [[buffer(1)]],
    const device bfloat* base [[buffer(2)]], constant Df2ConvArgs& a [[buffer(3)]],
    device bfloat* Y [[buffer(4)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.width || p.y >= a.rows) return;
  const uint ch = p.x, r = p.y;
  const uint g = ch / (a.width / a.groups);
  const device bfloat* d = D + (size_t)r * 4 * a.groups + a.side * 2 * a.groups;
  const float c0 = float(bfloat(float(base[(a.side * 2) * a.width + ch]) + float(d[g])));
  const float c1 = float(bfloat(float(base[(a.side * 2 + 1) * a.width + ch]) + float(d[a.groups + g])));
  float y = float(bfloat(c0 * float(X[(size_t)r * a.width + ch])));
  if (r % a.block >= 1) y = float(bfloat(y + float(bfloat(c1 * float(X[(size_t)(r - 1) * a.width + ch])))));
  Y[(size_t)r * a.width + ch] = bfloat(y);
}

inline void df2_head_norm_rope(const device bfloat* src, const device bfloat* w, uint pos, float eps, float theta,
                               device bfloat* dst, threadgroup float* nrm, uint lane) {
  float v[4];
  float sum = 0.0f;
  for (uint j = 0; j < 4; ++j) {
    v[j] = float(src[lane + 32 * j]);
    sum = fma(v[j], v[j], sum);
  }
  const float inv = precise::rsqrt(simd_sum(sum) / 128.0f + eps);
  for (uint j = 0; j < 4; ++j) nrm[lane + 32 * j] = float(bfloat(float(w[lane + 32 * j]) * float(bfloat(v[j] * inv))));
  simdgroup_barrier(mem_flags::mem_threadgroup);
  for (uint j = 0; j < 4; ++j) {
    const uint d = lane + 32 * j;
    const uint i = d % 64;
    const float angle = float(pos) * (1.0f / precise::pow(theta, float(2 * i) / 128.0f));
    const float c = float(bfloat(precise::cos(angle)));
    const float s = float(bfloat(precise::sin(angle)));
    const float rot = d < 64 ? -nrm[d + 64] : nrm[d - 64];
    dst[d] = bfloat(float(bfloat(nrm[d] * c)) + float(bfloat(rot * s)));
  }
}

// Normalize and rotate context keys into the layer's ring and copy context values.
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_ink2_df2_ctx_kv(
    const device bfloat* K_raw [[buffer(0)]], const device bfloat* V_raw [[buffer(1)]],
    const device bfloat* kw [[buffer(2)]], constant Df2CtxArgs& a [[buffer(3)]],
    device bfloat* K_ring [[buffer(4)]], device bfloat* V_ring [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float nrm[128];
  const uint head = tg.x, row = tg.y;
  if (head >= DF2_KVH || row >= a.rows) return;
  const uint pos = a.pos0 + row;
  const size_t slot = pos % a.ring;
  const device bfloat* ks = K_raw + ((size_t)row * DF2_KVH + head) * DF2_D;
  device bfloat* kd = K_ring + (slot * DF2_KVH + head) * DF2_D;
  df2_head_norm_rope(ks, kw, pos, a.eps, a.theta, kd, nrm, lane);
  const device bfloat* vs = V_raw + ((size_t)row * DF2_KVH + head) * DF2_D;
  device bfloat* vd = V_ring + (slot * DF2_KVH + head) * DF2_D;
  for (uint j = 0; j < 4; ++j) vd[lane + 32 * j] = vs[lane + 32 * j];
}

// Normalize and rotate block Q (32 heads) and K (8 heads), copying block V (8 heads).
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_ink2_df2_block_qkv(
    const device bfloat* Q_raw [[buffer(0)]], const device bfloat* K_raw [[buffer(1)]],
    const device bfloat* V_raw [[buffer(2)]], const device bfloat* qw [[buffer(3)]],
    const device bfloat* kw [[buffer(4)]], constant Df2BlockQkvArgs& a [[buffer(5)]],
    device bfloat* Q [[buffer(6)]], device bfloat* KB [[buffer(7)]], device bfloat* VB [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float nrm[128];
  const uint unit = tg.x, row = tg.y;
  if (unit >= DF2_QH + DF2_KVH || row >= a.rows) return;
  const uint pos = a.pos0 + (row % a.block);
  if (unit < DF2_QH) {
    const size_t off = ((size_t)row * DF2_QH + unit) * DF2_D;
    df2_head_norm_rope(Q_raw + off, qw, pos, a.eps, a.theta, Q + off, nrm, lane);
    return;
  }
  const uint head = unit - DF2_QH;
  const size_t off = ((size_t)row * DF2_KVH + head) * DF2_D;
  df2_head_norm_rope(K_raw + off, kw, pos, a.eps, a.theta, KB + off, nrm, lane);
  const device bfloat* vs = V_raw + off;
  device bfloat* vd = VB + off;
  for (uint j = 0; j < 4; ++j) vd[lane + 32 * j] = vs[lane + 32 * j];
}

inline float df2_max(float x, float y) { return isnan(x) || isnan(y) ? NAN : max(x, y); }

// Sliding-window bidirectional GQA 4:1 attention over context ring + block keys, 128 keys per chunk.
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_ink2_df2_attend(
    const device bfloat* Q [[buffer(0)]], const device bfloat* KB [[buffer(1)]],
    const device bfloat* VB [[buffer(2)]], const device bfloat* K_ring [[buffer(3)]],
    const device bfloat* V_ring [[buffer(4)]], constant Df2AttendArgs& a [[buffer(5)]],
    device float* part [[buffer(6)]], uint lane [[thread_index_in_simdgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint kh = tg.x, row = tg.y, chunk = tg.z;
  if (kh >= DF2_KVH || row >= a.rows || chunk * DF2_CHUNK >= a.keys) return;
  const uint reach = a.anchor + row + 1;
  const uint row_lo = max(a.from, reach > a.window ? reach - a.window : 0u);
  const device bfloat4* q = (const device bfloat4*)(Q + ((size_t)row * DF2_QH + kh * DF2_G) * DF2_D);
  float4 out[DF2_G];
  float maximum[DF2_G], total[DF2_G];
  for (uint g = 0; g < DF2_G; ++g) {
    maximum[g] = -INFINITY;
    total[g] = 0.0f;
    out[g] = float4(0.0f);
  }
  const uint first = chunk * DF2_CHUNK;
  const uint last = min(a.keys, first + DF2_CHUNK);
  for (uint base = first; base < last; base += 32) {
    const uint key = base + lane;
    const bool in_range = key < last;
    const bool is_ctx = key < a.retained;
    const bool live = in_range && (!is_ctx || (a.lo + key) >= row_lo);
    float dot[DF2_G] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint row_off = 0u;
    if (in_range) {
      const size_t krow = is_ctx ? (size_t)((a.lo + key) % a.ring) : (size_t)(key - a.retained);
      row_off = uint((krow * DF2_KVH + kh) * DF2_D);
      const device bfloat4* k = (const device bfloat4*)((is_ctx ? K_ring : KB) + row_off);
      for (uint d = 0; d < 32; ++d) {
        const float4 kv = float4(k[d]);
        for (uint g = 0; g < DF2_G; ++g) {
          const float4 qv = float4(q[g * 32 + d]);
          dot[g] = fma(qv.w, kv.w, fma(qv.z, kv.z, fma(qv.y, kv.y, fma(qv.x, kv.x, dot[g]))));
        }
      }
    }
    float p[DF2_G];
    for (uint g = 0; g < DF2_G; ++g) {
      const float score = live ? dot[g] * a.scale : -INFINITY;
      const bool nan = simd_any(isnan(score)) || isnan(maximum[g]);
      const float next = nan ? NAN : max(maximum[g], simd_max(score));
      const float old_w = maximum[g] == -INFINITY ? 0.0f : precise::exp(maximum[g] - next);
      p[g] = live ? precise::exp(score - next) : 0.0f;
      total[g] = fma(total[g], old_w, simd_sum(p[g]));
      out[g] *= old_w;
      maximum[g] = next;
    }
    const uint count = min(32u, last - base);
    for (uint j = 0; j < count; ++j) {
      const bool seen = simd_shuffle(uint(live), ushort(j)) != 0;
      const uint off_j = simd_shuffle(row_off, ushort(j));
      const device bfloat* vsrc = (base + j < a.retained ? V_ring : VB) + off_j;
      const float4 x = seen ? float4(((const device bfloat4*)vsrc)[lane]) : float4(0.0f);
      for (uint g = 0; g < DF2_G; ++g) out[g] = fma(float4(simd_shuffle(p[g], ushort(j))), x, out[g]);
    }
  }
  for (uint g = 0; g < DF2_G; ++g) {
    device float* o = part + ((((size_t)row * DF2_KVH + kh) * a.chunks + chunk) * DF2_G + g) * DF2_PART;
    ((device float4*)o)[lane] = out[g];
    if (lane == 0) { o[DF2_D] = maximum[g]; o[DF2_D + 1] = total[g]; }
  }
}

// Merge attention chunk partials for each (head, row) into BF16 output [rows, 4096].
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_ink2_df2_attend_merge(
    const device float* part [[buffer(0)]], constant Df2AttendArgs& a [[buffer(1)]],
    device bfloat* O [[buffer(2)]], uint lane [[thread_index_in_simdgroup]],
    uint2 tg [[threadgroup_position_in_grid]]) {
  const uint head = tg.x, row = tg.y;
  if (head >= DF2_QH || row >= a.rows) return;
  const uint kh = head / DF2_G, g = head % DF2_G;
  const device float* first = part + (((size_t)row * DF2_KVH + kh) * a.chunks * DF2_G + g) * DF2_PART;
  const size_t step = (size_t)DF2_G * DF2_PART;
  float maximum = -INFINITY;
  for (uint c = 0; c < a.chunks; ++c) maximum = df2_max(maximum, first[c * step + DF2_D]);
  float4 out = float4(0.0f);
  float total = 0.0f;
  for (uint c = 0; c < a.chunks; ++c) {
    const device float* p = first + c * step;
    if (p[DF2_D] == -INFINITY) continue;
    const float w = precise::exp(p[DF2_D] - maximum);
    total = fma(p[DF2_D + 1], w, total);
    out = fma(((const device float4*)p)[lane], float4(w), out);
  }
  const size_t qi = ((size_t)row * DF2_QH + head) * DF2_D;
  ((device bfloat4*)(O + qi))[lane] = bfloat4(total == 0.0f ? float4(0.0f) : out / total);
}

// Qwen3 SwiGLU over separate BF16 gate and up projections: bf16(bf16(silu(g)) * u).
[[kernel, max_total_threads_per_threadgroup(256)]] void tf_ink2_df2_silu_mul(
    const device bfloat* gate [[buffer(0)]], const device bfloat* up [[buffer(1)]],
    constant Df2ActArgs& a [[buffer(2)]], device bfloat* out [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= a.count) return;
  const float g = float(gate[i]);
  const float e = precise::exp(-fabs(g)), y = 1.0f / (1.0f + e);
  const float s = g < 0.0f ? e * y : y;
  out[i] = bfloat(float(bfloat(g * s)) * float(up[i]));
}

inline uint df2_key(ushort raw) {
  const uint bits = (raw & 0x7fffu) == 0 ? 0u : uint(raw);
  return (bits & 0x8000u) != 0 ? (~bits & 0xffffu) : (bits | 0x8000u);
}
inline bool df2_before(uint key, uint id, uint ok, uint oid) {
  return key > ok || (key == ok && id < oid);
}
inline void df2_bins(threadgroup atomic_uint* hist, threadgroup uint* bins, uint t) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < 256) {
    uint n = 0;
    for (uint s = 0; s < DF2_TOPK_SIMDS; ++s)
      n += (atomic_load_explicit(&hist[s * 128 + t / 2], memory_order_relaxed) >> (16 * (t & 1))) & 0xffffu;
    bins[t] = n;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
inline uint2 df2_cut(const threadgroup uint* bins, uint above, uint k) {
  uint cum = above;
  for (int b = 255; b >= 0; --b) {
    if (cum + bins[b] >= k) return uint2(uint(b), cum);
    cum += bins[b];
  }
  return uint2(0, cum);
}

// One-pass radix histogram top-k (k <= 16) over valid_vocab BF16 logits per row.
[[kernel, max_total_threads_per_threadgroup(DF2_TOPK_THREADS)]] void tf_ink2_df2_topk16(
    const device ushort* logits [[buffer(0)]], constant Df2TopkArgs& p [[buffer(1)]],
    device float* vals_out [[buffer(2)]], device uint* ids_out [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  if (row >= p.rows) return;
  threadgroup atomic_uint hist[DF2_TOPK_SIMDS * 128];
  threadgroup uint bins[256], keys[DF2_TOPK_PICKS], ids[DF2_TOPK_PICKS], scores[DF2_TOPK_PICKS], counts[DF2_TOPK_SIMDS];
  threadgroup atomic_uint flags, picked;
  threadgroup uint cut[4];
  const device ushort* x = logits + (size_t)row * p.stride;
  for (uint i = t; i < DF2_TOPK_SIMDS * 128; i += DF2_TOPK_THREADS)
    atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
  if (t == 0) {
    atomic_store_explicit(&flags, 0u, memory_order_relaxed);
    atomic_store_explicit(&picked, 0u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint mask = 0;
  for (uint tok = t; tok < p.vocab; tok += DF2_TOPK_THREADS) {
    const ushort raw = x[tok];
    if ((raw & 0x7f80u) == 0x7f80u) {
      mask |= 1u;
      continue;
    }
    const uint b = df2_key(raw) >> 8;
    atomic_fetch_add_explicit(&hist[sg * 128 + b / 2], 1u << (16 * (b & 1)), memory_order_relaxed);
  }
  mask = simd_or(mask);
  if (lane == 0 && mask != 0) atomic_fetch_or_explicit(&flags, mask, memory_order_relaxed);
  df2_bins(hist, bins, t);
  if (atomic_load_explicit(&flags, memory_order_relaxed) != 0) {
    if (t < p.k) {
      vals_out[(size_t)row * p.k + t] = -INFINITY;
      ids_out[(size_t)row * p.k + t] = 0xffffffffu;
    }
    return;
  }
  if (t == 0) {
    const uint2 h = df2_cut(bins, 0, p.k);
    cut[0] = h.x;
    cut[1] = h.y;
  }
  for (uint i = t; i < DF2_TOPK_SIMDS * 128; i += DF2_TOPK_THREADS)
    atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint high = cut[0];
  for (uint tok = t; tok < p.vocab; tok += DF2_TOPK_THREADS) {
    const uint key = df2_key(x[tok]);
    if ((key >> 8) != high) continue;
    const uint b = key & 0xffu;
    atomic_fetch_add_explicit(&hist[sg * 128 + b / 2], 1u << (16 * (b & 1)), memory_order_relaxed);
  }
  df2_bins(hist, bins, t);
  if (t == 0) {
    const uint2 l = df2_cut(bins, cut[1], p.k);
    cut[2] = (high << 8) | l.x;
    cut[3] = l.y + bins[l.x];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint kth = cut[2];
  if (cut[3] <= DF2_TOPK_PICKS) {
    for (uint tok = t; tok < p.vocab; tok += DF2_TOPK_THREADS) {
      const ushort raw = x[tok];
      const uint key = df2_key(raw);
      if (key < kth) continue;
      const uint at = atomic_fetch_add_explicit(&picked, 1u, memory_order_relaxed);
      keys[at] = key;
      ids[at] = tok;
      scores[at] = uint(raw) << 16;
    }
  } else {
    uint taken = 0;
    for (uint first = 0; first < p.vocab; first += DF2_TOPK_THREADS) {
      const uint tok = first + t;
      const ushort raw = tok < p.vocab ? x[tok] : ushort(0x7fc0);
      const uint key = (raw & 0x7f80u) == 0x7f80u ? 0u : df2_key(raw);
      const uint equal = key == kth ? 1u : 0u;
      if (key > kth) {
        const uint at = atomic_fetch_add_explicit(&picked, 1u, memory_order_relaxed);
        keys[at] = key;
        ids[at] = tok;
        scores[at] = uint(raw) << 16;
      }
      const uint before = simd_prefix_exclusive_sum(equal);
      if (lane == 31) counts[sg] = before + equal;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      uint rank = taken + before;
      for (uint s = 0; s < sg; ++s) rank += counts[s];
      uint all = 0;
      for (uint s = 0; s < DF2_TOPK_SIMDS; ++s) all += counts[s];
      if (equal != 0 && rank < DF2_TOPK_PICKS - DF2_MAX_K) {
        keys[DF2_TOPK_PICKS - 1 - rank] = key;
        ids[DF2_TOPK_PICKS - 1 - rank] = tok;
        scores[DF2_TOPK_PICKS - 1 - rank] = uint(raw) << 16;
      }
      taken += all;
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t == 0) {
      const uint above = atomic_load_explicit(&picked, memory_order_relaxed);
      const uint equal = min(taken, uint(DF2_TOPK_PICKS - DF2_MAX_K));
      for (uint i = 0; i < equal; ++i) {
        keys[above + i] = keys[DF2_TOPK_PICKS - 1 - i];
        ids[above + i] = ids[DF2_TOPK_PICKS - 1 - i];
        scores[above + i] = scores[DF2_TOPK_PICKS - 1 - i];
      }
      atomic_store_explicit(&picked, above + equal, memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint n = atomic_load_explicit(&picked, memory_order_relaxed);
  if (t < n) {
    uint place = 0;
    for (uint j = 0; j < n; ++j) place += df2_before(keys[j], ids[j], keys[t], ids[t]) ? 1u : 0u;
    if (place < p.k) {
      vals_out[(size_t)row * p.k + place] = as_type<float>(scores[t]);
      ids_out[(size_t)row * p.k + place] = ids[t];
    }
  }
}

// Candidate selector greedy walk with after-correction softcap: cap * tanh(score / cap).
[[kernel, max_total_threads_per_threadgroup(256)]] void tf_ink2_df2_walk(
    const device float* UV [[buffer(0)]], const device uint* UI [[buffer(1)]],
    const device bfloat* H [[buffer(2)]], const device bfloat* A [[buffer(3)]],
    const device bfloat* B [[buffer(4)]], constant Df2WalkArgs& a [[buffer(5)]],
    device uint* seq [[buffer(6)]], uint t [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[8][16];
  threadgroup uint prev;
  if (t == 0) prev = seq[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint e = 0; e < a.slots; ++e) {
    if (prev >= a.vocab) {
      if (t == 0) seq[1 + e] = 0xffffffffu;
      continue;
    }
    const float ah = float(bfloat(float(A[(size_t)prev * a.rank + t]) * float(H[(size_t)e * a.rank + t])));
    for (uint c = 0; c < a.top; ++c) {
      const uint cand = UI[e * a.top + c];
      const float bv = cand < a.vocab ? float(B[(size_t)cand * a.rank + t]) : 0.0f;
      const float s = simd_sum(ah * bv);
      if (lane == 0) part[sg][c] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
      float best = -INFINITY;
      uint best_id = 0xffffffffu;
      for (uint c = 0; c < a.top; ++c) {
        const uint cand = UI[e * a.top + c];
        if (cand >= a.vocab) continue;
        float s = UV[e * a.top + c];
        for (uint k = 0; k < a.rank / 32; ++k) s += part[k][c];
        if (a.softcap > 0.0f) s = a.softcap * tanh(s / a.softcap);
        if (isfinite(s) && s > best) { best = s; best_id = cand; }
      }
      prev = best_id;
      seq[1 + e] = prev;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
