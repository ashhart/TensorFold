// GLM's DFlash2 drafter top-k: one threadgroup a row finds the k-th key by its two bytes, then sorts the few above it.
#include <metal_stdlib>
using namespace metal;

// The Row ABI of core/bf16_topk.metal: every entry at or past count is { 0xffffffff, 0 }.
#define GLM_MAX_K 16
#define GLM_THREADS 512
#define GLM_SIMDS (GLM_THREADS / 32)
// candidates the sort takes in one pass: more ties at the k-th key take the ordered path
#define GLM_PICKS 64
struct Params { uint rows, vocab, k, stride; };
struct Entry { uint token, score_bits; };
struct Row { uint nonfinite, count; Entry entries[GLM_MAX_K]; };

// core's order: a larger key first, then the lower token; both zeros share one key
inline uint glm_key(ushort raw) {
  const uint bits = (raw & 0x7fffu) == 0 ? 0u : uint(raw);
  return (bits & 0x8000u) != 0 ? (~bits & 0xffffu) : (bits | 0x8000u);
}
inline bool glm_before(uint key, uint id, uint other_key, uint other_id) {
  return key > other_key || (key == other_key && id < other_id);
}

// Two 16-bit counters a word, one histogram a simdgroup; then the 256 bins' totals in `bins`.
inline void glm_bins(threadgroup atomic_uint* hist, threadgroup uint* bins, uint t) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < 256) {
    uint n = 0;
    for (uint s = 0; s < GLM_SIMDS; ++s)
      n += (atomic_load_explicit(&hist[s * 128 + t / 2], memory_order_relaxed) >> (16 * (t & 1))) & 0xffffu;
    bins[t] = n;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// From the top bin down: the bin holding the k-th key past `above` keys, and the keys in bins above it.
inline uint2 glm_cut(const threadgroup uint* bins, uint above, uint k) {
  uint cum = above;
  for (int b = 255; b >= 0; --b) {
    if (cum + bins[b] >= k) return uint2(uint(b), cum);
    cum += bins[b];
  }
  return uint2(0, cum);
}

[[kernel, max_total_threads_per_threadgroup(GLM_THREADS)]] void glm_draft_topk(
    const device ushort* logits [[buffer(0)]], device Row* output [[buffer(1)]], constant Params& p [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup atomic_uint hist[GLM_SIMDS * 128];
  threadgroup uint bins[256], keys[GLM_PICKS], ids[GLM_PICKS], scores[GLM_PICKS], counts[GLM_SIMDS];
  threadgroup atomic_uint flags, picked;
  threadgroup uint cut[4];
  const device ushort* x = logits + (size_t)row * p.stride;
  device Row& out = output[row];
  for (uint i = t; i < GLM_SIMDS * 128; i += GLM_THREADS) atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
  if (t == 0) {
    atomic_store_explicit(&flags, 0u, memory_order_relaxed);
    atomic_store_explicit(&picked, 0u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // the high byte of every finite key, and what the row holds that is not finite
  uint mask = 0;
  for (uint tok = t; tok < p.vocab; tok += GLM_THREADS) {
    const ushort raw = x[tok];
    if ((raw & 0x7f80u) == 0x7f80u) {
      mask |= (raw & 0x7fu) != 0 ? 1u : (raw & 0x8000u) == 0 ? 2u : 4u;
      continue;
    }
    const uint b = glm_key(raw) >> 8;
    atomic_fetch_add_explicit(&hist[sg * 128 + b / 2], 1u << (16 * (b & 1)), memory_order_relaxed);
  }
  mask = simd_or(mask);
  if (lane == 0 && mask != 0) atomic_fetch_or_explicit(&flags, mask, memory_order_relaxed);
  glm_bins(hist, bins, t);
  const uint held = atomic_load_explicit(&flags, memory_order_relaxed);
  if (held != 0) {
    if (t < GLM_MAX_K) out.entries[t] = { 0xffffffffu, 0u };
    if (t == 0) { out.nonfinite = held; out.count = 0; }
    return;
  }
  if (t == 0) {
    const uint2 h = glm_cut(bins, 0, p.k);
    cut[0] = h.x;
    cut[1] = h.y;
  }
  for (uint i = t; i < GLM_SIMDS * 128; i += GLM_THREADS) atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // the low byte of the keys in the k-th key's high bin
  const uint high = cut[0];
  for (uint tok = t; tok < p.vocab; tok += GLM_THREADS) {
    const uint key = glm_key(x[tok]);
    if ((key >> 8) != high) continue;
    const uint b = key & 0xffu;
    atomic_fetch_add_explicit(&hist[sg * 128 + b / 2], 1u << (16 * (b & 1)), memory_order_relaxed);
  }
  glm_bins(hist, bins, t);
  if (t == 0) {
    const uint2 l = glm_cut(bins, cut[1], p.k);
    cut[2] = (high << 8) | l.x;
    cut[3] = l.y + bins[l.x];  // keys above the k-th key's, and those equal to it
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint kth = cut[2];
  if (cut[3] <= GLM_PICKS) {
    // every key at or above the k-th: few enough to sort whole, whatever order they arrive in
    for (uint tok = t; tok < p.vocab; tok += GLM_THREADS) {
      const ushort raw = x[tok];
      const uint key = glm_key(raw);
      if (key < kth) continue;
      const uint at = atomic_fetch_add_explicit(&picked, 1u, memory_order_relaxed);
      keys[at] = key;
      ids[at] = tok;
      scores[at] = uint(raw) << 16;
    }
  } else {
    // many ties at the k-th key: the keys above it anywhere, then the lowest tokens equal to it, in token order
    uint taken = 0;
    for (uint first = 0; first < p.vocab; first += GLM_THREADS) {
      const uint tok = first + t;
      const ushort raw = tok < p.vocab ? x[tok] : ushort(0x7fc0);
      const uint key = (raw & 0x7f80u) == 0x7f80u ? 0u : glm_key(raw);
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
      for (uint s = 0; s < GLM_SIMDS; ++s) all += counts[s];
      // the equal ones fill the slots from the top down, after the at most k - 1 keys above
      if (equal != 0 && rank < GLM_PICKS - GLM_MAX_K) {
        keys[GLM_PICKS - 1 - rank] = key;
        ids[GLM_PICKS - 1 - rank] = tok;
        scores[GLM_PICKS - 1 - rank] = uint(raw) << 16;
      }
      taken += all;
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t == 0) {
      const uint above = atomic_load_explicit(&picked, memory_order_relaxed);
      const uint equal = min(taken, uint(GLM_PICKS - GLM_MAX_K));
      for (uint i = 0; i < equal; ++i) {
        keys[above + i] = keys[GLM_PICKS - 1 - i];
        ids[above + i] = ids[GLM_PICKS - 1 - i];
        scores[above + i] = scores[GLM_PICKS - 1 - i];
      }
      atomic_store_explicit(&picked, above + equal, memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // each candidate's place: the candidates before it in core's order
  const uint n = atomic_load_explicit(&picked, memory_order_relaxed);
  if (t < n) {
    uint place = 0;
    for (uint j = 0; j < n; ++j) place += glm_before(keys[j], ids[j], keys[t], ids[t]) ? 1u : 0u;
    if (place < p.k) out.entries[place] = { ids[t], scores[t] };
  }
  if (t >= p.k && t < GLM_MAX_K) out.entries[t] = { 0xffffffffu, 0u };
  if (t == 0) { out.nonfinite = 0; out.count = p.k; }
}
