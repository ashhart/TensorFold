// The keyed sampler the target head draws with: tf_sample_full's plan and race over bf16 logits.

// The host binds logits at 0, rules (header + one a row) at 1, picks at 2; GlmSampleRule mirrors draw_rule.zig's Rule.

#include <metal_stdlib>
using namespace metal;

struct GlmDraws {
  uint n;      // the host filled the first n rules
  uint pad[3];
};

struct GlmSampleRule {
  uint seed_lo, seed_hi;   // the keyed sampler's 64-bit seed
  uint position;           // this row's absolute token position: the noise is keyed by (seed, position, id)
  uint top_k;              // 0: off (the whole vocabulary races)
  float inv_t, top_p, near, min_log; // 1/T, the nucleus's mass, the candidate window, ln(min_p)
  uint vocab;              // this row's logit columns
};

constant uint GLM_GREEDY = 0xFFFFFFFFu; // a rule position no pass row ever has: the row draws argmax

inline ulong glm_mix(ulong x) {
  x ^= x >> 30; x *= 0xBF58476D1CE4E5B9UL; x ^= x >> 27; x *= 0x94D049BB133111EBUL; return x ^ (x >> 31);
}

inline float glm_uniform(ulong seed, uint pos, uint id) {
  ulong x = glm_mix(seed + 0x9E3779B97F4A7C15UL);
  x = glm_mix(x ^ (ulong(pos) * 0xD1B54A32D192ED03UL));
  x = glm_mix(x ^ ulong(id));
  return (float(uint(x >> 40)) + 0.5f) * (1.0f / 16777216.0f);
}

inline float glm_race_uniform(ulong seed, uint pos, uint id) {
  return min(glm_uniform(seed, pos, id), as_type<float>(0x3F7FFFFFu)); // below 1: a +inf score would win from anywhere
}

inline uint glm_key(float v) { uint b = as_type<uint>(v); return (b & 0x80000000u) ? ~b : (b | 0x80000000u); }
inline float glm_val(uint k) { uint b = (k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k; return as_type<float>(b); }

inline bool glm_beats(float s, uint k, uint i, float bs, uint bk, uint bi) {
  return s > bs || (s == bs && (k > bk || (k == bk && i < bi)));
}

constant constexpr uint TG = 1024;
constant constexpr uint NSG = TG / 32;
constant constexpr uint C = 1024;
constant constexpr float NEVER = 24.0f; // a race's noise spans under 19.5: tokens this far below the max never win

// A place in the (value desc, id asc) order; END is past every token.
struct Mark { uint key; uint id; };
constant constexpr Mark END = {0u, 0xFFFFFFFFu};

inline bool at_or_before(uint k, uint i, Mark m) { return k > m.key || (k == m.key && i <= m.id); }

inline bool after(uint k, uint i, Mark m) { return k < m.key || (k == m.key && i > m.id); }

// A threadgroup sum in tf_gpu_sample's order: simd sums, then simdgroups in order; every thread gets it.
inline float group_sum(float x, uint lane, uint sg, threadgroup float* fsh) {
  x = simd_sum(x);
  if (lane == 0) fsh[sg] = x;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float z = 0.0f;
  for (uint i = 0; i < NSG; i++) z += fsh[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return z;
}

// The row max in tf_gpu_sample's order.
inline float row_max(const device bfloat* base, uint V, float inv_t, uint t, uint lane, uint sg,
                     threadgroup float* fsh) {
  float lm = -INFINITY;
  for (uint i = t; i < V; i += TG) lm = max(lm, float(base[i]) * inv_t);
  lm = simd_max(lm);
  if (lane == 0) fsh[sg] = lm;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float m = -INFINITY;
  for (uint i = 0; i < NSG; i++) m = max(m, fsh[i]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return m;
}

// The probability mass (over `norm`) of the tokens after `b`, inside `top`, at or above `lo`, inside `cut`.
inline float mass(const device bfloat* base, uint V, float inv_t, float m, float lo, float norm,
                  Mark b, Mark top, Mark cut, bool strict, uint t, uint lane, uint sg,
                  threadgroup float* fsh) {
  float x = 0.0f;
  for (uint i = t; i < V; i += TG) {
    const float v = float(base[i]) * inv_t;
    if (!(v >= lo)) continue;
    const uint k = glm_key(v);
    const bool inside = strict ? (k > cut.key || (k == cut.key && i < cut.id)) : at_or_before(k, i, cut);
    if (after(k, i, b) && at_or_before(k, i, top) && inside) x += metal::exp(v - m) / norm;
  }
  return group_sum(x, lane, sg, fsh);
}

// The nucleus's last token past the block: the first place whose prefix reaches top_p, by bits of key then id.
inline Mark extend(const device bfloat* base, uint V, float inv_t, float m, float lo, float norm,
                   float top_p, float cum, Mark b, Mark top, uint t, uint lane, uint sg,
                   threadgroup float* fsh) {
  if (!(cum + mass(base, V, inv_t, m, lo, norm, b, top, END, false, t, lane, sg, fsh) >= top_p)) return END;
  uint key = 0u;
  for (int bit = 31; bit >= 0; bit--) {
    const uint cand = key | (1u << bit);
    if (cum + mass(base, V, inv_t, m, lo, norm, b, top, Mark{cand, 0xFFFFFFFFu}, false, t, lane, sg, fsh) >= top_p) key = cand;
  }
  uint id = 0u;
  for (int bit = 31 - int(clz(V - 1u)); bit >= 0; bit--) {
    const uint cand = id | (1u << bit);
    if (cand >= V) continue;
    if (cum + mass(base, V, inv_t, m, lo, norm, b, top, Mark{key, cand}, true, t, lane, sg, fsh) < top_p) id = cand;
  }
  return Mark{key, id};
}

// tf_gpu_sample's radix select of the `want` largest: the key they reach, how many lie above it.
inline void top_keys(const device bfloat* base, uint V, float inv_t, uint want, uint t,
                     threadgroup atomic_uint* hist, threadgroup uint* st,
                     thread uint& prefix, thread uint& above, thread uint& need) {
  uint pmask = 0u;
  prefix = 0u; need = min(want, V); above = 0u;
  for (int shift = 24; shift >= 0; shift -= 8) {
    if (t < 256) atomic_store_explicit(&hist[t], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = t; i < V; i += TG) {
      const uint k = glm_key(float(base[i]) * inv_t);
      if ((k & pmask) == prefix) atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
      uint cum = 0u;
      int b = 255;
      for (; b > 0; b--) {
        const uint c = atomic_load_explicit(&hist[b], memory_order_relaxed);
        if (cum + c >= need) break;
        cum += c;
      }
      st[0] = uint(b); st[1] = cum; st[2] = atomic_load_explicit(&hist[b], memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    prefix |= st[0] << shift; pmask |= 255u << shift;
    above += st[1]; need -= st[1];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// The ties at the select's key, cut by token id: tf_gpu_sample's second radix pass over id bits.
inline uint id_cut(const device bfloat* base, uint V, float inv_t, uint t, uint prefix, uint need,
                   threadgroup atomic_uint* hist, threadgroup uint* st) {
  uint ipre = 0u, imask = 0u, ineed = need;
  for (int shift = 24; shift >= 0; shift -= 8) {
    if (t < 256) atomic_store_explicit(&hist[t], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = t; i < V; i += TG) {
      if (glm_key(float(base[i]) * inv_t) == prefix && (i & imask) == ipre)
        atomic_fetch_add_explicit(&hist[(i >> shift) & 255u], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
      uint cum = 0u;
      uint b = 0u;
      for (; b < 255u; b++) {
        const uint c = atomic_load_explicit(&hist[b], memory_order_relaxed);
        if (cum + c >= ineed) break;
        cum += c;
      }
      st[0] = b; st[1] = cum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ipre |= st[0] << shift; imask |= 255u << shift; ineed -= st[1];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  return ipre;
}

// Select a complete boundary, including the lowest-id tie prefix, independently of candidate storage.
inline Mark select_mark(const device bfloat* base, uint V, float inv_t, uint want, uint t,
                        threadgroup atomic_uint* hist, threadgroup uint* st) {
  uint prefix, above, need;
  top_keys(base, V, inv_t, want, t, hist, st, prefix, above, need);
  const uint ties = st[2];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint cut = ties > need ? id_cut(base, V, inv_t, t, prefix, need, hist, st) : 0xFFFFFFFFu;
  return Mark{prefix, cut};
}

// Store all selected candidates, at most C; sorting removes the atomic fill order from the result.
inline uint candidates(const device bfloat* base, uint V, float inv_t, uint want, uint t,
                       threadgroup atomic_uint* hist, threadgroup uint* st,
                       threadgroup uint* ck, threadgroup uint* ci, threadgroup atomic_uint* fill) {
  const uint count = min(want, min(C, V));
  const Mark bound = select_mark(base, V, inv_t, count, t, hist, st);
  if (t == 0) atomic_store_explicit(&fill[0], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = t; i < V; i += TG) {
    const uint k = glm_key(float(base[i]) * inv_t);
    if (at_or_before(k, i, bound)) {
      const uint at = atomic_fetch_add_explicit(&fill[0], 1u, memory_order_relaxed);
      ck[at] = k; ci[at] = i;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return count;
}

// The nucleus uses the full top-k mass before min_p, then extends beyond the sorted candidate block.
inline Mark nucleus(const device bfloat* base, uint V, float inv_t, uint t, float m, float lo,
                    float top_p, uint n_cand, threadgroup uint* st, threadgroup uint* ck,
                    threadgroup uint* ci, uint lane, uint sg, threadgroup float* fsh, Mark top) {
  const Mark start = {0xFFFFFFFFu, 0xFFFFFFFFu};
  const float norm = mass(base, V, inv_t, m, -INFINITY, 1.0f, start, top, END, false, t, lane, sg, fsh);
  if (t == 0) {
    float cum = 0.0f;
    uint keep = 0u;
    for (uint j = 0; j < n_cand; j++) {
      cum += metal::exp(glm_val(ck[j]) - m) / norm;
      if (cum >= top_p) { keep = j + 1; break; }
    }
    st[0] = keep;
    fsh[0] = cum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint keep = st[0];
  const float cum = fsh[0];
  const Mark last = keep > 0u ? Mark{ck[keep - 1u], ci[keep - 1u]} : Mark{ck[n_cand - 1u], ci[n_cand - 1u]};
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return keep > 0u ? last : extend(base, V, inv_t, m, lo, norm, top_p, cum, last, top, t, lane, sg, fsh);
}

// The kept tokens' race (value - log(-log(u)), ties to the earlier in the order), written as the row's pick.
inline void race(const device bfloat* base, uint V, float inv_t, ulong seed, uint position, uint t,
                 uint lane, uint sg, float lo, Mark top, Mark cut, threadgroup float* fsh,
                 threadgroup uint* ush, threadgroup uint* ush2, device uint* out, uint row) {
  float bs = -INFINITY;
  uint bk = 0u, bi = 0xFFFFFFFFu;
  for (uint i = t; i < V; i += TG) {
    const float v = float(base[i]) * inv_t;
    if (!(v >= lo)) continue;
    const uint k = glm_key(v);
    if (!at_or_before(k, i, top) || !at_or_before(k, i, cut)) continue;
    const float score = v - metal::log(-metal::log(glm_race_uniform(seed, position, i)));
    if (glm_beats(score, k, i, bs, bk, bi)) { bs = score; bk = k; bi = i; }
  }
  for (ushort off = 16; off > 0; off >>= 1) {
    const float os = simd_shuffle_xor(bs, off);
    const uint ok = simd_shuffle_xor(bk, off), oi = simd_shuffle_xor(bi, off);
    if (glm_beats(os, ok, oi, bs, bk, bi)) { bs = os; bk = ok; bi = oi; }
  }
  if (lane == 0) { fsh[sg] = bs; ush[sg] = bk; ush2[sg] = bi; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    for (uint g = 0; g < NSG; g++) {
      if (glm_beats(fsh[g], ush[g], ush2[g], bs, bk, bi)) { bs = fsh[g]; bk = ush[g]; bi = ush2[g]; }
    }
    out[row] = bi;
  }
}

// A greedy row: argmax over the row's bf16 logits, glm_argmax's tie rule (larger value, lower id).
inline void greedy_row(const device bfloat* base, uint V, uint t, uint lane, uint sg,
                       threadgroup float* fsh, threadgroup uint* ush, device uint* out, uint row) {
  float best = -INFINITY;
  uint at = 0xFFFFFFFFu;
  for (uint i = t; i < V; i += TG) {
    const float v = float(base[i]);
    if (v > best) { best = v; at = i; }
  }
  for (ushort o = 16; o > 0; o >>= 1) {
    const float ob = simd_shuffle_xor(best, o);
    const uint oa = simd_shuffle_xor(at, o);
    if (ob > best || (ob == best && oa < at)) { best = ob; at = oa; }
  }
  if (lane == 0) { fsh[sg] = best; ush[sg] = at; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    for (uint g = 0; g < NSG; g++) {
      if (fsh[g] > best || (fsh[g] == best && ush[g] < at)) { best = fsh[g]; at = ush[g]; }
    }
    out[row] = min(at, V - 1u);
  }
}

// One row a 1,024-thread group; rules ride buffer 1 behind the header, one a row, every pass row transmitted.
[[max_total_threads_per_threadgroup(1024)]]
[[kernel]] void glm_sample(const device bfloat* L [[buffer(0)]],
                           constant GlmDraws& draws [[buffer(1)]],
                           device uint* out [[buffer(2)]], uint row [[threadgroup_position_in_grid]],
                           uint t [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float fsh[NSG];
  threadgroup float fsh2[NSG];
  threadgroup uint ush[NSG];
  threadgroup uint ush2[NSG];
  threadgroup atomic_uint hist[256];
  threadgroup uint st[4];
  threadgroup uint ck[C];
  threadgroup uint ci[C];
  threadgroup atomic_uint fill[2];
  if (row >= draws.n) return;
  constant GlmSampleRule* rules = (constant GlmSampleRule*)(constant uint*)(&draws + 1);
  const constant GlmSampleRule& rule = rules[row];
  const uint V = rule.vocab;
  const device bfloat* base = L + size_t(row) * V;
  if (rule.position == GLM_GREEDY) return greedy_row(base, V, t, lane, sg, fsh, ush, out, row);
  const ulong seed = ulong(rule.seed_lo) | (ulong(rule.seed_hi) << 32);
  const uint kc = rule.top_k;
  const uint want = kc == 0u ? V : min(kc, V);
  const float m = row_max(base, V, rule.inv_t, t, lane, sg, fsh);
  const float lo = max(m + rule.min_log, m - NEVER);
  Mark top = END;
  if (kc != 0u && kc < V) top = select_mark(base, V, rule.inv_t, kc, t, hist, st);
  const uint n_cand = candidates(base, V, rule.inv_t, want, t, hist, st, ck, ci, fill);
  // The stored block sorted (value desc, id asc); the fill wrote n_cand, the sort's span is initialized.
  uint sort_n = 1u;
  while (sort_n < n_cand) sort_n <<= 1;
  if (sort_n > C) sort_n = C;
  for (uint s = t; s < sort_n; s += TG) if (s >= n_cand) { ck[s] = 0u; ci[s] = 0xFFFFFFFFu; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint k = 2; k <= sort_n; k <<= 1) {
    for (uint j = k >> 1; j > 0; j >>= 1) {
      const uint p = t ^ j;
      if (p > t && p < sort_n) {
        const uint ka = ck[t], kb = ck[p], ia = ci[t], ib = ci[p];
        const bool a_first = ka > kb || (ka == kb && ia < ib);
        if (a_first != ((t & k) == 0)) { ck[t] = kb; ck[p] = ka; ci[t] = ib; ci[p] = ia; }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  const Mark cut = (rule.top_p > 0.0f && rule.top_p < 1.0f)
      ? nucleus(base, V, rule.inv_t, t, m, lo, rule.top_p, n_cand, st, ck, ci, lane, sg, fsh2, top)
      : END;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  race(base, V, rule.inv_t, seed, rule.position, t, lane, sg, lo, top, cut, fsh, ush, ush2, out, row);
}
