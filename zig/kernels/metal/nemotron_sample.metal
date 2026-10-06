// Keyed draws over the whole vocabulary for rows tf_gpu_sample would cut to 1,024 candidates (top_k off, or above 1,024).
#include <metal_stdlib>
using namespace metal;

constant constexpr uint TG = 1024;
constant constexpr uint NSG = TG / 32;
constant constexpr uint C = 1024;

// tokens this far (in logits / T) below the row's max never win the race: its noise spans under 19.5
constant constexpr float NEVER = 24.0f;

inline uint tf_key(float v) { uint b = as_type<uint>(v); return (b & 0x80000000u) ? ~b : (b | 0x80000000u); }
inline float tf_val(uint k) { uint b = (k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k; return as_type<float>(b); }
inline ulong tf_mix(ulong x) {
  x ^= x >> 30; x *= 0xBF58476D1CE4E5B9UL; x ^= x >> 27; x *= 0x94D049BB133111EBUL; return x ^ (x >> 31);
}
inline float tf_uniform(ulong seed, uint pos, uint id) {
  ulong x = tf_mix(seed + 0x9E3779B97F4A7C15UL);
  x = tf_mix(x ^ (ulong(pos) * 0xD1B54A32D192ED03UL));
  x = tf_mix(x ^ ulong(id));
  return (float(uint(x >> 40)) + 0.5f) * (1.0f / 16777216.0f);
}

// tf_uniform kept below 1: its top value rounds to 1.0, a score of +inf that would win from anywhere in the vocabulary
inline float race_uniform(ulong seed, uint pos, uint id) { return min(tf_uniform(seed, pos, id), as_type<float>(0x3F7FFFFFu)); }

// A place in the (value desc, id asc) order; END is past every token.
struct Mark { uint key; uint id; };
constant constexpr Mark END = {0u, 0xFFFFFFFFu};

inline bool at_or_before(uint k, uint i, Mark m) { return k > m.key || (k == m.key && i <= m.id); }

inline bool after(uint k, uint i, Mark m) { return k < m.key || (k == m.key && i > m.id); }

inline bool beats(float s, uint k, uint i, float bs, uint bk, uint bi) {
  return s > bs || (s == bs && (k > bk || (k == bk && i < bi)));
}

// One row of logits as the threadgroup reads it, and the thread's place in it.
struct Row {
  const device bfloat* L;
  size_t base;
  uint V;
  float inv_t;
  uint t, lane, sg;
  float v(uint i) const { return float(L[base + i]) * inv_t; }
};

// Threadgroup scratch shared by the steps.
struct Shared {
  threadgroup float* fsh;
  threadgroup float* fsh2;
  threadgroup uint* ush;
  threadgroup uint* ush2;
  threadgroup atomic_uint* hist;
  threadgroup uint* st;
  threadgroup uint* ck;
  threadgroup uint* ci;
  threadgroup atomic_uint* fill;
};

// A threadgroup sum in tf_gpu_sample's order: simd sums, then simdgroups in order; every thread gets it.
inline float group_sum(float x, thread const Row& r, thread const Shared& s) {
  x = simd_sum(x);
  if (r.lane == 0) s.fsh[r.sg] = x;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float z = 0.0f;
  for (uint i = 0; i < NSG; i++) z += s.fsh[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return z;
}

// The row max in tf_gpu_sample's order.
inline float row_max(thread const Row& r, thread const Shared& s) {
  float lm = -INFINITY;
  for (uint i = r.t; i < r.V; i += TG) lm = max(lm, r.v(i));
  lm = simd_max(lm);
  if (r.lane == 0) s.fsh[r.sg] = lm;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float m = -INFINITY;
  for (uint i = 0; i < NSG; i++) m = max(m, s.fsh[i]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return m;
}

// tf_gpu_sample's radix select of the `want` largest: the key they reach, how many lie above it, ties' id cut.
struct Top { uint key; uint above; uint need; uint idcut; };

inline Top top_keys(thread const Row& r, uint want, thread const Shared& s) {
  const uint t = r.t;
  uint prefix = 0u, pmask = 0u, need = min(want, r.V), above = 0u, ties = 0u;
  for (int shift = 24; shift >= 0; shift -= 8) {
    if (t < 256) atomic_store_explicit(&s.hist[t], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = t; i < r.V; i += TG) {
      const uint k = tf_key(r.v(i));
      if ((k & pmask) == prefix) atomic_fetch_add_explicit(&s.hist[(k >> shift) & 255u], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
      uint cum = 0u;
      int b = 255;
      for (; b > 0; b--) {
        const uint c = atomic_load_explicit(&s.hist[b], memory_order_relaxed);
        if (cum + c >= need) break;
        cum += c;
      }
      s.st[0] = uint(b); s.st[1] = cum; s.st[2] = atomic_load_explicit(&s.hist[b], memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    prefix |= s.st[0] << shift; pmask |= 255u << shift;
    above += s.st[1]; need -= s.st[1]; ties = s.st[2];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  uint idcut = 0xFFFFFFFFu;
  if (ties > need) {
    uint ipre = 0u, imask = 0u, ineed = need;
    for (int shift = 16; shift >= 0; shift -= 8) {
      if (t < 256) atomic_store_explicit(&s.hist[t], 0u, memory_order_relaxed);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint i = t; i < r.V; i += TG) {
        if (tf_key(r.v(i)) == prefix && (i & imask) == ipre)
          atomic_fetch_add_explicit(&s.hist[(i >> shift) & 255u], 1u, memory_order_relaxed);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (t == 0) {
        uint cum = 0u;
        uint b = 0u;
        for (; b < 255u; b++) {
          const uint c = atomic_load_explicit(&s.hist[b], memory_order_relaxed);
          if (cum + c >= ineed) break;
          cum += c;
        }
        s.st[0] = b; s.st[1] = cum;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      ipre |= s.st[0] << shift; imask |= 255u << shift; ineed -= s.st[1];
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    idcut = ipre;
  }
  return {prefix, above, need, idcut};
}

// tf_gpu_sample's candidates, sorted (value desc, id asc): a window near the max holding the rule's tokens, else the C largest.
inline uint first_block(thread const Row& r, float m, float top_p, float near, uint kw, thread float& z, thread const Shared& s) {
  const uint t = r.t;
  float ls = 0.0f, lnear[3] = {0.0f, 0.0f, 0.0f};
  uint near_count[3] = {0u, 0u, 0u};
  for (uint i = t; i < r.V; i += TG) {
    const float v = r.v(i);
    const float e = metal::exp(v - m);
    ls += e;
    for (int w = 0; w < 3; w++) {
      if (v >= m - near / float(1 << w)) { lnear[w] += e; near_count[w]++; }
    }
  }
  z = group_sum(ls, r, s);
  int window = -1;
  uint offset = 0u, n_near = 0u;
  for (int w = 0; w < 3; w++) {
    const float znear_part = simd_sum(lnear[w]);
    const uint before_in_simd = simd_prefix_exclusive_sum(near_count[w]);
    const uint simd_count = simd_sum(near_count[w]);
    if (r.lane == 0) { s.fsh2[r.sg] = znear_part; s.ush[r.sg] = simd_count; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float znear = 0.0f;
    uint off = before_in_simd, count = 0u;
    for (uint g = 0; g < NSG; g++) {
      znear += s.fsh2[g];
      if (g < r.sg) off += s.ush[g];
      count += s.ush[g];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool holds = count <= C && (kw == 0u
        ? (top_p > 0.0f && top_p < 1.0f && znear >= (top_p + 1e-4f) * z)
        : count >= min(kw, uint(C)));
    if (window < 0 && holds) { window = w; offset = off; n_near = count; }
  }
  const float floor_v = window < 0 ? INFINITY : m - near / float(1 << window);
  uint n_cand = 0u;
  uint sort_n = C;
  s.ck[t] = 0u;
  s.ci[t] = 0xFFFFFFFFu;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (window >= 0) {
    uint at = offset;
    for (uint i = t; i < r.V; i += TG) {
      const float v = r.v(i);
      if (v >= floor_v) { s.ck[at] = tf_key(v); s.ci[at] = i; at++; }
    }
    n_cand = n_near;
    sort_n = 32u;
    while (sort_n < n_near) sort_n <<= 1;
  } else {
    const Top top = top_keys(r, C, s);
    if (t == 0) {
      atomic_store_explicit(&s.fill[0], 0u, memory_order_relaxed);
      atomic_store_explicit(&s.fill[1], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = t; i < r.V; i += TG) {
      const uint k = tf_key(r.v(i));
      if (k > top.key) {
        const uint at = atomic_fetch_add_explicit(&s.fill[0], 1u, memory_order_relaxed);
        s.ck[at] = k; s.ci[at] = i;
      } else if (k == top.key && i <= top.idcut) {
        const uint at = top.above + atomic_fetch_add_explicit(&s.fill[1], 1u, memory_order_relaxed);
        s.ck[at] = k; s.ci[at] = i;
      }
    }
    n_cand = top.above + top.need;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint k = 2; k <= sort_n; k <<= 1) {
    for (uint j = k >> 1; j > 0; j >>= 1) {
      const uint p = t ^ j;
      if (p > t && p < sort_n) {
        const uint ka = s.ck[t], kb = s.ck[p], ia = s.ci[t], ib = s.ci[p];
        const bool a_first = ka > kb || (ka == kb && ia < ib);
        if (a_first != ((t & k) == 0)) { s.ck[t] = kb; s.ck[p] = ka; s.ci[t] = ib; s.ci[p] = ia; }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  return min(n_cand, C);
}

// The probability mass (over `norm`) of the tokens after `b`, inside `top`, at or above `lo`, and inside `cut`.
inline float mass(thread const Row& r, float m, float norm, float lo, Mark b, Mark top, Mark cut, bool strict, thread const Shared& s) {
  float x = 0.0f;
  for (uint i = r.t; i < r.V; i += TG) {
    const float v = r.v(i);
    if (!(v >= lo)) continue;
    const uint k = tf_key(v);
    const bool inside = strict ? (k > cut.key || (k == cut.key && i < cut.id)) : at_or_before(k, i, cut);
    if (after(k, i, b) && at_or_before(k, i, top) && inside) x += metal::exp(v - m) / norm;
  }
  return group_sum(x, r, s);
}

// The nucleus's last token past the block: the first place whose prefix reaches top_p, by bits of key then id.
inline Mark extend(thread const Row& r, float m, float norm, float lo, float top_p, float cum, Mark b, Mark top, thread const Shared& s) {
  if (!(cum + mass(r, m, norm, lo, b, top, END, false, s) >= top_p)) return END;
  uint key = 0u;
  for (int bit = 31; bit >= 0; bit--) {
    const uint cand = key | (1u << bit);
    if (cum + mass(r, m, norm, lo, b, top, Mark{cand, 0xFFFFFFFFu}, false, s) >= top_p) key = cand;
  }
  uint id = 0u;
  for (int bit = 31 - int(clz(r.V - 1u)); bit >= 0; bit--) {
    const uint cand = id | (1u << bit);
    if (cand >= r.V) continue;
    if (cum + mass(r, m, norm, lo, b, top, Mark{key, cand}, true, s) < top_p) id = cand;
  }
  return Mark{key, id};
}

// The nucleus's last token: tf_gpu_sample's sequential sum over its block, extended past the block when it falls short.
inline Mark nucleus(thread const Row& r, float m, float lo, float top_p, float near, uint kc, bool whole, Mark top, thread const Shared& s) {
  float z = 0.0f;
  const uint n = first_block(r, m, top_p, near, whole ? 0u : kc, z, s);
  const float norm = whole ? z : mass(r, m, 1.0f, -INFINITY, Mark{0xFFFFFFFFu, 0xFFFFFFFFu}, top, END, false, s);
  if (r.t == 0) {
    float cum = 0.0f;
    uint keep = 0u;
    for (uint j = 0; j < n; j++) {
      cum += metal::exp(tf_val(s.ck[j]) - m) / norm;
      if (cum >= top_p) { keep = j + 1; break; }
    }
    s.st[0] = keep;
    s.fsh2[0] = cum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint keep = s.st[0];
  const float cum = s.fsh2[0];
  const Mark last = keep > 0u ? Mark{s.ck[keep - 1u], s.ci[keep - 1u]} : Mark{s.ck[n - 1u], s.ci[n - 1u]};
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return keep > 0u ? last : extend(r, m, norm, lo, top_p, cum, last, top, s);
}

// The kept tokens' race (value - log(-log(u)), ties to the earlier in the order), written as the row's token.
template <bool MAP>
inline void race(thread const Row& r, float lo, Mark top, Mark cut, ulong seed, uint position, const device uint* IDS,
                 device uint* TOK, uint row, thread const Shared& s) {
  float bs = -INFINITY;
  uint bk = 0u, bi = 0xFFFFFFFFu;
  for (uint i = r.t; i < r.V; i += TG) {
    const float v = r.v(i);
    if (!(v >= lo)) continue;
    const uint k = tf_key(v);
    if (!at_or_before(k, i, top) || !at_or_before(k, i, cut)) continue;
    const float score = v - metal::log(-metal::log(race_uniform(seed, position, MAP ? IDS[i] : i)));
    if (beats(score, k, i, bs, bk, bi)) { bs = score; bk = k; bi = i; }
  }
  for (ushort off = 16; off > 0; off >>= 1) {
    const float os = simd_shuffle_xor(bs, off);
    const uint ok = simd_shuffle_xor(bk, off), oi = simd_shuffle_xor(bi, off);
    if (beats(os, ok, oi, bs, bk, bi)) { bs = os; bk = ok; bi = oi; }
  }
  if (r.lane == 0) { s.fsh[r.sg] = bs; s.ush[r.sg] = bk; s.ush2[r.sg] = bi; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (r.t == 0) {
    for (uint g = 0; g < NSG; g++) {
      if (beats(s.fsh[g], s.ush[g], s.ush2[g], bs, bk, bi)) { bs = s.fsh[g]; bk = s.ush[g]; bi = s.ush2[g]; }
    }
    const uint pick = min(bi, r.V - 1u);
    TOK[row] = MAP ? IDS[pick] : pick;
  }
}

// One row a 1,024-thread threadgroup, settings as tf_gpu_sample reads them; MAP keys the noise and the token by IDS.
template <bool MAP>
[[max_total_threads_per_threadgroup(1024)]]
[[kernel]] void tf_sample_full(
  const device bfloat* L [[buffer(0)]],
  const device uint* seeds [[buffer(1)]],
  const device uint* positions [[buffer(2)]],
  const device float* cfg [[buffer(3)]],
  const device uint* kcap [[buffer(4)]],
  device uint* TOK [[buffer(5)]],
  const device uint* IDS [[buffer(6)]],
  constant uint& V [[buffer(7)]],
  uint sg [[simdgroup_index_in_threadgroup]],
  uint lane [[thread_index_in_simdgroup]],
  uint3 tpos [[thread_position_in_threadgroup]],
  uint3 gpos [[threadgroup_position_in_grid]]) {
  threadgroup float fsh[NSG];
  threadgroup float fsh2[NSG];
  threadgroup uint ush[NSG];
  threadgroup uint ush2[NSG];
  threadgroup atomic_uint hist[256];
  threadgroup uint st[4];
  threadgroup uint ck[C];
  threadgroup uint ci[C];
  threadgroup atomic_uint fill[2];
  const Shared s = {fsh, fsh2, ush, ush2, hist, st, ck, ci, fill};
  const uint row = gpos.x;
  const Row r = {L, size_t(row) * V, V, cfg[4 * row], tpos.x, lane, sg};
  const float top_p = cfg[4 * row + 1];
  const float near = cfg[4 * row + 2];
  const float min_log = cfg[4 * row + 3];
  const uint kc = kcap[row];
  const ulong seed = ulong(seeds[2 * row]) | (ulong(seeds[2 * row + 1]) << 32);
  const float m = row_max(r, s);
  const float lo = max(m + min_log, m - NEVER);
  const bool whole = kc == 0u || kc >= V;
  Mark top = END;
  if (!whole) {
    const Top u = top_keys(r, kc, s);
    top = Mark{u.key, u.idcut};
  }
  const Mark cut = (top_p > 0.0f && top_p < 1.0f) ? nucleus(r, m, lo, top_p, near, kc, whole, top, s) : END;
  race<MAP>(r, lo, top, cut, seed, positions[row], IDS, TOK, row, s);
}

template [[host_name("tf_sample_full")]] [[kernel]] decltype(tf_sample_full<false>) tf_sample_full<false>;
template [[host_name("tf_sample_full_ids")]] [[kernel]] decltype(tf_sample_full<true>) tf_sample_full<true>;
