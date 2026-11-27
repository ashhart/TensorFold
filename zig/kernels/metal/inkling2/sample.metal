#include <metal_stdlib>
#pragma METAL fp contract(off)
using namespace metal;

// Inkling 2 Mini f32 top-k samplers matching lanes.gpu_rule.sample bit-for-bit.
constant constexpr uint INK2_PG = 32;
constant constexpr uint INK2_KMAX = 64;
constant constexpr uint INK2_C = 1024;
constant constexpr uint INK2_END_KEY = 0u;
constant constexpr uint INK2_END_ID = 0xFFFFFFFFu;

inline uint ink2_key(float v) { uint b = as_type<uint>(v); return (b & 0x80000000u) ? ~b : (b | 0x80000000u); }
inline float ink2_val(uint k) { uint b = (k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k; return as_type<float>(b); }
inline ulong ink2_mix(ulong x) {
  x ^= x >> 30; x *= 0xBF58476D1CE4E5B9UL; x ^= x >> 27; x *= 0x94D049BB133111EBUL; return x ^ (x >> 31);
}
inline float ink2_uniform(ulong seed, uint pos, uint id) {
  ulong x = ink2_mix(seed + 0x9E3779B97F4A7C15UL);
  x = ink2_mix(x ^ (ulong(pos) * 0xD1B54A32D192ED03UL));
  x = ink2_mix(x ^ ulong(id));
  return (float(uint(x >> 40)) + 0.5f) * 0x1.0p-24f;
}

// Sequential candidate-order norm, top_p nucleus, min_log floor (without NEVER), and un-clamped uniform24 race.
template <bool MAP, uint TG_N>
inline void ink2_eval_rule(
    uint n, float m, float top_p, float min_log, ulong seed, uint position,
    const device uint* IDS, device uint* TOK, uint row, uint t,
    threadgroup const uint* ck, threadgroup const uint* ci,
    threadgroup uint* st, threadgroup float* scores) {
  if (t == 0u) {
    float norm = 0.0f;
    for (uint j = 0u; j < n; j++) norm += precise::exp(ink2_val(ck[j]) - m);
    uint keep = n;
    if (top_p > 0.0f && top_p < 1.0f) {
      float cum = 0.0f;
      for (uint j = 0u; j < n; j++) {
        cum += precise::divide(precise::exp(ink2_val(ck[j]) - m), norm);
        if (cum >= top_p) { keep = j + 1u; break; }
      }
    }
    if (min_log > -INFINITY) {
      const float floor_v = m + min_log;
      uint j = 0u;
      while (j < keep && ink2_val(ck[j]) >= floor_v) j++;
      keep = j;
    }
    st[0] = keep;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint keep = st[0];
  for (uint j = t; j < keep; j += TG_N) {
    const uint id = ci[j];
    const float u = ink2_uniform(seed, position, MAP ? IDS[id] : id);
    scores[j] = ink2_val(ck[j]) - precise::log(-precise::log(u));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0u) {
    uint best = 0u;
    float best_score = -INFINITY;
    for (uint j = 0u; j < keep; j++) {
      const float s = scores[j];
      if (j == 0u || s > best_score) { best = j; best_score = s; }
    }
    const uint pick = ci[best];
    TOK[row] = MAP ? IDS[pick] : pick;
  }
}

// Merges tf_topk_slices_f32's sorted per-slice candidates for 1 <= top_k <= 64 and evaluates gpu_rule.sample.
template <bool MAP>
[[max_total_threads_per_threadgroup(64)]]
[[kernel]] void tf_ink2_topk_rule(
    const device uint* seeds [[buffer(1)]],
    const device uint* positions [[buffer(2)]],
    const device float* cfg [[buffer(3)]],
    const device uint* kcap [[buffer(4)]],
    device uint* TOK [[buffer(5)]],
    const device uint* IDS [[buffer(6)]],
    constant uint& V [[buffer(7)]],
    const device uint* CK [[buffer(8)]],
    const device uint* CI [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tpos [[thread_position_in_threadgroup]],
    uint3 gpos [[threadgroup_position_in_grid]]) {
  threadgroup uint sk[INK2_PG * INK2_KMAX], si[INK2_PG * INK2_KMAX];
  threadgroup uint ck[INK2_KMAX], ci[INK2_KMAX], st[4];
  threadgroup float scores[INK2_KMAX];
  const uint row = gpos.x, t = tpos.x;
  const uint kc = min(kcap[row], V);
  for (uint x = t; x < INK2_PG * kc; x += 64u) {
    const uint at = (x / kc) * INK2_KMAX + x % kc;
    const size_t from = size_t(row) * INK2_PG * INK2_KMAX + at;
    sk[at] = CK[from];
    si[at] = CI[from];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u) {
    uint p = 0u;
    uint hk = sk[lane * INK2_KMAX], hi = si[lane * INK2_KMAX];
    uint nk = kc > 1u ? sk[lane * INK2_KMAX + 1u] : INK2_END_KEY;
    uint ni = kc > 1u ? si[lane * INK2_KMAX + 1u] : INK2_END_ID;
    uint merged = 0u;
    for (uint n = 0u; n < kc; n++) {
      const uint mk = simd_max(hi == INK2_END_ID ? INK2_END_KEY : hk);
      const uint mi = simd_min(hi != INK2_END_ID && hk == mk ? hi : INK2_END_ID);
      if (mi == INK2_END_ID) break;
      merged = n + 1u;
      if (hi == mi) {
        ck[n] = hk;
        ci[n] = hi;
        p++;
        hk = nk;
        hi = ni;
        nk = p + 1u < kc ? sk[lane * INK2_KMAX + p + 1u] : INK2_END_KEY;
        ni = p + 1u < kc ? si[lane * INK2_KMAX + p + 1u] : INK2_END_ID;
      }
    }
    if (lane == 0u) st[1] = merged;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint n = st[1];
  const float m = ink2_val(ck[0]);
  const ulong seed = ulong(seeds[2 * row]) | (ulong(seeds[2 * row + 1u]) << 32);
  ink2_eval_rule<MAP, 64u>(
      n, m, cfg[4 * row + 1], cfg[4 * row + 3], seed, positions[row],
      IDS, TOK, row, t, ck, ci, st, scores);
}

template [[host_name("tf_ink2_topk_rule_f32")]] [[kernel]] decltype(tf_ink2_topk_rule<false>) tf_ink2_topk_rule<false>;
template [[host_name("tf_ink2_topk_rule_ids_f32")]] [[kernel]] decltype(tf_ink2_topk_rule<true>) tf_ink2_topk_rule<true>;

// Single-pass top-k selection (1 <= top_k <= 1024) matching lanes.gpu_rule.sample bit-for-bit.
template <bool MAP>
[[max_total_threads_per_threadgroup(1024)]]
[[kernel]] void tf_ink2_rule_1024(
    const device float* L [[buffer(0)]],
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
  threadgroup float fsh[32], scores[INK2_C];
  threadgroup atomic_uint hist[256], fill[2];
  threadgroup uint st[4], ck[INK2_C], ci[INK2_C];
  const uint row = gpos.x, t = tpos.x;
  const float inv_t = cfg[4 * row];
  const size_t base = size_t(row) * V;
  float lm = -INFINITY;
  for (uint i = t; i < V; i += 1024u) lm = max(lm, L[base + i] * inv_t);
  lm = simd_max(lm);
  if (lane == 0u) fsh[sg] = lm;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float m = -INFINITY;
  for (uint g = 0u; g < 32u; g++) m = max(m, fsh[g]);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint want = min(kcap[row], min(V, INK2_C));
  uint prefix = 0u, pmask = 0u, above = 0u, need = want, ties = 0u;
  for (int shift = 24; shift >= 0; shift -= 8) {
    if (t < 256u) atomic_store_explicit(&hist[t], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = t; i < V; i += 1024u) {
      const uint k = ink2_key(L[base + i] * inv_t);
      if ((k & pmask) == prefix) atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0u) {
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
    above += st[1]; need -= st[1]; ties = st[2];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  uint idcut = INK2_END_ID;
  if (ties > need) {
    uint ipre = 0u, imask = 0u, ineed = need;
    for (int shift = (V > (1u << 24)) ? 24 : 16; shift >= 0; shift -= 8) {
      if (t < 256u) atomic_store_explicit(&hist[t], 0u, memory_order_relaxed);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint i = t; i < V; i += 1024u) {
        if (ink2_key(L[base + i] * inv_t) == prefix && (i & imask) == ipre)
          atomic_fetch_add_explicit(&hist[(i >> shift) & 255u], 1u, memory_order_relaxed);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (t == 0u) {
        uint cum = 0u, b = 0u;
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
    idcut = ipre;
  }
  ck[t] = INK2_END_KEY;
  ci[t] = INK2_END_ID;
  if (t == 0u) {
    atomic_store_explicit(&fill[0], 0u, memory_order_relaxed);
    atomic_store_explicit(&fill[1], 0u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = t; i < V; i += 1024u) {
    const uint k = ink2_key(L[base + i] * inv_t);
    if (k > prefix) {
      const uint at = atomic_fetch_add_explicit(&fill[0], 1u, memory_order_relaxed);
      ck[at] = k; ci[at] = i;
    } else if (k == prefix && i <= idcut) {
      const uint at = above + atomic_fetch_add_explicit(&fill[1], 1u, memory_order_relaxed);
      ck[at] = k; ci[at] = i;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint sort_n = 32u;
  while (sort_n < want) sort_n <<= 1;
  for (uint k = 2u; k <= sort_n; k <<= 1) {
    for (uint j = k >> 1; j > 0u; j >>= 1) {
      const uint p = t ^ j;
      if (p > t && p < sort_n) {
        const uint ka = ck[t], kb = ck[p], ia = ci[t], ib = ci[p];
        const bool a_first = ka > kb || (ka == kb && ia < ib);
        if (a_first != ((t & k) == 0u)) { ck[t] = kb; ck[p] = ka; ci[t] = ib; ci[p] = ia; }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  const ulong seed = ulong(seeds[2 * row]) | (ulong(seeds[2 * row + 1u]) << 32);
  ink2_eval_rule<MAP, 1024u>(
      want, m, cfg[4 * row + 1], cfg[4 * row + 3], seed, positions[row],
      IDS, TOK, row, t, ck, ci, st, scores);
}

template [[host_name("tf_ink2_rule_1024_f32")]] [[kernel]] decltype(tf_ink2_rule_1024<false>) tf_ink2_rule_1024<false>;
template [[host_name("tf_ink2_rule_1024_ids_f32")]] [[kernel]] decltype(tf_ink2_rule_1024<true>) tf_ink2_rule_1024<true>;
