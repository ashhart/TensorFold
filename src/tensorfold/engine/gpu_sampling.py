"""Key GPU Gumbel draws by seed, position and token id so drafts match serial sampling on the same fp32 rule."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Sequence

import mlx.core as mx

from tensorfold.kernels import threads
from tensorfold.kernels.inputs import floats, ints, padded

CANDIDATES = 1024
# tokens this far (in logits / T) below the row's max are gathered directly when they hold the nucleus
NEAR = 20.0

_HEADER = r"""
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
"""

_SOURCE = r"""
  constexpr uint TG = 1024;
  constexpr uint NSG = TG / 32;
  const uint t = thread_position_in_threadgroup.x;
  const uint row = threadgroup_position_in_grid.x;
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const size_t base = size_t(row) * V;
  // the row's settings, read once (device memory: the compiler cannot keep them across the loops' stores)
  const float inv_t = cfg[4 * row];
  const float top_p = cfg[4 * row + 1];
  const float near = cfg[4 * row + 2];
  const float min_log = cfg[4 * row + 3];     // ln(min_p), -inf when off
  const uint kc = kcap[row];
  const uint cap = (kc == 0u || kc > C) ? C : kc;
  const ulong seed = ulong(seeds[2 * row]) | (ulong(seeds[2 * row + 1]) << 32);
  const uint position = positions[row];

  threadgroup float fsh[NSG];
  threadgroup float fsh2[NSG];
  threadgroup uint ush[NSG];
  threadgroup atomic_uint hist[256];
  threadgroup uint st[4];
  threadgroup uint ck[C];
  threadgroup uint ci[C];
  threadgroup atomic_uint fill_hi;
  threadgroup atomic_uint fill_tie;

  // row max (fixed order: each thread's stride, simd reduction, simdgroups in order)
  float lm = -INFINITY;
  for (uint i = t; i < V; i += TG) lm = max(lm, float(L[base + i]) * inv_t);
  lm = simd_max(lm);
  if (lane == 0) fsh[sg] = lm;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float m = -INFINITY;
  for (uint s = 0; s < NSG; s++) m = max(m, fsh[s]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // normalizer, and the count and mass of the tokens within NEAR, NEAR / 2 and NEAR / 4 of the max
  float ls = 0.0f, lnear[3] = {0.0f, 0.0f, 0.0f};
  uint near_count[3] = {0u, 0u, 0u};
  for (uint i = t; i < V; i += TG) {
    const float v = float(L[base + i]) * inv_t;
    const float e = metal::exp(v - m);
    ls += e;
    for (int w = 0; w < 3; w++) {
      if (v >= m - near / float(1 << w)) { lnear[w] += e; near_count[w]++; }
    }
  }
  ls = simd_sum(ls);
  if (lane == 0) fsh[sg] = ls;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float z = 0.0f;
  for (uint s = 0; s < NSG; s++) z += fsh[s];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // the widest window whose tokens are at most C and hold what the rule reads (top_p of the mass with a
  // margin, or the top_k tokens): those tokens are a top segment of the (value desc, id asc) order
  int window = -1;
  uint offset = 0u, n_near = 0u;
  for (int w = 0; w < 3; w++) {
    const float znear_part = simd_sum(lnear[w]);
    const uint before_in_simd = simd_prefix_exclusive_sum(near_count[w]);
    const uint simd_count = simd_sum(near_count[w]);
    if (lane == 0) { fsh2[sg] = znear_part; ush[sg] = simd_count; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float znear = 0.0f;
    uint off = before_in_simd, count = 0u;
    for (uint s = 0; s < NSG; s++) {
      znear += fsh2[s];
      if (s < sg) off += ush[s];
      count += ush[s];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool holds = count <= C && (kc == 0u
        ? (top_p > 0.0f && top_p < 1.0f && znear >= (top_p + 1e-4f) * z)
        : count >= min(kc, uint(C)));
    if (window < 0 && holds) { window = w; offset = off; n_near = count; }
  }
  const float floor_v = window < 0 ? INFINITY : m - near / float(1 << window);

  // When a window holds the rule's candidates they are gathered at offsets from the prefix sum above (no
  // atomics), then sorted; otherwise the radix path finds the C largest. Both give the same candidates for
  // the rule, so the same token.
  const bool fast = window >= 0;
  uint n_cand = 0u;
  uint sort_n = C;
  ck[t] = 0u;
  ci[t] = 0xFFFFFFFFu;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (fast) {
    uint at = offset;
    for (uint i = t; i < V; i += TG) {
      const float v = float(L[base + i]) * inv_t;
      if (v >= floor_v) { ck[at] = tf_key(v); ci[at] = i; at++; }
    }
    n_cand = n_near;
    sort_n = 32u;
    while (sort_n < n_near) sort_n <<= 1;
  } else {
    // key of the C-th largest value: radix select over the key's bytes, most significant first
    uint prefix = 0u, pmask = 0u, need = min(uint(C), uint(V)), above = 0u, ties = 0u;
    for (int shift = 24; shift >= 0; shift -= 8) {
      if (t < 256) atomic_store_explicit(&hist[t], 0u, memory_order_relaxed);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint i = t; i < V; i += TG) {
        const uint k = tf_key(float(L[base + i]) * inv_t);
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
      above += st[1]; need -= st[1]; ties = st[2];
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // more ties at that value than places left: the lowest ids (a second radix select, over ids)
    uint idcut = 0xFFFFFFFFu;
    if (ties > need) {
      uint ipre = 0u, imask = 0u, ineed = need;
      for (int shift = 16; shift >= 0; shift -= 8) {
        if (t < 256) atomic_store_explicit(&hist[t], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = t; i < V; i += TG) {
          if (tf_key(float(L[base + i]) * inv_t) == prefix && (i & imask) == ipre)
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
      idcut = ipre;
    }
    if (t == 0) {
      atomic_store_explicit(&fill_hi, 0u, memory_order_relaxed);
      atomic_store_explicit(&fill_tie, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = t; i < V; i += TG) {
      const uint k = tf_key(float(L[base + i]) * inv_t);
      if (k > prefix) {
        const uint s = atomic_fetch_add_explicit(&fill_hi, 1u, memory_order_relaxed);
        ck[s] = k; ci[s] = i;
      } else if (k == prefix && i <= idcut) {
        const uint s = above + atomic_fetch_add_explicit(&fill_tie, 1u, memory_order_relaxed);
        ck[s] = k; ci[s] = i;
      }
    }
    n_cand = above + need;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // bitonic sort by (value desc, id asc): the order does not depend on how the candidates were gathered
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

  // the nucleus: softmax over the whole row (top_k 0) or over the top_k, cut where it reaches top_p
  if (t == 0) {
    const uint n = min(n_cand, cap);
    float norm = z;
    if (kc != 0u) {
      norm = 0.0f;
      for (uint j = 0; j < n; j++) norm += metal::exp(tf_val(ck[j]) - m);
    }
    uint keep = n;
    if (top_p > 0.0f && top_p < 1.0f) {
      float cum = 0.0f;
      for (uint j = 0; j < n; j++) {
        cum += metal::exp(tf_val(ck[j]) - m) / norm;
        if (cum >= top_p) { keep = j + 1; break; }
      }
    }
    // min_p: the tokens within ln(min_p) of the top, a prefix of the order
    if (min_log > -INFINITY) {
      const float floor_p = m + min_log;
      uint j = 0;
      while (j < keep && tf_val(ck[j]) >= floor_p) j++;
      keep = j;
    }
    st[0] = keep;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint keep = st[0];

  // Gumbel-max over the kept candidates; ties go to the earlier candidate
  float score = -INFINITY;
  uint best = 0xFFFFFFFFu;
  if (t < keep) {
    score = tf_val(ck[t]) - metal::log(-metal::log(tf_uniform(seed, position, ci[t])));
    best = t;
  }
  const float sm = simd_max(score);
  const uint pick = simd_min(score == sm ? best : 0xFFFFFFFFu);
  if (lane == 0) { fsh[sg] = sm; ush[sg] = pick; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    float bs = -INFINITY;
    uint bj = 0xFFFFFFFFu;
    for (uint s = 0; s < NSG; s++) {
      if (fsh[s] > bs || (fsh[s] == bs && ush[s] < bj)) { bs = fsh[s]; bj = ush[s]; }
    }
    TOK[row] = ci[bj];
  }
"""

# Key subset-vocabulary noise by ascending token ids rather than columns, returning an id with the target's noise.
_SOURCE_IDS = (_SOURCE.replace("tf_uniform(seed, position, ci[t])", "tf_uniform(seed, position, IDS[ci[t]])")
               .replace("TOK[row] = ci[bj];", "TOK[row] = IDS[ci[bj]];"))
assert _SOURCE_IDS.count("IDS[") == 2

_kernels: dict[tuple[bool, int], Any] = {}


def _get_kernel(vocab: int, ids: bool = False) -> Any:
    """The kernel for this vocabulary, reserving its 1024 threads (the reductions' order is theirs) on every GPU."""

    kernel = _kernels.get((ids, vocab))
    if kernel is None:
        consts = f"  constexpr int V = {vocab};\n  constexpr int C = {CANDIDATES};\n"
        source = consts + (_SOURCE_IDS if ids else _SOURCE)
        header = _HEADER + threads.reserve(1024)
        digest = hashlib.sha256((header + source).encode()).hexdigest()[:16]
        kernel = _kernels[(ids, vocab)] = mx.fast.metal_kernel(
            name=f"tf_gpu_sample{'_ids' if ids else ''}_{digest}", output_names=["TOK"], source=source, header=header,
            input_names=["L", "seeds", "positions", "cfg", "kcap", *(["IDS"] if ids else [])])
    return kernel


def sample(logits: mx.array, sampling: Any, positions: Sequence[int] | mx.array, ids: mx.array | None = None
           ) -> mx.array:
    """Sample logits [R, V] at absolute positions; optional ascending ``ids`` map columns to returned token ids and key the noise."""

    logits = logits.reshape(-1, logits.shape[-1])
    return sample_rows(logits, [sampling] * int(logits.shape[0]), positions, ids)


def sample_rows(logits: mx.array, samplings: Sequence[Any], positions: Sequence[int] | mx.array,
                ids: mx.array | None = None) -> mx.array:
    """Sample rows with their own settings in one kernel, with the same result each row would get alone."""

    logits = logits.reshape(-1, logits.shape[-1])
    rows, vocab = logits.shape
    greedy = [s is None for s in samplings]
    picked = None
    if any(greedy):
        picked = mx.argmax(logits, axis=-1).astype(mx.uint32)
        if all(greedy):
            return picked if ids is None else ids[picked]
    seeds: list[int] = []
    cfg: list[float] = []
    caps: list[int] = []
    for s in samplings:
        seed = int(s.seed) & 0xFFFFFFFFFFFFFFFF if s is not None else 0
        seeds += [seed & 0xFFFFFFFF, seed >> 32]
        cfg += ([1.0 / max(float(s.temperature), 1e-6), float(s.top_p), NEAR, getattr(s, "min_log", -math.inf)]
                if s is not None else [1.0, 1.0, NEAR, -math.inf])
        caps.append(int(s.top_k or 0) if s is not None else 1)
    if isinstance(positions, mx.array):
        positions = padded(positions.astype(mx.uint32))
    else:
        positions = ints(positions, mx.uint32)
    inputs = [logits, ints(seeds, mx.uint32), positions, floats(cfg), ints(caps, mx.uint32)]
    out = _get_kernel(int(vocab), ids is not None)(inputs=inputs if ids is None else [*inputs, ids],
                                                  grid=(1024 * rows, 1, 1), threadgroup=(1024, 1, 1),
                                                  output_shapes=[(rows,)], output_dtypes=[mx.uint32])[0]
    if picked is not None:
        out = mx.where(mx.array(greedy), picked if ids is None else ids[picked], out)
    return out


def reference(values: Any, sampling: Any, position: int) -> int:
    """The same rule in numpy (fp32 where the kernel is fp32): for tests."""

    import numpy as np

    v = (np.asarray(values, dtype=np.float32) * np.float32(1.0 / max(float(sampling.temperature), 1e-6)))
    v = v.astype(np.float32)
    ids = np.arange(v.shape[0], dtype=np.int64)
    order = np.lexsort((ids, -v))[:CANDIDATES]
    cap = CANDIDATES if not sampling.top_k else min(int(sampling.top_k), CANDIDATES)
    m = v.max()
    top = order[:cap]
    norm = np.exp(v - m).astype(np.float32).sum(dtype=np.float32) if not sampling.top_k else \
        np.exp(v[top] - m).astype(np.float32).sum(dtype=np.float32)
    keep = len(top)
    if 0.0 < sampling.top_p < 1.0:
        cum = np.cumsum((np.exp(v[top] - m) / norm).astype(np.float32), dtype=np.float32)
        hit = np.nonzero(cum >= np.float32(sampling.top_p))[0]
        keep = int(hit[0]) + 1 if len(hit) else len(top)
    if getattr(sampling, "min_p", 0.0) > 0.0:
        keep = int((v[top[:keep]] >= np.float32(m + np.float32(sampling.min_log))).sum())
    kept = top[:keep]
    mask = np.uint64(0xFFFFFFFFFFFFFFFF)

    def mix(x):
        x = x ^ (x >> np.uint64(30)); x = x * np.uint64(0xBF58476D1CE4E5B9)
        x = x ^ (x >> np.uint64(27)); x = x * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))

    with np.errstate(over="ignore"):
        x = mix(np.uint64(int(sampling.seed) & 0xFFFFFFFFFFFFFFFF) + np.uint64(0x9E3779B97F4A7C15))
        x = mix(x ^ (np.uint64(position) * np.uint64(0xD1B54A32D192ED03)))
        x = mix(x ^ kept.astype(np.uint64))
    del mask
    u = ((x >> np.uint64(40)).astype(np.float32) + np.float32(0.5)) * np.float32(1.0 / 16777216.0)
    score = v[kept] - np.log(-np.log(u))
    return int(kept[int(np.argmax(score))])


__all__ = ["CANDIDATES", "reference", "sample"]
