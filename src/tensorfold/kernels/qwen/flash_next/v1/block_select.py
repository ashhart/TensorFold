"""Flash Next's block select: each row's TOP best complete blocks by score, as the keys they cover in position order."""

from __future__ import annotations

import mlx.core as mx

from tensorfold.kernels.qwen.flash_next.v1.base import ints, kernel

_SELECT_HEADER = r"""
inline uint tf_key(float v) { uint b = as_type<uint>(v); return (b & 0x80000000u) ? ~b : (b | 0x80000000u); }
// The cut bin of the 256 in `hist` (lane l scans bins 255 - 8l down to 248 - 8l): the cut key's next 8 bits, and
// how many of the `need` best keys lie in the bins above it.
inline void cut_bin(threadgroup atomic_uint* hist, uint lane, uint prefix, int shift, uint need,
                    threadgroup uint& cut, threadgroup uint& rest) {
  uint c[8], mine = 0u;
  for (int i = 0; i < 8; i++) {
    c[i] = atomic_load_explicit(&hist[255 - 8 * int(lane) - i], memory_order_relaxed);
    mine += c[i];
  }
  uint above = simd_prefix_exclusive_sum(mine);
  if (above < need && above + mine >= need) {
    int bin = 248 - 8 * int(lane);
    for (int i = 0; i < 8; i++) {
      if (above + c[i] >= need) { bin = 255 - 8 * int(lane) - i; break; }
      above += c[i];
    }
    cut = prefix | (uint(bin) << shift);
    rest = need - above;
  }
}
"""

CAP = 2048          # candidates a row's threadgroup memory holds
SAMPLED = 4096      # blocks from which a row samples a threshold first


def radix(n: str, key: str, ident: str) -> str:
    """Source of the radix select of the TOP best of ``n`` keys ``key(i)``, written as block ``ident(i)``'s keys."""

    k, b = key.format(i="i"), ident.format(i="i")
    return f"""
  {{
    uint prefix = 0u, mask = 0u, need = TOP;
    for (int shift = 24; shift >= 0; shift -= 8) {{
      if (t < 256) atomic_store_explicit(&hist[t], 0u, memory_order_relaxed);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int i = int(t); i < {n}; i += 1024) {{
        const uint k = {k};
        if ((k & mask) == prefix) atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], 1u, memory_order_relaxed);
      }}
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg == 0) cut_bin(hist, lane, prefix, shift, need, cut_t, need_t);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      prefix = cut_t;
      need = need_t;
      mask |= 255u << shift;
    }}
    const int chunk = ({n} + 1023) / 1024;
    const int lo = min({n}, int(t) * chunk), hi = min({n}, lo + chunk);
    int n_above = 0, n_equal = 0;
    for (int i = lo; i < hi; i++) {{
      const uint k = {k};
      n_above += k > prefix ? 1 : 0;
      n_equal += k == prefix ? 1 : 0;
    }}
    int pa = simd_prefix_exclusive_sum(n_above), pe = simd_prefix_exclusive_sum(n_equal);
    if (lane == 31) {{ tot_a[sg] = pa + n_above; tot_e[sg] = pe + n_equal; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {{
      const int a = tot_a[lane], e = tot_e[lane];
      tot_a[lane] = simd_prefix_exclusive_sum(a);
      tot_e[lane] = simd_prefix_exclusive_sum(e);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    pa += tot_a[sg];
    pe += tot_e[sg];
    int out = pa + min(pe, int(need));
    for (int i = lo; i < hi; i++) {{
      const uint k = {k};
      bool take = k > prefix;
      if (k == prefix) {{ take = pe < int(need); pe++; }}
      if (take) {{
        for (int j = 0; j < 4; j++) keys[out * 4 + j] = 4 * ({b}) + j;
        out++;
      }}
    }}
  }}
"""


_IDX_SELECT = r"""
  // One threadgroup (1024 threads) a row past TOP complete blocks. From SAMPLED blocks on, thread t samples block
  // t nb / 1024; the J-th best sample (J = 2 TOP 1024 / nb) is a threshold about 2 TOP blocks clear. Whenever TOP or
  // more clear it, every block at or above the TOP-th score does, so when TOP to CAP clear it they are kept in block
  // order in threadgroup memory and the radix select (8 bits a pass; among keys equal to the cut, the lowest block
  // ids) runs on them, else on every block: the same blocks in the same order. Then the tail keys [4 complete, ENDS).
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
  const int r = int(threadgroup_position_in_grid.x);
  const int nb = COMPLETE[r];
  if (nb <= TOP) return;
  const int ends = ENDS[r];
  const device float* sc = SC + size_t(r) * SC_shape[1];
  device int* keys = KEYS + size_t(r) * KW;
  threadgroup atomic_uint hist[256];
  threadgroup uint cut_t, need_t;
  threadgroup int tot_a[32], tot_e[32], count_t;
  threadgroup uint ck[CAP];
  threadgroup int cid[CAP];
  const int chunk0 = (nb + 1023) / 1024;
  const int lo0 = min(nb, int(t) * chunk0), hi0 = min(nb, lo0 + chunk0);
  uint thr = 0u;
  int at = 0, C = 0;
  if (nb >= SAMPLED) {                       // below SAMPLED blocks the radix over every block is as quick
    const uint sample = tf_key(sc[(long(t) * nb) / 1024]);
    uint prefix = 0u, mask = 0u, need = uint(clamp((2 * TOP * 1024) / nb, 1, 1024));
    for (int shift = 24; shift >= 0; shift -= 8) {
      if (t < 256) atomic_store_explicit(&hist[t], 0u, memory_order_relaxed);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if ((sample & mask) == prefix)
        atomic_fetch_add_explicit(&hist[(sample >> shift) & 255u], 1u, memory_order_relaxed);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg == 0) cut_bin(hist, lane, prefix, shift, need, cut_t, need_t);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      prefix = cut_t;
      need = need_t;
      mask |= 255u << shift;
    }
    thr = prefix;
    int cnt = 0;
    for (int b = lo0; b < hi0; b++) cnt += tf_key(sc[b]) >= thr ? 1 : 0;
    at = simd_prefix_exclusive_sum(cnt);
    if (lane == 31) tot_a[sg] = at + cnt;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      const int a = tot_a[lane], ex = simd_prefix_exclusive_sum(a);
      tot_a[lane] = ex;
      if (lane == 31) count_t = ex + a;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    at += tot_a[sg];
    C = count_t;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (C >= TOP && C <= CAP) {
    for (int b = lo0; b < hi0; b++) {
      const uint k = tf_key(sc[b]);
      if (k >= thr) { ck[at] = k; cid[at] = b; at++; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
""" + radix("C", "ck[{i}]", "cid[{i}]") + r"""
  } else {
""" + radix("nb", "tf_key(sc[{i}])", "{i}") + r"""
  }
  if (t == 0)
    for (int k = 4 * nb; k < ends; k++) keys[4 * TOP + (k - 4 * nb)] = k;
"""


def select_blocks(scores: mx.array, complete: list[int], ends: list[int], *, top: int) -> mx.array:
    """``index_select`` from block scores [R, NB] (fp32): each row's best ``top`` of its complete blocks."""

    rows = int(scores.shape[0])
    width = 4 * top + 3
    select = kernel("q4_idx_select", _IDX_SELECT, ["SC", "COMPLETE", "ENDS"], ["KEYS"], header=_SELECT_HEADER)
    return select(inputs=[scores, ints(complete), ints(ends)],
                  template=[("TOP", top), ("KW", width), ("CAP", CAP), ("SAMPLED", SAMPLED)],
                  grid=(1024 * rows, 1, 1), threadgroup=(1024, 1, 1),
                  output_shapes=[(rows, width)], output_dtypes=[mx.int32])[0]
