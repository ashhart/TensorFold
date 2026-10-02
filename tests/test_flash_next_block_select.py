"""Flash Next's block select keeps the bits of the radix select over every block (0.6.1), on any score shape."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

if not mx.metal.is_available():
    pytest.skip("needs a Metal GPU", allow_module_level=True)

from tensorfold.kernels.qwen.flash_next.v1 import block_select  # noqa: E402
from tensorfold.kernels.qwen.flash_next.v1.base import ints, kernel  # noqa: E402

TOP = 512
# the kernel the sampled-threshold select replaced (0.6.1): the radix select over every block; the reference bits
_RADIX_ALL = r"""
  // One threadgroup (1024 threads) a row past TOP complete blocks: its TOP best blocks by score (radix select over
  // order-preserving keys, 8 bits a pass; among scores equal to the cut, the lowest block ids), written as the keys
  // they cover (4 a block) in position order, then the row's tail keys [4 complete, ENDS).
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
  const int r = int(threadgroup_position_in_grid.x);
  const int nb = COMPLETE[r];
  if (nb <= TOP) return;
  const int ends = ENDS[r];
  const int stride = SC_shape[1];
  const device float* sc = SC + size_t(r) * stride;
  device int* keys = KEYS + size_t(r) * KW;
  threadgroup atomic_uint hist[256];
  threadgroup uint cut_t, need_t;
  threadgroup int tot_a[32], tot_e[32];
  uint prefix = 0u, mask = 0u, need = TOP;
  for (int shift = 24; shift >= 0; shift -= 8) {
    if (t < 256) atomic_store_explicit(&hist[t], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int b = int(t); b < nb; b += 1024) {
      const uint k = tf_key(sc[b]);
      if ((k & mask) == prefix) atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {                            // the cut bin: lane l scans bins 255 - 8l down to 248 - 8l
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
        cut_t = prefix | (uint(bin) << shift);
        need_t = need - above;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    prefix = cut_t;
    need = need_t;
    mask |= 255u << shift;
  }
  // `prefix` is the cut score's key: every block above it is taken, and the first `need` equal to it
  const int chunk = (nb + 1023) / 1024;
  const int lo = min(nb, int(t) * chunk), hi = min(nb, lo + chunk);
  int n_above = 0, n_equal = 0;
  for (int b = lo; b < hi; b++) {
    const uint k = tf_key(sc[b]);
    n_above += k > prefix ? 1 : 0;
    n_equal += k == prefix ? 1 : 0;
  }
  int pa = simd_prefix_exclusive_sum(n_above), pe = simd_prefix_exclusive_sum(n_equal);
  if (lane == 31) { tot_a[sg] = pa + n_above; tot_e[sg] = pe + n_equal; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const int a = tot_a[lane], e = tot_e[lane];
    tot_a[lane] = simd_prefix_exclusive_sum(a);
    tot_e[lane] = simd_prefix_exclusive_sum(e);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  pa += tot_a[sg];
  pe += tot_e[sg];
  int out = pa + min(pe, int(need));
  for (int b = lo; b < hi; b++) {
    const uint k = tf_key(sc[b]);
    bool take = k > prefix;
    if (k == prefix) { take = pe < int(need); pe++; }
    if (take) {
      for (int j = 0; j < 4; j++) keys[out * 4 + j] = 4 * b + j;
      out++;
    }
  }
  if (t == 0)
    for (int k = 4 * nb; k < ends; k++) keys[4 * TOP + (k - 4 * nb)] = k;
"""
_TF_KEY = ("inline uint tf_key(float v) { uint b = as_type<uint>(v); "
           "return (b & 0x80000000u) ? ~b : (b | 0x80000000u); }")


def _radix_all(scores, complete, ends):
    rows = int(scores.shape[0])
    run = kernel("test_idx_select_radix_all", _RADIX_ALL, ["SC", "COMPLETE", "ENDS"], ["KEYS"], header=_TF_KEY)
    return run(inputs=[scores, ints(complete), ints(ends)], template=[("TOP", TOP), ("KW", 4 * TOP + 3)],
               grid=(1024 * rows, 1, 1), threadgroup=(1024, 1, 1), output_shapes=[(rows, 4 * TOP + 3)],
               output_dtypes=[mx.int32])[0]


def _scores(kind, rows, nb, seed):
    key = mx.random.key(seed)
    if kind == "relu":                      # sums of relu dots: many zeros, a few -0.0
        x = mx.maximum(mx.random.normal((rows, nb), key=key), 0.0)
        return mx.where(mx.arange(nb)[None] % 97 == 5, mx.array(-0.0), x)
    if kind == "narrow":
        return 1.0 + mx.random.uniform(0.0, 1.0, (rows, nb), key=key) * 0.01
    if kind == "ties":                      # few distinct scores: the cut falls inside a run of equal ones
        return mx.round(mx.random.uniform(0.0, 3.0, (rows, nb), key=key) * 4) / 4
    if kind == "periodic":                  # the high scores only where the evenly spaced samples never look
        high = 2.0 + mx.random.uniform(0.0, 1.0, (rows, nb), key=key)
        return mx.where((mx.arange(nb)[None] % 16) == 7, high, 0.5)
    return mx.arange(nb)[None].astype(mx.float32) / nb + mx.random.uniform(0.0, 0.3, (rows, nb), key=key)


@pytest.mark.parametrize("kind", ["relu", "narrow", "ties", "periodic", "trend"])
@pytest.mark.parametrize("nb,rows", [(513, 1), (700, 3), (2049, 4), (4100, 2), (16387, 4), (33000, 1), (65536, 2)])
def test_select_keeps_the_radix_over_every_block(kind, nb, rows):
    scores = _scores(kind, rows, nb, nb * 7 + rows).astype(mx.float32)
    complete = [max(1, nb - (rows - 1 - r) // 4) for r in range(rows)]
    complete[0] = min(complete[0], TOP) if rows > 2 else complete[0]          # a row at the top: not selected
    ends = [4 * c + r % 4 for r, c in enumerate(complete)]
    got = block_select.select_blocks(scores, complete, ends, top=TOP)
    want = _radix_all(scores, complete, ends)
    for r, c in enumerate(complete):
        if c > TOP:
            n = 4 * TOP + ends[r] - 4 * c
            assert bool(mx.array_equal(got[r, :n], want[r, :n]).item()), (kind, nb, r)
