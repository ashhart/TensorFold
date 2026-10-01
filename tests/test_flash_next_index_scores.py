"""Flash Next's block scores read each pooled block once for several rows and keep one-block-a-simdgroup's bits."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

if not mx.metal.is_available():
    pytest.skip("needs a Metal GPU", allow_module_level=True)

from tensorfold.kernels.qwen.flash_next.v1 import attention  # noqa: E402
from tensorfold.kernels.qwen.flash_next.v1.base import ints, kernel, pick  # noqa: E402

# the kernel the multi-block one replaced (0.6.1): a simdgroup a block, rows in grid y; the reference bits
_ONE_BLOCK = r"""
  const uint lane = thread_index_in_simdgroup;
  const int b = int(threadgroup_position_in_grid.x) * 8 + int(simdgroup_index_in_threadgroup);
  const int r = int(threadgroup_position_in_grid.y);
  const int complete = COMPLETE[r];
  const int sb = SROW[r];
  if (complete <= TOP || b >= complete) return;
  constexpr int PER = DI / 32;
  const device bfloat* pb = POOLS + size_t(b) * DI + lane * PER;
  float p[PER];
  for (int i = 0; i < PER; i++) p[i] = float(pb[i]);
  float s = 0.0f;
  for (int h = 0; h < HI; h++) {
    const device bfloat* qh = Q + (r * HI + h) * DI + lane * PER;
    float dot = 0.0f;
    for (int i = 0; i < PER; i++) dot = fma(float(qh[i]), p[i], dot);
    s += metal::max(simd_sum(dot), 0.0f);
  }
  if (lane == 0) SC[size_t(r) * STRIDE[0] + b] = s / metal::precise::sqrt(float(DI));
"""


def _one_block(q, pooled, stream_of_row, complete, top):
    rows, heads, dims = q.shape
    nb = max(int(p.shape[0]) for p in pooled)
    names = ["Q"] + [f"POOLED{b}" for b in range(len(pooled))] + ["COMPLETE", "SROW", "STRIDE"]
    src = _ONE_BLOCK.replace("POOLS", pick("POOLED", len(pooled), "sb"))
    run = kernel(f"test_idx_scores_one_block{len(pooled)}", src, names, ["SC"])
    return run(inputs=[q, *pooled, ints(complete), ints(stream_of_row), mx.array([nb], dtype=mx.int32)],
               template=[("HI", heads), ("DI", dims), ("TOP", top)], grid=(-(-nb // 8) * 256, rows, 1),
               threadgroup=(256, 1, 1), output_shapes=[(rows, nb)], output_dtypes=[mx.float32])[0]


def test_blocks_a_simdgroup_keep_enough_threadgroups():
    assert [attention.score_blocks(b, 4) for b in (600, 2156, 4096, 8192, 16384, 65536)] == [1, 1, 2, 4, 8, 8]
    assert attention.score_blocks(65536, 1) == 1


def _scored(scores, complete, top):
    """Each scored row's complete blocks (the rest is unset)."""

    return [scores[r, :c] for r, c in enumerate(complete) if c > top]


def _inputs(rng, rows, blocks, heads=4, dims=128):
    q = mx.array(rng.normal(size=(rows, heads, dims)).astype(np.float32)).astype(mx.bfloat16)
    pooled = mx.array(rng.normal(size=(blocks, dims)).astype(np.float32)).astype(mx.bfloat16)
    return q, pooled


@pytest.mark.parametrize("rows,blocks,top", [(1, 40, 16), (3, 515, 512), (8, 2051, 512), (9, 700, 16),
                                             (16, 16387, 512), (4, 64, 70), (2, 4100, 512), (4, 8200, 512),
                                             (1, 16387, 512), (5, 33000, 512)])
def test_scores_equal_one_block_a_simdgroup(rows, blocks, top):
    rng = np.random.default_rng(rows * 1000 + blocks)
    q, pooled = _inputs(rng, rows, blocks)
    # rows of a window end at consecutive positions; one row sits at or below top (not scored)
    complete = [max(1, blocks - (rows - 1 - r) // 4) for r in range(rows)]
    if rows > 2:
        complete[0] = min(complete[0], top)
    ends = [4 * c + r % 4 for r, c in enumerate(complete)]
    new = attention.index_scores(q, pooled, complete, top=top)
    ref = _one_block(q, [pooled], [0] * rows, complete, top)
    for a, b in zip(_scored(new, complete, top), _scored(ref, complete, top)):
        assert bool(mx.array_equal(a, b).item())
    keys = attention.index_select(q, pooled, complete, ends, top=top)
    want = attention.select_blocks(ref, complete, ends, top=top)
    for r, c in enumerate(complete):
        if c > top:
            n = 4 * top + ends[r] - 4 * c
            assert bool(mx.array_equal(keys[r, :n], want[r, :n]).item()), r


@pytest.mark.parametrize("rows", [[2, 1, 3], [1, 9, 2], [4, 4, 4, 4]])
def test_scores_multi_equal_one_block_a_simdgroup(rows):
    rng = np.random.default_rng(sum(rows) * 7 + len(rows))
    top = 64
    sizes = [70 + 300 * b for b in range(len(rows))]
    qs, pooled = zip(*[_inputs(rng, n, blocks) for n, blocks in zip(rows, sizes)])
    q = mx.concatenate(qs)
    srow = [b for b, n in enumerate(rows) for _ in range(n)]
    complete = [sizes[b] - (rows[b] - 1 - r) // 4 for b, n in enumerate(rows) for r in range(n)]
    ends = [4 * c + 1 for c in complete]
    new = attention.index_scores_multi(q, list(pooled), srow, complete, top=top)
    ref = _one_block(q, list(pooled), srow, complete, top)
    for a, b in zip(_scored(new, complete, top), _scored(ref, complete, top)):
        assert bool(mx.array_equal(a, b).item())
    keys = attention.index_select_multi(q, list(pooled), srow, complete, ends, top=top)
    want = attention.select_blocks(ref, complete, ends, top=top)
    for r, c in enumerate(complete):
        if c > top:
            n = 4 * top + ends[r] - 4 * c
            assert bool(mx.array_equal(keys[r, :n], want[r, :n]).item()), r
