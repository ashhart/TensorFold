"""row_attention: a window row's (and a tree node's) attention has the bits of the one-row step at its position,
and matches MLX's attention to fp32 rounding."""

import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.kernels.qwen.dense.v1 import row_attention  # noqa: E402


def _same(a, b):
    return a.shape == b.shape and bool(mx.all(a.view(mx.uint16) == b.view(mx.uint16)).item())


@pytest.mark.parametrize("P", [0, 5, 63, 64, 200, 1000])
def test_tree_rows_equal_their_serial_steps(P):
    H, HKV, D = 24, 4, 256
    parents = [-1, 0, 0, 1, 2, 3, 3, 6]
    W = len(parents)
    cap = P + W + 7
    mx.random.seed(P)
    kb = (mx.random.normal((1, HKV, cap, D)) * 0.6).astype(mx.bfloat16)
    vb = (mx.random.normal((1, HKV, cap, D)) * 0.6).astype(mx.bfloat16)
    q = (mx.random.normal((1, H, W, D)) * 0.6).astype(mx.bfloat16)
    tree = row_attention.row_sdpa(q, kb, vb, 0.0625, P, parents)
    _, paths = row_attention.paths_of(parents)
    for node, path in enumerate(paths):
        # the serial step: the committed keys, then the node's path, contiguous, the node itself last
        rows = list(range(P)) + [P + r for r in path]
        idx = mx.array(rows, dtype=mx.int32)
        ks = mx.take(kb, idx, axis=2)
        vs = mx.take(vb, idx, axis=2)
        one = row_attention.row_sdpa(q[:, :, node:node + 1], ks, vs, 0.0625, len(rows) - 1, [-1])
        assert _same(tree[:, :, node:node + 1], one), f"node {node} at P={P} differs from its serial step"


def test_chains_match_mlx_attention():
    H, HKV, D, P, W = 24, 4, 256, 300, 5
    mx.random.seed(3)
    kb = (mx.random.normal((1, HKV, P + W, D)) * 0.6).astype(mx.bfloat16)
    vb = (mx.random.normal((1, HKV, P + W, D)) * 0.6).astype(mx.bfloat16)
    q = (mx.random.normal((1, H, W, D)) * 0.6).astype(mx.bfloat16)
    ours = row_attention.row_sdpa(q, kb, vb, 0.0625, P, list(range(-1, W - 1)))
    ref = mx.fast.scaled_dot_product_attention(q, kb, vb, scale=0.0625, mask="causal")
    diff = mx.abs(ours.astype(mx.float32) - ref.astype(mx.float32)).max().item()
    assert diff < 2e-2, diff


def test_one_kernel_signature_for_every_window():
    """row_sdpa's kernels keep one Metal signature for windows of 1 to 15 rows, chains and a tree."""

    from tests.kernel_signatures import changed, recording

    H, HKV, D, P = 24, 4, 256, 70
    windows = ([-1], [-1, 0], [-1, 0, 1], list(range(-1, 7)), list(range(-1, 8)), [-1, 0, 0, 1, 2, 3, 3, 6],
               list(range(-1, 14)), [-1])
    saved = dict(row_attention._kernels)
    try:
        with recording() as seen:
            row_attention._kernels.clear()                     # made again, inside the recorder
            for i, parents in enumerate(windows):
                W = len(parents)
                mx.random.seed(i)
                kb = (mx.random.normal((1, HKV, P + W, D)) * 0.6).astype(mx.bfloat16)
                vb = (mx.random.normal((1, HKV, P + W, D)) * 0.6).astype(mx.bfloat16)
                q = (mx.random.normal((1, H, W, D)) * 0.6).astype(mx.bfloat16)
                out = row_attention.row_sdpa(q, kb, vb, 0.0625, P, parents)
                assert bool(mx.all(mx.isfinite(out)).item())
    finally:
        row_attention._kernels.clear()
        row_attention._kernels.update(saved)
    assert {name.rsplit("_", 1)[0] for name, _ in seen} == {"row_attention_partial", "row_attention_merge"}
    assert not changed(seen), "kernels called with more than one signature: " + changed(seen)
