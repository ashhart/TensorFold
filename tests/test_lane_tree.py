"""Tree verification pieces: every tree node gets the bits of serial decoding along its path."""

import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.kernels import lane_tree  # noqa: E402


def _same(a, b):
    return bool(mx.all(a.view(mx.uint16) == b.view(mx.uint16)).item())


def test_tree_paths():
    depths, paths = lane_tree.tree_paths([-1, 0, 0, 1, 3, 2])
    assert depths == [0, 1, 1, 2, 3, 2]
    assert paths[4] == [0, 1, 3, 4] and paths[5] == [0, 2, 5]


def test_gated_delta_tree_matches_serial_steps():
    from mlx_lm.models.gated_delta import gated_delta_update

    mx.random.seed(0)
    W, Hk, Hv, Dk, Dv = 7, 16, 48, 128, 128
    parents = [-1, 0, 0, 1, 3, 2, 5]
    q = (mx.random.normal((1, W, Hk, Dk)) * 0.1).astype(mx.bfloat16)
    k = (mx.random.normal((1, W, Hk, Dk)) * 0.1).astype(mx.bfloat16)
    v = (mx.random.normal((1, W, Hv, Dv)) * 0.5).astype(mx.bfloat16)
    a = mx.random.normal((1, W, Hv)).astype(mx.bfloat16)
    b = mx.random.normal((1, W, Hv)).astype(mx.bfloat16)
    A_log = mx.random.normal((Hv,)).astype(mx.float32) * 0.1
    dt_bias = mx.random.normal((Hv,)).astype(mx.bfloat16)
    state = (mx.random.normal((1, Hv, Dv, Dk)) * 0.2).astype(mx.float32)
    try:
        from mlx_lm.models.gated_delta import compute_g

        g = compute_g(A_log, a, dt_bias)
        beta = mx.sigmoid(b)
        tree = lane_tree.gated_delta_tree(q, k, v, g, beta, state, parents)
        mx.eval(tree)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"metal kernels unavailable: {str(exc).splitlines()[0][:80]}")
    _, paths = lane_tree.tree_paths(parents)
    for node, path in enumerate(paths):
        s = state
        out = None
        for row in path:   # serial: one step at a time, mlx_lm's own kernel
            out, s = gated_delta_update(q[:, row:row + 1], k[:, row:row + 1], v[:, row:row + 1],
                                        a[:, row:row + 1], b[:, row:row + 1], A_log, dt_bias, s)
        mx.eval(out)
        assert _same(tree[:, node:node + 1], out), f"node {node} differs from its serial walk"


@pytest.mark.parametrize("P", [3, 250, 255, 256, 500, 700, 1020, 2047, 2048, 20000])
def test_tree_attention_matches_serial_paths(P):
    from tensorfold.kernels import lane_attention

    mx.random.seed(P)
    H, HKV, D = 24, 4, 256
    parents = [-1, 0, 0, 1, 1, 3, 2, 6, 7, 5]
    W = len(parents)
    L = P + W
    kb = (mx.random.normal((1, HKV, L + 40, D)) * 0.6).astype(mx.bfloat16)
    vb = (mx.random.normal((1, HKV, L + 40, D)) * 0.6).astype(mx.bfloat16)
    q = (mx.random.normal((1, H, W, D)) * 0.6).astype(mx.bfloat16)
    try:
        tree = lane_attention.lane_tree_sdpa(q, kb[:, :, :L], vb[:, :, :L], 0.0625, parents)
        mx.eval(tree)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"metal kernels unavailable: {str(exc).splitlines()[0][:80]}")
    _, paths = lane_tree.tree_paths(parents)
    for node, path in enumerate(paths):
        rows = list(range(P)) + [P + r for r in path]         # the keys serial decoding would hold
        idx = mx.array(rows, dtype=mx.int32)
        ks = mx.take(kb, idx, axis=2)
        vs = mx.take(vb, idx, axis=2)
        one = lane_attention.lane_sdpa(q[:, :, node:node + 1], ks, vs, 0.0625)
        mx.eval(one)
        assert _same(one, tree[:, :, node:node + 1]), f"node {node} differs from its serial path"


def test_chain_through_tree_kernel_equals_chain_kernel():
    from tensorfold.kernels import lane_attention

    mx.random.seed(9)
    H, HKV, D, W, P = 24, 4, 256, 8, 1000
    kb = (mx.random.normal((1, HKV, P + W, D)) * 0.6).astype(mx.bfloat16)
    vb = (mx.random.normal((1, HKV, P + W, D)) * 0.6).astype(mx.bfloat16)
    q = (mx.random.normal((1, H, W, D)) * 0.6).astype(mx.bfloat16)
    chain = lane_attention.lane_sdpa(q, kb, vb, 0.0625)
    tree = lane_attention.lane_tree_sdpa(q, kb, vb, 0.0625, [-1] + list(range(W - 1)))
    mx.eval(chain, tree)
    assert _same(chain, tree)


@pytest.mark.parametrize("W", [5, 40, 64, 128])
def test_long_chain_recurrence_equals_serial_steps(W):
    """Chains keep one state slot (up to 64 rows); every row equals mlx_lm's serial step."""
    from mlx_lm.models.gated_delta import compute_g, gated_delta_update

    mx.random.seed(W)
    Hk, Hv, Dk, Dv = 16, 48, 128, 128
    parents = [-1] + list(range(W - 1))
    q = (mx.random.normal((1, W, Hk, Dk)) * 0.1).astype(mx.bfloat16)
    k = (mx.random.normal((1, W, Hk, Dk)) * 0.1).astype(mx.bfloat16)
    v = (mx.random.normal((1, W, Hv, Dv)) * 0.5).astype(mx.bfloat16)
    a = mx.random.normal((1, W, Hv)).astype(mx.bfloat16)
    b = mx.random.normal((1, W, Hv)).astype(mx.bfloat16)
    A_log = mx.random.normal((Hv,)).astype(mx.float32) * 0.1
    dt_bias = mx.random.normal((Hv,)).astype(mx.bfloat16)
    state = (mx.random.normal((1, Hv, Dv, Dk)) * 0.2).astype(mx.float32)
    try:
        chain = lane_tree.gated_delta_tree(q, k, v, compute_g(A_log, a, dt_bias), mx.sigmoid(b), state, parents)
        mx.eval(chain)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"metal kernels unavailable: {str(exc).splitlines()[0][:80]}")
    s = state
    for row in range(W):
        out, s = gated_delta_update(q[:, row:row + 1], k[:, row:row + 1], v[:, row:row + 1],
                                    a[:, row:row + 1], b[:, row:row + 1], A_log, dt_bias, s)
        mx.eval(out)
        assert _same(chain[:, row:row + 1], out), f"row {row} of a {W}-row chain differs from serial"


def test_64_row_chain_attention_equals_single_queries():
    from tensorfold.kernels import lane_attention

    mx.random.seed(64)
    H, HKV, D, W, P = 24, 4, 256, 64, 1000          # the window crosses a chunk boundary
    kb = (mx.random.normal((1, HKV, P + W, D)) * 0.6).astype(mx.bfloat16)
    vb = (mx.random.normal((1, HKV, P + W, D)) * 0.6).astype(mx.bfloat16)
    q = (mx.random.normal((1, H, W, D)) * 0.6).astype(mx.bfloat16)
    tree = lane_attention.lane_tree_sdpa(q, kb, vb, 0.0625, [-1] + list(range(W - 1)))
    mx.eval(tree)
    for t in (0, 17, 40, 63):
        one = lane_attention.lane_sdpa(q[:, :, t:t + 1], kb[:, :, :P + t + 1], vb[:, :, :P + t + 1], 0.0625)
        assert _same(one, tree[:, :, t:t + 1]), f"row {t} of a 64-row chain differs from its single query"


def test_128_row_chain_attention_equals_single_queries():
    from tensorfold.kernels import lane_attention

    mx.random.seed(128)
    H, HKV, D, W, P = 24, 4, 256, 128, 3000
    kb = (mx.random.normal((1, HKV, P + W, D)) * 0.6).astype(mx.bfloat16)
    vb = (mx.random.normal((1, HKV, P + W, D)) * 0.6).astype(mx.bfloat16)
    q = (mx.random.normal((1, H, W, D)) * 0.6).astype(mx.bfloat16)
    tree = lane_attention.lane_tree_sdpa(q, kb, vb, 0.0625, [-1] + list(range(W - 1)))
    mx.eval(tree)
    for t in (0, 63, 64, 100, 127):
        one = lane_attention.lane_tree_sdpa(q[:, :, t:t + 1], kb[:, :, :P + t + 1], vb[:, :, :P + t + 1], 0.0625, [-1])
        assert _same(one, tree[:, :, t:t + 1]), f"row {t} of a 128-row chain differs from its single query"

