"""Tree verification pieces: every tree node gets the bits of serial decoding along its path."""

import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.kernels.qwen.dense.v1 import lane_tree  # noqa: E402


def _same(a, b):
    return bool(mx.all(a.view(mx.uint16) == b.view(mx.uint16)).item())


def _tree(q, k, v, g, beta, state, parents):
    """One stream's tree recurrence (``stream_gdn.tree`` with one stream)."""
    from tensorfold.kernels.qwen.dense.v1 import stream_gdn

    return stream_gdn.tree(q, k, v, g, beta, [state], stream_gdn.TreePlan([parents]))


def _tree_sdpa(q, kb, vb, start, parents):
    """One stream's tree attention: the window's rows sit at [start, start + len(parents)) of the buffers."""
    from tensorfold.kernels.qwen.dense.v1 import stream_attention

    plan = stream_attention.Plan([parents], [start], int(q.shape[1]), int(kb.shape[1]))
    return stream_attention.tree_sdpa(q, [(kb, vb)], 0.0625, plan)


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
        tree = _tree(q, k, v, g, beta, state, parents)
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
    from tensorfold.kernels.qwen.dense.v1 import lane_attention

    mx.random.seed(P)
    H, HKV, D = 24, 4, 256
    parents = [-1, 0, 0, 1, 1, 3, 2, 6, 7, 5]
    W = len(parents)
    L = P + W
    kb = (mx.random.normal((1, HKV, L + 40, D)) * 0.6).astype(mx.bfloat16)
    vb = (mx.random.normal((1, HKV, L + 40, D)) * 0.6).astype(mx.bfloat16)
    q = (mx.random.normal((1, H, W, D)) * 0.6).astype(mx.bfloat16)
    try:
        tree = _tree_sdpa(q, kb, vb, P, parents)
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
    from tensorfold.kernels.qwen.dense.v1 import lane_attention

    mx.random.seed(9)
    H, HKV, D, W, P = 24, 4, 256, 8, 1000
    kb = (mx.random.normal((1, HKV, P + W, D)) * 0.6).astype(mx.bfloat16)
    vb = (mx.random.normal((1, HKV, P + W, D)) * 0.6).astype(mx.bfloat16)
    q = (mx.random.normal((1, H, W, D)) * 0.6).astype(mx.bfloat16)
    chain = lane_attention.lane_sdpa(q, kb, vb, 0.0625)
    tree = _tree_sdpa(q, kb, vb, P, [-1] + list(range(W - 1)))
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
        chain = _tree(q, k, v, compute_g(A_log, a, dt_bias), mx.sigmoid(b), state, parents)
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
    from tensorfold.kernels.qwen.dense.v1 import lane_attention

    mx.random.seed(64)
    H, HKV, D, W, P = 24, 4, 256, 64, 1000          # the window crosses a chunk boundary
    kb = (mx.random.normal((1, HKV, P + W, D)) * 0.6).astype(mx.bfloat16)
    vb = (mx.random.normal((1, HKV, P + W, D)) * 0.6).astype(mx.bfloat16)
    q = (mx.random.normal((1, H, W, D)) * 0.6).astype(mx.bfloat16)
    tree = _tree_sdpa(q, kb, vb, P, [-1] + list(range(W - 1)))
    mx.eval(tree)
    for t in (0, 17, 40, 63):
        one = lane_attention.lane_sdpa(q[:, :, t:t + 1], kb[:, :, :P + t + 1], vb[:, :, :P + t + 1], 0.0625)
        assert _same(one, tree[:, :, t:t + 1]), f"row {t} of a 64-row chain differs from its single query"


def test_128_row_chain_attention_equals_single_queries():
    mx.random.seed(128)
    H, HKV, D, W, P = 24, 4, 256, 128, 3000
    kb = (mx.random.normal((1, HKV, P + W, D)) * 0.6).astype(mx.bfloat16)
    vb = (mx.random.normal((1, HKV, P + W, D)) * 0.6).astype(mx.bfloat16)
    q = (mx.random.normal((1, H, W, D)) * 0.6).astype(mx.bfloat16)
    tree = _tree_sdpa(q, kb, vb, P, [-1] + list(range(W - 1)))
    mx.eval(tree)
    for t in (0, 63, 64, 100, 127):
        one = _tree_sdpa(q[:, :, t:t + 1], kb, vb, P + t, [-1])
        assert _same(one, tree[:, :, t:t + 1]), f"row {t} of a 128-row chain differs from its single query"



@pytest.mark.parametrize("case", [([30, 700, 1405], [16, 5, 1]), ([64, 511, 512, 2047, 5000], [3, 16, 8, 1, 13]),
                                  ([10] * 8, [2] * 8), ([4000, 20000], [32, 9])])
def test_streams_in_one_launch_equal_one_stream_calls(case):
    """Every row of a multi-stream call has the bits of the same row from its own stream's call."""
    import random

    from tensorfold.kernels.qwen.dense.v1 import stream_attention

    starts, widths = case
    rng = random.Random(len(starts))
    mx.random.seed(len(starts))
    H, HKV, D = 24, 4, 256
    parents, kv, qs = [], [], []
    for P, W in zip(starts, widths):
        parents.append(list(range(-1, W - 1)) if W == 32 else [-1] + [rng.randrange(0, i) for i in range(1, W)])
        cap = 256 * (-(-(P + W + 7) // 256))
        kv.append(((mx.random.normal((1, HKV, cap, D)) * 0.6).astype(mx.bfloat16),
                   (mx.random.normal((1, HKV, cap, D)) * 0.6).astype(mx.bfloat16)))
        qs.append((mx.random.normal((1, H, W, D)) * 0.6).astype(mx.bfloat16))
    out = stream_attention.tree_sdpa(mx.concatenate(qs, axis=2), kv, 0.0625,
                                     stream_attention.Plan(parents, starts, H, HKV))
    mx.eval(out)
    first = 0
    for s, (P, W) in enumerate(zip(starts, widths)):
        one = _tree_sdpa(qs[s], kv[s][0], kv[s][1], P, parents[s])
        assert _same(out[:, :, first:first + W], one), f"stream {s} differs from its own call"
        first += W


def test_streams_replay_and_conv_tails_equal_serial_steps():
    """One launch replays every stream's kept rows from its own state (mlx_lm's serial steps, bit for bit) and takes
    each stream's next conv tail from its own conv state and kept rows."""
    from mlx_lm.models.gated_delta import compute_g, gated_delta_update

    from tensorfold.kernels.qwen.dense.v1 import stream_gdn

    mx.random.seed(21)
    Hk, Hv, Dk, Dv, C, n_keep = 16, 48, 128, 128, 64, 3
    widths, paths = [5, 1, 9], [[0, 2, 4], [0], [0, 1, 2, 3, 5, 8]]
    firsts = [0, 5, 6]
    R = sum(widths)
    q = (mx.random.normal((1, R, Hk, Dk)) * 0.1).astype(mx.bfloat16)
    k = (mx.random.normal((1, R, Hk, Dk)) * 0.1).astype(mx.bfloat16)
    v = (mx.random.normal((1, R, Hv, Dv)) * 0.5).astype(mx.bfloat16)
    a = mx.random.normal((1, R, Hv)).astype(mx.bfloat16)
    b = mx.random.normal((1, R, Hv)).astype(mx.bfloat16)
    A_log = mx.random.normal((Hv,)).astype(mx.float32) * 0.1
    dt_bias = mx.random.normal((Hv,)).astype(mx.bfloat16)
    g, beta = compute_g(A_log, a, dt_bias), mx.sigmoid(b)
    states = [(mx.random.normal((1, Hv, Dv, Dk)) * 0.2).astype(mx.float32) for _ in widths]
    convs = [mx.random.normal((1, n_keep, C)).astype(mx.bfloat16) for _ in widths]
    qkv = mx.random.normal((1, R, C)).astype(mx.bfloat16)
    plan = stream_gdn.CommitPlan(paths, firsts, n_keep)
    try:
        outs = stream_gdn.replay(q, k, v, g, beta, states, plan)
        tails = stream_gdn.conv_tails(convs, qkv, plan)
        mx.eval(outs, tails)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"metal kernels unavailable: {str(exc).splitlines()[0][:80]}")
    for s, (path, first) in enumerate(zip(paths, firsts)):
        state = states[s]
        for r in path:
            row = first + r
            _, state = gated_delta_update(q[:, row:row + 1], k[:, row:row + 1], v[:, row:row + 1],
                                          a[:, row:row + 1], b[:, row:row + 1], A_log, dt_bias, state)
        mx.eval(state)
        assert bool(mx.all(outs[s] == state).item()), f"stream {s}: replayed state differs from serial steps"
        seq = mx.concatenate([convs[s], qkv[:, [first + r for r in path]]], axis=1)
        assert _same(tails[s], seq[:, -n_keep:]), f"stream {s}: conv tail differs"


def test_streams_tree_recurrence_in_one_launch_equals_one_stream_calls():
    """Every row of a multi-stream recurrence launch has the bits of the same row from its own stream's launch."""
    from mlx_lm.models.gated_delta import compute_g

    from tensorfold.kernels.qwen.dense.v1 import stream_gdn

    mx.random.seed(33)
    Hk, Hv, Dk, Dv = 16, 48, 128, 128
    parents = [[-1, 0, 0, 1, 3], [-1], list(range(-1, 11)), [-1, 0, 1, 1, 2, 4, 4]]
    R = sum(len(p) for p in parents)
    q = (mx.random.normal((1, R, Hk, Dk)) * 0.1).astype(mx.bfloat16)
    k = (mx.random.normal((1, R, Hk, Dk)) * 0.1).astype(mx.bfloat16)
    v = (mx.random.normal((1, R, Hv, Dv)) * 0.5).astype(mx.bfloat16)
    g = compute_g(mx.random.normal((Hv,)).astype(mx.float32) * 0.1, mx.random.normal((1, R, Hv)).astype(mx.bfloat16),
                  mx.random.normal((Hv,)).astype(mx.bfloat16))
    beta = mx.sigmoid(mx.random.normal((1, R, Hv)).astype(mx.bfloat16))
    states = [(mx.random.normal((1, Hv, Dv, Dk)) * 0.2).astype(mx.float32) for _ in parents]
    try:
        together = stream_gdn.tree(q, k, v, g, beta, states, stream_gdn.TreePlan(parents))
        mx.eval(together)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"metal kernels unavailable: {str(exc).splitlines()[0][:80]}")
    first = 0
    for s, rp in enumerate(parents):
        W = len(rp)
        rows = slice(first, first + W)
        alone = _tree(q[:, rows], k[:, rows], v[:, rows], g[:, rows], beta[:, rows], states[s], rp)
        assert _same(together[:, rows], alone), f"stream {s} differs from its own launch"
        first += W
