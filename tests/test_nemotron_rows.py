"""Nemotron's row-exact kernels for Macs without tensor units: a row's bits do not depend on the other rows or on
how many ride together, and the results agree with an fp32 reference to bf16 rounding. They run on any Mac.

The rounding bound follows MLX's own one-row arithmetic, which the kernels keep: each run of 4 inputs is summed in
bf16 (3 roundings) before it meets the group bias, and outputs are stored in bf16. With u = 2^-8 (bf16's unit
roundoff), a 4-bit matvec is within u |y| + 3u (|x| @ |bias|.T) + fp32 slack of the exact product; MLX's own
kernels are checked against the same bound.
"""

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from tensorfold.kernels.nemotron.lightning.v1 import rows  # noqa: E402

U = 2.0 ** -8          # bf16 unit roundoff
SLACK = 1e-5           # fp32 summation, relative to sum |x w|


def _same(a, b):
    return a.shape == b.shape and bool(mx.all(a.view(mx.uint16) == b.view(mx.uint16)).item())


def _quantized(n, k, seed, group=64, experts=None):
    shape = (n, k) if experts is None else (experts, n, k)
    w = (mx.random.normal(shape, key=mx.random.key(seed)) * 0.05).astype(mx.bfloat16)
    return mx.quantize(w, group_size=group, bits=4)


def _fp32(q, s, b, group=64):
    """(dequantized weights, |bias| per input) in fp32, [.., N, K] each."""

    w = mx.dequantize(q, s, b, group_size=group, bits=4).astype(mx.float32)
    return w, mx.repeat(mx.abs(b.astype(mx.float32)), group, axis=-1)


def _mm(a, b):
    return mx.matmul(a, b, stream=mx.cpu)


def _exact_and_bound(x, w, bias):
    """x [R, K] (fp32 values of bf16 inputs) @ w.T in fp32 on the CPU, and the bf16 rounding bound of MLX's
    arithmetic around it."""

    y = _mm(x, w.T)
    ax = mx.abs(x)
    return y, U * mx.abs(y) + 3 * U * _mm(ax, bias.T) + SLACK * _mm(ax, mx.abs(w).T) + 1e-7


def _check(ours, exact, bound, what):
    err = mx.abs(ours.astype(mx.float32) - exact)
    worst = mx.max(err / bound).item()
    assert worst <= 1.0, f"{what}: error {mx.max(err).item():.3g} is {worst:.2f}x the bf16 rounding bound"


# K: Nemotron's 2,688 (in_proj, qkv, shared up, head), 4,096 (out_proj, o_proj), 3,712 (shared down), 1,856
# (routed down), and short ones that are only a tail or one step and a tail
@pytest.mark.parametrize("k", [2688, 4096, 3712, 1856, 576, 64])
def test_qmv_rows_do_not_depend_on_row_count(k):
    q, s, b = _quantized(48, k, seed=k)
    x = (mx.random.normal((16, k), key=mx.random.key(1)) * 0.5).astype(mx.bfloat16)
    full = rows.qmv(x, q, s, b, 64)
    mx.eval(full)
    for m in range(1, 17):
        assert _same(rows.qmv(x[:m], q, s, b, 64), full[:m]), f"rows 0..{m - 1} changed with {m} rows"
    for i in range(16):
        assert _same(rows.qmv(x[i:i + 1], q, s, b, 64), full[i:i + 1]), f"row {i} alone differs"
    # a window starting elsewhere, and a batched 3-d input
    assert _same(rows.qmv(x[5:12], q, s, b, 64), full[5:12])
    assert _same(rows.qmv(x[None, 2:6], q, s, b, 64), full[None, 2:6])


@pytest.mark.parametrize("group", [32, 64, 128])
@pytest.mark.parametrize("k", [2688, 4096, 3712, 1856])
def test_qmv_agrees_with_fp32(k, group):
    if k % group:
        pytest.skip("K must be a multiple of the group size")
    q, s, b = _quantized(64, k, seed=3 + k, group=group)
    x = (mx.random.normal((5, k), key=mx.random.key(2)) * 0.5).astype(mx.bfloat16)
    w, bias = _fp32(q, s, b, group)
    exact, bound = _exact_and_bound(x.astype(mx.float32), w, bias)
    _check(rows.qmv(x, q, s, b, group), exact, bound, "rows.qmv")
    one = mx.concatenate([mx.quantized_matmul(x[i:i + 1], q, s, b, transpose=True, group_size=group, bits=4)
                          for i in range(5)])
    _check(one, exact, bound, "MLX's one-row kernel")        # the bound holds for MLX's own arithmetic


def test_row_linear_routes_by_row_count():
    linear = nn.QuantizedLinear(2688, 64, bias=False, group_size=64, bits=4)
    q, s, b = _quantized(64, 2688, seed=11)
    linear.weight, linear.scales, linear.biases = q, s, b
    x = (mx.random.normal((40, 2688), key=mx.random.key(4)) * 0.5).astype(mx.bfloat16)
    mlx_out = linear(x)
    mlx_one = linear(x[:1])
    linear.__class__ = rows.RowLinear
    object.__setattr__(linear, "mlx_one_row", False)
    object.__setattr__(linear, "qmv_rows", 32)
    assert _same(linear(x[:7]), rows.qmv(x[:7], q, s, b, 64))
    assert _same(linear(x[:1]), rows.qmv(x[:1], q, s, b, 64))
    one_by_one = mx.concatenate([rows.qmv(x[i:i + 1], q, s, b, 64) for i in range(32)])
    assert _same(linear(x[:32]), one_by_one)             # a shared round past 16 rows: every row its one-row bits
    assert _same(linear(x), mlx_out)                     # past qmv_rows (prompt chunks): MLX's quantized matmul
    object.__setattr__(linear, "mlx_one_row", True)
    assert _same(linear(x[:1]), mlx_one)                 # one row: MLX's kernel when told its bits are the same
    assert _same(linear(x[:2]), rows.qmv(x[:2], q, s, b, 64))


class _Table:
    """A SwitchMLP's two expert tables (as mlx_lm's QuantizedSwitchLinear keeps them)."""

    def __init__(self, experts, dims, hidden, seed):
        self.fc1 = nn.QuantizedLinear(dims, hidden, bias=False, group_size=64, bits=4)
        self.fc2 = nn.QuantizedLinear(hidden, dims, bias=False, group_size=64, bits=4)
        self.fc1.weight, self.fc1.scales, self.fc1.biases = _quantized(hidden, dims, seed, experts=experts)
        self.fc2.weight, self.fc2.scales, self.fc2.biases = _quantized(dims, hidden, seed + 1, experts=experts)


def _routing(count, experts, top_k, seed):
    scores = mx.random.uniform(shape=(count, experts), key=mx.random.key(seed))
    return mx.argsort(-scores, axis=-1)[:, :top_k].astype(mx.uint32)


def test_experts_rows_do_not_depend_on_row_count():
    table = _Table(experts=12, dims=640, hidden=192, seed=5)
    x = (mx.random.normal((16, 640), key=mx.random.key(6)) * 0.5).astype(mx.bfloat16)
    ids = _routing(16, 12, 6, seed=7)                       # rows share experts, as consecutive tokens do
    full = rows.experts(table, x, ids)
    mx.eval(full)
    assert full.shape == (16, 6, 640)
    for m in range(1, 17):
        assert _same(rows.experts(table, x[:m], ids[:m]), full[:m]), f"rows 0..{m - 1} changed with {m} rows"
    for i in range(16):
        assert _same(rows.experts(table, x[i:i + 1], ids[i:i + 1]), full[i:i + 1]), f"row {i} alone differs"
    # the same (row, expert) pair gives the same bits in any slot and beside any other pairs
    swapped = mx.concatenate([ids[:, 3:], ids[:, :3]], axis=1)
    again = rows.experts(table, x, swapped)
    assert _same(again[:, :3], full[:, 3:]) and _same(again[:, 3:], full[:, :3])


def test_expert_groups_list_every_pair_once():
    import numpy as np

    for experts, count, seed in ((12, 16 * 6, 50), (130, 32 * 8, 51), (128, 1, 52), (128, 170 * 6, 53)):
        ids = mx.random.randint(0, experts, (count,), key=mx.random.key(seed)).astype(mx.uint32)
        uids, start, counts, members, used = rows.group(ids, experts)
        mx.eval(uids, start, counts, members, used)
        host = np.array(ids)
        want = sorted(set(host.tolist()))
        n = int(used.item())
        assert n == len(want) and np.array(uids)[:n].tolist() == want
        for u, e in enumerate(want):
            first, many = int(start[u].item()), int(counts[u].item())
            assert np.array(members)[first:first + many].tolist() == np.nonzero(host == e)[0].tolist()


@pytest.mark.parametrize("experts", [16, 130])
def test_grouped_experts_equal_pair_by_pair_at_every_threshold(experts, monkeypatch):
    """The same bits for every pair whatever the grouping: a pair alone in a one-row call, all pairs one by one,
    or grouped by expert, at thresholds that put every window size on either side."""

    table = _Table(experts=experts, dims=640, hidden=192, seed=60 + experts)
    x = (mx.random.normal((32, 640), key=mx.random.key(61)) * 0.5).astype(mx.bfloat16)
    ids = _routing(32, experts, 6, seed=62)                  # 32 rows x 6 slots: many experts picked several times
    alone = mx.concatenate([rows.experts(table, x[i:i + 1], ids[i:i + 1], grouped=False) for i in range(32)])
    mx.eval(alone)
    for m in (1, 2, 3, 4, 5, 8, 12, 16, 24, 32):
        assert _same(rows.experts(table, x[:m], ids[:m], grouped=True), alone[:m]), f"grouped, {m} rows"
        assert _same(rows.experts(table, x[:m], ids[:m], grouped=False), alone[:m]), f"pair by pair, {m} rows"
    for threshold in (1, 2, 4, 8, 16, 32, 33):
        monkeypatch.setattr(rows, "GROUP_ROWS", threshold)
        for m in (1, 4, 7, 8, 9, 16, 31, 32):
            assert _same(rows.experts(table, x[:m], ids[:m]), alone[:m]), f"threshold {threshold}, {m} rows"
        # a window starting elsewhere
        assert _same(rows.experts(table, x[9:30], ids[9:30]), alone[9:30]), f"threshold {threshold}, rows 9-29"


def _switch_mlp_exact_and_bound(table, x, ids):
    """fc2(relu(fc1 x)^2) per (row, slot) in fp32 without intermediate rounding, and the bf16 rounding bound of
    mlx_lm's computation (MLX's arithmetic in each matvec, fc1's output and relu2's output stored in bf16)."""

    w1, b1 = _fp32(table.fc1["weight"], table.fc1["scales"], table.fc1["biases"])
    w2, b2 = _fp32(table.fc2["weight"], table.fc2["scales"], table.fc2["biases"])
    xf = x.astype(mx.float32)
    exact, bound = [], []
    for r in range(x.shape[0]):
        ys, bs = [], []
        for e in [int(v) for v in ids[r].tolist()]:
            h, h_err = _exact_and_bound(xf[r:r + 1], w1[e], b1[e])
            h_err = h_err + U * (mx.abs(h) + h_err)             # fc1's output rounded to bf16
            a = mx.square(mx.maximum(h, 0))
            a_err = (2 * mx.abs(h) + h_err) * h_err + U * mx.square(mx.abs(h) + h_err)   # relu2, rounded
            y, y_err = _exact_and_bound(a, w2[e], b2[e])
            ys.append(y)
            bs.append(y_err + 3 * U * _mm(a_err, b2[e].T) + _mm(a_err, mx.abs(w2[e]).T))
        exact.append(mx.concatenate(ys))
        bound.append(mx.concatenate(bs))
    return mx.stack(exact), mx.stack(bound)


def test_experts_agree_with_fp32_switch_mlp():
    table = _Table(experts=4, dims=2688, hidden=1856, seed=21)      # Nemotron's widths, fewer experts
    x = (mx.random.normal((3, 2688), key=mx.random.key(22)) * 0.5).astype(mx.bfloat16)
    ids = _routing(3, 4, 3, seed=23)
    exact, bound = _switch_mlp_exact_and_bound(table, x, ids)
    _check(rows.experts(table, x, ids), exact, bound, "rows.experts")


def test_experts_agree_with_mlx_lm_switch_mlp():
    switch_layers = pytest.importorskip("mlx_lm.models.switch_layers")
    mlp = switch_layers.SwitchMLP(640, 192, 12, activation=nn.ReLU2())
    table = _Table(experts=12, dims=640, hidden=192, seed=31)
    mlp.fc1 = switch_layers.QuantizedSwitchLinear(640, 192, 12, bias=False, group_size=64, bits=4)
    mlp.fc2 = switch_layers.QuantizedSwitchLinear(192, 640, 12, bias=False, group_size=64, bits=4)
    for mine, theirs in ((table.fc1, mlp.fc1), (table.fc2, mlp.fc2)):
        theirs.weight, theirs.scales, theirs.biases = mine.weight, mine.scales, mine.biases
    x = (mx.random.normal((3, 640), key=mx.random.key(32)) * 0.5).astype(mx.bfloat16)
    ids = _routing(3, 12, 6, seed=33)
    exact, bound = _switch_mlp_exact_and_bound(mlp, x, ids)
    _check(rows.experts(mlp, x, ids), exact, bound, "rows.experts")
    _check(mlp(x, ids), exact, bound, "mlx_lm's SwitchMLP")          # the bound holds for mlx_lm's own path
