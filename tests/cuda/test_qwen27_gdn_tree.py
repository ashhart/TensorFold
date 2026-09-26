"""Tree nodes and commit replay use the same recurrence as serial steps."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.families.qwen3_5.cuda import gdn_tree


def _inputs(nodes: int):
    gen = torch.Generator(device="cuda").manual_seed(41 + nodes)
    kh, hv, dv, dk = 2, 6, 8, 128
    q = torch.randn((nodes, kh, dk), generator=gen, device="cuda").bfloat16() * 0.01
    k = torch.randn((nodes, kh, dk), generator=gen, device="cuda").bfloat16() * 0.01
    v = torch.randn((nodes, hv, dv), generator=gen, device="cuda").bfloat16()
    g = torch.rand((nodes, hv), generator=gen, device="cuda") * 0.4 + 0.5
    beta = torch.rand((nodes, hv), generator=gen, device="cuda")
    state = torch.randn((hv, dv, dk), generator=gen, device="cuda") * 0.05
    return q, k, v, g, beta, state


def _step(args, row, state):
    sliced = [x[row:row + 1].contiguous() for x in args[:5]]
    root = torch.tensor([-1], dtype=torch.int32, device="cuda")
    y = gdn_tree.tree(*sliced, state, root)
    rows = torch.zeros(1, dtype=torch.int32, device="cuda")
    count = torch.ones(1, dtype=torch.int32, device="cuda")
    state = gdn_tree.replay(*sliced, state, rows, count)
    return y[0], state


@pytest.mark.parametrize("nodes", [1, 16, 32, 128])
def test_tree_matches_serial_paths(nodes):
    args = _inputs(nodes)
    parents = [-1] + [(i - 1) // 2 for i in range(1, nodes)]
    tree_out = gdn_tree.tree(*args, torch.tensor(parents, dtype=torch.int32, device="cuda"))
    for node in range(nodes):
        path = []
        cur = node
        while cur >= 0:
            path.append(cur)
            cur = parents[cur]
        state = args[5]
        for row in reversed(path):
            out, state = _step(args, row, state)
        assert torch.equal(tree_out[node], out), f"node {node} differs from serial"
        padded = torch.tensor(list(reversed(path)) + [0] * (nodes - len(path)),
                              dtype=torch.int32, device="cuda")
        count = torch.tensor([len(path)], dtype=torch.int32, device="cuda")
        committed = gdn_tree.replay(*args, padded, count)
        assert torch.equal(committed, state), f"node {node} replay differs"


def test_128_node_depth_32_branch_stack():
    args = _inputs(128)
    # A 32-node spine and 96 later siblings at depth 31 force preorder to
    # leave and re-enter the deepest occupied stack slot.
    parents = [-1] + list(range(31)) + [30] * 96
    result = gdn_tree.tree(*args, torch.tensor(parents, dtype=torch.int32, device="cuda"))
    for node in (0, 15, 31, 32, 63, 96, 127):
        path = []
        cur = node
        while cur >= 0:
            path.append(cur)
            cur = parents[cur]
        state = args[5]
        for row in reversed(path):
            expected, state = _step(args, row, state)
        assert torch.equal(result[node], expected), f"deep branch node {node} differs"


def test_128_node_chain_and_zero_step_commit():
    args = _inputs(128)
    parents = torch.arange(-1, 127, dtype=torch.int32, device="cuda")
    full = gdn_tree.tree(*args, parents, chain=True)
    state = args[5]
    for row in range(128):
        out, state = _step(args, row, state)
        assert torch.equal(full[row], out), f"chain row {row} differs"
    rows = torch.arange(128, dtype=torch.int32, device="cuda")
    count = torch.tensor([128], dtype=torch.int32, device="cuda")
    assert torch.equal(gdn_tree.replay(*args, rows, count), state)
    count.zero_()
    assert torch.equal(gdn_tree.replay(*args, rows, count), args[5])


def test_checkpoint_head_layout():
    gen = torch.Generator(device="cuda").manual_seed(73)
    nodes, kh, hv, dv, dk = 4, 16, 48, 128, 128
    q = torch.randn(nodes, kh, dk, generator=gen, device="cuda").bfloat16() * 0.01
    k = torch.randn(nodes, kh, dk, generator=gen, device="cuda").bfloat16() * 0.01
    v = torch.randn(nodes, hv, dv, generator=gen, device="cuda").bfloat16()
    g = torch.rand(nodes, hv, generator=gen, device="cuda") * 0.4 + 0.5
    beta = torch.rand(nodes, hv, generator=gen, device="cuda")
    state0 = torch.randn(hv, dv, dk, generator=gen, device="cuda") * 0.05
    args = q, k, v, g, beta, state0
    parents = torch.tensor([-1, 0, 0, 1], dtype=torch.int32, device="cuda")
    result = gdn_tree.tree(*args, parents)
    state = state0
    for row in (0, 1, 3):
        out, state = _step(args, row, state)
        assert torch.equal(result[row], out)
    rows = torch.tensor([0, 1, 3, 0], dtype=torch.int32, device="cuda")
    count = torch.tensor([3], dtype=torch.int32, device="cuda")
    assert torch.equal(gdn_tree.replay(*args, rows, count), state)


def test_128_branch_checkpoint_head_layout():
    gen = torch.Generator(device="cuda").manual_seed(173)
    nodes, kh, hv, dv, dk = 128, 16, 48, 128, 128
    q = torch.randn(nodes, kh, dk, generator=gen, device="cuda").bfloat16() * 0.01
    k = torch.randn(nodes, kh, dk, generator=gen, device="cuda").bfloat16() * 0.01
    v = torch.randn(nodes, hv, dv, generator=gen, device="cuda").bfloat16()
    g = torch.rand(nodes, hv, generator=gen, device="cuda") * 0.4 + 0.5
    beta = torch.rand(nodes, hv, generator=gen, device="cuda")
    state0 = torch.randn(hv, dv, dk, generator=gen, device="cuda") * 0.05
    args = q, k, v, g, beta, state0
    parents = [-1] + [(i - 1) // 2 for i in range(1, nodes)]
    result = gdn_tree.tree(*args, torch.tensor(parents, dtype=torch.int32, device="cuda"))
    for node in (0, 1, 7, 31, 64, 127):
        path = []
        cur = node
        while cur >= 0:
            path.append(cur)
            cur = parents[cur]
        state = state0
        for row in reversed(path):
            expected, state = _step(args, row, state)
        assert torch.equal(result[node], expected), f"checkpoint head node {node} differs"


@pytest.mark.parametrize("keep", [1, 3, 12])
def test_replay_many_matches_replay_per_layer(keep):
    """One launch for every layer gives the same states, bit for bit, as one replay per layer."""

    layers = [_inputs(12) for _ in range(5)]
    for i, args in enumerate(layers):          # distinct inputs per layer
        for x in args:
            x.mul_(1.0 + 0.1 * i)
    rows = torch.tensor([0, 1, 3, 7, 2, 5, 9, 4, 6, 8, 10, 11][:keep] + [0] * (12 - keep),
                        dtype=torch.int32, device="cuda")
    count = torch.tensor([keep], dtype=torch.int32, device="cuda")
    single = [gdn_tree.replay(*args, rows, count) for args in layers]
    many = gdn_tree.replay_many(*[[args[j] for args in layers] for j in range(6)], rows, count)
    assert many.shape == (5, *single[0].shape)
    for i in range(5):
        assert torch.equal(many[i], single[i]), f"layer {i}"
