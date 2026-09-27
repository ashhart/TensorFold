"""The shared tree attention: a node gets the bits it gets alone (its path committed), in any window or stream."""

import math
import random

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda.kernels import attention as shared  # noqa: E402


def _inputs(w: int, p: int, *, h: int = 24, hk: int = 4, d: int = 256, seed: int = 0):
    gen = torch.Generator(device="cuda").manual_seed(910 + w + p + seed)
    q = torch.randn((w, h, d), generator=gen, device="cuda").bfloat16()
    kn = torch.randn((w, hk, d), generator=gen, device="cuda").bfloat16()
    vn = torch.randn((w, hk, d), generator=gen, device="cuda").bfloat16()
    kc = torch.randn((p + 5, hk, d), generator=gen, device="cuda").bfloat16()    # capacity past the committed keys
    vc = torch.randn((p + 5, hk, d), generator=gen, device="cuda").bfloat16()
    return q, kn, vn, kc, vc


def _attend(q, kn, vn, caches, trees, lengths, scale):
    plan = shared.plan(trees, lengths, q.shape[1] // kn.shape[1], "cuda")
    offs = torch.tensor(shared.offsets(caches, "cuda"), dtype=torch.int64, device="cuda").view(-1, 2)
    return shared.attention(q, kn, vn, offs, plan, scale=scale)


def _path(parents, node):
    rows = []
    while node >= 0:
        rows.append(node)
        node = parents[node]
    return rows[::-1]


def _serial(inputs, parents, p, node, scale):
    """The node alone: its ancestors committed after the cache's first p keys, the node as a one-row window."""

    q, kn, vn, kc, vc = inputs
    rows = _path(parents, node)
    keys = torch.cat((kc[:p], kn[rows[:-1]]), 0).contiguous()
    values = torch.cat((vc[:p], vn[rows[:-1]]), 0).contiguous()
    one = [x[node:node + 1].contiguous() for x in (q, kn, vn)]
    return _attend(*one, [(keys, values)], [[-1]], [keys.shape[0]], scale)[0]


@pytest.mark.parametrize("w,p", [(1, 0), (9, 13), (16, 511), (32, 512), (32, 1003), (128, 513)])
def test_branch_nodes_match_serial_bits(w, p):
    inputs = _inputs(w, p)
    parents = [-1] + [(i - 1) // 2 for i in range(1, w)]
    out = _attend(*inputs[:3], [inputs[3:]], [parents], [p], 1 / 16)
    for node in range(w):
        assert torch.equal(out[node], _serial(inputs, parents, p, node, 1 / 16)), f"node {node} differs"


def test_128_chain_matches_serial_bits_at_chunk_boundaries():
    w, p = 128, 499
    inputs = _inputs(w, p, h=12, hk=2, d=128)
    parents = list(range(-1, w - 1))
    out = _attend(*inputs[:3], [inputs[3:]], [parents], [p], 1 / math.sqrt(128))
    for node in (0, 1, 12, 13, 31, 63, 127):
        assert torch.equal(out[node], _serial(inputs, parents, p, node, 1 / math.sqrt(128))), f"node {node} differs"


def test_long_cache_matches_serial_bits():
    w, p = 128, 20501
    inputs = _inputs(w, p)
    parents = [-1] + [(i - 1) // 2 for i in range(1, w)]
    out = _attend(*inputs[:3], [inputs[3:]], [parents], [p], 1 / 16)
    for node in (0, 3, 8, 63, 127):
        assert torch.equal(out[node], _serial(inputs, parents, p, node, 1 / 16)), f"node {node} differs"


def test_streams_in_one_launch_equal_each_alone():
    rng = random.Random(5)
    shapes = [(rng.randint(1, 16), rng.choice([0, 1, 100, 512, 513, 1300, 2600])) for _ in range(9)]
    parts = [_inputs(w, p, seed=i) for i, (w, p) in enumerate(shapes)]
    trees = [[-1] + [rng.randint(max(0, i - 4), i - 1) for i in range(1, w)] for w, _ in shapes]
    q, kn, vn = (torch.cat([x[j] for x in parts]) for j in range(3))
    together = _attend(q, kn, vn, [x[3:] for x in parts], trees, [p for _, p in shapes], 1 / 16)
    base = 0
    for i, ((w, p), part) in enumerate(zip(shapes, parts)):
        alone = _attend(*part[:3], [part[3:]], [trees[i]], [p], 1 / 16)
        assert torch.equal(together[base:base + w], alone), f"stream {i}"
        base += w


def test_attention_matches_torch_reference():
    w, p = 7, 73
    q, kn, vn, kc, vc = _inputs(w, p)
    parents = [-1, 0, 0, 1, 1, 2, 3]
    out = _attend(q, kn, vn, [(kc, vc)], [parents], [p], 1 / 16)
    for node in range(w):
        rows = _path(parents, node)
        keys = torch.cat((kc[:p], kn[rows]), 0).float().repeat_interleave(6, 1)
        values = torch.cat((vc[:p], vn[rows]), 0).float().repeat_interleave(6, 1)
        scores = torch.einsum("hd,thd->ht", q[node].float(), keys) / 16
        ref = torch.einsum("ht,thd->hd", scores.softmax(-1), values).bfloat16()
        assert (out[node].float() - ref.float()).abs().max() < 0.035
