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


def _attend(q, kn, vn, caches, trees, lengths, scale, kv8=False):
    plan = shared.plan(trees, lengths, q.shape[1] // kn.shape[1], "cuda")
    offs = torch.tensor(shared.offsets(caches, "cuda"), dtype=torch.int64, device="cuda").view(-1, 2)
    return shared.attention(q, kn, vn, offs, plan, scale=scale, kv8=kv8)


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


@pytest.mark.parametrize("w,p,context", [(1, 0, 4096), (4, 1300, 4096), (3, 2047, 2048), (4, 4000, 8192)])
def test_a_padded_plan_gives_the_exact_plans_bits(w, p, context):
    """A graph's plan (items for every chunk below ``context``, the stream's keys set later) equals the exact plan."""

    inputs = _inputs(w, p)
    q, kn, vn, kc, vc = inputs
    parents = list(range(-1, w - 1))
    want = _attend(q, kn, vn, [(kc, vc)], [parents], [p], 1 / 16)
    flat, items, chunks = shared.padded_host(parents, context, q.shape[1] // kn.shape[1])
    flat[w + 2], flat[w + 3] = p, -(-(p + w) // shared.CHUNK)
    plan = shared.from_packed(torch.tensor(flat, dtype=torch.int32, device="cuda"), 1, w, items, chunks)
    offs = torch.tensor(shared.offsets([(kc, vc)], "cuda"), dtype=torch.int64, device="cuda").view(-1, 2)
    assert torch.equal(shared.attention(q, kn, vn, offs, plan, scale=1 / 16), want)


def test_strided_node_values_equal_contiguous_ones():
    """A fused [k | v] projection's value half, read in place, gives the bits of its contiguous copy."""

    w, p, hk, d = 9, 513, 4, 256
    q, kn, vn, kc, vc = _inputs(w, p)
    parents = [-1] + [(i - 1) // 2 for i in range(1, w)]
    kv = torch.zeros((w, 2 * hk * d), device="cuda", dtype=torch.bfloat16)
    kv[:, hk * d:] = vn.reshape(w, hk * d)
    view = kv[:, hk * d:].reshape(w, hk, d)
    assert not view.is_contiguous()
    assert torch.equal(_attend(q, kn, view, [(kc, vc)], [parents], [p], 1 / 16),
                       _attend(q, kn, vn, [(kc, vc)], [parents], [p], 1 / 16))


@pytest.mark.skipif(not getattr(torch.version, "hip", None), reason="ROCm's WMMA tree attention")
@pytest.mark.parametrize("w,p", [(12, 20501), (128, 1300), (1, 3000)])
def test_compute_waves_never_change_the_bits(monkeypatch, w, p):
    """A block of 1, 2, 5 or 8 compute waves (``TF_ROCM_TREE_CW``), with loader waves or without
    (``TF_ROCM_TREE_PIPE``), gives every pair the same bits, alone or in a launch of several streams."""

    inputs = _inputs(w, p)
    other = _inputs(3, 700, seed=1)
    parents = [-1] + list(range(w - 1))
    scale = 1 / math.sqrt(256)
    outs = []
    for cw, pipe in [(cw, pipe) for cw in ("1", "2", "5", "8") for pipe in ("0", "1")]:
        monkeypatch.setenv("TF_ROCM_TREE_CW", cw)
        monkeypatch.setenv("TF_ROCM_TREE_PIPE", pipe)
        alone = _attend(*inputs[:3], [inputs[3:]], [parents], [p], scale)
        both = _attend(torch.cat((inputs[0], other[0])), torch.cat((inputs[1], other[1])),
                       torch.cat((inputs[2], other[2])), [inputs[3:], other[3:]], [parents, [-1, 0, 1]], [p, 700],
                       scale)
        assert torch.equal(both[:w].view(torch.int16), alone.view(torch.int16))
        outs.append(both.view(torch.int16))
    assert all(torch.equal(outs[0], x) for x in outs[1:])



def _fp8(inputs):
    """The inputs as the FP8 cache's forward sees them: every key and value rounded (the window's own too), and the
    caches also as packed rows (``kv8.ROW8`` bytes a row)."""

    from tensorfold.cuda.kernels import kv8

    q, kn, vn, kc, vc = inputs
    packed = [kv8.reference_pack(t) for t in (kc, vc)]
    rounded = [kv8.unpack(t) for t in packed]
    kn, vn = (kv8.unpack(kv8.reference_pack(t)) for t in (kn, vn))
    return (q, kn, vn, *rounded), (q, kn, vn, *packed)


@pytest.mark.skipif(not getattr(torch.version, "hip", None), reason="ROCm's WMMA tree attention reads packed FP8")
@pytest.mark.parametrize("pipe", ["0", "1"])
@pytest.mark.parametrize("w,p", [(1, 0), (9, 13), (12, 511), (32, 1003), (128, 513), (12, 20501)])
def test_packed_fp8_caches_give_their_rounded_values_bits(monkeypatch, w, p, pipe):
    """Packed FP8 rows widen in shared memory to the bf16 values ``kv8.unpack`` gives, so a tree over packed caches
    gets the bits it gets over bf16 caches holding those values, in either schedule; nodes still match serial."""

    monkeypatch.setenv("TF_ROCM_TREE_PIPE", pipe)
    rounded, packed = _fp8(_inputs(w, p))
    parents = [-1] + [(i - 1) // 2 for i in range(1, w)]
    want = _attend(*rounded[:3], [rounded[3:]], [parents], [p], 1 / 16)
    got = _attend(*packed[:3], [packed[3:]], [parents], [p], 1 / 16, kv8=True)
    assert torch.equal(got, want)
    for node in {0, w // 2, w - 1}:
        assert torch.equal(got[node], _serial(rounded, parents, p, node, 1 / 16)), f"node {node} differs"


@pytest.mark.skipif(not getattr(torch.version, "hip", None), reason="ROCm's WMMA tree attention reads packed FP8")
def test_packed_fp8_streams_in_one_launch_equal_each_alone():
    rng = random.Random(9)
    shapes = [(rng.randint(1, 16), rng.choice([0, 1, 100, 512, 513, 1300, 2600])) for _ in range(6)]
    parts = [_fp8(_inputs(w, p, seed=i))[1] for i, (w, p) in enumerate(shapes)]
    trees = [[-1] + [rng.randint(max(0, i - 4), i - 1) for i in range(1, w)] for w, _ in shapes]
    q, kn, vn = (torch.cat([x[j] for x in parts]) for j in range(3))
    together = _attend(q, kn, vn, [x[3:] for x in parts], trees, [p for _, p in shapes], 1 / 16, kv8=True)
    base = 0
    for i, ((w, p), part) in enumerate(zip(shapes, parts)):
        alone = _attend(*part[:3], [part[3:]], [trees[i]], [p], 1 / 16, kv8=True)
        assert torch.equal(together[base:base + w], alone), f"stream {i}"
        base += w
